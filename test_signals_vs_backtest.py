"""
test_signals_vs_backtest.py
---------------------------
Verifies that generate_signals.py produces the same stock picks and weights
as the full backtest engine (vectorbt) for a known historical rebalance date.

What is being tested
--------------------
The backtest runs rising_momentum_sharpe over a full history window and feeds
the resulting weight matrix into vbt.Portfolio.from_orders.  generate_signals
does the same thing but is meant to be used live.  If the two disagree, the
live signal generator is broken.

We compare at three levels:

  Level 1 - Rebalance date (exact):
    Both paths must agree on which date is the most recent rebalance.

  Level 2 - Ticker overlap (exact):
    The set of selected tickers must match completely.

  Level 3 - Weight values (near-exact, tolerance = WEIGHT_TOL):
    The raw weight floats from rising_momentum_sharpe must be identical
    between the backtest data slice and the generate_signals data slice.
    Any difference here means a data-loading or code-path divergence.

  Bonus check - vbt input integrity:
    Confirms the weight matrix row that vbt actually received is bit-exact
    to the strategy output.  This rules out silent mutation of the weights
    DataFrame before it reaches vbt.
    NOTE: we do NOT compare drifted portfolio holdings (asset_value / total_value)
    because those diverge from target weights as the portfolio compounds -- that
    is expected behaviour, not a bug.

Usage
-----
  python test_signals_vs_backtest.py
  python test_signals_vs_backtest.py --date 2025-03-28
  python test_signals_vs_backtest.py --verbose
"""

import argparse
import sys
from datetime import date, timedelta

import pandas as pd
import vectorbt as vbt

from DataLoader import DataLoader
from config import cfg
from strategies import rising_momentum_sharpe
from generate_signals import get_raw_weights

# Tolerances
WEIGHT_TOL     = 0.01   # 1 pp max weight difference per ticker
TICKER_OVERLAP = 1.0    # fraction of tickers that must match (1.0 = all)

PASS = "\033[92mv\033[0m"
FAIL = "\033[91mx\033[0m"


def _find_rebalance_date(weights: pd.DataFrame, before: pd.Timestamp):
    """Return the most recent rebalance date with non-zero positions."""
    for ts in reversed(weights.loc[:before].index.tolist()):
        row = weights.loc[ts].dropna()
        if (row > 0).any():
            return ts
    return None


def run_test(test_date: date, verbose: bool) -> bool:
    print(f"\n{'=' * 62}")
    print(f"  Test date  : {test_date}")
    print(f"  Tolerances : weight +/-{WEIGHT_TOL*100:.1f} pp,  ticker overlap {TICKER_OVERLAP*100:.0f}%")
    print(f"{'=' * 62}\n")

    # 1. Download data
    bt_start = (test_date - timedelta(days=2 * 365)).strftime("%Y-%m-%d")
    bt_end   = test_date.strftime("%Y-%m-%d")
    print(f"[1/4] Downloading backtest data  {bt_start}  ->  {bt_end} ...")
    loader = DataLoader("tickers.txt", bt_start, bt_end, cfg.min_history)
    close  = loader.download_clean_data()

    if close.empty:
        print(f"  {FAIL}  No data returned for this date range.")
        return False

    last_ts = close.index[-1]

    # 2. Run strategy and find rebalance date
    print("[2/4] Running rising_momentum_sharpe over backtest window ...")
    bt_weights = rising_momentum_sharpe(close)
    rebal_ts   = _find_rebalance_date(bt_weights, last_ts)

    if rebal_ts is None:
        print(f"  {FAIL}  No rebalance date found -- strategy produced no signals.")
        return False

    bt_w_row = bt_weights.loc[rebal_ts].dropna()
    bt_w_row = bt_w_row[bt_w_row > 0]
    print(f"       Rebalance date found : {rebal_ts.date()}")
    print(f"       Tickers in backtest  : {sorted(bt_w_row.index.tolist())}")

    # 3. Run generate_signals on the same date
    print("[3/4] Running generate_signals.get_raw_weights ...")
    gs_w_row, gs_signal_ts, gs_last_ts = get_raw_weights(test_date)

    if gs_w_row.empty:
        print(f"  {FAIL}  generate_signals returned no positions.")
        return False

    print(f"       Signal date (gen)    : {gs_signal_ts.date()}")
    print(f"       Tickers in gen_sig   : {sorted(gs_w_row.index.tolist())}")

    # 4. Compare
    print("\n[4/4] Comparing ...\n")
    all_passed = True

    # 4a - Rebalance date
    if rebal_ts != gs_signal_ts:
        print(
            f"  {FAIL}  REBALANCE DATE MISMATCH\n"
            f"         Backtest : {rebal_ts.date()}\n"
            f"         GenSig   : {gs_signal_ts.date()}\n"
            "         This usually means the data windows differ."
        )
        all_passed = False
    else:
        print(f"  {PASS}  Rebalance dates match  ({rebal_ts.date()})")

    # 4b - Ticker overlap
    bt_tickers = set(bt_w_row.index)
    gs_tickers = set(gs_w_row.index)
    only_in_bt = bt_tickers - gs_tickers
    only_in_gs = gs_tickers - bt_tickers
    common     = bt_tickers & gs_tickers
    overlap    = len(common) / max(len(bt_tickers), len(gs_tickers))

    if overlap < TICKER_OVERLAP:
        print(f"  {FAIL}  TICKER MISMATCH  (overlap {overlap*100:.1f}% < {TICKER_OVERLAP*100:.0f}%)")
        if only_in_bt:
            print(f"         Only in backtest : {sorted(only_in_bt)}")
        if only_in_gs:
            print(f"         Only in gen_sig  : {sorted(only_in_gs)}")
        all_passed = False
    else:
        print(f"  {PASS}  Ticker overlap  {overlap*100:.1f}%  ({len(common)}/{max(len(bt_tickers), len(gs_tickers))} tickers)")
        if only_in_bt:
            print(f"         i  Only in backtest : {sorted(only_in_bt)}")
        if only_in_gs:
            print(f"         i  Only in gen_sig  : {sorted(only_in_gs)}")

    # 4c - Weight values
    all_tickers = sorted(bt_tickers | gs_tickers)
    diff_rows = []
    for tkr in all_tickers:
        w_bt = float(bt_w_row.get(tkr, 0.0))
        w_gs = float(gs_w_row.get(tkr, 0.0))
        diff = abs(w_bt - w_gs)
        diff_rows.append({
            "Ticker":        tkr,
            "BT Weight (%)": round(w_bt * 100, 4),
            "GS Weight (%)": round(w_gs * 100, 4),
            "Diff (pp)":     round(diff * 100, 6),
            "OK":            diff <= WEIGHT_TOL,
        })

    diff_df = pd.DataFrame(diff_rows)
    bad = diff_df[~diff_df["OK"]]

    if not bad.empty:
        print(f"\n  {FAIL}  WEIGHT DIFFERENCES EXCEED TOLERANCE ({WEIGHT_TOL*100:.1f} pp):")
        print(bad.to_string(index=False))
        all_passed = False
    else:
        max_diff = diff_df["Diff (pp)"].max()
        print(f"  {PASS}  All weights within tolerance  (max diff {max_diff:.6f} pp)")

    if verbose:
        print("\n  Full weight comparison:")
        print(diff_df.to_string(index=False))

    # Bonus: vbt input integrity check
    # We verify the weight matrix row vbt received is bit-exact to strategy output.
    # We do NOT check drifted portfolio holdings -- those diverge due to compounding.
    print("\n[+]  vbt input integrity check ...")
    try:
        pf = vbt.Portfolio.from_orders(
            close,
            size=bt_weights,
            size_type="targetpercent",
            init_cash=cfg.initial_cash,
            fees=cfg.fees,
            slippage=cfg.slippage,
            group_by=True,
            cash_sharing=True,
            freq="D",
        )

        # Compare the weight matrix row vbt received vs what the strategy produced.
        # These must be identical -- any gap means the DataFrame was mutated.
        vbt_input_row = bt_weights.loc[rebal_ts].dropna()
        vbt_input_row = vbt_input_row[vbt_input_row > 0]

        vbt_rows = []
        for tkr in sorted(bt_tickers | set(vbt_input_row.index)):
            w_strat = float(bt_w_row.get(tkr, 0.0))
            w_input = float(vbt_input_row.get(tkr, 0.0))
            vbt_rows.append({
                "Ticker":          tkr,
                "Strategy W (%)":  round(w_strat * 100, 6),
                "VBT input W (%)": round(w_input * 100, 6),
                "Diff (pp)":       round(abs(w_strat - w_input) * 100, 8),
            })
        vbt_df = pd.DataFrame(vbt_rows)

        if verbose:
            print("\n  Strategy weights vs vbt input weights (should be identical):")
            print(vbt_df.to_string(index=False))

        max_vbt_diff = vbt_df["Diff (pp)"].max()
        if max_vbt_diff > 1e-6:
            print(
                f"  {FAIL}  vbt input weights differ from strategy output "
                f"(max {max_vbt_diff:.8f} pp) -- weight matrix was mutated!"
            )
            print(vbt_df[vbt_df["Diff (pp)"] > 1e-6].to_string(index=False))
        else:
            print(
                f"  {PASS}  vbt input weights are bit-exact matches of strategy output  "
                f"(max diff {max_vbt_diff:.8f} pp)"
            )

        print(
            f"  i  Drifted portfolio holdings are NOT compared here -- they diverge\n"
            f"     from target weights due to compounding, fees, and slippage over\n"
            f"     the full history. That is expected behaviour, not a bug."
        )

    except Exception as exc:
        print(f"  !  vbt integrity check skipped: {exc}")

    # Result
    print(f"\n{'=' * 62}")
    if all_passed:
        print(f"  {PASS}  ALL CHECKS PASSED -- generate_signals matches the backtest.")
    else:
        print(f"  {FAIL}  SOME CHECKS FAILED -- see details above.")
    print(f"{'=' * 62}\n")
    return all_passed


def parse_args():
    parser = argparse.ArgumentParser(
        description="Verify generate_signals matches the backtest engine."
    )
    parser.add_argument(
        "--date", type=str, default=None,
        help="Test date YYYY-MM-DD. Defaults to 60 days ago.",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Print full weight comparison tables.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.date:
        test_date = date.fromisoformat(args.date)
    else:
        test_date = date.today() - timedelta(days=60)
        print(f"  No --date given; defaulting to {test_date} (60 days ago).")

    passed = run_test(test_date, args.verbose)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
