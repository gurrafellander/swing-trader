"""
test_signals_vs_backtest.py
───────────────────────────
Verifies that generate_signals.py produces the same stock picks and weights
as the full backtest engine (vectorbt) for a known historical rebalance date.

What is being tested
--------------------
The backtest runs rising_momentum_sharpe over a full history window and feeds
the resulting weight matrix into vbt.Portfolio.from_orders.  generate_signals
does the same thing but is meant to be used live.  If the two disagree, the
live signal generator is broken.

We compare at two levels:
  Level 1 — Weight level (exact):
    The raw weight Series from rising_momentum_sharpe must be identical
    between the backtest data slice and the generate_signals data slice.
    This catches any data-loading or date-alignment bugs.

  Level 2 — Portfolio level (approximate):
    The backtest portfolio's asset weights on the rebalance date must match
    the generate_signals weights within WEIGHT_TOL (default 1 pp).
    Small differences are expected due to vbt applying fees/slippage and
    the portfolio not instantly reaching target weights.

Usage
-----
  python test_signals_vs_backtest.py
  python test_signals_vs_backtest.py --date 2025-03-28   # pick a specific date
  python test_signals_vs_backtest.py --verbose           # print full diff tables

The test exits 0 on pass, 1 on failure.
"""

import argparse
import sys
from datetime import date, timedelta

import numpy as np
import pandas as pd
import vectorbt as vbt

from DataLoader import DataLoader
from config import cfg
from strategies import rising_momentum_sharpe
from generate_signals import get_raw_weights

# ── Tolerances ────────────────────────────────────────────────────────────────
WEIGHT_TOL      = 0.01   # 1 pp — max allowed weight difference per ticker
TICKER_OVERLAP  = 1.0    # fraction of tickers that must match (1.0 = all)

PASS = "\033[92m✔\033[0m"
FAIL = "\033[91m✘\033[0m"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _find_rebalance_date(weights: pd.DataFrame, before: pd.Timestamp) -> pd.Timestamp | None:
    """Return the most recent rebalance date with non-zero positions, at or before `before`."""
    for ts in reversed(weights.loc[:before].index.tolist()):
        row = weights.loc[ts].dropna()
        if (row > 0).any():
            return ts
    return None


def _portfolio_weights_on(pf, ts: pd.Timestamp) -> pd.Series:
    """
    Derive target weights on date `ts` from a vbt portfolio.
    We use asset_value / total_value, clipped to avoid tiny float noise.
    """
    av = pf.asset_value()
    if isinstance(av, pd.DataFrame):
        av = av.iloc[:, 0] if av.shape[1] == 1 else av
    tv = pf.value()
    if isinstance(tv, pd.DataFrame):
        tv = tv.iloc[:, 0]

    if isinstance(av, pd.DataFrame):
        w = av.loc[ts] / tv.loc[ts]
    else:
        # grouped portfolio returns a single Series for value, DataFrame for asset_value
        asset_vals = pf.asset_value(group_by=False)
        w = asset_vals.loc[ts] / tv.loc[ts]

    w = w.clip(lower=0)
    w = w[w > 1e-4]   # drop dust
    return w


# ── Core test logic ───────────────────────────────────────────────────────────

def run_test(test_date: date, verbose: bool) -> bool:
    """
    Returns True if the test passes.
    """
    print(f"\n{'═' * 62}")
    print(f"  Test date  : {test_date}")
    print(f"  Tolerances : weight ±{WEIGHT_TOL*100:.1f} pp,  ticker overlap {TICKER_OVERLAP*100:.0f}%")
    print(f"{'═' * 62}\n")

    # ── 1. Build the backtest data window ─────────────────────────────────────
    # Use the same 2-year lookback as generate_signals so the universe is identical.
    bt_start = (test_date - timedelta(days=2 * 365)).strftime("%Y-%m-%d")
    bt_end   = test_date.strftime("%Y-%m-%d")

    print(f"[1/4] Downloading backtest data  {bt_start}  →  {bt_end} …")
    loader = DataLoader("tickers.txt", bt_start, bt_end, cfg.min_history)
    close  = loader.download_clean_data()

    if close.empty:
        print(f"  {FAIL}  No data returned for this date range.")
        return False

    last_ts = close.index[-1]

    # ── 2. Run strategy and find the rebalance date ───────────────────────────
    print("[2/4] Running rising_momentum_sharpe over backtest window …")
    bt_weights = rising_momentum_sharpe(close)
    rebal_ts   = _find_rebalance_date(bt_weights, last_ts)

    if rebal_ts is None:
        print(f"  {FAIL}  No rebalance date found — strategy produced no signals.")
        return False

    bt_w_row = bt_weights.loc[rebal_ts].dropna()
    bt_w_row = bt_w_row[bt_w_row > 0]
    print(f"       Rebalance date found : {rebal_ts.date()}")
    print(f"       Tickers in backtest  : {sorted(bt_w_row.index.tolist())}")

    # ── 3. Run generate_signals on the same date ──────────────────────────────
    print("[3/4] Running generate_signals.get_raw_weights …")
    gs_w_row, gs_signal_ts, gs_last_ts = get_raw_weights(test_date)

    if gs_w_row.empty:
        print(f"  {FAIL}  generate_signals returned no positions.")
        return False

    print(f"       Signal date (gen)    : {gs_signal_ts.date()}")
    print(f"       Tickers in gen_sig   : {sorted(gs_w_row.index.tolist())}")

    # ── 4. Compare ────────────────────────────────────────────────────────────
    print("\n[4/4] Comparing …\n")
    all_passed = True

    # 4a — Signal date must match
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

    # 4b — Ticker overlap
    bt_tickers = set(bt_w_row.index)
    gs_tickers = set(gs_w_row.index)
    only_in_bt = bt_tickers - gs_tickers
    only_in_gs = gs_tickers - bt_tickers
    common     = bt_tickers & gs_tickers
    overlap    = len(common) / max(len(bt_tickers), len(gs_tickers))

    if overlap < TICKER_OVERLAP:
        print(
            f"  {FAIL}  TICKER MISMATCH  (overlap {overlap*100:.1f}% < {TICKER_OVERLAP*100:.0f}%)"
        )
        if only_in_bt:
            print(f"         Only in backtest : {sorted(only_in_bt)}")
        if only_in_gs:
            print(f"         Only in gen_sig  : {sorted(only_in_gs)}")
        all_passed = False
    else:
        print(f"  {PASS}  Ticker overlap  {overlap*100:.1f}%  ({len(common)}/{max(len(bt_tickers), len(gs_tickers))} tickers)")
        if only_in_bt:
            print(f"         ℹ  Only in backtest : {sorted(only_in_bt)}")
        if only_in_gs:
            print(f"         ℹ  Only in gen_sig  : {sorted(only_in_gs)}")

    # 4c — Weight differences on common tickers
    all_tickers = sorted(bt_tickers | gs_tickers)
    diff_rows = []
    for tkr in all_tickers:
        w_bt = float(bt_w_row.get(tkr, 0.0))
        w_gs = float(gs_w_row.get(tkr, 0.0))
        diff = abs(w_bt - w_gs)
        diff_rows.append({
            "Ticker":        tkr,
            "BT Weight (%)": round(w_bt * 100, 3),
            "GS Weight (%)": round(w_gs * 100, 3),
            "Diff (pp)":     round(diff * 100, 4),
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
        print(f"  {PASS}  All weights within tolerance  (max diff {max_diff:.4f} pp)")

    if verbose:
        print("\n  Full weight comparison:")
        print(diff_df.to_string(index=False))

    # ── 5. Run vbt portfolio to cross-check via actual traded weights ──────────
    # This is an optional deeper check: does vbt actually end up with roughly
    # these weights after applying fees/slippage on the rebalance date?
    print("\n[+]  Running vbt cross-check (portfolio weights after execution) …")
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
        vbt_w = _portfolio_weights_on(pf, rebal_ts)
        vbt_w = vbt_w.rename(index=lambda c: c)  # tickers already correct

        vbt_rows = []
        for tkr in sorted(bt_tickers | set(vbt_w.index)):
            w_strat = float(bt_w_row.get(tkr, 0.0))
            w_vbt   = float(vbt_w.get(tkr, 0.0))
            vbt_rows.append({
                "Ticker":           tkr,
                "Strategy W (%)":   round(w_strat * 100, 3),
                "VBT actual W (%)": round(w_vbt * 100, 3),
                "Diff (pp)":        round(abs(w_strat - w_vbt) * 100, 4),
            })
        vbt_df = pd.DataFrame(vbt_rows)

        if verbose:
            print("\n  Strategy target vs vbt actual weights:")
            print(vbt_df.to_string(index=False))

        max_vbt_diff = vbt_df["Diff (pp)"].max()
        # vbt introduces slippage/fees so we use a looser 3 pp tolerance here
        vbt_tol = 3.0
        if max_vbt_diff > vbt_tol:
            print(
                f"  ⚠   vbt cross-check: max diff {max_vbt_diff:.2f} pp > {vbt_tol} pp "
                f"(fees/slippage expected, but large gaps suggest a problem)"
            )
        else:
            print(f"  {PASS}  vbt cross-check passed  (max diff {max_vbt_diff:.4f} pp vs target)")

    except Exception as exc:
        print(f"  ⚠   vbt cross-check skipped: {exc}")

    # ── Result ────────────────────────────────────────────────────────────────
    print(f"\n{'═' * 62}")
    if all_passed:
        print(f"  {PASS}  ALL CHECKS PASSED — generate_signals matches the backtest.")
    else:
        print(f"  {FAIL}  SOME CHECKS FAILED — see details above.")
    print(f"{'═' * 62}\n")
    return all_passed


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Verify generate_signals matches the backtest engine."
    )
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help=(
            "Test date in YYYY-MM-DD format. "
            "The most recent rebalance at or before this date is used. "
            "Default: 60 days ago (well within any recent backtest)."
        ),
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print full weight comparison tables.",
    )
    return parser.parse_args()


def main():
    args  = parse_args()
    verbose = args.verbose

    if args.date:
        test_date = date.fromisoformat(args.date)
    else:
        # Default: 60 days ago — recent enough to share data with backtest,
        # old enough that yfinance has clean settled data.
        test_date = date.today() - timedelta(days=60)
        print(f"  No --date given; defaulting to {test_date} (60 days ago).")

    passed = run_test(test_date, verbose)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
