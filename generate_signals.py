"""
generate_signals.py
───────────────────
Runs rising_momentum_sharpe on the latest universe data up to a chosen date
and outputs a buy list with whole-share allocations.

Usage
-----
  python generate_signals.py                          # uses today, prompts for portfolio value
  python generate_signals.py --date 2024-05-09       # specific date
  python generate_signals.py --value 500000          # portfolio value in SEK
  python generate_signals.py --date 2024-05-09 --value 500000

Output
------
  signals_YYYY-MM-DD.csv   — one row per position with shares, cost, leftover cash
"""

import argparse
from datetime import date, timedelta

import pandas as pd

from DataLoader import DataLoader
from config import cfg
from strategies import rising_momentum_sharpe


# ── CLI args ──────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Generate rising_momentum_sharpe signals.")
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Signal date in YYYY-MM-DD format (default: today)",
    )
    parser.add_argument(
        "--value",
        type=float,
        default=None,
        help="Total portfolio value in SEK (default: prompted interactively)",
    )
    return parser.parse_args()


# ── Core logic ─────────────────────────────────────────────────────────────────

def get_raw_weights(signal_date: date) -> tuple[pd.Series, pd.Timestamp, pd.Timestamp]:
    """
    Lower-level helper used by tests.
    Returns (w_row, signal_ts, last_date) where:
      w_row      — ticker → weight (sums to ~1) from the most recent rebalance
      signal_ts  — the rebalance date the weights came from
      last_date  — the last available trading day (price reference)
    """
    history_start = (signal_date - timedelta(days=2 * 365)).strftime("%Y-%m-%d")
    history_end   = signal_date.strftime("%Y-%m-%d")
    loader = DataLoader("tickers.txt", history_start, history_end, cfg.min_history)
    close  = loader.download_clean_data()
    last_date = close.index[-1]
    weights = rising_momentum_sharpe(close)
    for ts in reversed(weights.loc[:last_date].index.tolist()):
        row = weights.loc[ts].dropna()
        row = row[row > 0]
        if not row.empty:
            return row, ts, last_date
    return pd.Series(dtype=float), None, last_date


def generate_signals(signal_date: date, portfolio_value: float) -> pd.DataFrame:
    """
    Run rising_momentum_sharpe up to signal_date and return a DataFrame
    with whole-share allocations for the given portfolio_value.
    """

    # We need enough history before signal_date for the strategy's lookback windows.
    # 2 years of buffer is generous for any momentum/Sharpe lookback.
    history_start = (signal_date - timedelta(days=2 * 365)).strftime("%Y-%m-%d")
    history_end   = signal_date.strftime("%Y-%m-%d")

    print(f"\nDownloading data  {history_start}  →  {history_end} …")
    loader = DataLoader("tickers.txt", history_start, history_end, cfg.min_history)
    close  = loader.download_clean_data()

    if close.empty:
        raise ValueError("No price data returned — check tickers.txt and the date range.")

    # Snap to the last available trading day on or before signal_date
    last_date = close.index[-1]
    if last_date.date() < signal_date:
        print(
            f"  ⚠  {signal_date} is not a trading day — "
            f"using last available date: {last_date.date()}"
        )
    signal_date = last_date.date()

    print(f"  Computing signals for  {signal_date} …")
    weights: pd.DataFrame = rising_momentum_sharpe(close)

    # The strategy only writes weights on rebalance days; all other rows are NaN.
    # Walk backwards from last_date to find the most recent rebalance day that
    # actually produced at least one non-zero position.
    signal_ts = None
    w_row = pd.Series(dtype=float)
    candidates = weights.loc[:last_date]          # everything up to our date
    for ts in reversed(candidates.index.tolist()):
        row = candidates.loc[ts].dropna()
        row = row[row > 0]
        if not row.empty:
            signal_ts = ts
            w_row = row
            break

    if w_row.empty:
        print("  ⚠  Strategy returned no positions in the entire history window.")
        return pd.DataFrame()

    if signal_ts != last_date:
        print(
            f"  ℹ  Last trading day ({last_date.date()}) is not a rebalance day — "
            f"using most recent rebalance signals from {signal_ts.date()}"
        )

    # Prices on the *last available* trading day (current prices, not rebalance-day prices)
    print(f"  Signals from rebalance: {signal_ts.date()}  |  Prices from: {last_date.date()}")
    prices: pd.Series = close.loc[last_date, w_row.index]

    # ── Allocation ──────────────────────────────────────────────────────────
    target_sek   = w_row * portfolio_value          # ideal SEK per ticker
    shares_exact = target_sek / prices              # fractional shares
    shares_whole = shares_exact.apply(             # floor to whole shares
        lambda x: int(x) if x >= 1 else 0
    )
    actual_sek   = shares_whole * prices            # what we actually spend
    leftover_pct = (w_row - actual_sek / portfolio_value) * 100  # slippage vs weight

    total_spent  = actual_sek.sum()
    cash_left    = portfolio_value - total_spent

    # ── Build output table ───────────────────────────────────────────────────
    out = pd.DataFrame(
        {
            "Ticker":       w_row.index,
            "Weight (%)":   (w_row * 100).round(2),
            "Target SEK":   target_sek.round(2),
            "Price (SEK)":  prices.round(2),
            "Shares":       shares_whole,
            "Actual SEK":   actual_sek.round(2),
            "Δ Weight (%)": leftover_pct.round(3),  # positive → we bought slightly less
        }
    ).sort_values("Weight (%)", ascending=False).reset_index(drop=True)

    # Drop tickers where we can't afford even 1 share
    zero_shares = out[out["Shares"] == 0]
    if not zero_shares.empty:
        print(
            f"  ⚠  Dropping {len(zero_shares)} ticker(s) with price > allocated budget "
            f"(cannot buy whole share):\n"
            + "\n".join(f"     {r.Ticker}  price={r['Price (SEK)']:.2f}  budget={r['Target SEK']:.2f}"
                        for _, r in zero_shares.iterrows())
        )
        out = out[out["Shares"] > 0].reset_index(drop=True)

    # ── Summary footer row ────────────────────────────────────────────────────
    summary = pd.DataFrame(
        [{
            "Ticker":       "── TOTAL / CASH ──",
            "Weight (%)":   out["Weight (%)"].sum().round(2),
            "Target SEK":   out["Target SEK"].sum().round(2),
            "Price (SEK)":  "",
            "Shares":       "",
            "Actual SEK":   total_spent.round(2),
            "Δ Weight (%)": "",
        }]
    )
    summary_cash = pd.DataFrame(
        [{
            "Ticker":       "Cash (leftover)",
            "Weight (%)":   round(cash_left / portfolio_value * 100, 2),
            "Target SEK":   "",
            "Price (SEK)":  "",
            "Shares":       "",
            "Actual SEK":   round(cash_left, 2),
            "Δ Weight (%)": "",
        }]
    )
    out_full = pd.concat([out, summary, summary_cash], ignore_index=True)

    return out_full, signal_ts.date(), cash_left


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # Resolve date
    if args.date:
        try:
            signal_date = date.fromisoformat(args.date)
        except ValueError:
            raise ValueError(f"Invalid date format: '{args.date}'. Use YYYY-MM-DD.")
    else:
        signal_date = date.today()

    # Resolve portfolio value
    if args.value is not None:
        portfolio_value = args.value
    else:
        raw = input("Enter portfolio value in SEK: ").strip().replace(",", "").replace(" ", "")
        portfolio_value = float(raw)

    if portfolio_value <= 0:
        raise ValueError("Portfolio value must be positive.")

    print(f"\n{'═' * 55}")
    print(f"  Strategy : Rising Momentum + Sharpe")
    print(f"  Date     : {signal_date}")
    print(f"  Portfolio: {portfolio_value:,.0f} SEK")
    print(f"{'═' * 55}")

    result = generate_signals(signal_date, portfolio_value)

    if isinstance(result, tuple):
        out_full, actual_date, cash_left = result
    else:
        print("No signals generated.")
        return

    # ── Save CSV ──────────────────────────────────────────────────────────────
    filename = f"signals_{actual_date}.csv"
    out_full.to_csv(filename, index=False)

    # ── Print to terminal ─────────────────────────────────────────────────────
    positions = out_full[~out_full["Ticker"].str.startswith("──") & (out_full["Ticker"] != "Cash (leftover)")]
    print(f"\n  {len(positions)} position(s) to buy:\n")
    print(out_full.to_string(index=False))
    print(f"\n  Cash left uninvested: {cash_left:,.2f} SEK")
    print(f"\n  Saved → {filename}")


if __name__ == "__main__":
    main()
