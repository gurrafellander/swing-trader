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

import numpy as np
import pandas as pd
import pandas_ta as ta

from DataLoader import DataLoader
from config import cfg
from strategies import rising_momentum_sharpe


# ── CLI args ──────────────────────────────────────────────────────────────────


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate rising_momentum_sharpe signals."
    )
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
    history_end = signal_date.strftime("%Y-%m-%d")
    loader = DataLoader("tickers.txt", history_start, history_end, cfg.min_history)
    close = loader.download_clean_data()
    last_date = close.index[-1]
    weights = rising_momentum_sharpe(close)
    for ts in reversed(weights.loc[:last_date].index.tolist()):
        row = weights.loc[ts].dropna()
        row = row[row > 0]
        if not row.empty:
            return row, ts, last_date
    return pd.Series(dtype=float), None, last_date


def compute_position_metrics(
    close: pd.DataFrame,
    tickers: list[str],
    signal_ts: pd.Timestamp,
) -> dict[str, dict]:
    """
    Compute per-ticker metrics over the same vol_lookback window the strategy used.

    Returns a dict keyed by ticker with:
      sharpe       — annualised Sharpe (rf=0)
      ann_return   — annualised return (%)
      ann_vol      — annualised volatility (%)
      max_dd       — max drawdown over the window (%)
      roc_short    — ROC over cfg.roc_short_lookback days (%)
      roc_long     — ROC over cfg.roc_long_lookback  days (%)
      acceleration — roc_short − roc_long (pp)
    """
    loc_i = close.index.get_loc(signal_ts)
    start_i = max(0, loc_i - cfg.vol_lookback)
    window = close.iloc[start_i : loc_i + 1][tickers].dropna(axis=1)

    daily_ret = window.pct_change().dropna()
    TRADING_DAYS = 252

    metrics = {}
    for tkr in tickers:
        if tkr not in daily_ret.columns:
            metrics[tkr] = {}
            continue

        r = daily_ret[tkr]
        p = window[tkr]

        ann_ret = float((1 + r.mean()) ** TRADING_DAYS - 1) * 100
        ann_vol = float(r.std() * np.sqrt(TRADING_DAYS)) * 100
        sharpe = (
            float(r.mean() / r.std() * np.sqrt(TRADING_DAYS)) if r.std() > 0 else 0.0
        )

        # Max drawdown over the window
        cum = (1 + r).cumprod()
        max_dd = float((cum / cum.cummax() - 1).min()) * 100

        # ROC values on the signal date (same formula as the strategy)
        roc_s_series = ta.roc(p, length=cfg.roc_short_lookback)
        roc_l_series = ta.roc(p, length=cfg.roc_long_lookback)
        roc_short = (
            float(roc_s_series.iloc[-1]) if roc_s_series is not None else float("nan")
        )
        roc_long = (
            float(roc_l_series.iloc[-1]) if roc_l_series is not None else float("nan")
        )
        accel = (
            roc_short - roc_long
            if not (np.isnan(roc_short) or np.isnan(roc_long))
            else float("nan")
        )

        metrics[tkr] = {
            "sharpe": round(sharpe, 2),
            "ann_return": round(ann_ret, 1),
            "ann_vol": round(ann_vol, 1),
            "max_dd": round(max_dd, 1),
            "roc_short": round(roc_short, 2),
            "roc_long": round(roc_long, 2),
            "acceleration": round(accel, 2),
        }

    return metrics


def generate_signals(signal_date: date, portfolio_value: float) -> pd.DataFrame:
    """
    Run rising_momentum_sharpe up to signal_date and return a DataFrame
    with whole-share allocations for the given portfolio_value.
    """

    # We need enough history before signal_date for the strategy's lookback windows.
    # 2 years of buffer is generous for any momentum/Sharpe lookback.
    history_start = (signal_date - timedelta(days=2 * 365)).strftime("%Y-%m-%d")
    history_end = signal_date.strftime("%Y-%m-%d")

    print(f"\nDownloading data  {history_start}  →  {history_end} …")
    loader = DataLoader("tickers.txt", history_start, history_end, cfg.min_history)
    close = loader.download_clean_data()

    if close.empty:
        raise ValueError(
            "No price data returned — check tickers.txt and the date range."
        )

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
    candidates = weights.loc[:last_date]  # everything up to our date
    for ts in reversed(candidates.index.tolist()):
        row = candidates.loc[ts].dropna()
        row = row[row > 0]
        if not row.empty:
            signal_ts = ts
            w_row = row
            break

    if w_row.empty:
        raise ValueError("Strategy returned no positions in the entire history window.")

    if signal_ts != last_date:
        print(
            f"  ℹ  Last trading day ({last_date.date()}) is not a rebalance day — "
            f"using most recent rebalance signals from {signal_ts.date()}"
        )

    # Prices on the *last available* trading day (current prices, not rebalance-day prices)
    print(
        f"  Signals from rebalance: {signal_ts.date()}  |  Prices from: {last_date.date()}"
    )
    prices: pd.Series = close.loc[last_date, w_row.index]

    # ── Allocation ──────────────────────────────────────────────────────────
    target_sek = w_row * portfolio_value  # ideal SEK per ticker
    shares_exact = target_sek / prices  # fractional shares
    shares_whole = shares_exact.apply(  # floor to whole shares
        lambda x: int(x) if x >= 1 else 0
    )
    actual_sek = shares_whole * prices  # what we actually spend
    leftover_pct = (w_row - actual_sek / portfolio_value) * 100  # slippage vs weight

    total_spent = actual_sek.sum()
    cash_left = portfolio_value - total_spent

    # ── Build output table ───────────────────────────────────────────────────
    out = (
        pd.DataFrame(
            {
                "Ticker": w_row.index,
                "Weight (%)": (w_row * 100).round(2),
                "Target SEK": target_sek.round(2),
                "Price (SEK)": prices.round(2),
                "Shares": shares_whole,
                "Actual SEK": actual_sek.round(2),
                "Δ Weight (%)": leftover_pct.round(
                    3
                ),  # positive → we bought slightly less
            }
        )
        .sort_values("Weight (%)", ascending=False)
        .reset_index(drop=True)
    )

    # Drop tickers where we can't afford even 1 share
    zero_shares = out[out["Shares"] == 0]
    if not zero_shares.empty:
        print(
            f"  ⚠  Dropping {len(zero_shares)} ticker(s) with price > allocated budget "
            f"(cannot buy whole share):\n"
            + "\n".join(
                f"     {r.Ticker}  price={r['Price (SEK)']:.2f}  budget={r['Target SEK']:.2f}"
                for _, r in zero_shares.iterrows()
            )
        )
        out = out[out["Shares"] > 0].reset_index(drop=True)

    # ── Per-position metrics ──────────────────────────────────────────────────
    metrics = compute_position_metrics(close, out["Ticker"].tolist(), signal_ts)
    out["Sharpe"] = out["Ticker"].map(
        lambda t: metrics.get(t, {}).get("sharpe", float("nan"))
    )
    out["Ann Ret (%)"] = out["Ticker"].map(
        lambda t: metrics.get(t, {}).get("ann_return", float("nan"))
    )
    out["Ann Vol (%)"] = out["Ticker"].map(
        lambda t: metrics.get(t, {}).get("ann_vol", float("nan"))
    )
    out["Max DD (%)"] = out["Ticker"].map(
        lambda t: metrics.get(t, {}).get("max_dd", float("nan"))
    )
    out["ROC Short"] = out["Ticker"].map(
        lambda t: metrics.get(t, {}).get("roc_short", float("nan"))
    )
    out["ROC Long"] = out["Ticker"].map(
        lambda t: metrics.get(t, {}).get("roc_long", float("nan"))
    )
    out["Accel (pp)"] = out["Ticker"].map(
        lambda t: metrics.get(t, {}).get("acceleration", float("nan"))
    )

    # ── Summary footer row ────────────────────────────────────────────────────
    metric_cols = [
        "Sharpe",
        "Ann Ret (%)",
        "Ann Vol (%)",
        "Max DD (%)",
        "ROC Short",
        "ROC Long",
        "Accel (pp)",
    ]
    summary = pd.DataFrame(
        [
            {
                "Ticker": "── TOTAL / CASH ──",
                "Weight (%)": out["Weight (%)"].sum().round(2),
                "Target SEK": out["Target SEK"].sum().round(2),
                "Price (SEK)": "",
                "Shares": "",
                "Actual SEK": total_spent.round(2),
                "Δ Weight (%)": "",
                **{c: "" for c in metric_cols},
            }
        ]
    )
    summary_cash = pd.DataFrame(
        [
            {
                "Ticker": "Cash (leftover)",
                "Weight (%)": round(cash_left / portfolio_value * 100, 2),
                "Target SEK": "",
                "Price (SEK)": "",
                "Shares": "",
                "Actual SEK": round(cash_left, 2),
                "Δ Weight (%)": "",
                **{c: "" for c in metric_cols},
            }
        ]
    )
    out_full = pd.concat([out, summary, summary_cash], ignore_index=True)

    return out_full, signal_ts.date(), cash_left, close, metrics


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
        raw = (
            input("Enter portfolio value in SEK: ")
            .strip()
            .replace(",", "")
            .replace(" ", "")
        )
        portfolio_value = float(raw)

    if portfolio_value <= 0:
        raise ValueError("Portfolio value must be positive.")

    print(f"\n{'═' * 55}")
    print(f"  Strategy : Rising Momentum + Sharpe")
    print(f"  Date     : {signal_date}")
    print(f"  Portfolio: {portfolio_value:,.0f} SEK")
    print(f"{'═' * 55}")

    try:
        out_full, actual_date, cash_left, _, _ = generate_signals(signal_date, portfolio_value)
    except ValueError as e:
        print(f"  No signals generated: {e}")
        return

    # ── Save CSV ──────────────────────────────────────────────────────────────
    filename = f"signals_{actual_date}.csv"
    out_full.to_csv(filename, index=False)

    # ── Print to terminal ─────────────────────────────────────────────────────
    positions = out_full[
        ~out_full["Ticker"].str.startswith("──")
        & (out_full["Ticker"] != "Cash (leftover)")
    ]
    print(f"\n  {len(positions)} position(s) to buy:\n")
    print(out_full.to_string(index=False))
    print(f"\n  Cash left uninvested: {cash_left:,.2f} SEK")
    print(f"\n  Saved → {filename}")


if __name__ == "__main__":
    main()
