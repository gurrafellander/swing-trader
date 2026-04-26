# backtest.py
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import vectorbt as vbt

from config import (
    START_DATE,
    END_DATE,
    MIN_HISTORY,
    INITIAL_CASH,
    FEES,
    SLIPPAGE,
    POSITION_SIZE,
    SIZE_TYPE,
)
from strategies import momentum_with_regime


def load_universe():
    with open("tickers.txt") as f:
        return [line.strip() for line in f if line.strip()]


def download_clean_data(tickers):
    print("Downloading data individually...")
    close_dict = {}
    for ticker in tickers:
        try:
            data = vbt.YFData.download(
                ticker, start=START_DATE, end=END_DATE, interval="1d")
            close = data.get("Close")
            if close is None or close.empty:
                print(f"Skipping {ticker} (no data)")
                continue
            close_dict[ticker] = close
        except Exception as e:
            print(f"Skipping {ticker}: {e}")

    if not close_dict:
        raise ValueError("No valid tickers downloaded.")

    close_df = pd.concat(close_dict.values(), axis=1)
    close_df.columns = list(close_dict.keys())
    close_df = close_df.sort_index().ffill().dropna(how="all")
    valid_cols = close_df.count() >= MIN_HISTORY
    close_df = close_df.loc[:, valid_cols]
    print(f"Remaining tickers after MIN_HISTORY filter: {close_df.shape[1]}")
    return close_df


def download_benchmark(index, ticker="^OMX"):
    print(f"Downloading benchmark ({ticker})...")
    try:
        data = vbt.YFData.download(ticker, start=START_DATE, end=END_DATE, interval="1d")
        bench = data.get("Close")
        bench = bench.reindex(index).ffill()
        # squeeze in case it comes back as DataFrame
        if isinstance(bench, pd.DataFrame):
            bench = bench.iloc[:, 0]
        return bench
    except Exception as e:
        print(f"Could not download benchmark: {e}")
        return None


def run_backtest():
    tickers = load_universe()
    close   = download_clean_data(tickers)
    print(f"Final dataset shape: {close.shape}")

    # Benchmark used for regime filter AND chart comparison
    benchmark = download_benchmark(close.index, ticker="^OMX")
    if benchmark is None:
        # Fallback: equal-weight universe as proxy
        print("WARNING: no benchmark downloaded, using universe mean as regime proxy")
        benchmark = close.mean(axis=1)

    # --- Strategy: cross-sectional momentum with vol weighting + regime filter ---
    target_weights = momentum_with_regime(close, benchmark)

    # --- Backtest with from_orders + targetpercent ---
    # momentum_with_regime returns a full weight matrix so from_orders is correct:
    # weights only change on monthly rebalance dates → very few trades.
    pf = vbt.Portfolio.from_orders(
        close,
        size=target_weights,
        size_type="targetpercent",
        init_cash=INITIAL_CASH,
        fees=FEES,
        slippage=SLIPPAGE,
        group_by=True,
        cash_sharing=True,
        call_seq="auto",          # sells before buys → frees cash correctly
        freq="1D",
    )

    print("\n===== BACKTEST RESULTS =====\n")
    print(pf.stats())

    # --- Equity curve plot ---
    portfolio_value = pf.value()
    if isinstance(portfolio_value, pd.DataFrame):
        portfolio_value = portfolio_value.iloc[:, 0]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=portfolio_value.index, y=portfolio_value.values,
        name="Momentum Strategy",
        line=dict(color="royalblue", width=2),
    ))

    if benchmark is not None:
        bench_norm = (benchmark / benchmark.dropna().iloc[0]) * INITIAL_CASH
        fig.add_trace(go.Scatter(
            x=bench_norm.index, y=bench_norm.values,
            name="OMXS30 (buy & hold)",
            line=dict(color="gray", width=1.5, dash="dash"),
        ))

    fig.update_layout(
        title="Portfolio Value Over Time — Momentum + Regime Filter",
        xaxis_title="Date",
        yaxis_title="Portfolio Value (SEK)",
        hovermode="x unified",
        template="plotly_white",
    )
    fig.write_html("portfolio_value.html")
    print("\nPlot saved to portfolio_value.html")
    return pf


if __name__ == "__main__":
    run_backtest()
