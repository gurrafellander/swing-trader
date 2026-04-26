# backtest.py

import pandas as pd
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
from strategies import rsi_mean_reversion


def load_universe():
    with open("tickers.txt") as f:
        return [line.strip() for line in f if line.strip()]


def download_clean_data(tickers):
    print("Downloading data individually...")

    close_dict = {}

    for ticker in tickers:
        try:
            data = vbt.YFData.download(
                ticker,
                start=START_DATE,
                end=END_DATE,
                interval="1d",
            )

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
    close_df.columns = close_dict.keys()
    close_df = close_df.sort_index()

    # Forward fill
    close_df = close_df.ffill()

    # Drop rows where all NaN
    close_df = close_df.dropna(how="all")

    # Minimum history filter
    valid_cols = close_df.count() >= MIN_HISTORY
    close_df = close_df.loc[:, valid_cols]

    print(f"Remaining tickers after MIN_HISTORY filter: {close_df.shape[1]}")

    return close_df


def download_spy(index):
    print("Downloading SPY benchmark...")

    spy_data = vbt.YFData.download("SPY", start=START_DATE, end=END_DATE, interval="1d")

    spy_close = spy_data.get("Close")

    # Align with universe index
    spy_close = spy_close.reindex(index)
    spy_close = spy_close.ffill()

    return spy_close


def run_backtest():
    tickers = load_universe()

    close = download_clean_data(tickers)

    print(f"Final dataset shape: {close.shape}")

    # spy_close = download_spy(close.index)

    entries, exits = rsi_mean_reversion(close)

    pf = vbt.Portfolio.from_signals(
        close,
        entries,
        exits,
        init_cash=INITIAL_CASH,
        fees=FEES,
        slippage=SLIPPAGE,
        size=POSITION_SIZE,
        size_type=SIZE_TYPE,
        direction="longonly",
    )

    print("\n===== BACKTEST RESULTS =====\n")
    print(pf.stats())

    pf.plot().show()

    return pf


if __name__ == "__main__":
    run_backtest()
