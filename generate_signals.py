# generate_signals.py

import vectorbt as vbt
import pandas as pd
from datetime import datetime

from config import (
    START_DATE,
    END_DATE,
    MAX_POSITIONS,
)

from strategies import rsi_mean_reversion


def load_universe():
    with open("tickers.txt") as f:
        tickers = [line.strip() for line in f if line.strip()]
    return tickers


def generate_today_signals():
    tickers = load_universe()

    data = vbt.YFData.download(
        tickers,
        start=START_DATE,
        end=END_DATE,
        interval="1d",
    )

    close = data.get("Close")

    entries, exits = rsi_mean_reversion(close)

    today = close.index[-1]

    today_entries = entries.loc[today]
    today_exits = exits.loc[today]

    buy_list = today_entries[today_entries].index.tolist()
    sell_list = today_exits[today_exits].index.tolist()

    print(f"\n===== SIGNALS FOR {today.date()} =====\n")

    print("BUY:")
    for s in buy_list:
        print(s)

    print("\nSELL:")
    for s in sell_list:
        print(s)

    # Optional: save to file
    with open("today_signals.txt", "w") as f:
        f.write(f"Signals for {today.date()}\n\n")
        f.write("BUY:\n")
        for s in buy_list:
            f.write(f"{s}\n")
        f.write("\nSELL:\n")
        for s in sell_list:
            f.write(f"{s}\n")


if __name__ == "__main__":
    generate_today_signals()
