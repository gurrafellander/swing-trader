# Swing Trading Backtesting Engine

## Overview

Modular RSI-based swing trading system using vectorbt.

Supports:

- Multi-stock universe
- Maximum position limit
- Ranking logic
- Equal-weight allocation
- Backtesting
- Daily signal generation

---

## Setup

1. Create virtual environment
2. Install requirements:

pip install -r requirements.txt

---

## Add Your Universe

Create `tickers.txt`:

AAPL
MSFT
NVDA
TSLA
...

One ticker per line.

---

## Run Backtest

python backtest.py

Outputs:

- Full performance stats
- Interactive equity and trade plots

---

## Generate Today's Signals

python generate_signals.py

Outputs:

- BUY list
- SELL list
- Saves `today_signals.txt`

---

## Modify Strategy

Edit `strategies.py`.

Each strategy must return:

entries (DataFrame)
exits (DataFrame)

Shape must match close prices.

---

## Modify Parameters

Edit `config.py`.

All allocation, RSI parameters, and limits are centralized there.

---

## Extend

You can:

- Add new strategy functions
- Loop strategies in backtest.py
- Add walk-forward validation
- Add Telegram/email notifications
- Connect to a PWA frontend
