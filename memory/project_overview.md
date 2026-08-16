---
name: project-overview
description: swing-trader project — what it is, what the two apps do, and key design decisions
metadata:
  type: project
---

Streamlit stock screener (`app.py`) + Flask signal generator dashboard (`dashboard.py`) for a Swedish/global momentum strategy.

- `app.py` — Streamlit UI with 3 views: Single Stock, Screener, Portfolio
- `dashboard.py` — Flask UI for live signal generation using `rising_momentum_sharpe` strategy
- `tickers.txt` — tradeable universe used by both backtest and screener
- `portfolio-cache/assets.json` — persisted portfolio watchlist

**Why:** User actively monitors and trades Swedish stocks (OMXS) plus global names.
**How to apply:** Keep tickers.txt clean (no benchmark indices) so backtest results aren't polluted. Load benchmark (^OMX) separately in app.py.
