# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Setup
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Launch the dashboard (Streamlit, http://localhost:8501)
streamlit run app.py
```

There is no test suite, linter, or formatter configured in this repo.

## Architecture

`app.py` is a single-page Streamlit app with three views (Single Stock, Screener, Portfolio) selected via a sidebar radio. Price data is loaded once per session and cached with `@st.cache_data`.

### Data flow

`tickers.txt` → `DataLoader.download_clean_data()` → close price DataFrame (cached) → indicators / screener filters / portfolio analysis, all computed on demand per view.

`DataLoader` (`DataLoader.py`) fetches individual tickers via plain `yfinance.download`, forward-fills gaps, and drops tickers with fewer than `cfg.min_history` rows. Downloaded history is cached to `price_cache/close_prices.parquet` and incrementally extended (backward, new tickers, tail) on each run rather than re-downloaded from scratch. The benchmark is `^OMX` (OMXS30), downloaded separately via `_download_raw` so it never pollutes the tradeable universe — adding it to `tickers.txt` would pull it into screener/portfolio calculations as if it were a tradeable asset.

Per-market timestamps from Yahoo Finance (Swedish stocks post at 22:00 UTC, US at 04:00 UTC next day, etc.) are normalized to Stockholm calendar dates in `_normalize()` before tickers are concatenated — otherwise `pd.concat` on individually downloaded tickers produces multiple rows per trading day.

### `config.py` — centralized singleton

All parameters live in `Config` and are imported as `cfg`. Never hardcode parameters in app code.

### `indicators.py`

Stateless indicator functions (`compute_ma`, `compute_bollinger`, `compute_rsi`, `compute_roc`, `compute_roc_accelerating`) plus an `Indicators` class that bundles the precomputed set used across views. `ROC_LOOKBACK` / `ROC_ACCEL_DAYS` are the shared rate-of-change window constants.

### `app.py` — views

- **Single Stock** — browse the universe or search an arbitrary Yahoo Finance ticker (`tab_browse` / `tab_search`), with a price chart overlaying the selected indicators. Searching a new ticker and adding it to the portfolio appends it to `tickers.txt` and clears the data cache so it's included in the universe on next app start.
- **Screener** — snapshot filter across the whole universe on the latest bar (see `TODO.txt` for the historical-query gap).
- **Portfolio** — delegates to `portfolio_view.render_portfolio_view`.

### `portfolio_view.py` — portfolio watchlist + risk view

Persists the watchlist to `portfolio-cache/assets.json` (`add_to_portfolio`, `remove_from_portfolio`, `remove_entire_portfolio`). Runs Monte Carlo risk simulation (via `portfolio_analysis.run_monte_carlo`) and renders it as a fan chart, correlation heatmap, return histogram, and risk-contribution breakdown.

### `portfolio_analysis.py`

Pure-numeric portfolio math: `fit_parameters` (drift/covariance from historical returns), `run_monte_carlo` (simulate terminal portfolio value distribution via correlated GBM with Cholesky-decomposed shocks), plus supporting metric/risk-contribution helpers. No Streamlit or I/O — safe to unit test in isolation.

### Adding a new view or indicator

1. New indicator: add a function to `indicators.py`, referencing `cfg` for any tunable parameter.
2. New view: add a branch in `app.py`'s navigation section, or a new module following the `portfolio_view.py` pattern (render function takes the cached `close` DataFrame and any pre-loaded indicators).

## Backtesting

The vectorbt-based backtesting engine (strategies, signal generation, the old Flask dashboard) has been split out into a separate, standalone project outside this repo. `DataLoader.py` here has no vectorbt dependency — it uses plain `yfinance` only.
