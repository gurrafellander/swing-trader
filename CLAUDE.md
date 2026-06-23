# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Setup
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Run full backtest (all strategies, saves HTML charts + CSV stats)
python backtest.py

# Generate live signals (interactive: prompts for date and portfolio value)
python generate_signals.py
python generate_signals.py --date 2025-06-01 --value 500000

# Launch web dashboard (Flask, http://localhost:5000)
python dashboard.py

# Run regression test (verifies signal generator matches backtest engine)
python test_signals_vs_backtest.py
python test_signals_vs_backtest.py --date 2025-03-28 --verbose
```

## Architecture

The system has two operational modes: **backtesting** (historical performance analysis) and **live signal generation** (current portfolio allocation).

### Data flow

`tickers.txt` → `DataLoader` → close price DataFrame → strategy function → target-weight DataFrame → `vbt.Portfolio.from_orders`

`DataLoader` fetches individual tickers via `vbt.YFData.download` (Yahoo Finance), forward-fills gaps, and drops tickers with fewer than `cfg.min_history` rows. The benchmark is `^OMX` (OMXS30).

### Weight matrix convention

Every strategy returns a DataFrame with the same shape as `close`. The values follow the vectorbt `targetpercent` convention:

- `NaN` — hold (no change this bar)
- `0.0` — close the position
- `float` — set allocation to this fraction of total portfolio

Strategies only write non-NaN values on rebalance days (monthly by default, `cfg.rebalance_freq = "MS"`). All other rows stay `NaN` so vectorbt treats them as hold.

### `config.py` — centralized singleton

All parameters live in `Config` and are imported as `cfg`. Strategy files, backtest, and signal generator all import `from config import cfg`. Never hardcode parameters in strategy code.

### `strategies.py` — strategy library

Six strategies, all returning target-weight DataFrames:

| Function | Type | Regime filter |
|---|---|---|
| `rsi_mean_reversion` | mean reversion | no |
| `momentum_with_regime` | cross-sectional momentum + inverse-vol weighting | yes (^OMX 200-day MA) |
| `momentum_sharpe_optimised` | momentum pre-filter → Sharpe optimisation | yes |
| `rising_momentum_sharpe` | accelerating momentum → Sharpe optimisation | no (always invested) |
| `ma_crossover` | fast/slow MA cross | no |
| `bollinger_mean_reversion` | lower BB entry, mid BB exit | no |

Shared helpers: `_rebalance_mask`, `_make_weights`, `_apply_ranking`, `_sharpe_weights`, `apply_stop_loss`.

`_sharpe_weights` runs mean-variance optimisation (SLSQP, long-only) then hard-caps to `cfg.top_n` positions by zeroing the smallest weights.

### `generate_signals.py` — live portfolio allocation

Only runs `rising_momentum_sharpe`. Downloads 2 years of history to satisfy lookback windows, then walks backwards from the last trading day to find the most recent rebalance that produced positions. Outputs whole-share counts given a portfolio value in SEK.

`get_raw_weights(signal_date)` is the lower-level helper used by both the CLI and the regression test.

### `dashboard.py` — Flask web UI

Wraps `generate_signals` in a single-page app. Two API endpoints:

- `GET /api/config` — returns current `cfg` values for the sidebar
- `POST /api/signals` — calls `generate_signals`, returns JSON positions

Note: `api_signals` downloads data twice (once in `generate_signals`, once again to derive `price_date`). This is a known inefficiency flagged in TODO.txt.

### `test_signals_vs_backtest.py` — regression test

Verifies that `generate_signals.get_raw_weights` agrees with the full backtest on: (1) rebalance date, (2) ticker set, (3) weight values (tolerance: 1 pp). Also checks that vectorbt received the unmodified weight matrix. Default test date is 60 days ago.

### Adding a new strategy

1. Add a function to `strategies.py` that returns a target-weight DataFrame.
2. Reference any new parameters via `cfg` (add them to `Config` in `config.py`).
3. Register it in the `strategies` list in `backtest.py`.
4. If the strategy should be live-tradeable, wire it into `generate_signals.py`.
