---
name: arch-notes
description: Key architecture decisions in app.py — benchmark loading, ticker search, portfolio persistence
metadata:
  type: project
---

## ^OMX benchmark handling
`^OMX` is NOT in tickers.txt — it's downloaded via a separate `load_benchmark()` cached function in app.py. This keeps the backtest universe clean (you can't trade an index).

**Why:** Adding ^OMX to tickers.txt would include it in the strategy portfolio optimisation.

## Ticker search / add-to-universe flow
- `_fetch_ticker(sym)` — cached, downloads any Yahoo Finance symbol on demand
- When user adds a searched ticker to portfolio → also appended to tickers.txt + `st.cache_data.clear()`
- Next app start picks it up in full universe with min_history filter applied

## Indicator computation
- `_adhoc_indicators(price_series)` — computes MA/BB/RSI/ROC for a single series (used for ^OMX + searched tickers, not pre-computed in the `Indicators` batch)

## Portfolio view OMX comparison
- `st.toggle("Show OMXS30 benchmark")` overlays yellow line
- Both portfolio and OMX rebased to 100 at first common date
- Uses `omx.reindex(port_close.index).ffill()` then divide by first valid value
