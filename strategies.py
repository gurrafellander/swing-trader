# strategies.py

import vectorbt as vbt
import pandas as pd

from config import (
    RSI_WINDOW,
    RSI_ENTRY,
    RSI_EXIT,
    MAX_POSITIONS,
    USE_RANKING,
    MOMENTUM_LOOKBACK,
    VOL_LOOKBACK,
    TOP_N,
    SPY_MA,
    REBALANCE_FREQ,
)


# ---------------------------------------------------
# 1. RSI Mean Reversion
# ---------------------------------------------------


def rsi_mean_reversion(close: pd.DataFrame):

    rsi = vbt.RSI.run(close, window=RSI_WINDOW)

    raw_entries = rsi.rsi < RSI_ENTRY
    exits = rsi.rsi > RSI_EXIT

    if USE_RANKING:
        score = 100 - rsi.rsi
        rank = score.rank(axis=1, ascending=False)
        entries = raw_entries & (rank <= MAX_POSITIONS)
    else:
        entries = raw_entries

    return entries, exits


# ---------------------------------------------------
# 2. Moving Average Crossover (Trend Following)
# ---------------------------------------------------


def ma_crossover(close: pd.DataFrame, fast_window=20, slow_window=50):

    fast_ma = vbt.MA.run(close, fast_window)
    slow_ma = vbt.MA.run(close, slow_window)

    raw_entries = fast_ma.ma_crossed_above(slow_ma)
    exits = fast_ma.ma_crossed_below(slow_ma)

    if USE_RANKING:
        # Rank strongest trend distance
        score = (fast_ma.ma - slow_ma.ma) / slow_ma.ma
        rank = score.rank(axis=1, ascending=False)
        entries = raw_entries & (rank <= MAX_POSITIONS)
    else:
        entries = raw_entries

    return entries, exits


# ---------------------------------------------------
# 3. Bollinger Band Mean Reversion
# ---------------------------------------------------


def bollinger_mean_reversion(close: pd.DataFrame, window=20, std=2):

    bb = vbt.BBANDS.run(close, window=window, alpha=std)

    lower = bb.lower
    upper = bb.upper
    mid = bb.middle

    raw_entries = close < lower
    exits = close > mid

    if USE_RANKING:
        # Strongest deviation below band
        score = (lower - close) / close
        rank = score.rank(axis=1, ascending=False)
        entries = raw_entries & (rank <= MAX_POSITIONS)
    else:
        entries = raw_entries

    return entries, exits


def momentum_pullback(close, spy_close):

    rsi = vbt.RSI.run(close, window=3)

    ma200 = vbt.MA.run(close, 200).ma
    ma50 = vbt.MA.run(close, 50).ma

    spy_ma200 = vbt.MA.run(spy_close, 200).ma

    market_filter = spy_close > spy_ma200

    trend_filter = close > ma200

    # 6 month momentum
    momentum = close.pct_change(126)

    rank = momentum.rank(axis=1, ascending=False)

    top_momentum = rank <= int(close.shape[1] * 0.3)

    entries = (
        market_filter.values.reshape(-1, 1)
        & trend_filter
        & top_momentum
        & (rsi.rsi < 20)
    )

    exits = (rsi.rsi > 60) | (close < ma50)

    return entries, exits


def _rebalance_mask(index):
    """Create a mask for rebalance timing (weekly/monthly/etc)."""
    s = pd.Series(index=index, data=1)
    return s.resample(REBALANCE_FREQ).first().reindex(index).notna()


# ----------------------------------------------------------
# Strategy 1
# Cross-Sectional Momentum (Top N performers)
# ----------------------------------------------------------


def momentum_rank_strategy(close):
    """
    Rank stocks by momentum and hold TOP_N.
    """

    momentum = close / close.shift(MOMENTUM_LOOKBACK) - 1

    rank = momentum.rank(axis=1, ascending=False)

    rebalance = _rebalance_mask(close.index)

    long_candidates = rank <= TOP_N

    entries = long_candidates & rebalance.values[:, None]

    exits = (~long_candidates) & rebalance.values[:, None]

    return entries, exits


# ----------------------------------------------------------
# Strategy 2
# Volatility Adjusted Momentum + Market Filter
# ----------------------------------------------------------


def vol_adjusted_momentum_strategy(close, spy_close):
    """
    Momentum normalized by volatility + SPY regime filter
    """

    returns = close.pct_change()

    momentum = close.pct_change(MOMENTUM_LOOKBACK)

    vol = returns.rolling(VOL_LOOKBACK).std()

    score = momentum / vol

    rank = score.rank(axis=1, ascending=False)

    rebalance = _rebalance_mask(close.index)

    long_candidates = rank <= TOP_N

    # Market regime filter
    spy_ma = spy_close.rolling(SPY_MA).mean()
    market_up = spy_close > spy_ma

    market_up = market_up.reindex(close.index).fillna(False)

    entries = long_candidates & rebalance.values[:, None] & market_up.values[:, None]

    exits = ((~long_candidates) | (~market_up.values[:, None])) & rebalance.values[
        :, None
    ]

    return entries, exits
