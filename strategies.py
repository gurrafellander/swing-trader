# strategies.py
import vectorbt as vbt
import pandas as pd
import numpy as np
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
    POSITION_SIZE,
)


# ---------------------------------------------------
# Helpers
# ---------------------------------------------------

def _flatten(df, close):
    """Flatten MultiIndex columns from vbt indicators to match close columns."""
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = close.columns
    return df


def _make_weights(entry: np.ndarray, exit_: np.ndarray, weight: float) -> np.ndarray:
    """NaN=hold, 0=close, weight=open/maintain."""
    out = np.full(entry.shape, np.nan, dtype=np.float32)
    out[exit_ & ~entry] = 0.0
    out[entry] = weight
    return out


def _rebalance_mask(index):
    """True only on the first trading day of each rebalance period."""
    freq = REBALANCE_FREQ if REBALANCE_FREQ != "W" else "W-FRI"
    dates_only = index.normalize()
    s = pd.Series(1, index=dates_only)
    rebalance_dates = s.resample(freq).first().index.normalize()
    return pd.Series(dates_only.isin(rebalance_dates), index=index)


# ---------------------------------------------------
# 1. RSI Mean Reversion (original, kept for reference)
# ---------------------------------------------------

def rsi_mean_reversion(close: pd.DataFrame):
    rsi = _flatten(vbt.RSI.run(close, window=RSI_WINDOW).rsi, close)
    rsi_prev = rsi.shift(1)
    entry_cross = (rsi < RSI_ENTRY) & (rsi_prev >= RSI_ENTRY)
    consecutive_stop = (rsi < 20).rolling(10).sum() >= 10
    exit_cross = ((rsi > RSI_EXIT) & (rsi_prev <= RSI_EXIT)) | consecutive_stop

    if USE_RANKING:
        score = np.where(entry_cross.values, RSI_ENTRY - rsi.values, np.nan)
        rank = pd.DataFrame(score, index=close.index, columns=close.columns) \
                 .rank(axis=1, ascending=False, na_option='bottom').values
        entries = entry_cross & pd.DataFrame(
            rank <= MAX_POSITIONS, index=close.index, columns=close.columns)
    else:
        entries = entry_cross

    return entries, exit_cross


# ---------------------------------------------------
# 2. Cross-Sectional Momentum with Vol Weighting + Regime Filter
#
# Thesis:
#   - Stocks that outperformed over the past 6 months tend to keep
#     outperforming over the next 1-3 months (momentum premium).
#   - Weighting by inverse volatility improves risk-adjusted returns
#     vs equal weight — you're not dominated by the most erratic names.
#   - A market regime filter (index > 200d MA) keeps you in cash during
#     bear markets, dramatically cutting drawdowns.
#   - Monthly rebalance gives momentum time to play out while staying
#     responsive to changes in leadership.
#
# Expected properties vs RSI mean reversion:
#   - ~120 trades over 8 years (vs 7800) → fees ~5k vs 57k
#   - Avg hold: 20-40 days (vs 10)
#   - Naturally long-biased → profits from the market's upward drift
# ---------------------------------------------------

def momentum_with_regime(close: pd.DataFrame, benchmark: pd.Series):
    """
    Parameters
    ----------
    close     : DataFrame of stock prices (your universe)
    benchmark : Series of index/benchmark prices for regime filter (e.g. OMXS30)

    Returns
    -------
    target_weights : DataFrame (NaN=hold, 0=close, float=target allocation)
                     for use with vbt.Portfolio.from_orders + size_type='targetpercent'
    """
    # --- Momentum score: skip last month to avoid short-term reversal ---
    # Classic impl: (t-252 to t-21) return, i.e. 12-1 month momentum
    mom_start = close.shift(21)           # exclude most recent month
    mom_end   = close.shift(MOMENTUM_LOOKBACK)  # 6 months ago (126 days)
    momentum  = (mom_start / mom_end) - 1  # return over the lookback window

    # --- Volatility: annualised daily return std over VOL_LOOKBACK ---
    daily_vol = close.pct_change().rolling(VOL_LOOKBACK).std() * np.sqrt(252)
    daily_vol = daily_vol.replace(0, np.nan)

    # --- Inverse-vol weights among top N ---
    inv_vol = 1.0 / daily_vol

    # --- Regime filter: benchmark above its 200-day MA ---
    bench_ma200 = benchmark.rolling(SPY_MA).mean()
    market_up   = (benchmark > bench_ma200).reindex(close.index).fillna(False)

    # --- Rebalance mask ---
    rebalance = _rebalance_mask(close.index)

    # --- Build target weight matrix ---
    # NaN everywhere by default (hold), updated only on rebalance days
    out = pd.DataFrame(np.nan, index=close.index, columns=close.columns,
                       dtype=np.float32)

    for date in close.index[rebalance]:
        mom_row    = momentum.loc[date]
        invvol_row = inv_vol.loc[date]

        if not market_up.loc[date]:
            # Bear regime: close all positions → set everything to 0
            out.loc[date, :] = 0.0
            continue

        # Drop tickers with insufficient data
        valid = mom_row.notna() & invvol_row.notna()
        if valid.sum() < TOP_N:
            out.loc[date, :] = 0.0
            continue

        # Rank by momentum, select top N
        ranked = mom_row[valid].rank(ascending=False)
        top_n  = ranked[ranked <= TOP_N].index

        # Inverse-vol weights, normalised to sum to 1
        weights = invvol_row[top_n]
        weights = weights / weights.sum()

        # Set target allocations; close everything not in top N
        out.loc[date, :] = 0.0
        out.loc[date, top_n] = weights.astype(np.float32)

    return out


# ---------------------------------------------------
# 3. Moving Average Crossover
# ---------------------------------------------------

def ma_crossover(close: pd.DataFrame, fast_window=20, slow_window=50):
    fast_ma = _flatten(vbt.MA.run(close, fast_window).ma, close)
    slow_ma = _flatten(vbt.MA.run(close, slow_window).ma, close)
    entry = (fast_ma > slow_ma).values
    exit_ = (fast_ma < slow_ma).values
    if USE_RANKING:
        score = np.where(entry, (fast_ma.values - slow_ma.values) / slow_ma.values, np.nan)
        rank = pd.DataFrame(score, index=close.index, columns=close.columns) \
                 .rank(axis=1, ascending=False, na_option='bottom').values
        entry = entry & (rank <= MAX_POSITIONS)
    weights = _make_weights(entry, exit_, POSITION_SIZE)
    return pd.DataFrame(weights, index=close.index, columns=close.columns)


# ---------------------------------------------------
# 4. Bollinger Band Mean Reversion
# ---------------------------------------------------

def bollinger_mean_reversion(close: pd.DataFrame, window=20, std=2):
    bb    = vbt.BBANDS.run(close, window=window, alpha=std)
    lower = _flatten(bb.lower,  close)
    mid   = _flatten(bb.middle, close)
    entry = (close < lower).values
    exit_ = (close > mid).values
    if USE_RANKING:
        score = np.where(entry, (lower.values - close.values) / close.values, np.nan)
        rank = pd.DataFrame(score, index=close.index, columns=close.columns) \
                 .rank(axis=1, ascending=False, na_option='bottom').values
        entry = entry & (rank <= MAX_POSITIONS)
    weights = _make_weights(entry, exit_, POSITION_SIZE)
    return pd.DataFrame(weights, index=close.index, columns=close.columns)
