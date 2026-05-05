"""
strategies.py
─────────────
All trading strategies return a target-weight DataFrame compatible with:
    vbt.Portfolio.from_orders(..., size_type='targetpercent')

Convention
----------
  NaN   → hold (no change)
  0.0   → close position
  float → set allocation to this fraction of portfolio
"""

import numpy as np
import pandas as pd
import pandas_ta as ta
import vectorbt as vbt
from scipy.optimize import minimize

from config import cfg

# ═══════════════════════════════════════════════════════════════════════════════
# Shared helpers
# ═══════════════════════════════════════════════════════════════════════════════


def _rebalance_mask(index: pd.DatetimeIndex) -> pd.Series:
    """
    Boolean Series that is True on the first trading day of each rebalance
    period (driven by cfg.rebalance_freq, e.g. 'MS' or 'W').
    """
    freq = "W-FRI" if cfg.rebalance_freq == "W" else cfg.rebalance_freq
    dates_only = index.normalize()
    rebalance_dates = (
        pd.Series(1, index=dates_only).resample(freq).first().index.normalize()
    )
    return pd.Series(dates_only.isin(rebalance_dates), index=index)


def _flatten(df: pd.DataFrame, close: pd.DataFrame) -> pd.DataFrame:
    """Collapse a vbt MultiIndex column frame to match close.columns."""
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = close.columns
    return df


def _make_weights(
    entry: np.ndarray,
    exit_: np.ndarray,
    weight: float,
) -> np.ndarray:
    """
    Build a weight array from boolean entry/exit masks.
    NaN=hold, 0=close, weight=target allocation.
    """
    out = np.full(entry.shape, np.nan, dtype=np.float32)
    out[exit_ & ~entry] = 0.0
    out[entry] = weight
    return out


def _apply_ranking(
    signal: np.ndarray,
    score: np.ndarray,
    close: pd.DataFrame,
) -> np.ndarray:
    """
    Keep only the top cfg.max_positions signals ranked by score.
    signal and score must have the same shape as close.
    """
    rank = (
        pd.DataFrame(score, index=close.index, columns=close.columns)
        .rank(axis=1, ascending=False, na_option="bottom")
        .values
    )
    return signal & (rank <= cfg.max_positions)


# ═══════════════════════════════════════════════════════════════════════════════
# Sharpe optimiser (used by momentum_sharpe_optimised)
# ═══════════════════════════════════════════════════════════════════════════════


def _sharpe_weights(return_matrix: pd.DataFrame) -> pd.Series:
    """
    Maximise the annualised Sharpe ratio via mean-variance optimisation
    (long-only, fully-invested).  Hard-caps to cfg.top_n positions.

    Falls back to equal weight if the optimiser does not converge.
    """
    cols = return_matrix.columns.tolist()
    n = len(cols)
    mu = return_matrix.mean().values * 252
    cov = return_matrix.cov().values * 252

    def neg_sharpe(w: np.ndarray) -> float:
        ret = w @ mu
        vol = np.sqrt(w @ cov @ w)
        return -(ret - cfg.rf_daily * 252) / (vol + 1e-9)

    result = minimize(
        neg_sharpe,
        x0=np.ones(n) / n,
        method="SLSQP",
        bounds=[(0, 1)] * n,
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
        options={"maxiter": 500, "ftol": 1e-9},
    )

    w = pd.Series(result.x if result.success else np.ones(n) / n, index=cols)

    # Hard-cap: keep only top_n positions by weight, renormalise
    if n > cfg.top_n:
        threshold = w.nlargest(cfg.top_n).iloc[-1]
        w[w < threshold] = 0.0
        if w.sum() > 0:
            w /= w.sum()

    return w


# ═══════════════════════════════════════════════════════════════════════════════
# Strategy 1 — RSI Mean Reversion
# ═══════════════════════════════════════════════════════════════════════════════


def rsi_mean_reversion(close: pd.DataFrame) -> pd.DataFrame:
    """
    Buy when RSI crosses below cfg.rsi_entry; sell when it crosses above
    cfg.rsi_exit (or after 10 consecutive days below 20).

    Returns a weight DataFrame for use with vbt.Portfolio.from_orders.
    """
    rsi = _flatten(vbt.RSI.run(close, window=cfg.rsi_window).rsi, close)
    rsi_prev = rsi.shift(1)

    entry_cross = (rsi < cfg.rsi_entry) & (rsi_prev >= cfg.rsi_entry)
    stuck_low = (rsi < 20).rolling(10).sum() >= 10
    exit_cross = ((rsi > cfg.rsi_exit) & (rsi_prev <= cfg.rsi_exit)) | stuck_low

    if cfg.use_ranking:
        score = np.where(entry_cross.values, cfg.rsi_entry - rsi.values, np.nan)
        entries = _apply_ranking(entry_cross.values, score, close)
    else:
        entries = entry_cross.values

    weights = _make_weights(entries, exit_cross.values, cfg.position_size)
    return pd.DataFrame(weights, index=close.index, columns=close.columns)


# ═══════════════════════════════════════════════════════════════════════════════
# Strategy 2 — Momentum + Inverse-Vol + Regime Filter
# ═══════════════════════════════════════════════════════════════════════════════


def momentum_with_regime(
    close: pd.DataFrame,
    benchmark: pd.Series,
) -> pd.DataFrame:
    """
    Classic cross-sectional momentum with inverse-volatility weighting.

    1. Score stocks by (t-21)/(t-lookback) return to skip the reversal month.
    2. Select top cfg.top_n; weight by inverse realised volatility.
    3. Regime filter: all-cash when benchmark < 200-day MA.
    4. Rebalance monthly.
    """
    mom_start = close.shift(21)
    mom_end = close.shift(cfg.momentum_lookback)
    momentum = (mom_start / mom_end) - 1

    daily_vol = close.pct_change().rolling(cfg.vol_lookback).std() * np.sqrt(252)
    daily_vol = daily_vol.replace(0, np.nan)
    inv_vol = 1.0 / daily_vol

    bench_ma = benchmark.rolling(cfg.spy_ma).mean()
    market_up = (benchmark > bench_ma).reindex(close.index).fillna(False)

    rebalance = _rebalance_mask(close.index)
    out = pd.DataFrame(
        np.nan, index=close.index, columns=close.columns, dtype=np.float32
    )

    for date in close.index[rebalance]:
        if not market_up.loc[date]:
            out.loc[date, :] = 0.0
            continue

        mom_row = momentum.loc[date]
        invvol_row = inv_vol.loc[date]
        valid = mom_row.notna() & invvol_row.notna()

        if valid.sum() < cfg.top_n:
            out.loc[date, :] = 0.0
            continue

        ranked = mom_row[valid].rank(ascending=False)
        top_n = ranked[ranked <= cfg.top_n].index
        weights = invvol_row[top_n]
        weights = weights / weights.sum()

        out.loc[date, :] = 0.0
        out.loc[date, top_n] = weights.astype(np.float32)

    return out


# ═══════════════════════════════════════════════════════════════════════════════
# Strategy 3 — Momentum + Sharpe Optimisation (drop-in for strategy 2)
# ═══════════════════════════════════════════════════════════════════════════════


def momentum_sharpe_optimised(
    close: pd.DataFrame,
    benchmark: pd.Series,
) -> pd.DataFrame:
    """
    Enhanced momentum strategy that replaces inverse-vol weighting with a
    full Sharpe-ratio optimisation over the top momentum candidates.

    1. Score stocks via pandas-ta ROC (with 1-month skip for reversal).
    2. Pre-filter to cfg.top_candidates highest-ROC stocks.
    3. Run mean-variance Sharpe optimisation on the candidate pool.
    4. Keep cfg.top_n positions; zero the rest.
    5. Regime filter: all-cash when benchmark < cfg.spy_ma-day MA.
    6. Rebalance monthly.
    """
    close_skipped = close.shift(21)
    roc = close_skipped.apply(
        lambda col: ta.roc(col, length=cfg.momentum_lookback), axis=0
    )

    daily_returns = close.pct_change()

    bench_ma = benchmark.rolling(cfg.spy_ma).mean()
    market_up = (benchmark > bench_ma).reindex(close.index).fillna(False)

    rebalance = _rebalance_mask(close.index)
    out = pd.DataFrame(
        np.nan, index=close.index, columns=close.columns, dtype=np.float32
    )

    for date in close.index[rebalance]:
        if not market_up.loc[date]:
            out.loc[date, :] = 0.0
            continue

        valid_roc = roc.loc[date].dropna()
        if len(valid_roc) < cfg.top_n:
            out.loc[date, :] = 0.0
            continue

        candidates = valid_roc.nlargest(cfg.top_candidates).index.tolist()

        loc_i = close.index.get_loc(date)
        start_i = max(0, loc_i - cfg.vol_lookback)
        ret_window = daily_returns.iloc[start_i : loc_i + 1][candidates].dropna(axis=1)

        if ret_window.shape[1] < cfg.top_n:
            out.loc[date, :] = 0.0
            continue

        weights = _sharpe_weights(ret_window)

        out.loc[date, :] = 0.0
        for ticker, w in weights.items():
            if w > 0:
                out.loc[date, ticker] = np.float32(w)

    return out


# ═══════════════════════════════════════════════════════════════════════════════
# Strategy 4 — Moving Average Crossover
# ═══════════════════════════════════════════════════════════════════════════════


def ma_crossover(
    close: pd.DataFrame,
    fast_window: int = 20,
    slow_window: int = 50,
) -> pd.DataFrame:
    """
    Enter when fast MA crosses above slow MA; exit on the reverse cross.
    Optionally limits to the cfg.max_positions strongest signals (by spread).
    """
    fast_ma = _flatten(vbt.MA.run(close, fast_window).ma, close)
    slow_ma = _flatten(vbt.MA.run(close, slow_window).ma, close)

    entry = (fast_ma > slow_ma).values
    exit_ = (fast_ma < slow_ma).values

    if cfg.use_ranking:
        score = np.where(
            entry, (fast_ma.values - slow_ma.values) / slow_ma.values, np.nan
        )
        entry = _apply_ranking(entry, score, close)

    weights = _make_weights(entry, exit_, cfg.position_size)
    return pd.DataFrame(weights, index=close.index, columns=close.columns)


# ═══════════════════════════════════════════════════════════════════════════════
# Strategy 5 — Bollinger Band Mean Reversion
# ═══════════════════════════════════════════════════════════════════════════════


def bollinger_mean_reversion(
    close: pd.DataFrame,
    window: int = 20,
    std: float = 2.0,
) -> pd.DataFrame:
    """
    Enter when price drops below the lower Bollinger Band;
    exit when price recovers above the middle band.
    """
    bb = vbt.BBANDS.run(close, window=window, alpha=std)
    lower = _flatten(bb.lower, close)
    mid = _flatten(bb.middle, close)

    entry = (close < lower).values
    exit_ = (close > mid).values

    if cfg.use_ranking:
        score = np.where(entry, (lower.values - close.values) / close.values, np.nan)
        entry = _apply_ranking(entry, score, close)

    weights = _make_weights(entry, exit_, cfg.position_size)
    return pd.DataFrame(weights, index=close.index, columns=close.columns)
