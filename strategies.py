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
    # resample().first() returns the first *actual trading day* in each period as a value;
    # using that instead of the calendar anchor labels (which may not be trading days).
    rebalance_dates = pd.Series(dates_only, index=dates_only).resample(freq).first()
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


def _sharpe_weights(return_matrix: pd.DataFrame) -> pd.Series:
    """
    Maximise the annualised Sharpe ratio via mean-variance optimisation
    (long-only, fully-invested). Hard-caps to cfg.top_n positions.

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
        bounds=[(0, cfg.max_position_weight)] * n,
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
        options={"maxiter": 500, "ftol": 1e-9},
    )

    w = pd.Series(result.x if result.success else np.ones(n) / n, index=cols)

    # Hard-cap: keep only top_n positions by weight, renormalise
    if n > cfg.top_n:
        top_tickers = w.nlargest(cfg.top_n).index
        w[~w.index.isin(top_tickers)] = 0.0
        if w.sum() > 0:
            w /= w.sum()

    # Zero out SLSQP numerical residuals (sub-0.1% weights are not meaningful)
    w[w < 1e-3] = 0.0
    if w.sum() > 0:
        w /= w.sum()

    return w


def apply_stop_loss(
    weights: pd.DataFrame,
    close: pd.DataFrame,
    stop: float = 0.05,
) -> pd.DataFrame:
    """
    Post-process a target-weight matrix to inject stop-loss exits.

    Tracks entry price for each open position and emits 0.0 (close) on
    the next bar when price falls more than `stop` below entry.

    Parameters
    ----------
    weights : target-weight DataFrame from any strategy
    close   : the same close prices used to build weights
    stop    : fractional loss from entry that triggers exit (default 0.05 = 5%)
    """
    out = weights.copy()
    entry_price = np.full(len(close.columns), np.nan)
    close_arr = close.to_numpy()
    out_arr = out.to_numpy(
        dtype=np.float32, copy=True
    )  # work on the underlying array directly

    for i in range(1, len(close)):
        w_prev = out_arr[i - 1]

        # Record entry prices for positions opened/held on previous bar
        for j, ep in enumerate(entry_price):
            if w_prev[j] > 0:
                # position was active on previous bar — lock in entry if not set
                if np.isnan(entry_price[j]):
                    entry_price[j] = close_arr[i - 1, j]
            elif not np.isnan(w_prev[j]):
                # explicit 0 on previous bar → strategy closed it
                entry_price[j] = np.nan

        # Check stop loss
        for j, ep in enumerate(entry_price):
            if np.isnan(ep):
                continue
            if close_arr[i, j] <= ep * (1 - stop):
                # Only override if strategy hasn't already acted on this bar
                if np.isnan(out_arr[i, j]):
                    out_arr[i, j] = 0.0
                entry_price[j] = np.nan

    # Write back — we modified out_arr in place so out already reflects changes
    return pd.DataFrame(out_arr, index=out.index, columns=out.columns)


# ═══════════════════════════════════════════════════════════════════════════════
# Strategy 1 — RSI Mean Reversion
# ═══════════════════════════════════════════════════════════════════════════════


def rsi_mean_reversion(close: pd.DataFrame) -> pd.DataFrame:
    """
    Buy when RSI crosses below cfg.rsi_entry; sell when it crosses above
    cfg.rsi_exit (or after 10 consecutive days below 20).
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

    1. Score stocks by (t-21)/(t-lookback) return (skip last month).
    2. Select top cfg.top_n; weight by inverse realised volatility.
    3. Regime filter: all-cash when benchmark < cfg.spy_ma-day MA.
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
# Strategy 3 — Momentum + Sharpe Optimisation + Regime Filter
# ═══════════════════════════════════════════════════════════════════════════════


def momentum_sharpe_optimised(
    close: pd.DataFrame,
    benchmark: pd.Series,
) -> pd.DataFrame:
    """
    Momentum pre-filter → Sharpe optimisation, with regime filter.

    1. Score stocks via pandas-ta ROC (1-month skip for reversal avoidance).
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
# Strategy 4 — Rising Momentum + Sharpe Optimisation  (always invested)
# ═══════════════════════════════════════════════════════════════════════════════


def rising_momentum_sharpe(close: pd.DataFrame) -> pd.DataFrame:
    """
    Always-invested strategy targeting stocks whose momentum is near zero
    and accelerating upward — i.e. stocks just starting to move.

    Selection logic
    ---------------
    Two ROC windows are computed on raw close prices (no reversal skip):
      • roc_short  — recent momentum  (cfg.roc_short_lookback days)
      • roc_long   — trend momentum   (cfg.roc_long_lookback  days)

    A stock qualifies as a *rising momentum* candidate when ALL of:
      1. roc_short > roc_long           momentum is accelerating (slope turning up)
      2. roc_long.abs() < cfg.roc_near_zero_threshold
                                        still near its zero baseline — not already
                                        priced in (default threshold: 5 %)
      3. roc_short > 0                  at least slightly positive right now

    Candidates are ranked by (roc_short − roc_long) acceleration spread.
    The top cfg.top_candidates are passed to the Sharpe optimiser, which
    uses a return window of cfg.rising_vol_lookback days (independent of
    cfg.vol_lookback, which the other momentum strategies use) and selects
    and sizes the final cfg.top_n holdings.

    No regime filter — always fully invested.
    Rebalances on cfg.rebalance_freq schedule.
    """
    roc_short = close.apply(
        lambda col: ta.roc(col, length=cfg.roc_short_lookback), axis=0
    )
    roc_long = close.apply(
        lambda col: ta.roc(col, length=cfg.roc_long_lookback), axis=0
    )
    acceleration = roc_short - roc_long
    daily_returns = close.pct_change()
    rebalance = _rebalance_mask(close.index)

    out = pd.DataFrame(
        np.nan, index=close.index, columns=close.columns, dtype=np.float32
    )

    for date in close.index[rebalance]:
        rs = roc_short.loc[date]
        rl = roc_long.loc[date]
        ac = acceleration.loc[date]

        # Boolean filter: accelerating, near-zero baseline, positive short-term
        rising_mask = (
            rs.notna()
            & rl.notna()
            & (rs > rl)  # accelerating
            & (rl.abs() < cfg.roc_near_zero_threshold)  # near zero baseline
            & (rs > 0)  # short-term positive
        )

        candidates_series = ac[rising_mask].dropna()

        if len(candidates_series) == 0:
            out.loc[date, :] = 0.0
            continue

        # Rank by acceleration; cap at top_candidates for the optimiser
        n_cands = min(len(candidates_series), cfg.top_candidates)
        candidates = candidates_series.nlargest(n_cands).index.tolist()

        # Trailing return window for Sharpe optimisation
        # Uses cfg.rising_vol_lookback (NOT cfg.vol_lookback) — kept separate
        # so this strategy's fast signal cadence and its Sharpe sizing window
        # stay matched, independent of whatever the other momentum strategies use.
        loc_i = close.index.get_loc(date)
        start_i = max(0, loc_i - cfg.rising_vol_lookback)
        ret_window = daily_returns.iloc[start_i : loc_i + 1][candidates].dropna(axis=1)

        # Need at least 2 stocks for a covariance matrix
        if ret_window.shape[1] < 2:
            out.loc[date, :] = 0.0
            continue

        weights = _sharpe_weights(ret_window)

        out.loc[date, :] = 0.0
        for ticker, w in weights.items():
            if w > 0:
                out.loc[date, ticker] = np.float32(w)

    return out


# ═══════════════════════════════════════════════════════════════════════════════
# Strategy 5 — Moving Average Crossover
# ═══════════════════════════════════════════════════════════════════════════════


def ma_crossover(
    close: pd.DataFrame,
    fast_window: int = 20,
    slow_window: int = 50,
) -> pd.DataFrame:
    """
    Enter when fast MA crosses above slow MA; exit on the reverse cross.
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
# Strategy 6 — Bollinger Band Mean Reversion
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


# ═══════════════════════════════════════════════════════════════════════════════
# Strategy 7 — Trend-Filtered RSI Dip Buy  (monthly rebalanced)
# ═══════════════════════════════════════════════════════════════════════════════


def trend_filtered_dip_buy(close: pd.DataFrame) -> pd.DataFrame:
    """
    Mean reversion *within* an uptrend: only buy RSI dips on stocks that
    are themselves above their own long-term moving average. This avoids
    "catching a falling knife" — the classic failure mode of plain
    Bollinger/RSI mean reversion, where a stock that looks oversold simply
    keeps falling because it's in a genuine downtrend.

    Selection logic (evaluated once per rebalance date)
    ----------------------------------------------------
    A stock qualifies as a dip-buy candidate when ALL of:
      1. close > MA(cfg.dip_trend_ma)        stock is in a long-term uptrend
      2. RSI(cfg.rsi_window) < cfg.rsi_entry  stock is short-term oversold

    Among qualifying stocks, the ones with the *lowest* RSI (most oversold)
    are ranked highest and the top cfg.top_n are selected. Positions are
    equal-weighted — this is a tactical dip-buy, not a risk-budgeted
    portfolio, so simple equal weight keeps the logic transparent.

    If fewer than cfg.top_n stocks qualify, available candidates are
    equal-weighted among themselves (not diluted by holding cash) — the
    point is to size into the dips that exist, not to mechanically fill
    cfg.top_n slots.

    If zero stocks qualify, the strategy goes to cash for the month.

    Rebalances on cfg.rebalance_freq schedule (monthly by default) — note
    this is intentionally slower than a typical RSI mean-reversion
    strategy, which usually checks daily. The monthly cadence means this
    captures the *existence* of a trend-confirmed dip at rebalance time
    rather than reacting to every daily wiggle, trading off some
    responsiveness for far lower turnover and fees.
    """
    rsi = _flatten(vbt.RSI.run(close, window=cfg.rsi_window).rsi, close)
    trend_ma = close.rolling(cfg.dip_trend_ma).mean()
    above_trend = close > trend_ma

    rebalance = _rebalance_mask(close.index)
    out = pd.DataFrame(
        np.nan, index=close.index, columns=close.columns, dtype=np.float32
    )

    for date in close.index[rebalance]:
        rsi_row = rsi.loc[date]
        trend_row = above_trend.loc[date]

        candidate_mask = trend_row & rsi_row.notna() & (rsi_row < cfg.rsi_entry)
        candidates_series = rsi_row[candidate_mask]

        if len(candidates_series) == 0:
            out.loc[date, :] = 0.0
            continue

        # Most oversold first (lowest RSI), cap at top_n
        n_take = min(len(candidates_series), cfg.top_n)
        chosen = candidates_series.nsmallest(n_take).index

        weight = 1.0 / n_take
        out.loc[date, :] = 0.0
        out.loc[date, chosen] = np.float32(weight)

    return out
