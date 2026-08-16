"""
portfolio_analysis.py
─────────────────────
Monte Carlo risk engine for the portfolio dashboard.

Runs correlated geometric Brownian motion (GBM) with Cholesky-decomposed
shocks, fitted entirely from historical price data already available in the app.
"""

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def fit_parameters(
    close: pd.DataFrame,
    tickers: list[str],
    lookback_days: int = 252,
) -> tuple[pd.Series, pd.Series, pd.DataFrame, pd.DataFrame]:
    """
    Estimate daily drift, daily vol, and correlation from historical log returns.

    Returns (daily_mean, daily_std, corr_matrix, log_ret_window).
    """
    data = close[tickers].dropna(how="all")
    log_ret = np.log(data / data.shift(1)).dropna()
    log_ret = log_ret.tail(lookback_days)

    if log_ret.shape[0] < 30:
        raise ValueError(
            f"Only {log_ret.shape[0]} days of overlapping history after dropping NaNs — "
            "need at least 30 to fit parameters."
        )

    # Drop any asset columns with too many NaNs in the window
    log_ret = log_ret.dropna(axis=1, how="any")
    tickers_clean = log_ret.columns.tolist()
    if set(tickers_clean) != set(tickers):
        missing = set(tickers) - set(tickers_clean)
        raise ValueError(
            f"Tickers with insufficient history in lookback window: {', '.join(missing)}"
        )

    daily_mean = log_ret.mean()
    daily_std = log_ret.std()
    corr_matrix = log_ret.corr()

    return daily_mean, daily_std, corr_matrix, log_ret


def run_monte_carlo(
    close: pd.DataFrame,
    tickers: list[str],
    weights: np.ndarray,
    portfolio_value: float,
    horizon_days: int = 21,
    n_paths: int = 100_000,
    lookback_days: int = 252,
    risk_free_rate: float = 0.02,
    seed: int = 42,
) -> dict:
    """
    Correlated GBM simulation for a multi-asset portfolio.

    Parameters
    ----------
    close            : DataFrame of close prices (all assets, any history)
    tickers          : list of tickers to simulate (must be columns in close)
    weights          : portfolio weights, must sum to 1, shape (n_assets,)
    portfolio_value  : starting portfolio value (reporting currency)
    horizon_days     : simulation horizon in trading days
    n_paths          : number of Monte Carlo paths
    lookback_days    : trailing window for fitting drift / vol / correlation
    risk_free_rate   : annualized risk-free rate for Sharpe calculation
    seed             : RNG seed for reproducibility

    Returns
    -------
    dict with keys:
        paths            : (n_paths, horizon_days+1) portfolio value array
        terminal_returns : (n_paths,) array of % returns at horizon
        metrics          : dict of risk statistics
        corr_matrix      : pd.DataFrame NxN correlation matrix
        ann_vols         : pd.Series annualized vol per ticker
        ann_drifts       : pd.Series annualized drift per ticker
        risk_contributions: pd.DataFrame per-position risk breakdown
        hist_portfolio   : pd.Series historical portfolio value (last ~1yr)
        tickers          : list of tickers in simulation
        weights          : weights array used
    """
    daily_mean, daily_std, corr_matrix, _ = fit_parameters(close, tickers, lookback_days)

    n_assets = len(tickers)
    mu = daily_mean.values       # (n_assets,)
    sigma = daily_std.values     # (n_assets,)

    # Regularize correlation matrix to ensure positive semi-definiteness
    corr_arr = corr_matrix.values.copy().astype(float)
    eigvals, eigvecs = np.linalg.eigh(corr_arr)
    eigvals = np.maximum(eigvals, 1e-8)
    corr_arr = eigvecs @ np.diag(eigvals) @ eigvecs.T
    # Re-normalize diagonal to exactly 1
    d = np.sqrt(np.diag(corr_arr))
    corr_arr = corr_arr / np.outer(d, d)
    L = np.linalg.cholesky(corr_arr)

    # Correlated standard-normal shocks: (n_paths, horizon_days, n_assets)
    rng = np.random.default_rng(seed)
    eps = rng.standard_normal((n_paths, horizon_days, n_assets))
    z_corr = eps @ L.T  # (n_paths, horizon_days, n_assets)

    # Daily log returns per step:
    # Under GBM, E[daily log return] = (mu - 0.5*sigma^2)*dt
    # Fitting daily_mean from history gives us exactly this term (since
    # daily_mean = sample mean of realized log returns).
    # So simulation step = daily_mean + daily_std * z_corr
    daily_log_ret = mu + sigma * z_corr  # (n_paths, horizon_days, n_assets)

    # Cumulative log returns → price relatives (relative to today = 1.0)
    cum_log_ret = np.cumsum(daily_log_ret, axis=1)   # (n_paths, horizon_days, n_assets)
    price_relatives = np.exp(cum_log_ret)             # (n_paths, horizon_days, n_assets)

    # Weighted portfolio value relative (equal-weight by default)
    port_rels = (price_relatives * weights).sum(axis=2)  # (n_paths, horizon_days)

    # Prepend day 0 = 1.0 (today)
    port_rels = np.concatenate([np.ones((n_paths, 1)), port_rels], axis=1)

    paths_value = port_rels * portfolio_value  # (n_paths, horizon_days+1)

    # Historical portfolio series — rebase so last row = portfolio_value
    hist_portfolio = _build_hist_portfolio(close, tickers, weights, portfolio_value)

    # Terminal statistics
    terminal_values = paths_value[:, -1]
    terminal_returns = (terminal_values - portfolio_value) / portfolio_value * 100

    metrics = _compute_metrics(
        paths_value, terminal_returns, terminal_values,
        portfolio_value, risk_free_rate, horizon_days,
    )

    risk_contrib = _compute_risk_contributions(weights, sigma, corr_arr, tickers)

    return {
        "paths": paths_value,
        "terminal_returns": terminal_returns,
        "metrics": metrics,
        "corr_matrix": corr_matrix,
        "ann_vols": daily_std * np.sqrt(TRADING_DAYS),
        "ann_drifts": daily_mean * TRADING_DAYS,
        "risk_contributions": risk_contrib,
        "hist_portfolio": hist_portfolio,
        "tickers": tickers,
        "weights": weights,
        "portfolio_value": portfolio_value,
        "horizon_days": horizon_days,
        "lookback_days": lookback_days,
    }


def _build_hist_portfolio(
    close: pd.DataFrame,
    tickers: list[str],
    weights: np.ndarray,
    portfolio_value: float,
    max_days: int = 252,
) -> pd.Series:
    """Build historical portfolio value series rebased to portfolio_value at the last row."""
    data = close[tickers].dropna(how="any")
    data = data.iloc[-max_days:]
    last_prices = data.iloc[-1]
    # Relative price at each row vs last row
    relatives = data.div(last_prices)
    port = (relatives * weights).sum(axis=1) * portfolio_value
    return port


def _compute_metrics(
    paths_value: np.ndarray,
    terminal_returns: np.ndarray,
    terminal_values: np.ndarray,
    portfolio_value: float,
    risk_free_rate: float,
    horizon_days: int,
) -> dict:
    mean_ret = float(terminal_returns.mean())
    median_ret = float(np.median(terminal_returns))
    std_ret = float(terminal_returns.std())

    var95 = float(np.percentile(terminal_returns, 5))
    var99 = float(np.percentile(terminal_returns, 1))
    cvar95 = float(terminal_returns[terminal_returns <= var95].mean())
    cvar99 = float(terminal_returns[terminal_returns <= var99].mean())

    prob_loss = float((terminal_returns < 0).mean() * 100)
    prob_loss_10 = float((terminal_returns < -10).mean() * 100)

    best = float(terminal_returns.max())
    worst = float(terminal_returns.min())

    # Annualize for Sharpe: scale horizon return and vol to annual
    ann_factor = TRADING_DAYS / horizon_days
    ann_mean = mean_ret / 100 * ann_factor
    ann_std = std_ret / 100 * np.sqrt(ann_factor)
    sharpe = (ann_mean - risk_free_rate) / ann_std if ann_std > 0 else 0.0

    # CAGR implied by the simulated mean terminal return
    cagr = (1 + mean_ret / 100) ** ann_factor - 1

    # Max drawdown per path
    running_max = np.maximum.accumulate(paths_value, axis=1)
    dd_pct = (paths_value - running_max) / running_max * 100  # ≤ 0
    path_max_dds = dd_pct.min(axis=1)
    avg_max_dd = float(path_max_dds.mean())
    tail_max_dd = float(np.percentile(path_max_dds, 5))  # 95th-pctile worst

    return {
        "mean_ret": round(mean_ret, 2),
        "median_ret": round(median_ret, 2),
        "std_ret": round(std_ret, 2),
        "mean_value": round(float(terminal_values.mean()), 0),
        "median_value": round(float(np.median(terminal_values)), 0),
        "var95": round(var95, 2),
        "var99": round(var99, 2),
        "cvar95": round(cvar95, 2),
        "cvar99": round(cvar99, 2),
        "prob_loss": round(prob_loss, 1),
        "prob_loss_10": round(prob_loss_10, 1),
        "best": round(best, 2),
        "worst": round(worst, 2),
        "sharpe": round(sharpe, 2),
        "cagr": round(cagr * 100, 2),
        "avg_max_dd": round(avg_max_dd, 2),
        "tail_max_dd": round(tail_max_dd, 2),
    }


def _compute_risk_contributions(
    weights: np.ndarray,
    daily_std: np.ndarray,
    corr_arr: np.ndarray,
    tickers: list[str],
) -> pd.DataFrame:
    """
    Compute each asset's % contribution to total portfolio variance.

    Variance contribution_i = w_i * (Cov @ w)_i / portfolio_variance
    """
    ann_vols = daily_std * np.sqrt(TRADING_DAYS)
    cov = corr_arr * np.outer(ann_vols, ann_vols)
    port_var = float(weights @ cov @ weights)
    cov_w = cov @ weights
    marginal = weights * cov_w
    pct_contrib = marginal / port_var * 100 if port_var > 0 else np.zeros(len(weights))

    return (
        pd.DataFrame(
            {
                "Ticker": tickers,
                "Weight (%)": weights * 100,
                "Ann Vol (%)": ann_vols * 100,
                "Risk Contrib (%)": pct_contrib,
            }
        )
        .sort_values("Risk Contrib (%)", ascending=False)
        .reset_index(drop=True)
    )
