"""
portfolio_view.py
─────────────────
Streamlit Portfolio view: holdings management, historical performance chart,
and full Monte Carlo risk dashboard.

Called from app.py via render_portfolio_view().
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pandas_ta as _ta
import plotly.graph_objects as go
import streamlit as st

from indicators import ROC_LOOKBACK
from portfolio_analysis import run_monte_carlo

# ── Portfolio persistence helpers ─────────────────────────────────────────────

_PORTFOLIO_DIR = Path("./portfolio-cache")
_PORTFOLIO_PATH = _PORTFOLIO_DIR / "assets.json"


def load_portfolio_file() -> list:
    """Read the persisted watchlist, tolerating a first-run/missing cache dir."""
    try:
        with open(_PORTFOLIO_PATH, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def _save_portfolio_file(data: list) -> None:
    _PORTFOLIO_DIR.mkdir(exist_ok=True)
    with open(_PORTFOLIO_PATH, "w") as f:
        json.dump(data, f)


def add_to_portfolio(ticker: str) -> None:
    portfolio = load_portfolio_file()
    if ticker not in portfolio:
        portfolio.append(ticker)
    st.session_state.portfolio = portfolio
    _save_portfolio_file(portfolio)


def remove_from_portfolio(ticker: str) -> None:
    portfolio = load_portfolio_file()
    if ticker in portfolio:
        portfolio.remove(ticker)
    st.session_state.portfolio = portfolio
    _save_portfolio_file(portfolio)


def remove_entire_portfolio() -> None:
    st.session_state.portfolio = []
    _save_portfolio_file([])


# ── Monte Carlo caching ───────────────────────────────────────────────────────


@st.cache_data(show_spinner="Running Monte Carlo simulation…")
def _cached_monte_carlo(
    _cache_key: str,  # hashable proxy for close data state
    tickers_key: tuple,
    weights_key: tuple,
    portfolio_value: float,
    horizon_days: int,
    n_paths: int,
    lookback_days: int,
    risk_free_rate: float,
    _close: pd.DataFrame,  # not hashed (underscore prefix), passed for compute only
) -> dict:
    return run_monte_carlo(
        close=_close,
        tickers=list(tickers_key),
        weights=np.array(weights_key),
        portfolio_value=portfolio_value,
        horizon_days=horizon_days,
        n_paths=n_paths,
        lookback_days=lookback_days,
        risk_free_rate=risk_free_rate,
    )


# ── Chart builders ────────────────────────────────────────────────────────────


def _fan_chart(
    hist_portfolio: pd.Series,
    paths: np.ndarray,
    horizon_days: int,
) -> go.Figure:
    """Continuous chart: historical portfolio line + simulated fan from today."""
    today = hist_portfolio.index[-1]

    # Future trading-day dates (horizon_days + 1 includes today as anchor point)
    future_dates = pd.bdate_range(start=today, periods=horizon_days + 1)
    # future_dates[0] = today, [1:] = forward trading days

    p5 = np.percentile(paths, 5, axis=0)
    p25 = np.percentile(paths, 25, axis=0)
    p50 = np.percentile(paths, 50, axis=0)
    p75 = np.percentile(paths, 75, axis=0)
    p95 = np.percentile(paths, 95, axis=0)

    sim_x = list(future_dates)

    fig = go.Figure()

    # 5–95 percentile band
    fig.add_trace(
        go.Scatter(
            x=sim_x + sim_x[::-1],
            y=list(p95) + list(p5[::-1]),
            fill="toself",
            fillcolor="rgba(61,255,160,0.07)",
            line=dict(color="rgba(0,0,0,0)"),
            name="5–95th pctile",
            hoverinfo="skip",
        )
    )

    # 25–75 percentile band
    fig.add_trace(
        go.Scatter(
            x=sim_x + sim_x[::-1],
            y=list(p75) + list(p25[::-1]),
            fill="toself",
            fillcolor="rgba(61,255,160,0.15)",
            line=dict(color="rgba(0,0,0,0)"),
            name="25–75th pctile",
            hoverinfo="skip",
        )
    )

    # Median simulation path
    fig.add_trace(
        go.Scatter(
            x=sim_x,
            y=p50,
            name="Median (sim)",
            line=dict(color="#3dffa0", width=1.5, dash="dash"),
            hovertemplate="%{x|%Y-%m-%d}<br>%{y:,.0f} SEK<extra>Median</extra>",
        )
    )

    # Historical line — drawn last so it sits on top
    fig.add_trace(
        go.Scatter(
            x=hist_portfolio.index,
            y=hist_portfolio.values,
            name="Historical",
            line=dict(color="#3dffa0", width=2),
            hovertemplate="%{x|%Y-%m-%d}<br>%{y:,.0f} SEK<extra>Historical</extra>",
        )
    )

    # Vertical "today" marker — pass as millisecond epoch so plotly handles it correctly
    fig.add_vline(
        x=int(today.timestamp() * 1000),
        line=dict(color="rgba(240,192,64,0.6)", dash="dash", width=1.5),
        annotation_text="Today",
        annotation_position="top left",
        annotation_font=dict(color="#f0c040", size=11),
    )

    fig.update_layout(
        title=f"Portfolio: Historical & {horizon_days}-Day Monte Carlo Projection",
        hovermode="x unified",
        height=460,
        template="plotly_dark",
        legend=dict(orientation="h", y=1.02, x=0),
        margin=dict(l=10, r=10, t=60, b=10),
        yaxis_title="Portfolio Value (SEK)",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(17,20,18,0.6)",
    )
    return fig


def _corr_heatmap(corr_matrix: pd.DataFrame) -> go.Figure:
    tickers = corr_matrix.columns.tolist()
    z = corr_matrix.values
    n = len(tickers)
    font_size = max(7, min(12, 100 // n))

    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=tickers,
            y=tickers,
            colorscale="RdBu_r",
            zmid=0,
            zmin=-1,
            zmax=1,
            text=np.round(z, 2),
            texttemplate="%{text:.2f}",
            textfont=dict(size=font_size),
            colorbar=dict(title="ρ", thickness=12, len=0.8),
        )
    )

    fig.update_layout(
        title="Pairwise Correlation Matrix",
        height=max(320, min(580, 70 * n + 80)),
        template="plotly_dark",
        margin=dict(l=10, r=10, t=50, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(17,20,18,0.6)",
    )
    return fig


def _return_histogram(terminal_returns: np.ndarray, var95: float) -> go.Figure:
    median_ret = float(np.median(terminal_returns))

    fig = go.Figure()

    # Full distribution
    fig.add_trace(
        go.Histogram(
            x=terminal_returns,
            nbinsx=40,
            name="Return dist.",
            marker_color="rgba(61,255,160,0.45)",
            marker_line=dict(color="rgba(61,255,160,0.7)", width=0.4),
        )
    )

    # Left tail highlighted
    tail = terminal_returns[terminal_returns <= var95]
    if len(tail) > 0:
        fig.add_trace(
            go.Histogram(
                x=tail,
                nbinsx=12,
                name="VaR 95% tail",
                marker_color="rgba(255,95,95,0.65)",
                marker_line=dict(color="rgba(255,95,95,0.85)", width=0.4),
            )
        )

    fig.add_vline(
        x=var95,
        line=dict(color="#ff5f5f", dash="dash", width=1.5),
        annotation_text=f"VaR 95%: {var95:.1f}%",
        annotation_font=dict(color="#ff5f5f", size=11),
        annotation_position="top right",
    )
    fig.add_vline(
        x=median_ret,
        line=dict(color="#3dffa0", dash="dot", width=1.5),
        annotation_text=f"Median: {median_ret:.1f}%",
        annotation_font=dict(color="#3dffa0", size=11),
        annotation_position="top left",
    )

    fig.update_layout(
        title="Distribution of Simulated Returns",
        xaxis_title="Return (%)",
        yaxis_title="Paths",
        barmode="overlay",
        height=350,
        template="plotly_dark",
        legend=dict(orientation="h", y=1.02, x=0),
        margin=dict(l=10, r=10, t=50, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(17,20,18,0.6)",
    )
    return fig


def _risk_contribution_chart(risk_contrib: pd.DataFrame) -> go.Figure:
    df = risk_contrib.sort_values("Risk Contrib (%)", ascending=True)
    n = len(df)
    equal_w = 100.0 / n

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            y=df["Ticker"],
            x=df["Risk Contrib (%)"],
            orientation="h",
            name="Risk Contribution",
            marker_color="rgba(255,95,95,0.65)",
            marker_line=dict(color="rgba(255,95,95,0.85)", width=0.5),
            hovertemplate="%{y}: %{x:.1f}% of variance<extra></extra>",
        )
    )

    # Equal-weight reference
    fig.add_vline(
        x=equal_w,
        line=dict(color="rgba(61,255,160,0.5)", dash="dot", width=1.5),
        annotation_text=f"Equal {equal_w:.1f}%",
        annotation_font=dict(color="#3dffa0", size=10),
        annotation_position="top right",
    )

    fig.update_layout(
        title="Risk Contribution by Position (% of Portfolio Variance)",
        xaxis_title="% of Portfolio Variance",
        height=max(280, 38 * n + 100),
        template="plotly_dark",
        margin=dict(l=10, r=10, t=50, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(17,20,18,0.6)",
    )
    return fig


# ── Main render function ──────────────────────────────────────────────────────


def render_portfolio_view(
    close: pd.DataFrame,
    omx,
    price_indicator_chart_fn,
    portfolio: list,
    remove_fn,
    clear_fn,
) -> None:
    """
    Render the full Portfolio view.

    Parameters
    ----------
    close                   : full universe close prices
    omx                     : OMXS30 benchmark series (may be None)
    price_indicator_chart_fn: reference to app._price_indicator_chart
    portfolio               : current portfolio ticker list
    remove_fn               : callback(ticker) to remove one ticker
    clear_fn                : callback() to clear all tickers
    """
    st.header("Portfolio View")

    if not portfolio:
        st.info(
            "Your portfolio is empty. Add tickers from the Single Stock or Screener views."
        )
        return

    # ── Holdings bar ──────────────────────────────────────────────────────────
    st.markdown("**Holdings:**")
    row_len = min(len(portfolio), 8)
    rm_cols = st.columns(row_len + 1)
    for i, t in enumerate(list(portfolio)):
        with rm_cols[i % row_len]:
            if st.button(f"✕ {t}", key=f"rm_{t}"):
                remove_fn(t)
                st.rerun()
    with rm_cols[-1]:
        if st.button("✕ Clear all", type="primary"):
            clear_fn()
            st.rerun()

    valid = [t for t in portfolio if t in close.columns]
    missing = [t for t in portfolio if t not in close.columns]
    if missing:
        st.warning(f"Not in universe (ignoring): {', '.join(missing)}")

    if not valid:
        st.info("No valid tickers remaining.")
        return

    port_close = close[valid].dropna(how="any")
    if port_close.empty:
        st.warning("No overlapping data across all holdings.")
        return

    # ── Historical performance chart ──────────────────────────────────────────
    st.caption(
        "Equal-weighted, buy-and-hold from earliest common date. "
        "Weights set once and drift naturally — no periodic rebalancing."
    )

    rebased = port_close.div(port_close.iloc[0]) * 100
    port_indexed = rebased.mean(axis=1)
    port_indexed.name = "Portfolio"

    show_omx = st.toggle("Show OMXS30 benchmark", value=False)
    omx_rebased = None
    if show_omx and omx is not None:
        omx_aligned = omx.reindex(port_close.index).ffill()
        first_omx = omx_aligned.first_valid_index()
        if first_omx is not None:
            omx_rebased = omx_aligned / omx_aligned[first_omx] * 100
            omx_rebased.name = "OMXS30"
    elif show_omx and omx is None:
        st.warning("Could not load OMXS30 benchmark data.")

    port_roc = _ta.roc(port_indexed, length=ROC_LOOKBACK)
    fig_perf = price_indicator_chart_fn(
        price_series=port_indexed,
        rsi_series=None,
        roc_series=port_roc.dropna(),
        benchmark_series=omx_rebased,
        title=f"Portfolio ({len(valid)} holdings, rebased to 100)",
        show_rsi=False,
    )
    st.plotly_chart(fig_perf, width="stretch")
    st.caption(
        f"Start date: {port_close.index[0].date()}  |  Holdings: {', '.join(valid)}"
    )

    st.divider()

    # ── Monte Carlo Risk Dashboard ─────────────────────────────────────────────
    st.subheader("Monte Carlo Risk Analysis")

    # Parameters
    with st.expander("Simulation Parameters", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            portfolio_value = st.number_input(
                "Portfolio Value (SEK)",
                min_value=10_000,
                max_value=100_000_000,
                value=1_000_000,
                step=50_000,
                format="%d",
            )
        with c2:
            horizon_days = st.number_input(
                "Horizon (trading days)",
                min_value=5,
                max_value=126,
                value=21,
                step=1,
            )
        with c3:
            lookback_days = st.selectbox(
                "Vol/correlation lookback",
                [126, 252, 504],
                index=1,
                format_func=lambda x: (
                    f"{x}d (~6m)"
                    if x == 126
                    else f"{x}d (~1y)" if x == 252 else f"{x}d (~2y)"
                ),
            )
        with c4:
            risk_free_pct = st.number_input(
                "Risk-free rate (%)",
                min_value=0.0,
                max_value=10.0,
                value=2.0,
                step=0.25,
            )
            risk_free_rate = risk_free_pct / 100.0

        fast_mode = st.checkbox(
            "Fast preview mode (5k paths — instant, less accurate)", value=False
        )
        n_paths = 5_000 if fast_mode else 100_000
        st.caption(f"Paths: {n_paths:,}  ·  Assets: {len(valid)}  ·  Equal-weighted")

    # Equal weights
    weights = np.array([1.0 / len(valid)] * len(valid))

    # Cache key proxy — encodes data state without hashing the full DataFrame
    cache_key = (
        f"{port_close.index[-1].date()}|{port_close.shape}|"
        f"{','.join(sorted(valid))}"
    )

    try:
        results = _cached_monte_carlo(
            _cache_key=cache_key,
            tickers_key=tuple(valid),
            weights_key=tuple(float(w) for w in weights),
            portfolio_value=float(portfolio_value),
            horizon_days=int(horizon_days),
            n_paths=int(n_paths),
            lookback_days=int(lookback_days),
            risk_free_rate=float(risk_free_rate),
            _close=port_close,
        )
    except ValueError as e:
        st.error(f"Simulation failed: {e}")
        return

    m = results["metrics"]

    # ── Fan chart ──────────────────────────────────────────────────────────────
    fig_fan = _fan_chart(results["hist_portfolio"], results["paths"], int(horizon_days))
    st.plotly_chart(fig_fan, width="stretch")

    # ── Key metrics ────────────────────────────────────────────────────────────
    st.subheader("Key Risk Metrics")

    def _sign(v: float) -> str:
        return f"{v:+.1f}%" if abs(v) < 1000 else f"{v:+.0f}%"

    r1c1, r1c2, r1c3, r1c4 = st.columns(4)
    r2c1, r2c2, r2c3, r2c4 = st.columns(4)
    r3c1, r3c2, r3c3, r3c4 = st.columns(4)

    with r1c1:
        st.metric(
            "Expected Return", _sign(m["mean_ret"]), f"→ {m['mean_value']:,.0f} SEK"
        )
    with r1c2:
        st.metric(
            "Median Return", _sign(m["median_ret"]), f"→ {m['median_value']:,.0f} SEK"
        )
    with r1c3:
        st.metric(
            "30d Volatility (σ)", f"{m['std_ret']:.1f}%", "std dev of simulated returns"
        )
    with r1c4:
        st.metric(
            "Sim. Sharpe (ann.)", f"{m['sharpe']:.2f}", f"rf = {risk_free_pct:.2f}%"
        )

    with r2c1:
        st.metric("VaR 95%", _sign(m["var95"]), f"CVaR 95%: {m['cvar95']:+.1f}%")
    with r2c2:
        st.metric("VaR 99%", _sign(m["var99"]), f"CVaR 99%: {m['cvar99']:+.1f}%")
    with r2c3:
        st.metric(
            "P(Any Loss)",
            f"{m['prob_loss']:.1f}%",
            f"P(>10% loss): {m['prob_loss_10']:.1f}%",
        )
    with r2c4:
        st.metric("Implied CAGR", _sign(m["cagr"]), "from simulated drift")

    with r3c1:
        st.metric("Best Outcome", _sign(m["best"]))
    with r3c2:
        st.metric("Worst Outcome", _sign(m["worst"]))
    with r3c3:
        st.metric("Avg Max Drawdown", f"{m['avg_max_dd']:.1f}%", "mean across paths")
    with r3c4:
        st.metric(
            "Tail Max DD (95th)",
            f"{m['tail_max_dd']:.1f}%",
            "95th-pctile worst intra-period",
        )

    st.divider()

    # ── Histogram + Correlation side-by-side ──────────────────────────────────
    col_hist, col_corr = st.columns(2)

    with col_hist:
        fig_hist = _return_histogram(results["terminal_returns"], m["var95"])
        st.plotly_chart(fig_hist, width="stretch")

    with col_corr:
        fig_corr = _corr_heatmap(results["corr_matrix"])
        st.plotly_chart(fig_corr, width="stretch")

    st.divider()

    # ── Risk contribution ──────────────────────────────────────────────────────
    st.subheader("Risk Contribution by Position")

    col_bar, col_tbl = st.columns([3, 2])
    with col_bar:
        fig_risk = _risk_contribution_chart(results["risk_contributions"])
        st.plotly_chart(fig_risk, width="stretch")

    with col_tbl:
        rc = results["risk_contributions"].copy()
        rc["Weight (%)"] = rc["Weight (%)"].round(1)
        rc["Ann Vol (%)"] = rc["Ann Vol (%)"].round(1)
        rc["Risk Contrib (%)"] = rc["Risk Contrib (%)"].round(1)
        st.caption("Sorted by risk contribution (highest first)")
        st.dataframe(rc, hide_index=True, width="stretch")

    # ── Methodology disclosure ─────────────────────────────────────────────────
    non_sek = [
        t
        for t in valid
        if not (
            t.endswith(".ST")
            or t.endswith(".CO")
            or t.endswith(".HE")
            or t.endswith(".OL")
        )
    ]
    with st.expander("Methodology & Limitations"):
        st.markdown(f"""
**Simulation method:** Correlated geometric Brownian motion (GBM) with Cholesky-decomposed
shocks across {len(valid)} assets, {n_paths:,} paths over {int(horizon_days)} trading days.

**Parameter estimation:** Daily log-return mean, volatility, and the full {len(valid)}×{len(valid)}
correlation matrix are fitted from the trailing **{int(lookback_days)} trading-day**
({'~6 months' if lookback_days == 126 else '~1 year' if lookback_days == 252 else '~2 years'}) window.
All parameters come from realized history — no prior assumptions or hardcoded values.

**Currency handling:** {"No non-SEK holdings detected — all tickers appear to be SEK-denominated." if not non_sek else
f"Non-SEK holdings detected: **{', '.join(non_sek)}**. These are treated as if their returns are already in SEK "
"(no FX conversion is applied). This ignores currency risk: if SEK appreciates against USD (or other currencies), "
"the actual SEK return would be lower than simulated. This is a known simplification — FX is not modeled as a "
"separate stochastic process in this version."}

**Weights:** Equal-weighted ({100/len(valid):.1f}% per position).

**Not a forecast.** GBM assumes constant drift and volatility, and ignores fat tails, jumps,
mean-reversion, and regime changes. Past correlation and volatility may not predict future behavior.
This is a probabilistic risk tool, not an investment recommendation.
        """)
