"""
Stock Screener & Portfolio Dashboard
Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from config import cfg
from DataLoader import DataLoader
from indicators import Indicators, ROC_LOOKBACK

st.set_page_config(page_title="Stock Screener", layout="wide")


# ── Data loading (cached across all views) ────────────────────────────────────


@st.cache_data(show_spinner="Downloading price data…")
def load_data():
    loader = DataLoader(
        ticker_path=cfg.ticker_path,
        start_date=cfg.start_date,
        end_date=cfg.end_date,
        min_history=cfg.min_history,
    )
    close = loader.download_clean_data()
    return close


@st.cache_data(show_spinner="Computing indicators…")
def load_indicators(_close: pd.DataFrame) -> Indicators:
    # Underscore prefix on _close tells Streamlit not to hash the DataFrame arg.
    return Indicators(_close)


close = load_data()
ind = load_indicators(close)
tickers = list(close.columns)


# ── Session state for portfolio watchlist ─────────────────────────────────────

if "portfolio" not in st.session_state:
    st.session_state.portfolio: list[str] = []


def add_to_portfolio(ticker: str):
    if ticker not in st.session_state.portfolio:
        st.session_state.portfolio.append(ticker)


def remove_from_portfolio(ticker: str):
    if ticker in st.session_state.portfolio:
        st.session_state.portfolio.remove(ticker)


# ── Shared chart helpers ──────────────────────────────────────────────────────


def _price_indicator_chart(
    price_series: pd.Series,
    rsi_series: pd.Series | None,
    roc_series: pd.Series,
    ma50: pd.Series | None = None,
    ma200: pd.Series | None = None,
    bb_upper: pd.Series | None = None,
    bb_mid: pd.Series | None = None,
    bb_lower: pd.Series | None = None,
    title: str = "",
    show_rsi: bool = True,
) -> go.Figure:
    """
    Two-row Plotly figure: price (with optional overlays) + momentum panel.
    Both rows share the x-axis so zoom/pan stays in sync.
    Bottom row uses dual y-axis: RSI (left) and ROC (right).
    """
    row_heights = [0.65, 0.35]
    specs = [[{}], [{"secondary_y": True}]]

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=row_heights,
        specs=specs,
        vertical_spacing=0.04,
    )

    # ── Top row: price ────────────────────────────────────────────────────────
    fig.add_trace(
        go.Scatter(
            x=price_series.index,
            y=price_series.values,
            name="Price",
            line=dict(color="#1f77b4", width=1.5),
        ),
        row=1,
        col=1,
    )
    if ma50 is not None:
        fig.add_trace(
            go.Scatter(
                x=ma50.index,
                y=ma50.values,
                name="MA50",
                line=dict(color="#ff7f0e", width=1, dash="dot"),
            ),
            row=1,
            col=1,
        )
    if ma200 is not None:
        fig.add_trace(
            go.Scatter(
                x=ma200.index,
                y=ma200.values,
                name="MA200",
                line=dict(color="#d62728", width=1, dash="dash"),
            ),
            row=1,
            col=1,
        )
    if bb_upper is not None and bb_lower is not None:
        # Draw upper band; lower band fills back to upper to shade the region.
        fig.add_trace(
            go.Scatter(
                x=list(bb_upper.index) + list(bb_lower.index[::-1]),
                y=list(bb_upper.values) + list(bb_lower.values[::-1]),
                fill="toself",
                fillcolor="rgba(148,103,189,0.08)",
                line=dict(color="rgba(0,0,0,0)"),
                name="BB Band",
                hoverinfo="skip",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=bb_upper.index,
                y=bb_upper.values,
                name="BB Upper",
                line=dict(color="#9467bd", width=0.8, dash="dot"),
                showlegend=False,
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=bb_lower.index,
                y=bb_lower.values,
                name="BB Lower",
                line=dict(color="#9467bd", width=0.8, dash="dot"),
                showlegend=False,
            ),
            row=1,
            col=1,
        )
    if bb_mid is not None:
        fig.add_trace(
            go.Scatter(
                x=bb_mid.index,
                y=bb_mid.values,
                name="BB Mid",
                line=dict(color="#9467bd", width=0.8, dash="longdash"),
            ),
            row=1,
            col=1,
        )

    # ── Bottom row: RSI (left y) + ROC (right y) ──────────────────────────────
    if show_rsi and rsi_series is not None:
        fig.add_trace(
            go.Scatter(
                x=rsi_series.index,
                y=rsi_series.values,
                name="RSI(14)",
                line=dict(color="#2ca02c", width=1.2),
            ),
            row=2,
            col=1,
            secondary_y=False,
        )
        # Reference lines at 30 and 70
        for level, label in [(30, "Oversold 30"), (70, "Overbought 70")]:
            fig.add_hline(
                y=level,
                line=dict(color="rgba(44,160,44,0.4)", dash="dot", width=1),
                row=2,
                col=1,
                annotation_text=label,
                annotation_position="right",
                annotation_font_size=10,
            )

    fig.add_trace(
        go.Scatter(
            x=roc_series.index,
            y=roc_series.values,
            name=f"ROC({ROC_LOOKBACK}d) %",
            line=dict(color="#e377c2", width=1.2),
        ),
        row=2,
        col=1,
        secondary_y=True,
    )

    # ── Layout ────────────────────────────────────────────────────────────────
    fig.update_layout(
        title=title,
        hovermode="x unified",
        legend=dict(orientation="h", y=1.02, x=0),
        margin=dict(l=10, r=60, t=60, b=10),
        height=600,
        template="plotly_white",
    )
    fig.update_xaxes(showgrid=True, gridcolor="#f0f0f0")
    fig.update_yaxes(
        title_text="Price", row=1, col=1, showgrid=True, gridcolor="#f0f0f0"
    )

    if show_rsi:
        fig.update_yaxes(
            title_text="RSI",
            row=2,
            col=1,
            secondary_y=False,
            range=[0, 100],
            showgrid=True,
            gridcolor="#f0f0f0",
        )
    fig.update_yaxes(title_text="ROC %", row=2, col=1, secondary_y=True, showgrid=False)

    return fig


# ── Navigation ────────────────────────────────────────────────────────────────

view = st.sidebar.radio(
    "View",
    ["Single Stock", "Screener", "Portfolio"],
    index=0,
)
st.sidebar.markdown("---")
st.sidebar.caption(
    f"Universe: {len(tickers)} tickers  \n"
    f"Data: {close.index[0].date()} → {close.index[-1].date()}  \n"
    f"ROC lookback: {ROC_LOOKBACK} days"
)


# ══════════════════════════════════════════════════════════════════════════════
# 1. SINGLE STOCK VIEW
# ══════════════════════════════════════════════════════════════════════════════

if view == "Single Stock":
    st.header("Single Stock View")

    col_pick, col_btn = st.columns([4, 1])
    with col_pick:
        ticker = st.selectbox("Select ticker", tickers, index=0)
    with col_btn:
        st.write("")  # vertical alignment
        st.write("")
        if st.button("＋ Add to portfolio", key="single_add"):
            add_to_portfolio(ticker)
            st.toast(f"{ticker} added to portfolio.")

    price = close[ticker].dropna()
    fig = _price_indicator_chart(
        price_series=price,
        rsi_series=ind.rsi[ticker].dropna(),
        roc_series=ind.roc[ticker].dropna(),
        ma50=ind.ma50[ticker].dropna(),
        ma200=ind.ma200[ticker].dropna(),
        bb_upper=ind.bb_upper[ticker].dropna(),
        bb_mid=ind.bb_mid[ticker].dropna(),
        bb_lower=ind.bb_lower[ticker].dropna(),
        title=ticker,
        show_rsi=True,
    )
    st.plotly_chart(fig, width="stretch")


# ══════════════════════════════════════════════════════════════════════════════
# 2. SCREENER VIEW
# ══════════════════════════════════════════════════════════════════════════════

elif view == "Screener":
    st.header("Screener")
    st.caption(f"Snapshot as of {close.index[-1].date()} — one row per ticker.")

    snap = ind.latest_snapshot()

    with st.expander("Filters", expanded=True):
        f1, f2, f3, f4, f5 = st.columns(5)
        with f1:
            rsi_range = st.slider("RSI(14)", 0.0, 100.0, (0.0, 100.0), step=1.0)
        with f2:
            roc_col = f"ROC({ROC_LOOKBACK}d) %"
            roc_min_v = float(snap[roc_col].min())
            roc_max_v = float(snap[roc_col].max())
            roc_range = st.slider(
                roc_col,
                roc_min_v,
                roc_max_v,
                (roc_min_v, roc_max_v),
                step=0.5,
            )
        with f3:
            vs_ma50_min = st.number_input(
                "vs MA50 % ≥", value=float(snap["vs MA50 %"].min()), step=1.0
            )
            vs_ma50_max = st.number_input(
                "vs MA50 % ≤", value=float(snap["vs MA50 %"].max()), step=1.0
            )
        with f4:
            vs_ma200_min = st.number_input(
                "vs MA200 % ≥", value=float(snap["vs MA200 %"].min()), step=1.0
            )
            vs_ma200_max = st.number_input(
                "vs MA200 % ≤", value=float(snap["vs MA200 %"].max()), step=1.0
            )
        with f5:
            above_ma50 = st.checkbox("Above MA50 only")
            above_ma200 = st.checkbox("Above MA200 only")

    mask = (
        snap["RSI(14)"].between(*rsi_range)
        & snap[roc_col].between(*roc_range)
        & snap["vs MA50 %"].between(vs_ma50_min, vs_ma50_max)
        & snap["vs MA200 %"].between(vs_ma200_min, vs_ma200_max)
    )
    if above_ma50:
        mask &= snap["vs MA50 %"] > 0
    if above_ma200:
        mask &= snap["vs MA200 %"] > 0

    filtered = snap[mask].copy()
    st.caption(f"{len(filtered)} / {len(snap)} tickers shown")

    # Render table with per-row Add buttons
    # Use a two-column layout: table + buttons column
    if filtered.empty:
        st.info("No tickers match the current filters.")
    else:
        # Format numerics for display
        display = filtered.copy()
        for col in display.columns:
            if col != "Ticker":
                display[col] = display[col].round(2)

        col_table, col_actions = st.columns([10, 1])
        with col_table:
            st.dataframe(
                display,
                width="stretch",
                hide_index=True,
                column_config={
                    "Ticker": st.column_config.TextColumn("Ticker"),
                    "Last Close": st.column_config.NumberColumn(
                        "Last Close", format="%.2f"
                    ),
                    "RSI(14)": st.column_config.NumberColumn("RSI(14)", format="%.1f"),
                    roc_col: st.column_config.NumberColumn(roc_col, format="%.1f%%"),
                    "vs MA50 %": st.column_config.NumberColumn(
                        "vs MA50 %", format="%.1f%%"
                    ),
                    "vs MA200 %": st.column_config.NumberColumn(
                        "vs MA200 %", format="%.1f%%"
                    ),
                    "vs BB Upper %": st.column_config.NumberColumn(
                        "vs BB Upper %", format="%.1f%%"
                    ),
                    "vs BB Lower %": st.column_config.NumberColumn(
                        "vs BB Lower %", format="%.1f%%"
                    ),
                    "BB Width %": st.column_config.NumberColumn(
                        "BB Width %", format="%.1f%%"
                    ),
                },
            )

        st.markdown("**Add tickers to portfolio:**")
        add_cols = st.columns(6)
        for i, row_ticker in enumerate(filtered["Ticker"].tolist()):
            with add_cols[i % 6]:
                if st.button(f"+ {row_ticker}", key=f"add_{row_ticker}"):
                    add_to_portfolio(row_ticker)
                    st.toast(f"{row_ticker} added to portfolio.")


# ══════════════════════════════════════════════════════════════════════════════
# 3. PORTFOLIO VIEW
# ══════════════════════════════════════════════════════════════════════════════

elif view == "Portfolio":
    st.header("Portfolio View")
    st.caption(
        "Equal-weighted, buy-and-hold from earliest common date. "
        "Weights are set once and drift naturally — no periodic rebalancing."
    )

    portfolio = st.session_state.portfolio

    if not portfolio:
        st.info(
            "Your portfolio is empty. Add tickers from the Single Stock or Screener views."
        )
    else:
        # Remove button per ticker
        st.markdown("**Holdings:**")
        rm_cols = st.columns(min(len(portfolio), 8))
        for i, t in enumerate(list(portfolio)):
            with rm_cols[i % 8]:
                if st.button(f"✕ {t}", key=f"rm_{t}"):
                    remove_from_portfolio(t)
                    st.rerun()

        # Filter to tickers that exist in our universe
        valid = [t for t in portfolio if t in close.columns]
        missing = [t for t in portfolio if t not in close.columns]
        if missing:
            st.warning(f"Not in universe (ignoring): {', '.join(missing)}")

        if not valid:
            st.info("No valid tickers remaining.")
        else:
            # Align to common date range
            port_close = close[valid].dropna(how="any")
            if port_close.empty:
                st.warning("No overlapping data across all holdings.")
            else:
                # Rebase each series to 100 at first common date, equal-weight average
                rebased = port_close.div(port_close.iloc[0]) * 100
                portfolio_value = rebased.mean(axis=1)
                portfolio_value.name = "Portfolio"

                # ROC on the portfolio value series (same definition)
                import pandas_ta as ta_local

                port_roc = ta_local.roc(portfolio_value, length=ROC_LOOKBACK)

                fig = _price_indicator_chart(
                    price_series=portfolio_value,
                    rsi_series=None,
                    roc_series=port_roc.dropna(),
                    title=f"Portfolio ({len(valid)} holdings, rebased to 100)",
                    show_rsi=False,
                )
                st.plotly_chart(fig, width="stretch")

                st.caption(
                    f"Start date: {port_close.index[0].date()}  |  "
                    f"Holdings: {', '.join(valid)}"
                )
