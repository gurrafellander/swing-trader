"""
backtest.py
───────────
Runs all strategies and plots equity curves + drawdown on a single HTML chart.
"""

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import vectorbt as vbt

from DataLoader import DataLoader
from config import cfg
from strategies import (
    bollinger_mean_reversion,
    ma_crossover,
    momentum_sharpe_optimised,
    momentum_with_regime,
    rising_momentum_sharpe,
    rsi_mean_reversion,
)

# ── Colour palette (one per curve) ───────────────────────────────────────────
COLOURS = {
    "RSI Mean Reversion": "#4C9BE8",
    "Momentum + InvVol + Regime": "#F4A261",
    "Momentum + Sharpe + Regime": "#2A9D8F",
    "Rising Momentum + Sharpe": "#E76F51",
    "MA Crossover": "#9B5DE5",
    "Bollinger Mean Reversion": "#F15BB5",
    "Buy & Hold (Universe)": "#aaaaaa",
    "OMXS30 Buy & Hold": "#666666",
}


def _run_portfolio(weights: pd.DataFrame, close: pd.DataFrame):
    """Shared vbt.Portfolio.from_orders call."""
    return vbt.Portfolio.from_orders(
        close,
        size=weights,
        size_type="targetpercent",
        init_cash=cfg.initial_cash,
        fees=cfg.fees,
        slippage=cfg.slippage,
        group_by=True,
        cash_sharing=True,
        freq="D",
    )


def _equity(pf) -> pd.Series:
    v = pf.value()
    return v.iloc[:, 0] if isinstance(v, pd.DataFrame) else v


def _drawdown(equity: pd.Series) -> pd.Series:
    return (equity / equity.cummax() - 1) * 100


def run_backtest():
    # ── Data ─────────────────────────────────────────────────────────────────
    loader = DataLoader("tickers.txt", cfg.start_date, cfg.end_date, cfg.min_history)
    close = loader.download_clean_data()
    benchmark = loader.download_benchmark(close.index, ticker="^OMX")

    if benchmark is None:
        print("WARNING: no benchmark downloaded — using universe mean as regime proxy")
        benchmark = close.mean(axis=1)

    bm_returns = benchmark.pct_change().reindex(close.index).fillna(0)

    # ── Universe buy & hold (equal weight, rebalanced monthly) ───────────────
    ew_weight = 1.0 / close.shape[1]
    ew_weights = pd.DataFrame(
        ew_weight, index=close.index, columns=close.columns, dtype="float32"
    )
    # Only set weights on rebalance days so vbt treats intervening rows as hold
    rebal_mask = pd.Series(False, index=close.index)
    rebal_mask.iloc[0] = True  # first day to open positions
    ew_out = pd.DataFrame(
        float("nan"), index=close.index, columns=close.columns, dtype="float32"
    )
    ew_out.loc[rebal_mask[rebal_mask].index] = ew_weight

    # ── Define all strategies ─────────────────────────────────────────────────
    # Each entry: (display_name, callable, kwargs)
    strategies = [
        ("RSI Mean Reversion", lambda: rsi_mean_reversion(close)),
        ("Momentum + InvVol + Regime", lambda: momentum_with_regime(close, benchmark)),
        (
            "Momentum + Sharpe + Regime",
            lambda: momentum_sharpe_optimised(close, benchmark),
        ),
        ("Rising Momentum + Sharpe", lambda: rising_momentum_sharpe(close)),
        ("MA Crossover", lambda: ma_crossover(close)),
        ("Bollinger Mean Reversion", lambda: bollinger_mean_reversion(close)),
    ]

    # ── Run all strategies ────────────────────────────────────────────────────
    results: dict[str, dict] = {}

    for name, build_weights in strategies:
        print(f"  Running: {name} ...", end=" ", flush=True)
        weights = build_weights()
        pf = _run_portfolio(weights, close)
        eq = _equity(pf)
        results[name] = {
            "pf": pf,
            "equity": eq,
            "stats": pf.stats(settings=dict(benchmark_rets=bm_returns)),
        }
        print(
            f"done  →  {eq.iloc[-1]:,.0f} SEK  ({(eq.iloc[-1]/cfg.initial_cash - 1)*100:.1f} %)"
        )

    # Universe B&H
    pf_bh = _run_portfolio(ew_out, close)
    eq_bh = _equity(pf_bh)
    results["Buy & Hold (Universe)"] = {
        "pf": pf_bh,
        "equity": eq_bh,
        "stats": pf_bh.stats(settings=dict(benchmark_rets=bm_returns)),
    }
    print(
        f"  Buy & Hold (Universe): {eq_bh.iloc[-1]:,.0f} SEK  ({(eq_bh.iloc[-1]/cfg.initial_cash - 1)*100:.1f} %)"
    )

    # ── Print stats table ─────────────────────────────────────────────────────
    print("\n" + "═" * 60)
    stat_keys = [
        "Total Return [%]",
        "Sharpe Ratio",
        "Sortino Ratio",
        "Calmar Ratio",
        "Max Drawdown [%]",
        "Total Trades",
        "Total Fees Paid",
    ]
    stats_df = pd.DataFrame(
        {name: d["stats"][stat_keys] for name, d in results.items()}
    ).T
    print(stats_df.to_string())
    stats_df.to_csv("all_stats.csv")
    print("\nStats saved to all_stats.csv")

    # ── Build combined chart ───────────────────────────────────────────────────
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.68, 0.32],
        vertical_spacing=0.04,
        subplot_titles=("Portfolio Value (SEK)", "Drawdown (%)"),
    )

    # Strategy equity curves
    for name, data in results.items():
        eq = data["equity"]
        colour = COLOURS.get(name, "#888888")
        width = 1.5 if "Hold" in name else 2
        dash = "dot" if "Hold" in name else "solid"
        fig.add_trace(
            go.Scatter(
                x=eq.index,
                y=eq.values,
                name=name,
                line=dict(color=colour, width=width, dash=dash),
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=eq.index,
                y=_drawdown(eq).values,
                name=name,
                line=dict(color=colour, width=1, dash=dash),
                showlegend=False,
            ),
            row=2,
            col=1,
        )

    # OMXS30 buy & hold
    bench_norm = (benchmark / benchmark.dropna().iloc[0]) * cfg.initial_cash
    fig.add_trace(
        go.Scatter(
            x=bench_norm.index,
            y=bench_norm.values,
            name="OMXS30 Buy & Hold",
            line=dict(color=COLOURS["OMXS30 Buy & Hold"], width=1.5, dash="dash"),
        ),
        row=1,
        col=1,
    )

    fig.update_layout(
        title="Strategy Comparison — All Strategies vs Benchmarks",
        hovermode="x unified",
        template="plotly_white",
        legend=dict(orientation="v", x=1.01, y=1),
        height=750,
    )
    fig.update_yaxes(title_text="Value (SEK)", row=1, col=1)
    fig.update_yaxes(title_text="Drawdown %", row=2, col=1)

    fig.write_html("strategy_comparison.html")
    print("Chart saved to strategy_comparison.html")

    return results


if __name__ == "__main__":
    run_backtest()
