# backtest.py
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import vectorbt as vbt

from DataLoader import DataLoader
from config import cfg
from strategies import momentum_sharpe_optimised, momentum_with_regime

STRATEGY = momentum_sharpe_optimised  # swap to momentum_with_regime to compare
STRATEGY_NAME = STRATEGY.__name__


def run_backtest():
    loader = DataLoader("tickers.txt", cfg.start_date, cfg.end_date, cfg.min_history)
    close = loader.download_clean_data()

    benchmark = loader.download_benchmark(close.index, ticker="^OMX")
    if benchmark is None:
        print("WARNING: no benchmark downloaded — using universe mean as regime proxy")
        benchmark = close.mean(axis=1)

    target_weights = STRATEGY(close, benchmark)

    pf = vbt.Portfolio.from_orders(
        close,
        size=target_weights,
        size_type="targetpercent",
        init_cash=cfg.initial_cash,
        fees=cfg.fees,
        slippage=cfg.slippage,
        group_by=True,
        cash_sharing=True,
        freq="D",  # business-day frequency — avoids weekend gap errors
    )

    print(f"\n===== BACKTEST RESULTS: {STRATEGY_NAME} =====\n")
    stats = pf.stats()
    print(stats)
    stats.to_csv(f"stats_{STRATEGY_NAME}.csv")

    # --- Plot: equity curve + drawdown ----------------------------------------
    portfolio_value = pf.value()
    if isinstance(portfolio_value, pd.DataFrame):
        portfolio_value = portfolio_value.iloc[:, 0]

    drawdown = (portfolio_value / portfolio_value.cummax() - 1) * 100  # percent

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.7, 0.3],
        vertical_spacing=0.04,
    )

    fig.add_trace(
        go.Scatter(
            x=portfolio_value.index,
            y=portfolio_value.values,
            name=STRATEGY_NAME,
            line=dict(color="royalblue", width=2),
        ),
        row=1,
        col=1,
    )

    bench_norm = (benchmark / benchmark.dropna().iloc[0]) * cfg.initial_cash
    fig.add_trace(
        go.Scatter(
            x=bench_norm.index,
            y=bench_norm.values,
            name="OMXS30 (buy & hold)",
            line=dict(color="gray", width=1.5, dash="dash"),
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=drawdown.index,
            y=drawdown.values,
            name="Drawdown %",
            fill="tozeroy",
            line=dict(color="crimson", width=1),
        ),
        row=2,
        col=1,
    )

    fig.update_layout(
        title=f"Portfolio Value — {STRATEGY_NAME}",
        yaxis_title="Portfolio Value (SEK)",
        yaxis2_title="Drawdown %",
        hovermode="x unified",
        template="plotly_white",
    )

    out_html = f"portfolio_{STRATEGY_NAME}.html"
    fig.write_html(out_html)
    print(f"\nPlot saved to {out_html}")

    return pf


if __name__ == "__main__":
    run_backtest()
