from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Config:
    # ── Backtest ──────────────────────────────────────────────────────────────
    initial_cash: float = 100_000
    fees: float = 0.001  # 0.10 % per trade
    slippage: float = 0.0005  # 0.05 % per trade
    rf_daily: float = 0.0

    # ── Portfolio constraints ─────────────────────────────────────────────────
    max_positions: int = 10
    size_type: str = "percent"

    @property
    def position_size(self) -> float:
        return 1.0 / self.max_positions

    # ── Data ─────────────────────────────────────────────────────────────────
    start_date: str = "2018-01-01"
    end_date: Optional[str] = None  # None → today
    min_history: int = 200

    # ── RSI mean-reversion ────────────────────────────────────────────────────
    rsi_window: int = 14
    rsi_entry: int = 30
    rsi_exit: int = 55

    # ── Momentum / Sharpe strategy ────────────────────────────────────────────
    top_n: int = 10  # final portfolio size
    top_candidates: int = 20  # pre-filter pool fed into Sharpe optimiser
    momentum_lookback: int = 126  # ~6 months; last 21 days skipped (reversal)
    vol_lookback: int = 63  # 3-month window for vol / return estimation
    spy_ma: int = 200  # regime filter: benchmark must be above this MA

    # ── Rebalance ─────────────────────────────────────────────────────────────
    # "MS" = first trading day of each month (best for momentum)
    # "W"  = weekly
    rebalance_freq: str = "MS"
    rebal_freq: int = 21  # fallback cadence in trading days

    # ── Misc ──────────────────────────────────────────────────────────────────
    use_ranking: bool = True


# Shared singleton — import this everywhere
cfg = Config()
