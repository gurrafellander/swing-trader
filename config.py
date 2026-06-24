from dataclasses import dataclass
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

    # ── Trend-filtered dip buy ────────────────────────────────────────────────
    dip_trend_ma: int = 150  # long-term MA defining "uptrend" for dip-buy

    # ── Momentum / Sharpe strategies ──────────────────────────────────────────
    top_n: int = 10  # final portfolio size
    top_candidates: int = 20  # pre-filter pool fed into Sharpe optimiser
    max_position_weight: float = 0.4  # per-stock ceiling fed to the Sharpe optimiser
    momentum_lookback: int = 126  # ~6 months; last 21 days skipped (reversal)
    vol_lookback: int = 63  # trailing window for return/vol estimation
    spy_ma: int = 200  # regime filter MA length

    # ── Rising Momentum strategy ──────────────────────────────────────────────
    # Signal detection uses fast ROC windows (catches early breakouts), but
    # the Sharpe optimiser needs its own appropriately-scaled return/covariance
    # window — kept separate so tuning one doesn't silently break the other.
    roc_short_lookback: int = 10  # ~2 weeks  — captures recent turn
    roc_long_lookback: int = 31  # ~6 weeks  — trend baseline
    roc_near_zero_threshold: float = 5.0  # |roc_long| must be below this (%)
    # to qualify as "not yet priced in"
    rising_vol_lookback: int = 21  # ~1 month — return window for THIS strategy's
    # Sharpe optimiser (scaled to match its fast
    # signal cadence, independent of vol_lookback)

    # ── Rebalance ─────────────────────────────────────────────────────────────
    # "MS" = first trading day of each month (best for momentum)
    # "W"  = weekly
    rebalance_freq: str = "MS"
    rebal_freq: int = 21  # fallback cadence in trading days

    # ── Misc ──────────────────────────────────────────────────────────────────
    use_ranking: bool = True


# Shared singleton — import this everywhere
cfg = Config()
