from dataclasses import dataclass
from typing import Optional


@dataclass
class Config:
    # ── Data ─────────────────────────────────────────────────────────────────
    start_date: str = "2018-01-01"
    end_date: Optional[str] = None  # None → today
    min_history: int = 200
    ticker_path: str = "tickers.txt"


# Shared singleton — import this everywhere
cfg = Config()
