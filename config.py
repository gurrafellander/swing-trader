# config.py

# Backtest Settings
INITIAL_CASH = 100000
FEES         = 0.001    # 0.10% per trade
SLIPPAGE     = 0.0005   # 0.05% per trade

# Portfolio Constraints
MAX_POSITIONS = 10
POSITION_SIZE = 1 / MAX_POSITIONS   # used by RSI strategy
SIZE_TYPE     = "percent"

# Data Settings
START_DATE = "2018-01-01"
END_DATE   = None   # None = today

# Minimum history requirement (days)
MIN_HISTORY = 200

# RSI Strategy Parameters
RSI_WINDOW = 14
RSI_ENTRY  = 30
RSI_EXIT   = 55

# Momentum Strategy Parameters
TOP_N             = 15     # number of stocks to hold
MOMENTUM_LOOKBACK = 252    # ~6 months (skip last 21 days to avoid reversal)
VOL_LOOKBACK      = 63     # 3-month vol for inverse-vol weighting
SPY_MA            = 200    # regime filter: benchmark must be above this MA
USE_RANKING       = True

# Rebalance frequency
# "MS" = monthly (first trading day of month) — best for momentum
# "W"  = weekly
REBALANCE_FREQ = "MS"
