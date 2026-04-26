# config.py

# Backtest Settings
INITIAL_CASH = 100000
FEES = 0.001
SLIPPAGE = 0.0005

# Portfolio Constraints
MAX_POSITIONS = 10
POSITION_SIZE = 1 / MAX_POSITIONS  # equal weight
SIZE_TYPE = "percent"

# Data Settings
START_DATE = "2018-01-01"
END_DATE = None  # None = today
TIMEFRAME = "1d"

# Minimum history requirement (days)
MIN_HISTORY = 200

# RSI Strategy Parameters
RSI_WINDOW = 14
RSI_ENTRY = 30
RSI_EXIT = 55

# Ranking enabled (True/False)
USE_RANKING = True


TOP_N = 10
MOMENTUM_LOOKBACK = 126  # ~6 months
VOL_LOOKBACK = 63  # 3 months volatility

SPY_MA = 200  # regime filter

REBALANCE_FREQ = "W"  # weekly rebalance
