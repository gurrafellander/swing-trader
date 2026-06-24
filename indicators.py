"""
Vectorized indicator computation for the full ticker universe.
All functions operate on a close-price DataFrame (dates × tickers).
"""

import pandas as pd
import pandas_ta as ta

# Single ROC lookback used everywhere — change here to tune globally.
ROC_LOOKBACK = 21


def compute_ma(close: pd.DataFrame, window: int) -> pd.DataFrame:
    return close.rolling(window).mean()


def compute_bollinger(close: pd.DataFrame, window: int = 20, std: float = 2.0):
    """Returns (upper, mid, lower) DataFrames."""
    mid = close.rolling(window).mean()
    sigma = close.rolling(window).std()
    return mid + std * sigma, mid, mid - std * sigma


def compute_rsi(close: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """RSI computed column-by-column via pandas_ta."""
    result = pd.DataFrame(index=close.index, columns=close.columns, dtype=float)
    for col in close.columns:
        rsi = ta.rsi(close[col], length=window)
        if rsi is not None:
            result[col] = rsi
    return result


def compute_roc(close: pd.DataFrame, length: int = ROC_LOOKBACK) -> pd.DataFrame:
    """Rate of Change (%) via pandas_ta — same definition everywhere."""
    result = pd.DataFrame(index=close.index, columns=close.columns, dtype=float)
    for col in close.columns:
        roc = ta.roc(close[col], length=length)
        if roc is not None:
            result[col] = roc
    return result


class Indicators:
    """Holds all pre-computed indicator DataFrames for the full universe."""

    def __init__(self, close: pd.DataFrame):
        self.close = close
        self.ma50 = compute_ma(close, 50)
        self.ma200 = compute_ma(close, 200)
        self.bb_upper, self.bb_mid, self.bb_lower = compute_bollinger(close)
        self.rsi = compute_rsi(close)
        self.roc = compute_roc(close)

    def latest_snapshot(self) -> pd.DataFrame:
        """One-row-per-ticker snapshot of the most recent indicator values."""
        last = self.close.index[-1]

        def _last(df):
            return df.loc[last] if last in df.index else df.iloc[-1]

        close_s = _last(self.close)
        ma50_s = _last(self.ma50)
        ma200_s = _last(self.ma200)
        bb_upper_s = _last(self.bb_upper)
        bb_lower_s = _last(self.bb_lower)
        bb_mid_s = _last(self.bb_mid)
        rsi_s = _last(self.rsi)
        roc_s = _last(self.roc)

        snap = pd.DataFrame(
            {
                "Last Close": close_s,
                "RSI(14)": rsi_s,
                f"ROC({ROC_LOOKBACK}d) %": roc_s,
                "vs MA50 %": (close_s / ma50_s - 1) * 100,
                "vs MA200 %": (close_s / ma200_s - 1) * 100,
                "vs BB Upper %": (close_s / bb_upper_s - 1) * 100,
                "vs BB Lower %": (close_s / bb_lower_s - 1) * 100,
                "BB Width %": (bb_upper_s - bb_lower_s) / bb_mid_s * 100,
            }
        )
        snap.index.name = "Ticker"
        return snap.reset_index()
