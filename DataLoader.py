from datetime import date
from pathlib import Path

import pandas as pd
import vectorbt as vbt

# Local price cache — gitignored, built on first run and kept up to date.
_CACHE_DIR = Path("price_cache")
_CACHE_FILE = _CACHE_DIR / "close_prices.parquet"


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse a tz-aware DataFrame to one tz-naive row per Stockholm calendar date.

    Yahoo Finance returns per-market UTC timestamps (Swedish stocks at 22:00 UTC,
    UK stocks at 23:00 UTC, US stocks at 04:00 UTC next day), so pd.concat on
    individually downloaded tickers creates ~3 rows per trading day.  Converting
    to Stockholm time maps all sessions on the same calendar day to the same date
    before groupby collapses them.
    """
    if df.index.tz is not None:
        df.index = df.index.tz_convert("Europe/Stockholm").normalize().tz_localize(None)
    else:
        df.index = df.index.normalize()
    return df.groupby(df.index).last()


def _download_raw(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """Download tickers from Yahoo Finance and return a normalized close DataFrame."""
    close_dict = {}
    for ticker in tickers:
        try:
            data = vbt.YFData.download(ticker, start=start, end=end, interval="1d")
            close = data.get("Close")
            if close is None or close.empty:
                print(f"  Skipping {ticker} (no data)")
                continue
            close_dict[ticker] = close
        except Exception as e:
            print(f"  Skipping {ticker}: {e}")

    if not close_dict:
        return pd.DataFrame()

    df = pd.concat(close_dict.values(), axis=1)
    df.columns = list(close_dict.keys())
    return _normalize(df)


class DataLoader:
    def __init__(self, ticker_path, start_date, end_date, min_history):
        self.ticker_path = ticker_path
        self.tickers = self.load_universe()
        self.start_date = start_date
        self.end_date = end_date
        self.min_history = min_history

    def load_universe(self):
        with open(self.ticker_path) as f:
            return [line.strip() for line in f if line.strip()]

    # ── cache helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _load_cache() -> pd.DataFrame | None:
        try:
            return pd.read_parquet(_CACHE_FILE)
        except Exception:
            return None

    @staticmethod
    def _save_cache(df: pd.DataFrame) -> None:
        _CACHE_DIR.mkdir(exist_ok=True)
        df.to_parquet(_CACHE_FILE)

    # ── main entry point ───────────────────────────────────────────────────────

    def download_clean_data(self) -> pd.DataFrame:
        today = date.today()
        today_str = str(today)
        request_start = date.fromisoformat(self.start_date)

        cached = self._load_cache()

        if cached is None:
            # ── First run: download everything and cache it ──────────────────
            print(f"No price cache — downloading full history ({self.start_date} → {today_str})…")
            cached = _download_raw(self.tickers, self.start_date, today_str)
            if not cached.empty:
                self._save_cache(cached)

        else:
            cache_first = cached.index[0].date()
            cache_last  = cached.index[-1].date()
            changed = False

            # ── 1. Extend backward if caller needs older data ────────────────
            if request_start < cache_first:
                print(f"Extending cache backward ({self.start_date} → {cache_first})…")
                early = _download_raw(self.tickers, self.start_date, str(cache_first))
                if not early.empty:
                    cached = pd.concat([early, cached])
                    cached = cached[~cached.index.duplicated(keep="last")].sort_index()
                    changed = True

            # ── 2. Add tickers that are new in tickers.txt ───────────────────
            missing = [t for t in self.tickers if t not in cached.columns]
            if missing:
                print(f"Adding {len(missing)} new ticker(s) to cache…")
                new_cols = _download_raw(missing, self.start_date, today_str)
                if not new_cols.empty:
                    cached = cached.join(new_cols, how="outer").sort_index()
                    changed = True

            # ── 3. Bring tail up to today ────────────────────────────────────
            if cache_last < today:
                print(f"Updating cache ({cache_last} → {today_str})…")
                tail = _download_raw(self.tickers, str(cache_last), today_str)
                if not tail.empty:
                    cached = pd.concat([cached, tail])
                    cached = cached[~cached.index.duplicated(keep="last")].sort_index()
                    changed = True
            else:
                print("Price cache is up to date — skipping download.")

            if changed:
                self._save_cache(cached)

        # ── Slice to requested window, apply min_history filter ───────────────
        end_dt = pd.Timestamp(self.end_date) if self.end_date else pd.Timestamp(today)
        available = [t for t in self.tickers if t in cached.columns]
        close_df = cached.loc[pd.Timestamp(self.start_date):end_dt, available]

        close_df = close_df.sort_index().ffill().dropna(how="all")
        valid_cols = close_df.count() >= self.min_history
        close_df = close_df.loc[:, valid_cols]

        print(f"Remaining tickers after MIN_HISTORY filter: {close_df.shape[1]}")
        print(f"Final dataset shape: {close_df.shape}")
        return close_df

    # ── benchmark (not cached — single ticker, fast) ──────────────────────────

    def download_benchmark(self, index, ticker="^OMX"):
        print(f"Downloading benchmark ({ticker})…")
        try:
            data = vbt.YFData.download(
                ticker, start=self.start_date, end=self.end_date, interval="1d"
            )
            bench = data.get("Close")
            bench = bench.reindex(index).ffill()
            if isinstance(bench, pd.DataFrame):
                bench = bench.iloc[:, 0]
            return bench
        except Exception as e:
            print(f"Could not download benchmark: {e}")
            return None
