import pandas as pd
import vectorbt as vbt


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

    def download_clean_data(self):
        print("Downloading data individually...")
        close_dict = {}
        if len(self.tickers) == 0:
            raise ValueError("No tickers in tickers list.")
        for ticker in self.tickers:
            try:
                data = vbt.YFData.download(
                    ticker, start=self.start_date, end=self.end_date, interval="1d"
                )
                close = data.get("Close")
                if close is None or close.empty:
                    print(f"Skipping {ticker} (no data)")
                    continue
                close_dict[ticker] = close
            except Exception as e:
                print(f"Skipping {ticker}: {e}")

        if not close_dict:
            raise ValueError("No valid tickers downloaded.")

        close_df = pd.concat(close_dict.values(), axis=1)
        close_df.columns = list(close_dict.keys())
        close_df = close_df.sort_index().ffill().dropna(how="all")
        valid_cols = close_df.count() >= self.min_history
        close_df = close_df.loc[:, valid_cols]
        print(f"Remaining tickers after MIN_HISTORY filter: {close_df.shape[1]}")
        print(f"Final dataset shape: {close_df.shape}")
        return close_df

    def download_benchmark(self, index, ticker="^OMX"):
        print(f"Downloading benchmark ({ticker})...")
        try:
            data = vbt.YFData.download(
                ticker, start=self.start_date, end=self.end_date, interval="1d"
            )
            bench = data.get("Close")
            bench = bench.reindex(index).ffill()
            # squeeze in case it comes back as DataFrame
            if isinstance(bench, pd.DataFrame):
                bench = bench.iloc[:, 0]
            return bench
        except Exception as e:
            print(f"Could not download benchmark: {e}")
            return None
