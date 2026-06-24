"""
Regression tests: verify generate_signals matches the backtest engine for
known historical rebalance months.

Both paths are run with the *same* 2-year data window (anchored to month_start)
so the comparison is apples-to-apples.  On first run these fetch from Yahoo
Finance; subsequent runs are fast because DataLoader serves from the local cache.

Run with:
    pytest tests/test_backtest_consistency.py -v -s

Note on divergence from backtest.py
------------------------------------
backtest.py uses cfg.start_date = "2018-01-01" (full history).
generate_signals uses a 2-year rolling window anchored to month_start.
Running the strategy on different-length histories can produce different
ticker sets because the min_history filter sees different row counts.
These tests use the 2-year window for *both* paths, which is the correct
reference for live-trading decisions.
"""
from datetime import date, timedelta

import pandas as pd
import pytest

from DataLoader import DataLoader
from config import cfg
from strategies import rising_momentum_sharpe
from generate_signals import get_raw_weights

WEIGHT_TOL = 0.01  # 1 percentage point

# Settled historical dates — each represents a different rebalance month.
# Pick dates well in the past so prices are final and cache stays stable.
TEST_DATES = [
    date(2025, 3, 28),   # mid-month — should resolve to 2025-03-03 rebalance
    date(2025, 1, 15),   # early in month
    date(2024, 11, 20),  # Q4
    date(2024, 9, 3),    # early-month edge: rebalance is 2024-09-02
]


# ── helpers ────────────────────────────────────────────────────────────────────

def _backtest_row(test_date: date) -> tuple[pd.Series, pd.Timestamp | None]:
    """
    Run rising_momentum_sharpe with the same 2-year window generate_signals uses.
    Returns (weight_row, rebalance_ts).
    """
    month_start = test_date.replace(day=1)
    bt_start = (month_start - timedelta(days=2 * 365)).strftime("%Y-%m-%d")
    bt_end = test_date.strftime("%Y-%m-%d")
    loader = DataLoader("tickers.txt", bt_start, bt_end, cfg.min_history)
    close = loader.download_clean_data()
    weights = rising_momentum_sharpe(close)
    last_ts = close.index[-1]
    for ts in reversed(weights.loc[:last_ts].index.tolist()):
        row = weights.loc[ts].dropna()
        row = row[row > 0]
        if not row.empty:
            return row, ts
    return pd.Series(dtype=float), None


# ── module-level fixture — download once, reuse across all tests ───────────────

@pytest.fixture(scope="module")
def all_results():
    """
    Collect (backtest_row, bt_ts, gs_row, gs_ts) for every TEST_DATE.
    The DataLoader cache means only the first run hits the network.
    """
    out = {}
    for d in TEST_DATES:
        bt_row, bt_ts = _backtest_row(d)
        gs_row, gs_ts, _ = get_raw_weights(d)
        out[d] = {"bt_row": bt_row, "bt_ts": bt_ts, "gs_row": gs_row, "gs_ts": gs_ts}
    return out


# ── tests ──────────────────────────────────────────────────────────────────────

class TestBacktestVsSignals:
    """
    Parameterised tests that run for each date in TEST_DATES.
    Each test receives the pre-downloaded fixture so there is no repeated I/O.
    """

    @pytest.mark.parametrize("test_date", TEST_DATES)
    def test_both_paths_produced_positions(self, all_results, test_date):
        r = all_results[test_date]
        assert not r["bt_row"].empty, f"[{test_date}] Backtest returned no positions"
        assert not r["gs_row"].empty, f"[{test_date}] generate_signals returned no positions"

    @pytest.mark.parametrize("test_date", TEST_DATES)
    def test_rebalance_date_matches(self, all_results, test_date):
        r = all_results[test_date]
        assert r["bt_ts"] == r["gs_ts"], (
            f"[{test_date}] Rebalance date mismatch:\n"
            f"  backtest       → {r['bt_ts']}\n"
            f"  generate_signals → {r['gs_ts']}"
        )

    @pytest.mark.parametrize("test_date", TEST_DATES)
    def test_ticker_set_matches(self, all_results, test_date):
        r = all_results[test_date]
        bt_tickers = set(r["bt_row"].index)
        gs_tickers = set(r["gs_row"].index)
        only_bt = sorted(bt_tickers - gs_tickers)
        only_gs = sorted(gs_tickers - bt_tickers)
        assert bt_tickers == gs_tickers, (
            f"[{test_date}] Ticker set mismatch:\n"
            f"  backtest only:        {only_bt}\n"
            f"  generate_signals only: {only_gs}\n"
            f"  backtest:             {sorted(bt_tickers)}\n"
            f"  generate_signals:     {sorted(gs_tickers)}"
        )

    @pytest.mark.parametrize("test_date", TEST_DATES)
    def test_weights_within_tolerance(self, all_results, test_date):
        r = all_results[test_date]
        common = set(r["bt_row"].index) & set(r["gs_row"].index)
        mismatches = []
        for tkr in sorted(common):
            bt_w = float(r["bt_row"].get(tkr, 0.0))
            gs_w = float(r["gs_row"].get(tkr, 0.0))
            diff = abs(bt_w - gs_w)
            if diff > WEIGHT_TOL:
                mismatches.append(
                    f"  {tkr}: bt={bt_w*100:.2f}%  gs={gs_w*100:.2f}%  diff={diff*100:.2f}pp"
                )
        assert not mismatches, (
            f"[{test_date}] Weight differences exceed {WEIGHT_TOL*100:.0f}pp:\n"
            + "\n".join(mismatches)
        )

    @pytest.mark.parametrize("test_date", TEST_DATES)
    def test_weights_sum_to_one(self, all_results, test_date):
        r = all_results[test_date]
        for label, row in (("backtest", r["bt_row"]), ("generate_signals", r["gs_row"])):
            total = float(row.sum())
            assert abs(total - 1.0) < 0.02, (
                f"[{test_date}] {label} weights sum to {total:.4f}, expected ~1.0"
            )
