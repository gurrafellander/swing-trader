"""
Unit tests for generate_signals.py

DataLoader is mocked so these tests run without network access.
"""
import math
from datetime import date
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

import generate_signals as gs


# ─── fixture ─────────────────────────────────────────────────────────────────

@pytest.fixture
def synthetic_close():
    """
    500 business-day price history for 15 tickers.
    Long enough for roc_long_lookback (63) + vol_lookback (63) + several monthly
    rebalances to fire.
    """
    rng = np.random.default_rng(7)
    n_days, n_tickers = 500, 15
    tickers = [f"S{i:02d}.ST" for i in range(n_tickers)]
    # Start early enough that by the end we have at least 2 rebalance cycles
    dates = pd.bdate_range("2022-01-01", periods=n_days)
    log_ret = rng.standard_normal((n_days, n_tickers)) * 0.012
    prices = 100.0 * np.exp(np.cumsum(log_ret, axis=0))
    return pd.DataFrame(prices, index=dates, columns=tickers)


def patch_loader(monkeypatch, close_data):
    """Replace DataLoader so download_clean_data() returns close_data."""
    mock_cls = MagicMock()
    mock_cls.return_value.download_clean_data.return_value = close_data
    monkeypatch.setattr(gs, "DataLoader", mock_cls)


# ─── get_raw_weights ─────────────────────────────────────────────────────────

class TestGetRawWeights:
    def test_returns_three_tuple(self, synthetic_close, monkeypatch):
        patch_loader(monkeypatch, synthetic_close)
        result = gs.get_raw_weights(synthetic_close.index[-1].date())
        assert len(result) == 3

    def test_weights_are_positive(self, synthetic_close, monkeypatch):
        patch_loader(monkeypatch, synthetic_close)
        w, ts, _ = gs.get_raw_weights(synthetic_close.index[-1].date())
        if ts is None:
            pytest.skip("No rebalance found in synthetic data")
        assert (w > 0).all(), f"Non-positive weights: {w[w <= 0]}"

    def test_weights_sum_to_one(self, synthetic_close, monkeypatch):
        patch_loader(monkeypatch, synthetic_close)
        w, ts, _ = gs.get_raw_weights(synthetic_close.index[-1].date())
        if ts is None:
            pytest.skip("No rebalance found in synthetic data")
        assert abs(w.sum() - 1.0) < 1e-3, f"Weights sum to {w.sum():.6f}"

    def test_signal_ts_is_before_or_on_last_date(self, synthetic_close, monkeypatch):
        patch_loader(monkeypatch, synthetic_close)
        last = synthetic_close.index[-1].date()
        _, ts, last_date = gs.get_raw_weights(last)
        if ts is None:
            pytest.skip("No rebalance found in synthetic data")
        assert ts <= last_date

    def test_no_residual_weights(self, synthetic_close, monkeypatch):
        """No weight in (0, 1e-3) — residuals must be zeroed by _sharpe_weights."""
        patch_loader(monkeypatch, synthetic_close)
        w, ts, _ = gs.get_raw_weights(synthetic_close.index[-1].date())
        if ts is None:
            pytest.skip("No rebalance found in synthetic data")
        residuals = w[(w > 0) & (w < 1e-3)]
        assert residuals.empty, f"Residual weights: {residuals.to_dict()}"


class TestSameMonthConsistency:
    """
    Two signal dates in the same calendar month must yield the same rebalance
    date and the same set of selected tickers.

    This is the key regression test for the bugs fixed in _rebalance_mask and
    the DataLoader window anchoring.
    """

    def _two_dates_in_last_month(self, close):
        """Return (first, last) trading day of the most recent month that has ≥2 days."""
        months = sorted(set(zip(close.index.year, close.index.month)), reverse=True)
        for year, month in months:
            days = close.index[(close.index.year == year) & (close.index.month == month)]
            if len(days) >= 2:
                return days[0].date(), days[-1].date()
        return None, None

    def test_same_rebalance_date(self, synthetic_close, monkeypatch):
        d1, d2 = self._two_dates_in_last_month(synthetic_close)
        if d1 is None:
            pytest.skip("Need ≥2 trading days in the last month")

        # Return the same close slice for both calls (anchored to d1 and d2 respectively)
        close_d1 = synthetic_close.loc[:pd.Timestamp(d1)]
        close_d2 = synthetic_close.loc[:pd.Timestamp(d2)]

        call_count = {"n": 0}
        mock_cls = MagicMock()

        def side_effect(*args, **kwargs):
            inst = MagicMock()
            inst.download_clean_data.return_value = (
                close_d1 if call_count["n"] == 0 else close_d2
            )
            call_count["n"] += 1
            return inst

        mock_cls.side_effect = side_effect
        monkeypatch.setattr(gs, "DataLoader", mock_cls)

        _, ts1, _ = gs.get_raw_weights(d1)
        _, ts2, _ = gs.get_raw_weights(d2)

        assert ts1 is not None and ts2 is not None
        assert ts1 == ts2, f"Same month gave different rebalance dates: {ts1} vs {ts2}"

    def test_same_ticker_set(self, synthetic_close, monkeypatch):
        d1, d2 = self._two_dates_in_last_month(synthetic_close)
        if d1 is None:
            pytest.skip("Need ≥2 trading days in the last month")

        close_d1 = synthetic_close.loc[:pd.Timestamp(d1)]
        close_d2 = synthetic_close.loc[:pd.Timestamp(d2)]

        call_count = {"n": 0}
        mock_cls = MagicMock()

        def side_effect(*args, **kwargs):
            inst = MagicMock()
            inst.download_clean_data.return_value = (
                close_d1 if call_count["n"] == 0 else close_d2
            )
            call_count["n"] += 1
            return inst

        mock_cls.side_effect = side_effect
        monkeypatch.setattr(gs, "DataLoader", mock_cls)

        w1, ts1, _ = gs.get_raw_weights(d1)
        w2, ts2, _ = gs.get_raw_weights(d2)

        if ts1 is None or ts2 is None:
            pytest.skip("No rebalance found")

        assert set(w1.index) == set(w2.index), (
            f"Ticker sets differ within same month:\n"
            f"  d1-only: {set(w1.index) - set(w2.index)}\n"
            f"  d2-only: {set(w2.index) - set(w1.index)}"
        )


# ─── generate_signals ────────────────────────────────────────────────────────

class TestGenerateSignals:
    def test_output_dataframe_has_required_columns(self, synthetic_close, monkeypatch):
        patch_loader(monkeypatch, synthetic_close)
        portfolio_value = 100_000.0
        last = synthetic_close.index[-1].date()
        out, actual_date, cash_left, close, metrics = gs.generate_signals(last, portfolio_value)

        required = {"Ticker", "Weight (%)", "Price (SEK)", "Shares", "Actual SEK"}
        assert required.issubset(set(out.columns))

    def test_shares_are_whole_numbers(self, synthetic_close, monkeypatch):
        patch_loader(monkeypatch, synthetic_close)
        last = synthetic_close.index[-1].date()
        out, *_ = gs.generate_signals(last, 100_000.0)

        positions = out[
            ~out["Ticker"].str.startswith(("──", "--"))
            & (out["Ticker"] != "Cash (leftover)")
        ]
        for _, row in positions.iterrows():
            shares = row["Shares"]
            if shares != "":
                assert float(shares) == int(float(shares)), (
                    f"{row['Ticker']} has fractional shares: {shares}"
                )

    def test_actual_sek_plus_cash_equals_portfolio(self, synthetic_close, monkeypatch):
        patch_loader(monkeypatch, synthetic_close)
        last = synthetic_close.index[-1].date()
        out, _, cash_left, _, _ = gs.generate_signals(last, 100_000.0)

        positions = out[
            ~out["Ticker"].str.startswith(("──", "--"))
            & (out["Ticker"] != "Cash (leftover)")
        ]
        total_invested = float(positions["Actual SEK"].sum())
        assert abs(total_invested + cash_left - 100_000.0) < 1.0, (
            f"invested ({total_invested:.2f}) + cash ({cash_left:.2f}) "
            f"!= 100 000"
        )

    def test_cash_left_non_negative(self, synthetic_close, monkeypatch):
        patch_loader(monkeypatch, synthetic_close)
        last = synthetic_close.index[-1].date()
        _, _, cash_left, _, _ = gs.generate_signals(last, 100_000.0)
        assert cash_left >= 0, f"Negative cash: {cash_left}"
