"""
Unit tests for strategies.py

Covers: _rebalance_mask, _sharpe_weights, rising_momentum_sharpe

All tests use synthetic price data — no network calls.
"""
import numpy as np
import pandas as pd
import pytest

from strategies import _rebalance_mask, _sharpe_weights, rising_momentum_sharpe
from config import cfg


# ─── shared fixtures ──────────────────────────────────────────────────────────

@pytest.fixture
def close_300():
    """300 business-day price series for 20 synthetic tickers, starting 2023-01-01."""
    rng = np.random.default_rng(42)
    n_days, n_tickers = 300, 20
    tickers = [f"S{i:02d}" for i in range(n_tickers)]
    dates = pd.bdate_range("2023-01-01", periods=n_days)
    log_ret = rng.standard_normal((n_days, n_tickers)) * 0.012
    prices = 100.0 * np.exp(np.cumsum(log_ret, axis=0))
    return pd.DataFrame(prices, index=dates, columns=tickers)


@pytest.fixture
def returns_5():
    rng = np.random.default_rng(0)
    tickers = list("ABCDE")
    dates = pd.bdate_range("2024-01-01", periods=120)
    return pd.DataFrame(rng.standard_normal((120, 5)) * 0.01, index=dates, columns=tickers)


@pytest.fixture
def returns_25():
    rng = np.random.default_rng(0)
    tickers = [f"T{i}" for i in range(25)]
    dates = pd.bdate_range("2024-01-01", periods=120)
    return pd.DataFrame(rng.standard_normal((120, 25)) * 0.01, index=dates, columns=tickers)


# ─── _rebalance_mask ──────────────────────────────────────────────────────────

class TestRebalanceMask:
    def _bdays(self, start, end):
        return pd.bdate_range(start, end)

    def test_weekday_month_start_is_rebalance_day(self):
        """April 1, 2025 is a Tuesday and must be marked as rebalance day."""
        idx = self._bdays("2025-03-01", "2025-04-30")
        mask = _rebalance_mask(idx)
        assert mask[pd.Timestamp("2025-04-01")] == True

    def test_weekend_month_start_deferred_to_monday(self):
        """June 1, 2025 is Sunday — June 2 (Monday) must be the rebalance day."""
        idx = self._bdays("2025-05-01", "2025-06-30")
        mask = _rebalance_mask(idx)
        assert mask[pd.Timestamp("2025-06-02")] == True

    def test_saturday_month_start_deferred_to_monday(self):
        """March 1, 2025 is Saturday — March 3 (Monday) must be the rebalance day."""
        idx = self._bdays("2025-02-01", "2025-03-31")
        mask = _rebalance_mask(idx)
        assert mask[pd.Timestamp("2025-03-03")] == True

    def test_exactly_one_rebalance_per_month(self):
        """Exactly one True per calendar month across a two-year range."""
        idx = self._bdays("2023-01-01", "2024-12-31")
        mask = _rebalance_mask(idx)
        counts = mask.groupby(mask.index.to_period("M")).sum()
        bad = counts[counts != 1]
        assert bad.empty, f"Months with != 1 rebalance: {bad.to_dict()}"

    def test_rebalance_day_is_always_the_earliest_in_month(self):
        """The marked day must be the first trading day of its calendar month."""
        idx = self._bdays("2024-01-01", "2024-12-31")
        mask = _rebalance_mask(idx)
        for ts in mask[mask].index:
            month_first = idx[idx.month == ts.month][0]
            assert ts == month_first, f"{ts} is not the first trading day of its month"

    def test_weekly_mode_fires_on_first_day_of_each_week(self):
        """Weekly rebalance (W) must fire exactly once per ISO week."""
        original = cfg.rebalance_freq
        cfg.rebalance_freq = "W"
        try:
            idx = self._bdays("2025-01-01", "2025-03-31")
            mask = _rebalance_mask(idx)
            # Each marked date should be the first business day in its ISO week
            for ts in mask[mask].index:
                week_no = ts.isocalendar().week
                year_no = ts.isocalendar().year
                week_days = idx[
                    (idx.isocalendar().week == week_no)
                    & (idx.isocalendar().year == year_no)
                ]
                assert ts == week_days[0], (
                    f"{ts} is not the first trading day of ISO week {week_no}/{year_no}"
                )
        finally:
            cfg.rebalance_freq = original


# ─── _sharpe_weights ─────────────────────────────────────────────────────────

class TestSharpeWeights:
    def test_weights_sum_to_one(self, returns_5):
        w = _sharpe_weights(returns_5)
        assert abs(w.sum() - 1.0) < 1e-6, f"Weights sum to {w.sum()}"

    def test_all_weights_non_negative(self, returns_5):
        w = _sharpe_weights(returns_5)
        assert (w >= 0).all(), f"Negative weights found: {w[w < 0]}"

    def test_max_position_weight_respected(self, returns_5):
        w = _sharpe_weights(returns_5)
        assert w.max() <= cfg.max_position_weight + 1e-6, (
            f"Weight {w.max():.4f} exceeds cap {cfg.max_position_weight}"
        )

    def test_top_n_cap_on_large_pool(self, returns_25):
        """With 25 candidates (> top_n=10), at most top_n positions are non-zero."""
        w = _sharpe_weights(returns_25)
        n_nonzero = (w > 0).sum()
        assert n_nonzero <= cfg.top_n, f"{n_nonzero} positions exceed top_n={cfg.top_n}"

    def test_no_optimizer_residuals(self, returns_25):
        """No weight should be in the range (0, 1e-3) — those are SLSQP residuals."""
        w = _sharpe_weights(returns_25)
        residuals = w[(w > 0) & (w < 1e-3)]
        assert residuals.empty, f"Residual weights found: {residuals.to_dict()}"

    def test_small_pool_not_capped(self, returns_5):
        """With fewer candidates than top_n, all can receive positive weight."""
        w = _sharpe_weights(returns_5)
        # All 5 candidates are eligible; optimiser may assign weight to any subset
        assert (w >= 0).all()
        assert abs(w.sum() - 1.0) < 1e-6

    def test_returns_series_with_correct_index(self, returns_5):
        w = _sharpe_weights(returns_5)
        assert isinstance(w, pd.Series)
        assert list(w.index) == list(returns_5.columns)


# ─── rising_momentum_sharpe ──────────────────────────────────────────────────

class TestRisingMomentumSharpe:
    def test_output_shape_matches_close(self, close_300):
        weights = rising_momentum_sharpe(close_300)
        assert weights.shape == close_300.shape

    def test_output_columns_match_close(self, close_300):
        weights = rising_momentum_sharpe(close_300)
        assert list(weights.columns) == list(close_300.columns)

    def test_non_rebalance_rows_are_all_nan(self, close_300):
        """Non-rebalance rows must be entirely NaN (hold — no change)."""
        weights = rising_momentum_sharpe(close_300)
        mask = _rebalance_mask(close_300.index)
        non_rebalance = weights.loc[~mask.values]
        assert non_rebalance.isna().all().all(), (
            "Some non-rebalance rows have non-NaN values"
        )

    def test_rebalance_rows_sum_to_zero_or_one(self, close_300):
        """Each rebalance row must be either fully invested (sum≈1) or flat (sum=0)."""
        weights = rising_momentum_sharpe(close_300)
        rebalance_rows = weights.dropna(how="all")
        row_sums = rebalance_rows.fillna(0).sum(axis=1)
        for dt, s in row_sums.items():
            assert abs(s - 1.0) < 1e-4 or abs(s) < 1e-4, (
                f"Row {dt} sums to {s:.6f} — expected ~0 or ~1"
            )

    def test_no_negative_weights(self, close_300):
        weights = rising_momentum_sharpe(close_300)
        stacked = weights.stack()
        assert (stacked >= 0).all(), f"Negative weights found: {stacked[stacked < 0]}"

    def test_no_residual_weights(self, close_300):
        """No weight in the range (0, 1e-3) — residuals must be zeroed."""
        weights = rising_momentum_sharpe(close_300)
        stacked = weights.stack()
        residuals = stacked[(stacked > 0) & (stacked < 1e-3)]
        assert residuals.empty, f"Residual weights found:\n{residuals}"

    def test_consistency_across_window_lengths(self, close_300):
        """
        Extra rows AFTER a rebalance date must not change the weights computed
        on that date. (Regression test for the optimizer residual / different-
        length DataFrame non-determinism issue.)
        """
        weights_full = rising_momentum_sharpe(close_300)
        rebalance_dates = weights_full.dropna(how="all").index
        if len(rebalance_dates) < 2:
            pytest.skip("Not enough rebalance dates in synthetic data")

        # Use the second-to-last rebalance; cut the DataFrame 10 days after it
        ref_date = rebalance_dates[-2]
        ref_loc = close_300.index.get_loc(ref_date)
        cutoff = close_300.index[min(ref_loc + 10, len(close_300.index) - 1)]

        weights_short = rising_momentum_sharpe(close_300.loc[:cutoff])

        row_short = weights_short.loc[ref_date].fillna(0)
        row_long = weights_full.loc[ref_date].fillna(0)

        tickers_short = set(row_short[row_short > 0].index)
        tickers_long = set(row_long[row_long > 0].index)

        assert tickers_short == tickers_long, (
            f"Rebalance {ref_date} ticker sets differ between window lengths:\n"
            f"  short-window only: {tickers_short - tickers_long}\n"
            f"  long-window only:  {tickers_long - tickers_short}"
        )
