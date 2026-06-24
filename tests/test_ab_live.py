"""
Live A/B consistency test — makes real Yahoo Finance downloads.

Run with:
    pytest tests/test_ab_live.py -v -s

Tests that June 2 and June 23 2026 (same rebalance month) return the same
signal date, the same ticker set, and the same weights.
"""
from datetime import date

import pytest

from generate_signals import get_raw_weights

DATE_A = date(2026, 6, 2)
DATE_B = date(2026, 6, 23)


@pytest.fixture(scope="module")
def ab_results():
    """Download once per module run to avoid fetching twice per test."""
    w_a, ts_a, last_a = get_raw_weights(DATE_A)
    w_b, ts_b, last_b = get_raw_weights(DATE_B)
    return (w_a, ts_a, last_a), (w_b, ts_b, last_b)


def test_same_rebalance_date(ab_results):
    (_, ts_a, _), (_, ts_b, _) = ab_results
    print(f"\n  June 2  → rebalance {ts_a}")
    print(f"  June 23 → rebalance {ts_b}")
    assert ts_a == ts_b, f"Different rebalance dates: {ts_a} vs {ts_b}"


def test_same_ticker_set(ab_results):
    (w_a, _, _), (w_b, _, _) = ab_results
    only_a = sorted(set(w_a.index) - set(w_b.index))
    only_b = sorted(set(w_b.index) - set(w_a.index))
    print(f"\n  June 2  tickers: {sorted(w_a.index.tolist())}")
    print(f"  June 23 tickers: {sorted(w_b.index.tolist())}")
    if only_a or only_b:
        print(f"  Only in June 2:  {only_a}")
        print(f"  Only in June 23: {only_b}")
    assert not only_a and not only_b, (
        f"Ticker sets differ!\n"
        f"  Only in June 2:  {only_a}\n"
        f"  Only in June 23: {only_b}"
    )


def test_same_weights(ab_results):
    (w_a, _, _), (w_b, _, _) = ab_results
    common = set(w_a.index) & set(w_b.index)
    for tkr in common:
        diff = abs(float(w_a[tkr]) - float(w_b[tkr]))
        assert diff < 1e-3, f"{tkr}: weight differs by {diff:.6f} ({w_a[tkr]:.4f} vs {w_b[tkr]:.4f})"
