"""
Unit tests for dashboard.py API endpoints.

generate_signals is mocked — no network calls, no DataLoader.
"""
import json
import math
from datetime import date
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

import dashboard as db


# ─── fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def client():
    db.app.config["TESTING"] = True
    with db.app.test_client() as c:
        yield c


@pytest.fixture
def synthetic_close():
    rng = np.random.default_rng(99)
    n_days, n_tickers = 400, 8
    tickers = [f"S{i:02d}.ST" for i in range(n_tickers)]
    dates = pd.bdate_range("2023-06-01", periods=n_days)
    log_ret = rng.standard_normal((n_days, n_tickers)) * 0.012
    prices = 100.0 * np.exp(np.cumsum(log_ret, axis=0))
    return pd.DataFrame(prices, index=dates, columns=tickers)


def _make_mock_return(close: pd.DataFrame, portfolio_value: float = 100_000.0):
    """
    Build the 5-tuple that generate_signals returns, using signal-date
    prices for share sizing (matching the simplified generate_signals logic).
    """
    tickers = close.columns[:5].tolist()
    weights = np.ones(5) / 5

    # Signal (rebalance) date is 10 rows before the end
    signal_ts = close.index[-10].date()
    signal_prices = close.iloc[-10][tickers].values

    shares = np.floor(weights * portfolio_value / signal_prices).astype(int)
    actual_sek = (shares * signal_prices).round(2)
    cash_left = float(portfolio_value - actual_sek.sum())

    metric_cols = ["Sharpe", "Ann Ret (%)", "Ann Vol (%)", "Max DD (%)",
                   "ROC Short", "ROC Long", "Accel (pp)"]
    out = pd.DataFrame({
        "Ticker": tickers,
        "Weight (%)": (weights * 100).round(2),
        "Target SEK": (weights * portfolio_value).round(2),
        "Price (SEK)": signal_prices.round(2),   # signal-date prices
        "Shares": shares,
        "Actual SEK": actual_sek,
        "Δ Weight (%)": np.zeros(5),
        **{c: np.ones(5) for c in metric_cols},
    })
    metrics = {
        t: {"sharpe": 1.0, "ann_return": 10.0, "ann_vol": 15.0,
            "max_dd": -8.0, "roc_short": 2.0, "roc_long": 1.0, "acceleration": 1.0}
        for t in tickers
    }
    return out, signal_ts, cash_left, close, metrics


def _post_signals(client, signal_date: str, value: float = 100_000.0):
    resp = client.post(
        "/api/signals",
        data=json.dumps({"date": signal_date, "value": value}),
        content_type="application/json",
    )
    return resp, resp.get_json()


# ─── api_config ───────────────────────────────────────────────────────────────

class TestApiConfig:
    def test_returns_200(self, client):
        assert client.get("/api/config").status_code == 200

    def test_contains_expected_keys(self, client):
        data = client.get("/api/config").get_json()
        for key in ("roc_short", "roc_long", "top_n", "rebal_freq"):
            assert key in data, f"Missing key: {key}"


# ─── api_signals — structure ──────────────────────────────────────────────────

class TestApiSignalsStructure:
    def test_returns_200_on_valid_input(self, client, synthetic_close):
        with patch.object(db, "generate_signals", return_value=_make_mock_return(synthetic_close)):
            resp, _ = _post_signals(client, "2025-06-02")
        assert resp.status_code == 200

    def test_response_has_required_top_level_keys(self, client, synthetic_close):
        with patch.object(db, "generate_signals", return_value=_make_mock_return(synthetic_close)):
            _, data = _post_signals(client, "2025-06-02")
        for key in ("positions", "signal_date", "cash_left", "portfolio_value",
                    "current_value", "gains_sek", "price_date", "universe_size"):
            assert key in data, f"Missing top-level key: {key}"

    def test_positions_have_required_fields(self, client, synthetic_close):
        with patch.object(db, "generate_signals", return_value=_make_mock_return(synthetic_close)):
            _, data = _post_signals(client, "2025-06-02")
        required = {"ticker", "weight_pct", "shares", "actual_sek",
                    "signal_day_price", "current_day_price"}
        for pos in data["positions"]:
            missing = required - set(pos.keys())
            assert not missing, f"Position missing keys: {missing}"

    def test_bad_input_returns_400(self, client):
        resp = client.post(
            "/api/signals",
            data=json.dumps({"date": "not-a-date"}),
            content_type="application/json",
        )
        assert resp.status_code == 400

    def test_universe_size_matches_close_columns(self, client, synthetic_close):
        with patch.object(db, "generate_signals", return_value=_make_mock_return(synthetic_close)):
            _, data = _post_signals(client, "2025-06-02")
        assert data["universe_size"] == synthetic_close.shape[1]


# ─── api_signals — prices ─────────────────────────────────────────────────────

class TestPrices:
    def test_signal_day_price_equals_price_sek_column(self, client, synthetic_close):
        """signal_day_price comes from generate_signals' Price (SEK) column."""
        mock_return = _make_mock_return(synthetic_close)
        out_df = mock_return[0]
        with patch.object(db, "generate_signals", return_value=mock_return):
            _, data = _post_signals(client, "2025-06-02")
        for pos in data["positions"]:
            expected = float(out_df.loc[out_df["Ticker"] == pos["ticker"], "Price (SEK)"].iloc[0])
            assert abs(pos["signal_day_price"] - expected) < 1e-3

    def test_current_day_price_comes_from_last_close_row(self, client, synthetic_close):
        """current_day_price is close.iloc[-1], which may differ from signal price."""
        mock_return = _make_mock_return(synthetic_close)
        ticker = mock_return[0]["Ticker"].iloc[0]
        expected_current = float(synthetic_close.iloc[-1][ticker])
        with patch.object(db, "generate_signals", return_value=mock_return):
            _, data = _post_signals(client, "2025-06-02")
        pos = next(p for p in data["positions"] if p["ticker"] == ticker)
        assert abs(pos["current_day_price"] - expected_current) < 1e-3

    def test_price_date_is_last_close_date(self, client, synthetic_close):
        mock_return = _make_mock_return(synthetic_close)
        expected_date = str(synthetic_close.index[-1].date())
        with patch.object(db, "generate_signals", return_value=mock_return):
            _, data = _post_signals(client, "2025-06-02")
        assert data["price_date"] == expected_date


# ─── api_signals — gains ─────────────────────────────────────────────────────

class TestGains:
    def test_zero_gains_when_signal_equals_last_price(self, client, synthetic_close):
        """When signal_ts is the last row, signal_price == current_price → gains ≈ 0."""
        mock_return = _make_mock_return(synthetic_close)
        out, _, cash_left, close, metrics = mock_return
        # Override signal date to last row and re-compute prices at last row
        signal_ts_last = close.index[-1].date()
        last_prices = close.iloc[-1][out["Ticker"].tolist()].values
        shares = out["Shares"].values
        actual_sek = (shares * last_prices).round(2)
        out = out.copy()
        out["Price (SEK)"] = last_prices.round(2)
        out["Actual SEK"] = actual_sek
        cash_left = float(100_000.0 - actual_sek.sum())

        with patch.object(db, "generate_signals",
                          return_value=(out, signal_ts_last, cash_left, close, metrics)):
            _, data = _post_signals(client, "2025-06-02")
        assert abs(data["gains_sek"]) < 1.0, f"Expected ~0 gains, got {data['gains_sek']}"

    def test_gains_sek_is_finite(self, client, synthetic_close):
        with patch.object(db, "generate_signals", return_value=_make_mock_return(synthetic_close)):
            _, data = _post_signals(client, "2025-06-02")
        assert math.isfinite(data["gains_sek"])

    def test_current_value_equals_invested_plus_gains(self, client, synthetic_close):
        """current_value = invested_capital + gains_sek."""
        with patch.object(db, "generate_signals", return_value=_make_mock_return(synthetic_close)):
            _, data = _post_signals(client, "2025-06-02")
        invested = data["portfolio_value"] - data["cash_left"]
        assert abs(data["current_value"] - (invested + data["gains_sek"] + data["cash_left"])) < 1.0

    def test_portfolio_value_matches_request(self, client, synthetic_close):
        value = 250_000.0
        with patch.object(db, "generate_signals",
                          return_value=_make_mock_return(synthetic_close, value)):
            _, data = _post_signals(client, "2025-06-02", value=value)
        assert data["portfolio_value"] == pytest.approx(value)
