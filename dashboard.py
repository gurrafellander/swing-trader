"""
dashboard.py
------------
Flask web dashboard for generate_signals.py.

Usage
-----
  pip install flask
  python dashboard.py
  # open http://localhost:5000
"""

from flask import Flask, render_template_string, request, jsonify
from datetime import date
import traceback

from generate_signals import generate_signals

app = Flask(__name__)

HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>Signal Generator — Rising Momentum + Sharpe</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500;600&family=IBM+Plex+Sans:wght@300;400;500&display=swap" rel="stylesheet">
<style>
  :root {
    --bg:        #0d0f0e;
    --surface:   #131614;
    --border:    #1f2421;
    --border-hi: #2a3330;
    --text:      #c8d5c9;
    --muted:     #556b57;
    --accent:    #3dffa0;
    --accent-dim:#1a6644;
    --warn:      #f0c040;
    --danger:    #ff5f5f;
    --mono:      'IBM Plex Mono', monospace;
    --sans:      'IBM Plex Sans', sans-serif;
  }

  * { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    background: var(--bg);
    color: var(--text);
    font-family: var(--mono);
    font-size: 13px;
    min-height: 100vh;
    line-height: 1.6;
  }

  /* Subtle scanline overlay */
  body::before {
    content: '';
    position: fixed;
    inset: 0;
    background: repeating-linear-gradient(
      0deg,
      transparent,
      transparent 2px,
      rgba(0,0,0,0.03) 2px,
      rgba(0,0,0,0.03) 4px
    );
    pointer-events: none;
    z-index: 999;
  }

  header {
    border-bottom: 1px solid var(--border);
    padding: 20px 32px;
    display: flex;
    align-items: baseline;
    gap: 16px;
  }

  header .logo {
    font-size: 11px;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--accent);
    font-weight: 600;
  }

  header .subtitle {
    font-size: 11px;
    color: var(--muted);
    font-family: var(--sans);
    font-weight: 300;
    letter-spacing: 0.05em;
  }

  .layout {
    display: grid;
    grid-template-columns: 280px 1fr;
    min-height: calc(100vh - 57px);
  }

  /* ── Sidebar ── */
  aside {
    border-right: 1px solid var(--border);
    padding: 28px 24px;
    display: flex;
    flex-direction: column;
    gap: 28px;
  }

  .section-label {
    font-size: 10px;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 12px;
  }

  .field {
    display: flex;
    flex-direction: column;
    gap: 6px;
    margin-bottom: 16px;
  }

  .field label {
    font-size: 11px;
    color: var(--muted);
    letter-spacing: 0.08em;
  }

  .field input {
    background: var(--surface);
    border: 1px solid var(--border);
    color: var(--text);
    font-family: var(--mono);
    font-size: 13px;
    padding: 9px 12px;
    border-radius: 3px;
    outline: none;
    transition: border-color 0.15s;
    width: 100%;
  }

  .field input:focus {
    border-color: var(--accent-dim);
    box-shadow: 0 0 0 1px var(--accent-dim);
  }

  .field input::placeholder { color: var(--muted); }

  .btn-run {
    background: var(--accent);
    color: #061a0e;
    border: none;
    font-family: var(--mono);
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    padding: 12px 0;
    width: 100%;
    border-radius: 3px;
    cursor: pointer;
    transition: background 0.15s, transform 0.1s;
    margin-top: 4px;
  }

  .btn-run:hover  { background: #5fffb8; }
  .btn-run:active { transform: scale(0.98); }
  .btn-run:disabled {
    background: var(--accent-dim);
    color: var(--muted);
    cursor: not-allowed;
  }

  .divider {
    border: none;
    border-top: 1px solid var(--border);
  }

  /* Config display */
  .config-row {
    display: flex;
    justify-content: space-between;
    font-size: 11px;
    padding: 4px 0;
    border-bottom: 1px solid var(--border);
  }
  .config-row:last-child { border-bottom: none; }
  .config-key   { color: var(--muted); }
  .config-value { color: var(--text); font-weight: 500; }

  /* ── Main content ── */
  main {
    padding: 28px 32px;
    display: flex;
    flex-direction: column;
    gap: 24px;
  }

  .status-bar {
    font-size: 11px;
    color: var(--muted);
    height: 18px;
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .status-bar .dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    background: var(--muted);
    flex-shrink: 0;
  }
  .status-bar.running .dot { background: var(--warn); animation: pulse 0.8s infinite; }
  .status-bar.ok .dot      { background: var(--accent); }
  .status-bar.error .dot   { background: var(--danger); }
  .status-bar.ok    { color: var(--accent); }
  .status-bar.error { color: var(--danger); }

  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50%       { opacity: 0.3; }
  }

  /* Summary cards */
  .cards {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
  }

  .card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 14px 16px;
    opacity: 0;
    transform: translateY(6px);
    transition: opacity 0.3s, transform 0.3s;
  }
  .card.visible { opacity: 1; transform: translateY(0); }

  .card-label {
    font-size: 10px;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 6px;
  }

  .card-value {
    font-size: 20px;
    font-weight: 500;
    color: var(--accent);
    letter-spacing: -0.02em;
  }

  .card-sub {
    font-size: 10px;
    color: var(--muted);
    margin-top: 2px;
  }

  /* Metadata strip */
  .meta-strip {
    display: flex;
    gap: 24px;
    font-size: 11px;
    color: var(--muted);
    padding: 10px 14px;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 3px;
    opacity: 0;
    transition: opacity 0.3s 0.15s;
  }
  .meta-strip.visible { opacity: 1; }
  .meta-strip span b { color: var(--text); font-weight: 500; }

  /* Table */
  .table-wrap {
    overflow-x: auto;
    opacity: 0;
    transform: translateY(8px);
    transition: opacity 0.35s 0.2s, transform 0.35s 0.2s;
  }
  .table-wrap.visible { opacity: 1; transform: translateY(0); }

  table {
    width: 100%;
    border-collapse: collapse;
    font-size: 12px;
  }

  thead th {
    text-align: left;
    font-size: 10px;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--muted);
    padding: 8px 14px;
    border-bottom: 1px solid var(--border);
    white-space: nowrap;
  }

  thead th.num { text-align: right; }

  tbody tr {
    border-bottom: 1px solid var(--border);
    transition: background 0.1s;
  }
  tbody tr:hover { background: rgba(61,255,160,0.03); }

  tbody td {
    padding: 10px 14px;
    white-space: nowrap;
  }
  tbody td.num { text-align: right; }

  .ticker-cell {
    font-weight: 500;
    color: var(--accent);
    letter-spacing: 0.06em;
  }

  .weight-bar-wrap {
    display: flex;
    align-items: center;
    gap: 10px;
    justify-content: flex-end;
  }

  .weight-bar {
    height: 3px;
    background: var(--accent);
    border-radius: 2px;
    min-width: 2px;
    opacity: 0.5;
  }

  /* Error box */
  .error-box {
    background: rgba(255,95,95,0.07);
    border: 1px solid rgba(255,95,95,0.25);
    border-radius: 4px;
    padding: 16px 20px;
    font-size: 12px;
    color: var(--danger);
    white-space: pre-wrap;
    display: none;
  }
  .error-box.visible { display: block; }

  /* Empty state */
  .empty {
    color: var(--muted);
    font-size: 12px;
    padding: 40px 0;
    text-align: center;
    letter-spacing: 0.05em;
  }

  .empty .prompt {
    font-size: 32px;
    margin-bottom: 10px;
    opacity: 0.15;
  }
</style>
</head>
<body>

<header>
  <span class="logo">Signal Generator</span>
  <span class="subtitle">Rising Momentum + Sharpe &nbsp;/&nbsp; OMXS + Global Universe</span>
</header>

<div class="layout">

  <!-- Sidebar -->
  <aside>
    <div>
      <div class="section-label">Parameters</div>

      <div class="field">
        <label>AS OF DATE</label>
        <input type="date" id="inp-date" value="" />
      </div>

      <div class="field">
        <label>PORTFOLIO VALUE (SEK)</label>
        <input type="number" id="inp-value" placeholder="e.g. 500000" min="1000" step="1000" />
      </div>

      <button class="btn-run" id="btn-run" onclick="runSignals()">
        Run &#x25B6;
      </button>
    </div>

    <hr class="divider" />

    <div>
      <div class="section-label">Strategy Config</div>
      <div id="cfg-table">
        <!-- filled by /api/config -->
        <div class="config-row"><span class="config-key">loading…</span></div>
      </div>
    </div>
  </aside>

  <!-- Main -->
  <main>
    <div class="status-bar" id="status-bar">
      <div class="dot"></div>
      <span id="status-text">awaiting input</span>
    </div>

    <div id="error-box" class="error-box"></div>

    <!-- Summary cards (hidden until result) -->
    <div class="cards" id="cards" style="display:none">
      <div class="card" id="card-positions">
        <div class="card-label">Positions</div>
        <div class="card-value" id="cv-positions">—</div>
        <div class="card-sub">selected stocks</div>
      </div>
      <div class="card" id="card-invested">
        <div class="card-label">Invested</div>
        <div class="card-value" id="cv-invested">—</div>
        <div class="card-sub" id="cs-invested-pct">of portfolio</div>
      </div>
      <div class="card" id="card-cash">
        <div class="card-label">Cash Left</div>
        <div class="card-value" id="cv-cash">—</div>
        <div class="card-sub">uninvested</div>
      </div>
      <div class="card" id="card-date">
        <div class="card-label">Signal Date</div>
        <div class="card-value" id="cv-date" style="font-size:14px">—</div>
        <div class="card-sub" id="cs-price-date">prices from —</div>
      </div>
    </div>

    <!-- Meta strip -->
    <div class="meta-strip" id="meta-strip">
      <span>Rebalance: <b id="m-rebal">—</b></span>
      <span>Price date: <b id="m-price">—</b></span>
      <span>Universe: <b id="m-universe">—</b></span>
      <span>Total weight: <b id="m-weight">—</b></span>
    </div>

    <!-- Table -->
    <div class="table-wrap" id="table-wrap">
      <div class="empty" id="empty-state">
        <div class="prompt">//</div>
        enter a date and portfolio value, then run
      </div>
      <table id="result-table" style="display:none">
        <thead>
          <tr>
            <th>Ticker</th>
            <th class="num">Weight %</th>
            <th class="num">Target SEK</th>
            <th class="num">Price</th>
            <th class="num">Shares</th>
            <th class="num">Actual SEK</th>
            <th class="num">Δ Weight pp</th>
          </tr>
        </thead>
        <tbody id="result-body"></tbody>
      </table>
    </div>

  </main>
</div>

<script>
// Set default date to today
document.getElementById('inp-date').value = new Date().toISOString().split('T')[0];

// Load config on startup
fetch('/api/config')
  .then(r => r.json())
  .then(cfg => {
    const el = document.getElementById('cfg-table');
    el.innerHTML = '';
    for (const [k, v] of Object.entries(cfg)) {
      el.innerHTML += `<div class="config-row">
        <span class="config-key">${k}</span>
        <span class="config-value">${v}</span>
      </div>`;
    }
  })
  .catch(() => {
    document.getElementById('cfg-table').innerHTML =
      '<div class="config-row"><span class="config-key">config unavailable</span></div>';
  });

function setStatus(state, text) {
  const bar = document.getElementById('status-bar');
  const txt = document.getElementById('status-text');
  bar.className = 'status-bar ' + state;
  txt.textContent = text;
}

function fmtSEK(v) {
  return new Intl.NumberFormat('sv-SE', {
    style: 'decimal', maximumFractionDigits: 0
  }).format(v) + ' SEK';
}

function showCards(data) {
  document.getElementById('cards').style.display = 'grid';
  document.getElementById('meta-strip').classList.add('visible');

  const positions = data.positions.length;
  const invested  = data.positions.reduce((s, r) => s + r.actual_sek, 0);
  const pct       = (invested / data.portfolio_value * 100).toFixed(1);

  document.getElementById('cv-positions').textContent = positions;
  document.getElementById('cv-invested').textContent  = fmtSEK(invested);
  document.getElementById('cs-invested-pct').textContent = pct + '% of portfolio';
  document.getElementById('cv-cash').textContent      = fmtSEK(data.cash_left);
  document.getElementById('cv-date').textContent      = data.signal_date;
  document.getElementById('cs-price-date').textContent = 'prices from ' + data.price_date;

  document.getElementById('m-rebal').textContent    = data.signal_date;
  document.getElementById('m-price').textContent    = data.price_date;
  document.getElementById('m-universe').textContent = data.universe_size + ' tickers';
  const totalW = data.positions.reduce((s, r) => s + r.weight_pct, 0);
  document.getElementById('m-weight').textContent  = totalW.toFixed(2) + '%';

  // Stagger card reveals
  ['card-positions','card-invested','card-cash','card-date'].forEach((id, i) => {
    setTimeout(() => document.getElementById(id).classList.add('visible'), i * 60);
  });
}

function maxWeight(positions) {
  return Math.max(...positions.map(r => r.weight_pct));
}

function buildTable(positions) {
  const tbody = document.getElementById('result-body');
  tbody.innerHTML = '';
  const maxW = maxWeight(positions);

  positions.forEach(row => {
    const barWidth = Math.round((row.weight_pct / maxW) * 60);
    const delta    = row.delta_weight_pp;
    const deltaCol = Math.abs(delta) < 0.1 ? 'var(--muted)' :
                     delta > 0 ? 'var(--warn)' : 'var(--accent)';

    tbody.innerHTML += `<tr>
      <td class="ticker-cell">${row.ticker}</td>
      <td class="num">
        <div class="weight-bar-wrap">
          <span>${row.weight_pct.toFixed(2)}</span>
          <div class="weight-bar" style="width:${barWidth}px"></div>
        </div>
      </td>
      <td class="num">${row.target_sek.toLocaleString('sv-SE', {maximumFractionDigits:0})}</td>
      <td class="num">${row.price_sek.toLocaleString('sv-SE', {minimumFractionDigits:2, maximumFractionDigits:2})}</td>
      <td class="num" style="font-weight:600;color:var(--accent)">${row.shares}</td>
      <td class="num">${row.actual_sek.toLocaleString('sv-SE', {maximumFractionDigits:0})}</td>
      <td class="num" style="color:${deltaCol}">${delta > 0 ? '+' : ''}${delta.toFixed(3)}</td>
    </tr>`;
  });
}

async function runSignals() {
  const dateVal  = document.getElementById('inp-date').value;
  const valueVal = document.getElementById('inp-value').value;

  if (!dateVal || !valueVal || parseFloat(valueVal) <= 0) {
    setStatus('error', 'please enter a valid date and portfolio value');
    return;
  }

  // Reset UI
  document.getElementById('error-box').classList.remove('visible');
  document.getElementById('result-table').style.display = 'none';
  document.getElementById('empty-state').style.display  = 'none';
  document.getElementById('cards').style.display = 'none';
  document.getElementById('meta-strip').classList.remove('visible');
  document.getElementById('table-wrap').classList.remove('visible');
  ['card-positions','card-invested','card-cash','card-date'].forEach(id =>
    document.getElementById(id).classList.remove('visible')
  );

  const btn = document.getElementById('btn-run');
  btn.disabled = true;
  setStatus('running', 'downloading data and computing signals…');

  try {
    const resp = await fetch('/api/signals', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ date: dateVal, value: parseFloat(valueVal) })
    });

    const data = await resp.json();

    if (!resp.ok || data.error) {
      throw new Error(data.error || 'unknown server error');
    }

    showCards(data);
    buildTable(data.positions);

    document.getElementById('result-table').style.display = 'table';
    document.getElementById('table-wrap').classList.add('visible');

    setStatus('ok', `${data.positions.length} positions computed  ·  signal date ${data.signal_date}`);

  } catch (err) {
    const box = document.getElementById('error-box');
    box.textContent = 'Error: ' + err.message;
    box.classList.add('visible');
    document.getElementById('empty-state').style.display = 'block';
    setStatus('error', err.message);
  } finally {
    btn.disabled = false;
  }
}

// Allow Enter key to trigger run
document.addEventListener('keydown', e => {
  if (e.key === 'Enter') runSignals();
});
</script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/api/config")
def api_config():
    """Return relevant strategy config values for display in the sidebar."""
    try:
        from config import cfg

        return jsonify(
            {
                "roc_short": getattr(cfg, "roc_short_lookback", "—"),
                "roc_long": getattr(cfg, "roc_long_lookback", "—"),
                "near_zero_thr": f"{getattr(cfg, 'roc_near_zero_threshold', '—')} %",
                "top_candidates": getattr(cfg, "top_candidates", "—"),
                "top_n": getattr(cfg, "top_n", "—"),
                "vol_lookback": getattr(cfg, "vol_lookback", "—"),
                "rebal_freq": getattr(cfg, "rebalance_freq", "—"),
                "fees": f"{getattr(cfg, 'fees', 0) * 100:.3f} %",
                "slippage": f"{getattr(cfg, 'slippage', 0) * 100:.3f} %",
            }
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/signals", methods=["POST"])
def api_signals():
    """Run generate_signals and return structured JSON."""
    body = request.get_json(force=True)
    try:
        signal_date = date.fromisoformat(body["date"])
        portfolio_value = float(body["value"])
    except (KeyError, ValueError) as e:
        return jsonify({"error": f"Bad input: {e}"}), 400

    try:
        result = generate_signals(signal_date, portfolio_value)
    except Exception:
        return jsonify({"error": traceback.format_exc()}), 500

    if not isinstance(result, tuple):
        return jsonify({"error": "Strategy returned no positions for this date."}), 200

    out_full, actual_date, cash_left = result

    # Strip summary rows — keep only real positions
    positions_df = out_full[
        ~out_full["Ticker"].str.startswith("──")
        & (out_full["Ticker"] != "Cash (leftover)")
    ].copy()

    positions = []
    for _, row in positions_df.iterrows():
        positions.append(
            {
                "ticker": row["Ticker"],
                "weight_pct": float(row["Weight (%)"]),
                "target_sek": float(row["Target SEK"]),
                "price_sek": float(row["Price (SEK)"]),
                "shares": int(row["Shares"]),
                "actual_sek": float(row["Actual SEK"]),
                "delta_weight_pp": float(row["Δ Weight (%)"]),
            }
        )

    # Derive price_date — last trading day used for prices
    from datetime import timedelta
    from DataLoader import DataLoader
    from config import cfg as _cfg

    history_start = (signal_date - timedelta(days=2 * 365)).strftime("%Y-%m-%d")
    history_end = signal_date.strftime("%Y-%m-%d")
    loader = DataLoader("tickers.txt", history_start, history_end, _cfg.min_history)
    close = loader.download_clean_data()
    price_date = str(close.index[-1].date())

    return jsonify(
        {
            "positions": positions,
            "signal_date": str(actual_date),
            "price_date": price_date,
            "cash_left": round(cash_left, 2),
            "portfolio_value": portfolio_value,
            "universe_size": close.shape[1],
        }
    )


if __name__ == "__main__":
    print("\n  Signal Generator Dashboard")
    print("  http://localhost:5000\n")
    app.run(debug=True, port=5000)
