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

import logging
import math
import traceback
from datetime import date, timedelta

import pandas as pd
from flask import Flask, jsonify, render_template_string, request

from config import cfg
from DataLoader import DataLoader
from generate_signals import generate_signals

log = logging.getLogger(__name__)

app = Flask(__name__)

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>Signal Generator</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500;600&family=IBM+Plex+Sans:wght@300;400;500&display=swap" rel="stylesheet">
<style>
  :root {
    --bg:        #0d0f0e;
    --surface:   #111412;
    --surface2:  #161917;
    --border:    #1e2420;
    --border-hi: #283229;
    --text:      #c4d4c5;
    --muted:     #4e6450;
    --accent:    #3dffa0;
    --accent-dim:#1a5e40;
    --warn:      #f0c040;
    --danger:    #ff5f5f;
    --mono:      'IBM Plex Mono', monospace;
    --sans:      'IBM Plex Sans', sans-serif;
    --r:         4px;
  }

  * { box-sizing: border-box; margin: 0; padding: 0; }
  html, body { height: 100%; background: var(--bg); color: var(--text); font-family: var(--mono); font-size: 13px; line-height: 1.55; }

  body::after {
    content: ''; position: fixed; inset: 0;
    background: repeating-linear-gradient(0deg, transparent, transparent 2px, rgba(0,0,0,0.025) 2px, rgba(0,0,0,0.025) 4px);
    pointer-events: none; z-index: 9999;
  }

  header { border-bottom: 1px solid var(--border); padding: 16px 28px; display: flex; align-items: center; gap: 14px; }
  .logo { font-size: 11px; letter-spacing: .2em; text-transform: uppercase; color: var(--accent); font-weight: 600; }
  .logo-sub { font-size: 11px; color: var(--muted); font-family: var(--sans); font-weight: 300; }

  .layout { display: grid; grid-template-columns: 268px 1fr; height: calc(100vh - 49px); overflow: hidden; }

  aside { border-right: 1px solid var(--border); padding: 24px 20px; overflow-y: auto; display: flex; flex-direction: column; gap: 24px; }
  main  { padding: 24px 28px; overflow-y: auto; display: flex; flex-direction: column; gap: 20px; }

  .sec-label { font-size: 10px; letter-spacing: .16em; text-transform: uppercase; color: var(--muted); margin-bottom: 10px; }

  .field { display: flex; flex-direction: column; gap: 5px; margin-bottom: 14px; }
  .field label { font-size: 10px; color: var(--muted); letter-spacing: .08em; }
  .field input {
    background: var(--surface2); border: 1px solid var(--border); color: var(--text);
    font-family: var(--mono); font-size: 13px; padding: 8px 11px; border-radius: var(--r);
    outline: none; width: 100%; transition: border-color .15s, box-shadow .15s;
  }
  .field input:focus { border-color: var(--accent-dim); box-shadow: 0 0 0 1px var(--accent-dim); }
  .field input::placeholder { color: var(--muted); }

  .btn-run {
    background: var(--accent); color: #061a0e; border: none; font-family: var(--mono);
    font-size: 11px; font-weight: 600; letter-spacing: .14em; text-transform: uppercase;
    padding: 11px 0; width: 100%; border-radius: var(--r); cursor: pointer;
    transition: background .12s, transform .1s;
  }
  .btn-run:hover  { background: #5fffc0; }
  .btn-run:active { transform: scale(.98); }
  .btn-run:disabled { background: var(--accent-dim); color: var(--muted); cursor: not-allowed; }

  hr.div { border: none; border-top: 1px solid var(--border); }

  .cfg-row { display: flex; justify-content: space-between; font-size: 11px; padding: 4px 0; border-bottom: 1px solid var(--border); }
  .cfg-row:last-child { border-bottom: none; }
  .cfg-k { color: var(--muted); }
  .cfg-v { color: var(--text); font-weight: 500; }

  .status { font-size: 11px; color: var(--muted); display: flex; align-items: center; gap: 8px; height: 16px; }
  .status .dot { width: 6px; height: 6px; border-radius: 50%; background: var(--muted); flex-shrink: 0; }
  .status.running .dot { background: var(--warn); animation: blink .7s infinite; }
  .status.ok  .dot { background: var(--accent); }
  .status.err .dot { background: var(--danger); }
  .status.ok  { color: var(--accent); }
  .status.err { color: var(--danger); }
  @keyframes blink { 0%,100%{opacity:1} 50%{opacity:.2} }

  .cards { display: grid; grid-template-columns: repeat(5, 1fr); gap: 10px; }
  .card { background: var(--surface); border: 1px solid var(--border); border-radius: var(--r); padding: 13px 15px; opacity: 0; transform: translateY(5px); transition: opacity .28s, transform .28s; }
  .card.vis { opacity: 1; transform: translateY(0); }
  .card-label { font-size: 9px; letter-spacing: .16em; text-transform: uppercase; color: var(--muted); margin-bottom: 5px; }
  .card-val { font-size: 19px; font-weight: 500; letter-spacing: -.02em; color: var(--accent); }
  .card-val.warn  { color: var(--warn); }
  .card-val.neg   { color: var(--danger); }
  .card-val.neu   { color: var(--text); }
  .card-sub { font-size: 10px; color: var(--muted); margin-top: 2px; }

  .meta { display: flex; gap: 20px; flex-wrap: wrap; font-size: 11px; color: var(--muted); padding: 9px 13px; background: var(--surface); border: 1px solid var(--border); border-radius: var(--r); opacity: 0; transition: opacity .3s .1s; }
  .meta.vis { opacity: 1; }
  .meta b { color: var(--text); font-weight: 500; }
  .meta b.pos { color: var(--accent); }
  .meta b.neg { color: var(--danger); }

  .err-box { background: rgba(255,95,95,.06); border: 1px solid rgba(255,95,95,.22); border-radius: var(--r); padding: 14px 18px; font-size: 12px; color: var(--danger); white-space: pre-wrap; display: none; }
  .err-box.vis { display: block; }

  .tbl-wrap { overflow-x: auto; opacity: 0; transform: translateY(6px); transition: opacity .32s .18s, transform .32s .18s; }
  .tbl-wrap.vis { opacity: 1; transform: translateY(0); }

  table { width: 100%; border-collapse: collapse; font-size: 12px; }

  thead th {
    text-align: left; font-size: 9px; letter-spacing: .13em; text-transform: uppercase;
    color: var(--muted); padding: 7px 12px; border-bottom: 1px solid var(--border);
    white-space: nowrap; cursor: pointer; user-select: none;
  }
  thead th:hover { color: var(--text); }
  thead th.r { text-align: right; }
  thead th .arr { margin-left: 4px; opacity: .35; }
  thead th.sorted .arr { opacity: 1; color: var(--accent); }

  tbody tr { border-bottom: 1px solid var(--border); transition: background .08s; }
  tbody tr:last-child { border-bottom: none; }
  tbody tr:hover { background: rgba(61,255,160,.025); }
  td { padding: 9px 12px; white-space: nowrap; }
  td.r { text-align: right; }

  .tkr { font-weight: 600; color: var(--accent); letter-spacing: .06em; }

  .wbar-cell { display: flex; align-items: center; gap: 8px; justify-content: flex-end; }
  .wbar { height: 3px; border-radius: 2px; background: var(--accent); opacity: .4; min-width: 1px; }

  .pos { color: var(--accent); }
  .neg { color: var(--danger); }
  .wrn { color: var(--warn); }
  .neu { color: var(--text); }

  .sp { display: inline-block; padding: 1px 7px; border-radius: 10px; font-size: 11px; font-weight: 600; }
  .sp.hi  { background: rgba(61,255,160,.1);  color: var(--accent); border: 1px solid rgba(61,255,160,.2); }
  .sp.med { background: rgba(240,192,64,.08); color: var(--warn);   border: 1px solid rgba(240,192,64,.2); }
  .sp.low { background: rgba(255,95,95,.07);  color: var(--danger); border: 1px solid rgba(255,95,95,.2); }

  .empty { color: var(--muted); font-size: 12px; padding: 48px 0; text-align: center; letter-spacing: .05em; }
  .empty .glyph { font-size: 36px; margin-bottom: 10px; opacity: .12; }
</style>
</head>
<body>

<header>
  <span class="logo">Signal Generator</span>
  <span class="logo-sub">Rising Momentum + Sharpe &nbsp;/&nbsp; OMXS + Global</span>
</header>

<div class="layout">
<aside>
  <div>
    <div class="sec-label">Parameters</div>
    <div class="field">
      <label>AS OF DATE</label>
      <input type="date" id="inp-date" />
    </div>
    <div class="field">
      <label>PORTFOLIO VALUE (SEK)</label>
      <input type="number" id="inp-value" placeholder="e.g. 500 000" min="1000" step="1000" />
    </div>
    <button class="btn-run" id="btn-run" onclick="runSignals()">Run &#x25B6;</button>
  </div>
  <hr class="div" />
  <div>
    <div class="sec-label">Strategy Config</div>
    <div id="cfg-table"><div class="cfg-row"><span class="cfg-k">loading…</span></div></div>
  </div>
</aside>

<main>
  <div class="status" id="status"><div class="dot"></div><span id="status-txt">awaiting input</span></div>
  <div class="err-box" id="err-box"></div>

  <div class="cards" id="cards" style="display:none">
    <div class="card" id="c-pos">
      <div class="card-label">Positions</div>
      <div class="card-val" id="cv-pos">—</div>
      <div class="card-sub">selected stocks</div>
    </div>
    <div class="card" id="c-sharpe">
      <div class="card-label">Port. Sharpe</div>
      <div class="card-val" id="cv-sharpe">—</div>
      <div class="card-sub">weighted avg</div>
    </div>
    <div class="card" id="c-ret">
      <div class="card-label">Port. Ann. Return</div>
      <div class="card-val" id="cv-ret">—</div>
      <div class="card-sub">weighted avg</div>
    </div>
    <div class="card" id="c-vol">
      <div class="card-label">Port. Ann. Vol</div>
      <div class="card-val neu" id="cv-vol">—</div>
      <div class="card-sub">weighted avg</div>
    </div>
    <div class="card" id="c-cash">
      <div class="card-label">Cash Left</div>
      <div class="card-val neu" id="cv-cash">—</div>
      <div class="card-sub" id="cs-cash-pct">uninvested</div>
    </div>
  </div>

  <div class="meta" id="meta">
    <span>Signal date: <b id="m-sig">—</b></span>
    <span>Price date: <b id="m-price">—</b></span>
    <span>Universe: <b id="m-uni">—</b> tickers</span>
    <span>Total invested: <b id="m-inv">—</b></span>
    <span>Current value: <b id="m-curr">—</b></span>
    <span>Gains vs invested: <b id="m-gains">—</b></span>
  </div>

  <div class="tbl-wrap" id="tbl-wrap">
    <div class="empty" id="empty"><div class="glyph">//</div>enter a date and portfolio value, then run</div>
    <table id="tbl" style="display:none">
      <thead id="thead">
        <tr>
          <th data-key="ticker">Ticker <span class="arr">↕</span></th>
          <th class="r" data-key="weight_pct">Weight % <span class="arr">↕</span></th>
          <th class="r" data-key="sharpe">Sharpe <span class="arr">↕</span></th>
          <th class="r" data-key="ann_return">Ann Ret % <span class="arr">↕</span></th>
          <th class="r" data-key="ann_vol">Ann Vol % <span class="arr">↕</span></th>
          <th class="r" data-key="max_dd">Max DD % <span class="arr">↕</span></th>
          <th class="r" data-key="acceleration">Accel pp <span class="arr">↕</span></th>
          <th class="r" data-key="signal_day_price">Signal Price <span class="arr">↕</span></th>
          <th class="r" data-key="current_day_price">Current Price <span class="arr">↕</span></th>
          <th class="r" data-key="shares">Shares <span class="arr">↕</span></th>
          <th class="r" data-key="actual_sek">Actual SEK <span class="arr">↕</span></th>
        </tr>
      </thead>
      <tbody id="tbody"></tbody>
    </table>
  </div>
</main>
</div>

<script>
document.getElementById('inp-date').value = new Date().toISOString().split('T')[0];

fetch('/api/config').then(r=>r.json()).then(cfg=>{
  const el = document.getElementById('cfg-table');
  el.innerHTML = Object.entries(cfg)
    .map(([k,v])=>`<div class="cfg-row"><span class="cfg-k">${k}</span><span class="cfg-v">${v}</span></div>`)
    .join('');
}).catch(()=>{
  document.getElementById('cfg-table').innerHTML='<div class="cfg-row"><span class="cfg-k">unavailable</span></div>';
});

// Sort setup
let _positions=[], _sortKey='weight_pct', _sortAsc=false;

document.querySelectorAll('#thead th').forEach(th=>{
  th.addEventListener('click',()=>{
    const key=th.dataset.key;
    if(_sortKey===key) _sortAsc=!_sortAsc;
    else { _sortKey=key; _sortAsc=(key==='ticker'); }
    document.querySelectorAll('#thead th').forEach(t=>{ t.classList.remove('sorted'); t.querySelector('.arr').textContent='↕'; });
    th.classList.add('sorted');
    th.querySelector('.arr').textContent=_sortAsc?'↑':'↓';
    renderTable();
  });
});

function setStatus(s,txt){
  const el=document.getElementById('status');
  el.className='status '+s;
  document.getElementById('status-txt').textContent=txt;
}

function fmtSEK(v){ return new Intl.NumberFormat('sv-SE',{maximumFractionDigits:0}).format(v)+' SEK'; }

function sharpePill(s){
  if(s===null||s===undefined) return '<span class="sp low">n/a</span>';
  const cls=s>=1?'hi':s>=0.5?'med':'low';
  return `<span class="sp ${cls}">${s.toFixed(2)}</span>`;
}

function colNum(v,higherBetter,fmt){
  if(v===null||v===undefined) return '<span class="neu">—</span>';
  fmt=fmt||(x=>x.toFixed(2));
  const good=higherBetter?v>0:v<0;
  return `<span class="${good?'pos':'neg'}">${v>0?'+':''}${fmt(v)}</span>`;
}

function portMetrics(pos){
  let tw=0,ws=0,wr=0,wv=0;
  for(const p of pos){ const w=p.weight_pct/100; tw+=w; if(p.sharpe!=null)ws+=w*p.sharpe; if(p.ann_return!=null)wr+=w*p.ann_return; if(p.ann_vol!=null)wv+=w*p.ann_vol; }
  return tw>0?{sharpe:ws/tw,ann_return:wr/tw,ann_vol:wv/tw}:{sharpe:null,ann_return:null,ann_vol:null};
}

function showCards(data){
  document.getElementById('cards').style.display='grid';
  document.getElementById('meta').classList.add('vis');
  const pm=portMetrics(data.positions);
  const invested=data.positions.reduce((s,r)=>s+r.actual_sek,0);
  const cashPct=(data.cash_left/data.portfolio_value*100).toFixed(1);

  document.getElementById('cv-pos').textContent=data.positions.length;

  const sEl=document.getElementById('cv-sharpe');
  sEl.textContent=pm.sharpe!=null?pm.sharpe.toFixed(2):'—';
  sEl.className='card-val '+(pm.sharpe>=1?'pos':pm.sharpe>=0.5?'warn':'neg');

  const rEl=document.getElementById('cv-ret');
  rEl.textContent=pm.ann_return!=null?(pm.ann_return>0?'+':'')+pm.ann_return.toFixed(1)+'%':'—';
  rEl.className='card-val '+(pm.ann_return>=0?'':'warn');

  document.getElementById('cv-vol').textContent=pm.ann_vol!=null?pm.ann_vol.toFixed(1)+'%':'—';
  document.getElementById('cv-cash').textContent=fmtSEK(data.cash_left);
  document.getElementById('cs-cash-pct').textContent=cashPct+'% uninvested';

  document.getElementById('m-sig').textContent=data.signal_date;
  document.getElementById('m-price').textContent=data.price_date;
  document.getElementById('m-uni').textContent=data.universe_size;
  document.getElementById('m-inv').textContent=fmtSEK(invested + data.cash_left);

  const currEl=document.getElementById('m-curr');
  currEl.textContent=fmtSEK(data.current_value);
  currEl.className=data.gains_sek>=0?'pos':'neg';

  const gainsEl=document.getElementById('m-gains');
  gainsEl.textContent=(data.gains_sek>=0?'+':'')+fmtSEK(data.gains_sek);
  gainsEl.className=data.gains_sek>=0?'pos':'neg';

  ['c-pos','c-sharpe','c-ret','c-vol','c-cash'].forEach((id,i)=>{
    setTimeout(()=>document.getElementById(id).classList.add('vis'),i*55);
  });
}

function renderTable(){
  const sorted=[..._positions].sort((a,b)=>{
    const va=a[_sortKey],vb=b[_sortKey];
    if(typeof va==='string') return _sortAsc?va.localeCompare(vb):vb.localeCompare(va);
    const an=(va===null||va===undefined)?-Infinity:va;
    const bn=(vb===null||vb===undefined)?-Infinity:vb;
    return _sortAsc?an-bn:bn-an;
  });
  const maxW=Math.max(...sorted.map(r=>r.weight_pct));
  document.getElementById('tbody').innerHTML=sorted.map(r=>{
    const barW=Math.round((r.weight_pct/maxW)*56);
    const signalPrice=r.signal_day_price;
    const currentPrice=r.current_day_price;
    let priceChangePct=null;
    if(signalPrice!=null && currentPrice!=null && signalPrice>0)
      priceChangePct=(currentPrice-signalPrice)/signalPrice*100;

    return `<tr>
      <td class="tkr">${r.ticker}</td>
      <td class="r"><div class="wbar-cell"><span>${r.weight_pct.toFixed(2)}</span><div class="wbar" style="width:${barW}px"></div></div></td>
      <td class="r">${sharpePill(r.sharpe)}</td>
      <td class="r">${colNum(r.ann_return,true,x=>x.toFixed(1)+'%')}</td>
      <td class="r">${r.ann_vol!=null?r.ann_vol.toFixed(1)+'%':'—'}</td>
      <td class="r">${colNum(r.max_dd,false,x=>x.toFixed(1)+'%')}</td>
      <td class="r">${colNum(r.acceleration,true,x=>x.toFixed(2))}</td>
      <td class="r">${signalPrice!=null?signalPrice.toLocaleString('sv-SE',{minimumFractionDigits:2,maximumFractionDigits:2}):'—'}</td>
      <td class="r">${currentPrice!=null?(priceChangePct!=null?`<span class="${priceChangePct>=0?'pos':'neg'}">${currentPrice.toLocaleString('sv-SE',{minimumFractionDigits:2,maximumFractionDigits:2})} (${priceChangePct>=0?'+':''}${priceChangePct.toFixed(1)}%)</span>`:`<span class="neu">${currentPrice.toLocaleString('sv-SE',{minimumFractionDigits:2,maximumFractionDigits:2})}</span>`):'—'}</td>
      <td class="r" style="font-weight:600;color:var(--accent)">${r.shares}</td>
      <td class="r">${r.actual_sek!=null?r.actual_sek.toLocaleString('sv-SE',{maximumFractionDigits:0}):'—'}</td>
    </tr>`;
  }).join('');
}

async function runSignals(){
  const dateVal=document.getElementById('inp-date').value;
  const valVal=document.getElementById('inp-value').value;
  if(!dateVal||!valVal||parseFloat(valVal)<=0){ setStatus('err','enter a valid date and portfolio value'); return; }

  document.getElementById('err-box').classList.remove('vis');
  document.getElementById('tbl').style.display='none';
  document.getElementById('empty').style.display='none';
  document.getElementById('cards').style.display='none';
  document.getElementById('meta').classList.remove('vis');
  document.getElementById('tbl-wrap').classList.remove('vis');
  ['c-pos','c-sharpe','c-ret','c-vol','c-cash'].forEach(id=>document.getElementById(id).classList.remove('vis'));

  const btn=document.getElementById('btn-run');
  btn.disabled=true;
  setStatus('running','downloading data and computing signals…');

  try{
    const resp=await fetch('/api/signals',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({date:dateVal,value:parseFloat(valVal)})});
    const data=await resp.json();
    if(!resp.ok||data.error) throw new Error(data.error||'server error');

    _positions=data.positions;
    _sortKey='weight_pct'; _sortAsc=false;

    showCards(data);
    renderTable();
    document.getElementById('tbl').style.display='table';
    document.getElementById('tbl-wrap').classList.add('vis');

    const pm=portMetrics(data.positions);
    setStatus('ok',`${data.positions.length} positions  ·  port. Sharpe ${pm.sharpe!=null?pm.sharpe.toFixed(2):'n/a'}  ·  signal ${data.signal_date}`);
  }catch(err){
    const box=document.getElementById('err-box');
    box.textContent='Error: '+err.message;
    box.classList.add('vis');
    document.getElementById('empty').style.display='block';
    setStatus('err',err.message);
  }finally{ btn.disabled=false; }
}

document.addEventListener('keydown',e=>{ if(e.key==='Enter') runSignals(); });
</script>
</body>
</html>
"""


def _safe_float(v):
    """Return a rounded float or None for NaN / non-numeric values."""
    try:
        f = float(v)
        return None if math.isnan(f) else round(f, 6)
    except (TypeError, ValueError):
        return None


def _asof_index(index, target_date):
    """
    Return the integer position of the last timestamp in *index* whose
    .date() <= target_date, or None if no such timestamp exists.

    Using asof() avoids iterating the entire index and correctly handles
    weekends / holidays where target_date itself is not present.
    """
    ts = pd.Timestamp(target_date)
    # Match the index's timezone (if any) to avoid tz-naive vs tz-aware errors
    if index.tz is not None:
        ts = ts.tz_localize(index.tz)
    label = index.asof(ts)
    if pd.isna(label):
        return None
    # get_loc returns a scalar for exact matches (which asof guarantees)
    return index.get_loc(label)


@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/api/config")
def api_config():
    try:
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
        log.exception("api_config failed")
        return jsonify({"error": str(e)}), 500


@app.route("/api/signals", methods=["POST"])
def api_signals():
    body = request.get_json(force=True)
    try:
        signal_date = date.fromisoformat(body["date"])
        portfolio_value = float(body["value"])
    except (KeyError, ValueError) as e:
        return jsonify({"error": f"Bad input: {e}"}), 400

    try:
        result = generate_signals(signal_date, portfolio_value)
    except Exception:
        log.exception("generate_signals raised an exception")
        return jsonify({"error": traceback.format_exc()}), 500

    if not isinstance(result, tuple):
        return jsonify({"error": "Strategy returned no positions for this date."}), 200

    out_full, actual_date, cash_left, close, metrics = result

    # ------------------------------------------------------------------ #
    # Filter out separator / summary rows                                  #
    # ------------------------------------------------------------------ #
    positions_df = out_full[
        ~out_full["Ticker"].str.startswith(("--", "\u2500"))
        & (out_full["Ticker"] != "Cash (leftover)")
    ].copy()

    # ------------------------------------------------------------------ #
    # Signal-date prices                                                   #
    # Use asof() so weekends / holidays fall back to the nearest prior     #
    # trading day rather than silently missing the date.                   #
    # ------------------------------------------------------------------ #
    signal_date_idx = _asof_index(close.index, actual_date)

    if signal_date_idx is not None:
        signal_prices = close.iloc[signal_date_idx]
        # Invested capital = sum of (shares × price-paid) for each position
        invested_capital = cash_left + sum(
            int(row["Shares"]) * signal_prices[row["Ticker"]]
            for _, row in positions_df.iterrows()
            if row["Ticker"] in signal_prices.index
            and int(row["Shares"]) > 0
            and not (
                isinstance(signal_prices[row["Ticker"]], float)
                and math.isnan(signal_prices[row["Ticker"]])
            )
        )
    else:
        log.warning(
            "actual_date %s not found in close index; falling back to Actual SEK.",
            actual_date,
        )
        signal_prices = None
        # Fallback: use the pre-computed Actual SEK column (already at signal prices)
        invested_capital = cash_left + float(positions_df["Actual SEK"].sum())

    # ------------------------------------------------------------------ #
    # Current prices                                                       #
    # Download data from actual_date forward so we get the latest quote.  #
    # Keep min_history consistent with the original DataLoader call so    #
    # we don't introduce tickers absent from the signal universe.          #
    # ------------------------------------------------------------------ #
    original_min_history = getattr(cfg, "min_history", 252)
    today = date.today()
    end_date = today + timedelta(days=5)  # buffer for non-trading days

    latest_prices = None
    latest_date = None
    try:
        current_loader = DataLoader(
            "tickers.txt",
            str(actual_date),
            str(end_date),  # date, not datetime — no .date() needed
            original_min_history,
        )
        current_close = current_loader.download_clean_data()

        if not current_close.empty:
            latest_prices = current_close.iloc[-1]
            latest_date = current_close.index[-1].date()
            log.info("Current prices loaded from %s", latest_date)
        else:
            log.warning("current_close is empty; will fall back to signal-date prices.")
    except Exception:
        log.exception("Error downloading current prices; falling back.")

    # If the download failed or returned nothing, use the last row of the
    # original close DataFrame as a graceful fallback.
    if latest_prices is None:
        latest_prices = close.iloc[-1]
        latest_date = close.index[-1].date()
        log.info("Using fallback prices from %s", latest_date)

    # ------------------------------------------------------------------ #
    # Current portfolio value                                              #
    # ------------------------------------------------------------------ #
    current_value = cash_left
    for _, row in positions_df.iterrows():
        tkr = row["Ticker"]
        shares = int(row["Shares"])
        if tkr in latest_prices.index and shares > 0:
            cp = latest_prices[tkr]
            if not (isinstance(cp, float) and math.isnan(cp)):
                current_value += shares * cp

    # Gains are relative to what was actually invested at signal prices,
    # keeping the metric internally consistent regardless of rounding.
    gains_sek = current_value - invested_capital

    # ------------------------------------------------------------------ #
    # Build positions list                                                 #
    # ------------------------------------------------------------------ #
    positions = []
    for _, row in positions_df.iterrows():
        tkr = row["Ticker"]
        m = metrics.get(tkr, {})

        signal_day_price = None
        if signal_prices is not None and tkr in signal_prices.index:
            sp = signal_prices[tkr]
            if not (isinstance(sp, float) and math.isnan(sp)):
                signal_day_price = _safe_float(sp)

        current_day_price = None
        if tkr in latest_prices.index:
            cp = latest_prices[tkr]
            if not (isinstance(cp, float) and math.isnan(cp)):
                current_day_price = _safe_float(cp)

        delta_weight = _safe_float(
            row.get("Delta Weight (%)")
            if "Delta Weight (%)" in row
            else row.get("\u0394 Weight (%)")
        )

        positions.append(
            {
                "ticker": tkr,
                "weight_pct": _safe_float(row["Weight (%)"]),
                "target_sek": _safe_float(row["Target SEK"]),
                "price_sek": _safe_float(row["Price (SEK)"]),
                "shares": int(row["Shares"]),
                "actual_sek": _safe_float(row["Actual SEK"]),
                "delta_weight_pp": delta_weight,
                "signal_day_price": signal_day_price,
                "current_day_price": current_day_price,
                "sharpe": m.get("sharpe"),
                "ann_return": m.get("ann_return"),
                "ann_vol": m.get("ann_vol"),
                "max_dd": m.get("max_dd"),
                "roc_short": m.get("roc_short"),
                "roc_long": m.get("roc_long"),
                "acceleration": m.get("acceleration"),
            }
        )

    return jsonify(
        {
            "positions": positions,
            "signal_date": str(actual_date),
            "price_date": str(latest_date),
            "cash_left": round(cash_left, 2),
            "portfolio_value": portfolio_value,
            "current_value": round(current_value, 2),
            "gains_sek": round(gains_sek, 2),
            "universe_size": close.shape[1],
        }
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s  %(name)s  %(message)s"
    )
    print("\n  Signal Generator Dashboard")
    print("  http://localhost:5000\n")
    app.run(debug=True, port=5000)
