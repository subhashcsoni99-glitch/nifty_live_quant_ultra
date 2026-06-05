#!/usr/bin/env python3
"""
Test all 5 suggestions vs baseline.
Each test runs independently; results shown side-by-side.
"""
import sys, os, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from datetime import datetime
from nifty_core import GOOD_STOCKS, ATR_CONFIG, get_ohlc, add_features, get_signal as core_get_signal, ai_opinion_pipeline, get_sector, build_ml_features
import joblib

SLIPPAGE = 0.001
MAX_HOLD = 15

BLACKLIST = {'SBIN', 'BHEL', 'TITAN'}

def _cat(ai, ml, div):
    if ai is None: return 'C'
    ai_out = ai.get('outlook', 'NEUTRAL')
    ai_conf = ai.get('confidence', 'LOW')
    ml_dir = (ml.get('direction', None) if ml else None)
    ml_up = ml_dir == 'UP'
    ai_bull = ai_out == 'BULLISH'
    bear_downgrade = (div == 'BEARISH')
    if bear_downgrade: return 'C'
    if ai_bull and ml_up: return 'A'
    if ai_bull and (ai_conf in ('HIGH', 'MEDIUM') or (ml is None and ai_conf in ('HIGH', 'MEDIUM'))): return 'B'
    if ml_up: return 'D'
    return 'C'

def get_ml(sym, df):
    mp = f"models/{sym.upper()}_model.joblib"
    if not os.path.exists(mp): return None
    try:
        m = joblib.load(mp)
        feat = build_ml_features(df)
        p = m.predict_proba(feat)[0]; d = m.predict(feat)[0]
        return {'direction': 'UP' if d == 1 else 'DOWN', 'confidence': round(max(p)*100, 1)}
    except: return None

def run_mode(mode, rev_hold=3, rev_loss=1.0, blacklist=False, t1_override=None, sig_exit=True, rsi_filter=False):
    cfg = dict(ATR_CONFIG[mode])
    if t1_override is not None: cfg['t1'] = t1_override
    sl_mult, t1_mult, t2_mult = cfg['sl'], cfg['t1'], cfg['t2']
    tbc = {'A':[], 'B':[], 'C':[], 'D':[]}
    per_stock = []
    start, end = datetime(2025,5,15), datetime(2026,5,15)
    for sym in GOOD_STOCKS:
        if blacklist and sym in BLACKLIST: continue
        name = sym.replace('.NS', '')
        df = get_ohlc(sym, days=1095)
        if df is None: continue
        df = add_features(df)
        if hasattr(df.index, 'tz') and df.index.tz: df = df.tz_localize(None)
        df = df[df.index <= pd.Timestamp(end).tz_localize(None)]
        df = df[df.index >= pd.Timestamp(start).tz_localize(None)]
        if len(df) < 200: continue

        capital = 100000.0; shares = 0; position = None; entry_price = 0; entry_date = None
        tsl = 0; shares_remaining = 0; t1_triggered = False; partial_exits = 0
        trades = []; pnl_list = []; realized_pnl_sum = 0.0; peak_realized = capital
        entry_cat = 'C'

        for i in range(200, len(df)):
            sig_val, sig_name, div = core_get_signal(df, i)
            price = df['Close'].iloc[i]; atr = df['atr'].iloc[i]
            if pd.isna(atr) or atr == 0: atr = price * 0.02
            rsi = df['rsi'].iloc[i]
            slip_entry = price * (1 + SLIPPAGE); slip_exit = price * (1 - SLIPPAGE)

            # Entry filter: skip if RSI > 60 (overbought)
            if sig_val == 1 and position is None:
                if rsi_filter and not (pd.isna(rsi) or rsi < 60): continue
                risk = capital * 0.01
                sl_dist = atr * sl_mult
                raw_shares = max(1, int(risk / sl_dist))
                pos_value = raw_shares * slip_entry
                max_pos = capital * 0.2
                if pos_value > max_pos: raw_shares = max(1, int(max_pos / slip_entry))
                shares = raw_shares; shares_remaining = raw_shares
                t1_triggered = False; position = 'LONG'
                entry_price = slip_entry; entry_date = df.index[i]
                tsl = entry_price - atr * 1.5; capital -= shares * entry_price
                macd = df['macd'].iloc[i]; macd_sig = df['macd_sig'].iloc[i]; vol_ratio = df['vol_ratio'].iloc[i]; ret5 = df['ret5'].iloc[i]
                ai = ai_opinion_pipeline(name, price, rsi, macd, macd_sig, atr, vol_ratio, ret5, df)
                ml = get_ml(name, df)
                entry_cat = _cat(ai, ml, div)

            # REVERSAL exit with configurable min hold and loss threshold
            elif sig_val == -1 and position == 'LONG' and sig_exit:
                pullback_pct = (entry_price - price) / entry_price * 100
                hold_days = (df.index[i] - entry_date).days if entry_date else 0
                unreal_pnl = ((price - entry_price) / entry_price) * 100 if entry_price > 0 else 0
                if pullback_pct >= rev_loss and hold_days >= rev_hold:
                    if shares_remaining > 0: capital += shares_remaining * slip_exit
                    pnl = ((slip_exit - entry_price) / entry_price) * 100
                    pnl_list.append(pnl); realized_pnl_sum += shares_remaining * (slip_exit - entry_price)
                    peak_realized = max(peak_realized, capital)
                    trades.append({'pnl': round(pnl,2), 'type': 'REVERSAL', 'cat': entry_cat, 'date': str(df.index[i].date()), 'partial': partial_exits > 0, 'pullback_pct': round(pullback_pct,2)})
                    shares = 0; shares_remaining = 0; t1_triggered = False; position = None; entry_date = None

            elif position == 'LONG':
                sl = entry_price - atr * sl_mult
                t1 = entry_price + atr * t1_mult
                t2 = entry_price + atr * t2_mult

                if shares_remaining > 0 and price >= t1 and not t1_triggered:
                    exit_shares = shares_remaining // 2
                    capital += exit_shares * slip_exit; shares_remaining -= exit_shares
                    partial_exits += 1; t1_triggered = True; peak_realized = max(peak_realized, capital)
                    tsl = max(tsl, entry_price); tsl = max(tsl, price - atr * 1.5)

                if t1_triggered and shares_remaining > 0:
                    new_tsl = price - atr * 1.5
                    if new_tsl > tsl: tsl = new_tsl
                    if tsl > 0 and price <= tsl:
                        pnl = ((slip_exit - entry_price) / entry_price) * 100
                        pnl_list.append(pnl); realized_pnl_sum += shares_remaining * (slip_exit - entry_price)
                        peak_realized = max(peak_realized, capital)
                        trades.append({'pnl': round(pnl,2), 'type': 'TSL', 'cat': entry_cat, 'date': str(df.index[i].date()), 'partial': partial_exits > 0})
                        shares = 0; shares_remaining = 0; t1_triggered = False; position = None; entry_date = None; continue

                if t1_triggered and price >= t2 and shares_remaining > 0:
                    pnl = ((slip_exit - entry_price) / entry_price) * 100
                    pnl_list.append(pnl); realized_pnl_sum += shares_remaining * (slip_exit - entry_price)
                    peak_realized = max(peak_realized, capital)
                    trades.append({'pnl': round(pnl,2), 'type': 'T2', 'cat': entry_cat, 'date': str(df.index[i].date()), 'partial': partial_exits > 0})
                    shares = 0; shares_remaining = 0; t1_triggered = False; position = None; entry_date = None; continue

                hold_days = (df.index[i] - entry_date).days if entry_date else 0
                unreal_pnl = ((price - entry_price) / entry_price) * 100 if entry_price > 0 else 0
                if hold_days > MAX_HOLD and unreal_pnl < 0:
                    if shares_remaining > 0: capital += shares_remaining * slip_exit
                    pnl = ((slip_exit - entry_price) / entry_price) * 100
                    pnl_list.append(pnl); realized_pnl_sum += shares_remaining * (slip_exit - entry_price)
                    peak_realized = max(peak_realized, capital)
                    trades.append({'pnl': round(pnl,2), 'type': 'TIME', 'cat': entry_cat, 'date': str(df.index[i].date()), 'partial': partial_exits > 0, 'hold_days': hold_days, 'unreal_pct': round(unreal_pnl,2)})
                    shares = 0; shares_remaining = 0; t1_triggered = False; position = None; entry_date = None; continue

                abs_sl = entry_price * (0.985 if t1_triggered else 0.97)
                if price <= abs_sl:
                    if shares_remaining > 0: capital += shares_remaining * slip_exit
                    pnl = ((slip_exit - entry_price) / entry_price) * 100
                    pnl_list.append(pnl); realized_pnl_sum += shares_remaining * (slip_exit - entry_price)
                    peak_realized = max(peak_realized, capital)
                    trades.append({'pnl': round(pnl,2), 'type': 'ABSSL', 'cat': entry_cat, 'date': str(df.index[i].date()), 'partial': partial_exits > 0})
                    shares = 0; shares_remaining = 0; t1_triggered = False; position = None; entry_date = None; continue

                if price <= sl:
                    if shares_remaining > 0: capital += shares_remaining * slip_exit
                    pnl = ((slip_exit - entry_price) / entry_price) * 100
                    pnl_list.append(pnl); realized_pnl_sum += shares_remaining * (slip_exit - entry_price)
                    peak_realized = max(peak_realized, capital)
                    trades.append({'pnl': round(pnl,2), 'type': 'SL', 'cat': entry_cat, 'date': str(df.index[i].date()), 'partial': partial_exits > 0})
                    shares = 0; shares_remaining = 0; t1_triggered = False; position = None; entry_date = None

        if position == 'LONG' and shares_remaining > 0:
            se = df['Close'].iloc[-1] * (1 - SLIPPAGE)
            capital += shares_remaining * se
            pnl = ((se - entry_price) / entry_price) * 100
            pnl_list.append(pnl); realized_pnl_sum += shares_remaining * (se - entry_price)
            peak_realized = max(peak_realized, capital)
            trades.append({'pnl': round(pnl,2), 'type': 'CLOSED', 'cat': entry_cat, 'date': str(df.index[-1].date()), 'partial': partial_exits > 0})

        if trades:
            for t in trades: tbc.setdefault(t.get('cat','C'), []).append(t)
            per_stock.append({'symbol': name, 'trades': trades})

    total_trades = sum(len(v) for v in tbc.values())
    wins = sum(len([t for t in v if t['pnl'] > 0]) for v in tbc.values())
    wr = round(wins/total_trades*100, 1) if total_trades else 0
    avg_ret = round(sum(t['pnl'] for v in tbc.values() for t in v)/total_trades, 2) if total_trades else 0
    exit_totals = {}
    for v in tbc.values():
        for t in v: exit_totals[t['type']] = exit_totals.get(t['type'], 0) + 1

    cat_a = tbc.get('A', []); cat_b = tbc.get('B', [])
    a_wr = round(len([t for t in cat_a if t['pnl'] > 0])/len(cat_a)*100, 1) if cat_a else 0
    b_wr = round(len([t for t in cat_b if t['pnl'] > 0])/len(cat_b)*100, 1) if cat_b else 0
    a_ret = round(sum(t['pnl'] for t in cat_a)/len(cat_a), 2) if cat_a else 0
    b_ret = round(sum(t['pnl'] for t in cat_b)/len(cat_b), 2) if cat_b else 0

    return {'wr': wr, 'avg_ret': avg_ret, 'total': total_trades, 'wins': wins,
            'cat_a_wr': a_wr, 'cat_b_wr': b_wr, 'cat_a_ret': a_ret, 'cat_b_ret': b_ret,
            'exit_totals': exit_totals, 'tbc': tbc, 'per_stock': per_stock}

def fmt(res):
    return f"{res['wr']}%  {res['avg_ret']:+.2f}%  {res['total']}trds  A_WR={res['cat_a_wr']}%  B_WR={res['cat_b_wr']}%"

tests = [
    ("1️⃣ BASELINE",         dict()),
    ("2️⃣ REVERSAL stricter", dict(rev_hold=5, rev_loss=2.0)),
    ("3️⃣ BLACKLIST",         dict(blacklist=True)),
    ("4️⃣ T1=1.5× INTRADAY", dict(t1_override=1.5)),
    ("5️⃣ RSI filter",        dict(rsi_filter=True)),
    ("6️⃣ NO SIG EXIT",      dict(sig_exit=False)),
    ("7️⃣ COMBINED BEAR",    dict(rev_hold=5, rev_loss=2.0, blacklist=True, t1_override=1.5, sig_exit=True)),
]

results = {}
for label, params in tests:
    m = params.get('mode', 'intraday')
    res = run_mode(m, **params)
    results[label] = res

print("=" * 90)
print("📊 ALL 5 SUGGESTIONS TEST — INTRADAY | May 2025 → May 2026")
print("=" * 90)
print(f"\n{'Test':<22} {'WR':>8} {'AvgRet':>9} {'Trades':>7}  {'A_WR':>7} {'A_Ret':>8} {'B_WR':>7} {'B_Ret':>8}")
print("-" * 90)
for label, res in results.items():
    print(f"  {label:<20} {res['wr']:>7} {res['avg_ret']:>+8.2f} {res['total']:>6}  {res['cat_a_wr']:>6.1f}% {res['cat_a_ret']:>+7.2f}% {res['cat_b_wr']:>6.1f}% {res['cat_b_ret']:>+7.2f}%")

print("\n" + "=" * 90)
print("🔍 EXIT BREAKDOWN PER TEST")
print("=" * 90)
for label, res in results.items():
    exits = res['exit_totals']
    total = res['total']
    rev_pct = round(exits.get('REVERSAL', 0)/total*100, 1) if total else 0
    abssl_pct = round(exits.get('ABSSL', 0)/total*100, 1) if total else 0
    exit_str = ' | '.join(f"{k}:{v}" for k, v in sorted(exits.items()))
    print(f"\n  {label}")
    print(f"    REVERSAL={exits.get('REVERSAL',0)} ({rev_pct}%) | ABSSL={exits.get('ABSSL',0)} ({abssl_pct}%) | {exit_str}")

print("\n" + "=" * 90)
print("📈 TOP STOCKS PER TEST (sorted by avg return)")
print("=" * 90)
for label, res in results.items():
    top = sorted(res['per_stock'], key=lambda x: sum(t['pnl'] for t in x['trades'])/len(x['trades']), reverse=True)[:5]
    if not top: print(f"\n  {label}: no trades"); continue
    parts = []
    for s in top:
        avg = sum(t['pnl'] for t in s['trades'])/len(s['trades'])
        wr = len([t for t in s['trades'] if t['pnl'] > 0])/len(s['trades'])*100
        parts.append(f"{s['symbol']}:{avg:+.1f}%({wr:.0f}%)")
    print(f"\n  {label}: {' | '.join(parts)}")

print("\n" + "=" * 90)
print("📊 WORST STOCKS PER TEST")
print("=" * 90)
for label, res in results.items():
    worst = sorted(res['per_stock'], key=lambda x: sum(t['pnl'] for t in x['trades'])/len(x['trades']))[:5]
    if not worst: continue
    parts = []
    for s in worst:
        avg = sum(t['pnl'] for t in s['trades'])/len(s['trades'])
        wr = len([t for t in s['trades'] if t['pnl'] > 0])/len(s['trades'])*100
        parts.append(f"{s['symbol']}:{avg:+.1f}%({wr:.0f}%)")
    print(f"\n  {label}: {' | '.join(parts)}")