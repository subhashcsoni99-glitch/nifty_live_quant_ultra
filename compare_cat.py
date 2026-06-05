#!/usr/bin/env python3
"""
Intraday vs Swing Backtest — PER CATEGORY (Cat A/B/C/D)
All review fixes applied:
  1. BLACKLIST: SBIN, BHEL, TITAN skipped
  2. REVERSAL: 5-day hold + 2% loss required
  3. RSI ENTRY FILTER: skip BUY if RSI > 60
  4. BEAR REGIME SL: SL at 80% of normal ATR
  5. T1/T2 per mode multipliers
  6. ABSSL adaptive: 3% before T1, 1.5% after T1 partial
  7. TSL always active after T1 partial
"""
import sys, os, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from datetime import datetime
from nifty_core import (
    GOOD_STOCKS, ATR_CONFIG,
    get_ohlc, add_features,
    get_signal as core_get_signal,
    ai_opinion_pipeline, get_sector,
    build_ml_features,
)
import joblib

MAX_HOLD_DAYS = 15
SLIPPAGE = 0.001

# ─── NEW CONFIG ─────────────────────────────────────────────────────────────
BLACKLIST = {'SBIN', 'BHEL', 'TITAN'}
REVERSAL_MIN_HOLD_DAYS = 5
REVERSAL_MIN_LOSS_PCT = 2.0
RSI_ENTRY_MAX = 60
BEAR_REGIME_SL_FACTOR = 0.8

# ─── ML Prediction ─────────────────────────────────────────────────────────
def get_ml_prediction(sym, df):
    model_path = f"models/{sym.upper()}_model.joblib"
    if not os.path.exists(model_path): return None
    try:
        model = joblib.load(model_path)
        features = build_ml_features(df)
        proba = model.predict_proba(features)[0]
        direction = model.predict(features)[0]
        return {'direction': 'UP' if direction == 1 else 'DOWN', 'confidence': round(max(proba)*100, 1)}
    except: return None

def _categorize(ai, ml, div):
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

def backtest_stock(symbol, start, end, mode='intraday', max_position_pct=0.2, no_sig_exit=False):
    name = symbol.replace('.NS', '')
    df = get_ohlc(symbol, days=1095)
    if df is None: return None
    df = add_features(df)
    if hasattr(df.index, 'tz') and df.index.tz is not None: df = df.tz_localize(None)
    ts_end = pd.Timestamp(end); ts_end = ts_end.tz_localize(None) if ts_end.tzinfo else ts_end
    df = df[df.index <= ts_end]
    ts_start = pd.Timestamp(start); ts_start = ts_start.tz_localize(None) if ts_start.tzinfo else ts_start
    df = df[df.index >= ts_start]
    if len(df) < 200: return None

    cfg = ATR_CONFIG[mode]
    sl_mult = cfg['sl'] * BEAR_REGIME_SL_FACTOR  # bear-regime adjustment
    t1_mult, t2_mult = cfg['t1'], cfg['t2']

    initial_capital = 100000.0
    capital = initial_capital
    shares = 0; position = None; entry_price = 0; entry_date = None
    tsl = 0; shares_remaining = 0; partial_exits = 0; t1_triggered = False
    trades = []; pnl_list = []; realized_pnl_sum = 0.0; peak_realized = capital
    entry_cat = 'C'

    for i in range(200, len(df)):
        sig_val, sig_name, div = core_get_signal(df, i)
        price = df['Close'].iloc[i]
        atr = df['atr'].iloc[i]
        if pd.isna(atr) or atr == 0: atr = price * 0.02
        slip_entry = price * (1 + SLIPPAGE)
        slip_exit  = price * (1 - SLIPPAGE)
        rsi = df['rsi'].iloc[i]

        # ── Entry ──────────────────────────────────────────────────────
        if sig_val == 1 and position is None:
            # RSI entry filter — skip overbought entries
            if not (pd.isna(rsi) or rsi < RSI_ENTRY_MAX): continue
            # Blacklist check
            if name in BLACKLIST: continue

            risk = capital * 0.01
            sl_dist = atr * sl_mult
            raw_shares = max(1, int(risk / sl_dist))
            pos_value = raw_shares * slip_entry
            max_pos = capital * max_position_pct
            if pos_value > max_pos: raw_shares = max(1, int(max_pos / slip_entry))
            shares = raw_shares; shares_remaining = raw_shares
            t1_triggered = False; position = 'LONG'
            entry_price = slip_entry; entry_date = df.index[i]
            tsl = entry_price - atr * 1.5; capital -= shares * entry_price
            macd = df['macd'].iloc[i]; macd_sig = df['macd_sig'].iloc[i]; vol_ratio = df['vol_ratio'].iloc[i]; ret5 = df['ret5'].iloc[i]
            ai = ai_opinion_pipeline(name, price, rsi, macd, macd_sig, atr, vol_ratio, ret5, df)
            ml = get_ml_prediction(name, df)
            entry_cat = _categorize(ai, ml, div)

        # ── REVERSAL Exit: 5-day + 2% loss required ────────────────────────────
        elif sig_val == -1 and position == 'LONG' and not no_sig_exit:
            pullback_pct = (entry_price - price) / entry_price * 100
            hold_days = (df.index[i] - entry_date).days if entry_date else 0
            if pullback_pct >= REVERSAL_MIN_LOSS_PCT and hold_days >= REVERSAL_MIN_HOLD_DAYS:
                if shares_remaining > 0: capital += shares_remaining * slip_exit
                pnl = ((slip_exit - entry_price) / entry_price) * 100
                pnl_list.append(pnl); realized_pnl_sum += shares_remaining * (slip_exit - entry_price)
                peak_realized = max(peak_realized, capital)
                trades.append({'pnl': round(pnl, 2), 'type': 'REVERSAL', 'cat': entry_cat,
                               'date': str(df.index[i].date()), 'partial': partial_exits > 0,
                               'pullback_pct': round(pullback_pct, 2)})
                shares = 0; shares_remaining = 0; t1_triggered = False
                position = None; entry_date = None

        # ── In Position ─────────────────────────────────────────────────
        elif position == 'LONG':
            sl = entry_price - atr * sl_mult
            t1 = entry_price + atr * t1_mult
            t2 = entry_price + atr * t2_mult

            # T1 Partial Exit
            if shares_remaining > 0 and price >= t1 and not t1_triggered:
                exit_shares = shares_remaining // 2
                capital += exit_shares * slip_exit; shares_remaining -= exit_shares
                partial_exits += 1; tsl = max(tsl, entry_price)
                t1_triggered = True; peak_realized = max(peak_realized, capital)
                tsl = max(tsl, price - atr * 1.5)

            # TSL — always active after T1 partial
            if t1_triggered and shares_remaining > 0:
                new_tsl = price - atr * 1.5
                if new_tsl > tsl: tsl = new_tsl
                if tsl > 0 and price <= tsl:
                    pnl = ((slip_exit - entry_price) / entry_price) * 100
                    pnl_list.append(pnl); realized_pnl_sum += shares_remaining * (slip_exit - entry_price)
                    peak_realized = max(peak_realized, capital)
                    trades.append({'pnl': round(pnl, 2), 'type': 'TSL', 'cat': entry_cat,
                                   'date': str(df.index[i].date()), 'partial': partial_exits > 0})
                    shares = 0; shares_remaining = 0; t1_triggered = False
                    position = None; entry_date = None; continue

            # T2 — only after T1 partial
            if t1_triggered and price >= t2 and shares_remaining > 0:
                pnl = ((slip_exit - entry_price) / entry_price) * 100
                pnl_list.append(pnl); realized_pnl_sum += shares_remaining * (slip_exit - entry_price)
                peak_realized = max(peak_realized, capital)
                trades.append({'pnl': round(pnl, 2), 'type': 'T2', 'cat': entry_cat,
                               'date': str(df.index[i].date()), 'partial': partial_exits > 0})
                shares = 0; shares_remaining = 0; t1_triggered = False
                position = None; entry_date = None; continue

            # Time Exit — only cut losers
            hold_days = (df.index[i] - entry_date).days if entry_date else 0
            unreal_pnl = ((price - entry_price) / entry_price) * 100 if entry_price > 0 else 0
            if hold_days > MAX_HOLD_DAYS and unreal_pnl < 0:
                if shares_remaining > 0: capital += shares_remaining * slip_exit
                pnl = ((slip_exit - entry_price) / entry_price) * 100
                pnl_list.append(pnl); realized_pnl_sum += shares_remaining * (slip_exit - entry_price)
                peak_realized = max(peak_realized, capital)
                trades.append({'pnl': round(pnl, 2), 'type': 'TIME', 'cat': entry_cat,
                               'date': str(df.index[i].date()), 'partial': partial_exits > 0,
                               'hold_days': hold_days, 'unreal_pct': round(unreal_pnl, 2)})
                shares = 0; shares_remaining = 0; t1_triggered = False
                position = None; entry_date = None; continue

            # ABSSL — adaptive: 1.5% after T1, 3% before
            abs_sl = entry_price * (0.985 if t1_triggered else 0.97)
            if price <= abs_sl:
                if shares_remaining > 0: capital += shares_remaining * slip_exit
                pnl = ((slip_exit - entry_price) / entry_price) * 100
                pnl_list.append(pnl); realized_pnl_sum += shares_remaining * (slip_exit - entry_price)
                peak_realized = max(peak_realized, capital)
                trades.append({'pnl': round(pnl, 2), 'type': 'ABSSL', 'cat': entry_cat,
                               'date': str(df.index[i].date()), 'partial': partial_exits > 0})
                shares = 0; shares_remaining = 0; t1_triggered = False
                position = None; entry_date = None; continue

            # Fixed SL
            if price <= sl:
                if shares_remaining > 0: capital += shares_remaining * slip_exit
                pnl = ((slip_exit - entry_price) / entry_price) * 100
                pnl_list.append(pnl); realized_pnl_sum += shares_remaining * (slip_exit - entry_price)
                peak_realized = max(peak_realized, capital)
                trades.append({'pnl': round(pnl, 2), 'type': 'SL', 'cat': entry_cat,
                               'date': str(df.index[i].date()), 'partial': partial_exits > 0})
                shares = 0; shares_remaining = 0; t1_triggered = False
                position = None; entry_date = None

    if position == 'LONG' and shares_remaining > 0:
        slip_exit_end = df['Close'].iloc[-1] * (1 - SLIPPAGE)
        capital += shares_remaining * slip_exit_end
        pnl = ((slip_exit_end - entry_price) / entry_price) * 100
        pnl_list.append(pnl); realized_pnl_sum += shares_remaining * (slip_exit_end - entry_price)
        peak_realized = max(peak_realized, capital)
        trades.append({'pnl': round(pnl, 2), 'type': 'CLOSED', 'cat': entry_cat,
                       'date': str(df.index[-1].date()), 'partial': partial_exits > 0})

    return {'symbol': name, 'trades': trades, 'capital': capital, 'initial': initial_capital}


def print_summary(trades_by_cat, mode):
    cats = ['A', 'B', 'C', 'D']
    tw = tl = tt = 0
    cfg = ATR_CONFIG[mode]
    sl_mult = cfg['sl'] * BEAR_REGIME_SL_FACTOR
    print(f"\n{'='*70}")
    print(f"📊 {mode.upper()} — SL={sl_mult:.1f}× | T1={cfg['t1']}× | T2={cfg['t2']}× | RSI<{RSI_ENTRY_MAX} | REV:{REVERSAL_MIN_HOLD_DAYS}d+{REVERSAL_MIN_LOSS_PCT}%")
    print(f"{'='*70}")
    for cat in cats:
        trades = trades_by_cat.get(cat, [])
        if not trades:
            print(f"\n  🏷️ Cat {cat}: -- (no trades)"); continue
        wins = [t for t in trades if t['pnl'] > 0]; losses = [t for t in trades if t['pnl'] <= 0]
        wr = round(len(wins)/len(trades)*100, 1); succ = len(wins); fail = len(losses); total = len(trades)
        tw += succ; tl += fail; tt += total
        avg_ret = round(sum(t['pnl'] for t in trades)/len(trades), 2)
        exit_types = {}
        for t in trades: exit_types[t['type']] = exit_types.get(t['type'], 0) + 1
        exit_str = ' | '.join(f"{k}:{v}" for k, v in sorted(exit_types.items()))
        print(f"\n  🏷️ Cat {cat}: {total} trades | ✅{succ} ({wr}%) | ❌{fail} | Ret:{avg_ret:+.2f}%")
        print(f"     Exits: {exit_str}")
    if tt > 0: print(f"\n  📋 OVERALL: {tt} trades | ✅{tw} ({tw/tt*100:.1f}%) | ❌{tl}")
    return tw, tl, tt


end_dt = datetime(2026, 5, 15); start_dt = datetime(2025, 5, 15)
results = {}
for mode in ['intraday', 'swing']:
    print(f"\n{'#'*70}")
    print(f"# ⚙️  MODE: {mode.upper()}  |  {start_dt.date()} → {end_dt.date()}")
    print(f"# BLACKLIST: {BLACKLIST} | REVERSAL: {REVERSAL_MIN_HOLD_DAYS}d+{REVERSAL_MIN_LOSS_PCT}% | RSI<{RSI_ENTRY_MAX}")
    print(f"{'#'*70}")
    tbc = {'A': [], 'B': [], 'C': [], 'D': []}; per_stock = []
    for sym in GOOD_STOCKS:
        res = backtest_stock(sym, start_dt, end_dt, mode=mode)
        if res and res['trades']:
            per_stock.append(res)
            for t in res['trades']: tbc.setdefault(t.get('cat', 'C'), []).append(t)
    tw, tl, tt = print_summary(tbc, mode)
    print(f"\n  {'─'*65}")
    print(f"  {'Stock':<15} {'Dom':>4} {'Trds':>5} {'WR%':>6} {'Ret%':>7} {'W':>3} {'L':>3} {'Exits':>22}")
    print(f"  {'─'*65}")
    for res in sorted(per_stock, key=lambda x: -len(x['trades'])):
        cc = {}
        for t in res['trades']: cc[t.get('cat','C')] = cc.get(t.get('cat','C'), 0) + 1
        dom = max(cc, key=lambda c: cc[c]) if cc else '-'
        tds = res['trades']; wins = len([t for t in tds if t['pnl'] > 0]); losses = len([t for t in tds if t['pnl'] <= 0])
        wr = round(wins/len(tds)*100, 1); avg_ret = round(sum(t['pnl'] for t in tds)/len(tds), 2)
        et = {}; [et.__setitem__(t['type'], et.get(t['type'], 0) + 1) for t in tds]
        exit_str = '/'.join(f"{k}:{v}" for k, v in sorted(et.items()))
        print(f"  {res['symbol']:<15} {dom:>4} {len(tds):>5} {wr:>6.1f} {avg_ret:>+7.2f} {wins:>3} {losses:>3}  {exit_str}")
    results[mode] = {'wins': tw, 'losses': tl, 'trades': tt, 'per_stock': per_stock, 'tbc': tbc}

print(f"\n{'#'*70}")
print(f"# 📊 COMPARISON — INTRADAY vs SWING")
print(f"# ATR config: INTRADAY (SL=3×→2.4× bear, T1=2×, T2=3.5×) | SWING (SL=2×→1.6× bear, T1=3×, T2=6×)")
print(f"# BLACKLIST: {BLACKLIST} | REVERSAL: {REVERSAL_MIN_HOLD_DAYS}d+{REVERSAL_MIN_LOSS_PCT}%")
print(f"# RSI ENTRY: <{RSI_ENTRY_MAX} | BEAR REGIME SL FACTOR: {BEAR_REGIME_SL_FACTOR}")
print(f"# ABSSL: 3% before T1, 1.5% after T1 partial")
print(f"# TSL: always active after T1 partial")
print(f"{'#'*70}")
print(f"\n{'Metric':<28} {'💥 INTRADAY':>15} {'🎯 SWING':>15}")
print(f"{'-'*58}")
cats = ['A', 'B', 'C', 'D']
for cat in cats:
    id_t = results['intraday']['tbc'].get(cat, []); sw_t = results['swing']['tbc'].get(cat, [])
    id_w = len([t for t in id_t if t['pnl'] > 0]); id_l = len([t for t in id_t if t['pnl'] <= 0])
    sw_w = len([t for t in sw_t if t['pnl'] > 0]); sw_l = len([t for t in sw_t if t['pnl'] <= 0])
    id_wr = f"{id_w/len(id_t)*100:.1f}%" if id_t else "0%"; sw_wr = f"{sw_w/len(sw_t)*100:.1f}%" if sw_t else "0%"
    id_ret = f"{sum(t['pnl'] for t in id_t)/len(id_t):+.2f}%" if id_t else "0%"
    sw_ret = f"{sum(t['pnl'] for t in sw_t)/len(sw_t):+.2f}%" if sw_t else "0%"
    print(f"  Cat {cat} WR%              {id_wr:>14} {sw_wr:>14}")
    print(f"  Cat {cat} Avg Ret%         {id_ret:>14} {sw_ret:>14}")
    print(f"  Cat {cat} Trades           {len(id_t):>14} {len(sw_t):>14}")
    print(f"  {'─'*58}")