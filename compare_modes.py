#!/usr/bin/env python3
"""
Intraday vs Swing Backtest Comparison
Runs both level modes on 1 year of data and compares success/failure rates.
"""
import sys, os, warnings, json, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings('ignore')

from nifty_core import (
    NIFTY50_STOCKS, EXCLUDED_STOCKS, GOOD_STOCKS,
    ATR_CONFIG,
    get_ohlc, add_features,
    get_signal, get_sector, check_sector_limit, MAX_PER_SECTOR,
)
from datetime import datetime, timedelta

MIN_TRADES = 3

def run_backtest(symbol, start, end, mode='intraday', use_t1_partial=True,
                 max_hold_days=5, slippage_pct=0.001, max_position_pct=0.2,
                 sector_limits=False, use_trailing=False, verbose=False):
    name = symbol.replace('.NS', '')
    df = get_ohlc(symbol, days=1095)
    if df is None:
        return None
    df = add_features(df)
    if hasattr(df.index, 'tz') and df.index.tz is not None:
        df = df.tz_localize(None)
    if end:
        ts = pd.Timestamp(end); ts = ts.tz_localize(None) if ts.tzinfo else ts
        df = df[df.index <= ts]
    if start:
        ts = pd.Timestamp(start); ts = ts.tz_localize(None) if ts.tzinfo else ts
        df = df[df.index >= ts]
    if len(df) < 200:
        return None

    cfg = ATR_CONFIG[mode]
    sl_mult = cfg['sl']
    t1_mult = cfg['t1']
    t2_mult = cfg['t2']

    initial_capital = 100000.0
    capital = initial_capital
    shares = 0
    position = None
    entry_price = 0
    entry_date = None
    tsl = 0
    shares_remaining = 0
    partial_exits = 0
    t1_triggered = False
    trades = []
    sector_counts = {} if sector_limits else {}
    pnl_list = []
    realized_pnl_sum = 0.0
    peak_realized = capital

    for i in range(200, len(df)):
        sig_val, sig_name, div = get_signal(df, i)
        price = df['Close'].iloc[i]
        atr = df['atr'].iloc[i]
        if pd.isna(atr) or atr == 0:
            atr = price * 0.02

        slip_entry = price * (1 + slippage_pct)
        slip_exit  = price * (1 - slippage_pct)

        if sig_val == 1 and position is None:
            if sector_limits:
                sect = get_sector(name)
                if check_sector_limit(sect, sector_counts, MAX_PER_SECTOR):
                    continue
            risk = capital * 0.01
            sl_dist = atr * sl_mult
            raw_shares = max(1, int(risk / sl_dist))
            pos_value = raw_shares * slip_entry
            max_pos = capital * max_position_pct
            if pos_value > max_pos:
                raw_shares = max(1, int(max_pos / slip_entry))
            shares = raw_shares
            shares_remaining = raw_shares
            t1_triggered = False
            position = 'LONG'
            entry_price = slip_entry
            entry_date = df.index[i]
            tsl = entry_price - atr * 1.5
            capital -= shares * entry_price
            if sector_limits:
                sector_counts[sect] = sector_counts.get(sect, 0) + 1

        elif sig_val == -1 and position == 'LONG':
            pullback_pct = (entry_price - price) / entry_price * 100
            if pullback_pct >= 1.0:
                if shares_remaining > 0:
                    capital += shares_remaining * slip_exit
                pnl = ((slip_exit - entry_price) / entry_price) * 100
                pnl_list.append(pnl)
                realized_pnl_sum += shares_remaining * (slip_exit - entry_price)
                peak_realized = max(peak_realized, capital)
                trades.append({'pnl': round(pnl, 2), 'type': 'REVERSAL', 'date': str(df.index[i].date()),
                               'partial': partial_exits > 0, 'pullback_pct': round(pullback_pct, 2)})
                shares = 0; shares_remaining = 0; t1_triggered = False
                position = None; entry_date = None
                if sector_limits:
                    sector_counts[get_sector(name)] = max(0, sector_counts.get(get_sector(name), 0) - 1)

        elif position == 'LONG':
            sl = entry_price - atr * sl_mult
            t1 = entry_price + atr * t1_mult
            t2 = entry_price + atr * t2_mult

            if use_t1_partial and shares_remaining > 0 and price >= t1 and not t1_triggered:
                exit_shares = shares_remaining // 2
                capital += exit_shares * slip_exit
                shares_remaining -= exit_shares
                partial_exits += 1
                tsl = max(tsl, entry_price)
                t1_triggered = True
                peak_realized = max(peak_realized, capital)

            hold_days = (df.index[i] - entry_date).days if entry_date else 0
            unreal_pnl = ((price - entry_price) / entry_price) * 100 if entry_price > 0 else 0
            if hold_days > max_hold_days and sig_val != 1 and unreal_pnl < 2:
                if shares_remaining > 0:
                    capital += shares_remaining * slip_exit
                pnl = ((slip_exit - entry_price) / entry_price) * 100
                pnl_list.append(pnl)
                realized_pnl_sum += shares_remaining * (slip_exit - entry_price)
                peak_realized = max(peak_realized, capital)
                trades.append({'pnl': round(pnl, 2), 'type': 'TIME', 'date': str(df.index[i].date()),
                               'partial': partial_exits > 0, 'hold_days': hold_days, 'unreal_pct': round(unreal_pnl, 2)})
                shares = 0; shares_remaining = 0; t1_triggered = False
                position = None; entry_date = None
                if sector_limits:
                    sector_counts[get_sector(name)] = max(0, sector_counts.get(get_sector(name), 0) - 1)
                continue

            abs_sl = entry_price * 0.97
            if price <= abs_sl:
                if shares_remaining > 0:
                    capital += shares_remaining * slip_exit
                pnl = ((slip_exit - entry_price) / entry_price) * 100
                pnl_list.append(pnl)
                realized_pnl_sum += shares_remaining * (slip_exit - entry_price)
                peak_realized = max(peak_realized, capital)
                trades.append({'pnl': round(pnl, 2), 'type': 'ABSSL', 'date': str(df.index[i].date()), 'partial': partial_exits > 0})
                shares = 0; shares_remaining = 0; t1_triggered = False
                position = None; entry_date = None
                if sector_limits:
                    sector_counts[get_sector(name)] = max(0, sector_counts.get(get_sector(name), 0) - 1)
                continue

            if use_trailing:
                new_tsl = price - atr * 1.5
                if new_tsl > tsl and price >= entry_price + atr * 0.5:
                    tsl = new_tsl
                if tsl > 0 and price <= tsl:
                    if shares_remaining > 0:
                        capital += shares_remaining * slip_exit
                    pnl = ((slip_exit - entry_price) / entry_price) * 100
                    pnl_list.append(pnl)
                    realized_pnl_sum += shares_remaining * (slip_exit - entry_price)
                    peak_realized = max(peak_realized, capital)
                    trades.append({'pnl': round(pnl, 2), 'type': 'TSL', 'date': str(df.index[i].date()), 'partial': partial_exits > 0})
                    shares = 0; shares_remaining = 0; t1_triggered = False
                    position = None; entry_date = None
                    if sector_limits:
                        sector_counts[get_sector(name)] = max(0, sector_counts.get(get_sector(name), 0) - 1)
                    continue

            if price <= sl:
                if shares_remaining > 0:
                    capital += shares_remaining * slip_exit
                pnl = ((slip_exit - entry_price) / entry_price) * 100
                pnl_list.append(pnl)
                realized_pnl_sum += shares_remaining * (slip_exit - entry_price)
                peak_realized = max(peak_realized, capital)
                trades.append({'pnl': round(pnl, 2), 'type': 'SL', 'date': str(df.index[i].date()), 'partial': partial_exits > 0})
                shares = 0; shares_remaining = 0; t1_triggered = False
                position = None; entry_date = None
                if sector_limits:
                    sector_counts[get_sector(name)] = max(0, sector_counts.get(get_sector(name), 0) - 1)

    if position == 'LONG' and shares_remaining > 0:
        slip_exit_end = df['Close'].iloc[-1] * (1 - slippage_pct)
        capital += shares_remaining * slip_exit_end
        pnl = ((slip_exit_end - entry_price) / entry_price) * 100
        pnl_list.append(pnl)
        realized_pnl_sum += shares_remaining * (slip_exit_end - entry_price)
        peak_realized = max(peak_realized, capital)
        trades.append({'pnl': round(pnl, 2), 'type': 'CLOSED', 'date': str(df.index[-1].date()), 'partial': partial_exits > 0})

    if not trades:
        return None

    wins = [t['pnl'] for t in trades if t['pnl'] > 0]
    losses = [t['pnl'] for t in trades if t['pnl'] <= 0]
    total_ret = ((capital - initial_capital) / initial_capital) * 100
    realized_return = (realized_pnl_sum / initial_capital * 100)
    max_drawdown = max(0.0, (peak_realized - capital) / peak_realized * 100) if peak_realized > 0 else 0.0

    return {
        'symbol': name, 'trades': len(trades), 'wins': len(wins), 'losses': len(losses),
        'win_rate': round(len(wins) / len(trades) * 100, 1) if trades else 0.0,
        'realized_return': round(realized_return, 2),
        'return': round(total_ret, 2),
        'max_drawdown': round(max_drawdown, 2),
        'mode': mode,
    }


import pandas as pd
from datetime import datetime

end_dt = datetime(2026, 5, 15)
start_dt = datetime(2025, 5, 15)

stocks = GOOD_STOCKS

print(f"Running Intraday vs Swing backtest: {start_dt.date()} → {end_dt.date()}")
print("=" * 70)

results = {'intraday': [], 'swing': []}
for mode in ['intraday', 'swing']:
    print(f"\n{'─'*70}")
    print(f"⚙️  MODE: {mode.upper()}")
    print(f"{'─'*70}")
    for sym in stocks:
        r = run_backtest(sym, start_dt, end_dt, mode=mode)
        if r and r['trades'] > 0:
            results[mode].append(r)
            print(f"  {r['symbol']:15s} | Trds:{r['trades']:3d} | WR:{r['win_rate']:5.1f}% | "
                  f"Ret:{r['realized_return']:+6.2f}% | DD:{r['max_drawdown']:5.2f}%")

def summarize(mode_results, label):
    if not mode_results:
        return
    total_trades = sum(r['trades'] for r in mode_results)
    all_wins = sum(r['wins'] for r in mode_results)
    all_losses = sum(r['losses'] for r in mode_results)
    avg_ret = sum(r['realized_return'] for r in mode_results) / len(mode_results)
    avg_wr = sum(r['win_rate'] for r in mode_results) / len(mode_results)
    avg_dd = sum(r['max_drawdown'] for r in mode_results) / len(mode_results)
    success_pct = (all_wins / total_trades * 100) if total_trades else 0
    fail_pct = (all_losses / total_trades * 100) if total_trades else 0
    print(f"\n{'='*70}")
    print(f"📊 {label} SUMMARY")
    print(f"{'='*70}")
    print(f"  Stocks analyzed : {len(mode_results)}")
    print(f"  Total trades    : {total_trades}")
    print(f"  Wins   (✅)    : {all_wins}  ({success_pct:.1f}%)")
    print(f"  Losses (❌)    : {all_losses} ({fail_pct:.1f}%)")
    print(f"  Avg Win Rate   : {avg_wr:.1f}%")
    print(f"  Avg Return     : {avg_ret:+.2f}%")
    print(f"  Avg Drawdown   : {avg_dd:.2f}%")
    return success_pct, fail_pct, avg_wr, avg_ret, avg_dd

print("\n")
summarize(results['intraday'], "💥 INTRADAY")
summarize(results['swing'], "🎯 SWING")

print(f"\n{'='*70}")
print(f"📊 COMPARISON: INTRADAY vs SWING")
print(f"{'='*70}")
print(f"{'Metric':<25} {'💥 INTRADAY':>15} {'🎯 SWING':>15}")
print(f"{'-'*70}")
for metric, key in [('Success Rate %','wr'),('Avg Return %','ret'),('Avg Drawdown %','dd')]:
    id_val = next((r[key] for r in results['intraday']), 0)
    sw_val = next((r[key] for r in results['swing']), 0)
    label = {'wr':'Win Rate %','ret':'Avg Return %','dd':'Avg Drawdown %'}[key]
    id_disp = f"{id_val:.1f}%" if key == 'wr' else f"{id_val:+.2f}%"
    sw_disp = f"{sw_val:.1f}%" if key == 'wr' else f"{sw_val:+.2f}%"
    print(f"{label:<25} {id_disp:>15} {sw_disp:>15}")
print(f"{'-'*70}")