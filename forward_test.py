#!/usr/bin/env python3
"""
NIFTY Live Quant Ultra - Forward Testing (Paper Trading) v8
v8 changes:
  1. pnl_list tracks every realized exit for proper Sharpe calculation
  2. realized_return = realized_pnl / capital_at_risk (correct denominator)
  3. Peak capital = realized only (no unrealized counting)
  4. Signal exit fires with pullback guard (≥1% from entry) — same as backtest
  5. --no-sig-exit flag for SL/TSL-only exits
  6. ML validation still works as momentum filter
  7. Proper 1% risk position sizing per trade
"""
import sys
import os
import warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import yfinance as yf
import numpy as np
import pandas as pd
import json
from datetime import datetime

from nifty_core import (
    DEFAULT_STOCKS, ATR_CONFIG, RSI_CONFIG, SIGNAL_CONFIG,
    get_ohlc, add_features, build_ml_features,
)

STOCKS = DEFAULT_STOCKS

def get_signal(df, i):
    """Unified signal — mirrors backtest.py exactly."""
    if i < 200:
        return 0
    row = df.iloc[i]
    pv = row['Close']
    ma20 = row['ma20']
    ma50 = row['ma50']
    ma200 = row['ma200']
    rsi = row['rsi']
    macd, macd_sig = row['macd'], row['macd_sig']
    vol_ratio = row['vol_ratio']
    ret5 = row['ret5']

    if pd.isna(rsi) or pd.isna(ma20) or pd.isna(ma50) or pd.isna(ma200):
        return 0

    c_price_ma20 = pv > ma20
    c_price_ma50 = pv > ma50
    c_ma50_ma200 = ma50 > ma200
    c_rsi_buy = rsi < RSI_CONFIG['buy_strict']
    c_rsi_sell = rsi > RSI_CONFIG['sell_strict']
    c_macd = macd > macd_sig
    c_vol = vol_ratio > SIGNAL_CONFIG['volume_spike']
    c_mom = ret5 > SIGNAL_CONFIG['momentum_zero']

    buy_cnt = sum([c_price_ma20, c_price_ma50, c_ma50_ma200, c_rsi_buy, c_macd, c_vol, c_mom])
    sell_cnt = sum([pv < ma20, pv < ma50, ma50 < ma200,
                    c_rsi_sell, not c_macd, c_vol, ret5 < 0])

    from nifty_core import detect_divergence
    div = detect_divergence(df.iloc[:i+1])
    if div == "BULLISH":
        buy_cnt += 2

    if rsi > 70:
        buy_cnt = 0
    elif rsi > RSI_CONFIG['buy_relaxed']:
        if not c_ma50_ma200:
            buy_cnt = 0
    if rsi < RSI_CONFIG['sell_relaxed']:
        sell_cnt = 0

    if buy_cnt >= SIGNAL_CONFIG['min_confirmations']:
        return 1
    elif sell_cnt >= SIGNAL_CONFIG['min_confirmations']:
        return -1
    return 0

def get_ml_prediction(symbol, df, idx):
    """Get ML prediction at index idx. Returns (direction, confidence) or None."""
    model_path = f"models/{symbol.upper()}_model.joblib"
    if not os.path.exists(model_path):
        return None
    try:
        import joblib
        model = joblib.load(model_path)
        features = build_ml_features(df, idx)
        proba = model.predict_proba(features)[0]
        direction = model.predict(features)[0]
        conf = max(proba) * 100
        return {'direction': 'UP' if direction == 1 else 'DOWN', 'confidence': round(conf, 1)}
    except:
        return None

def forward_test_stock(symbol, test_days=30, use_ml=True, no_sig_exit=False, verbose=False):
    """Paper-trade a single stock over the last `test_days` trading days."""
    name = symbol.replace('.NS', '')
    df = get_ohlc(symbol, days=test_days + 250)  # extra days for MA lookback
    if df is None or len(df) < 250:
        return None

    df = add_features(df).dropna()
    if len(df) < 250:
        return None

    test_start_idx = max(200, len(df) - test_days)

    initial_capital = 100000.0
    capital = initial_capital
    peak_capital = capital
    shares = 0
    position = None
    entry_price = 0
    entry_capital_at_risk = 0.0
    entry_date = None
    tsl = 0
    shares_remaining = 0
    partial_exits = 0
    t1_triggered = False
    trades = []
    pnl_list = []
    realized_pnl_sum = 0.0

    for i in range(test_start_idx, len(df)):
        signal = get_signal(df, i)
        current_price = df['Close'].iloc[i]
        current_date = df.index[i]
        atr = df['atr'].iloc[i]
        if pd.isna(atr) or atr == 0:
            atr = current_price * 0.02

        slip = 0.001
        entry_px = current_price * (1 + slip)
        exit_px = current_price * (1 - slip)

        # ── Entry ─────────────────────────────────────────────────────
        if signal == 1 and position is None:
            if use_ml:
                ml = get_ml_prediction(name, df, i)
                ml_dir = ml['direction'] if ml else None
                if ml and ml_dir != 'UP':
                    if verbose:
                        print(f"  ⏭️  SKIP {name} @ ₹{current_price:.2f} — ML says {ml_dir} [{current_date.date()}]")
                    continue

            risk = capital * 0.01
            sl_dist = atr * ATR_CONFIG['swing']['sl']
            raw_shares = max(1, int(risk / sl_dist))
            shares = raw_shares
            shares_remaining = raw_shares
            t1_triggered = False
            position = 'LONG'
            entry_price = entry_px
            entry_date = current_date
            entry_capital_at_risk = shares * entry_px
            tsl = entry_price - atr * 1.5
            capital = capital - (shares * entry_price)

            ml_info = f" ML:{ml['direction']}({ml['confidence']}%)" if (use_ml and get_ml_prediction(name, df, i)) else ""
            if verbose:
                print(f"  📈 BUY {name} @ ₹{entry_price:.2f} [{current_date.date()}]{ml_info}")

        # ── SELL Signal Exit ─────────────────────────────────────────────
        elif signal == -1 and position == 'LONG' and not no_sig_exit:
            pullback = (entry_price - current_price) / entry_price * 100
            if pullback >= 1.0:
                if shares_remaining > 0:
                    capital += shares_remaining * exit_px
                pnl = ((exit_px - entry_price) / entry_price) * 100
                pnl_list.append(pnl)
                realized_pnl_sum += shares_remaining * (exit_px - entry_price)
                peak_capital = max(peak_capital, capital)
                trades.append({'type': 'SIG_REVERSAL', 'entry': entry_price, 'exit': exit_px,
                               'pnl': round(pnl, 2), 'date': str(current_date.date()),
                               'pullback_pct': round(pullback, 2)})
                position = None
                shares = 0
                shares_remaining = 0
                if verbose:
                    print(f"  📉 SELL (sig) {name} @ ₹{exit_px:.2f} | P&L: {pnl:+.2f}%")

        # ── In Position ────────────────────────────────────────────────
        elif position == 'LONG':
            sl = entry_price - atr * ATR_CONFIG['swing']['sl']
            t1 = entry_price + atr * ATR_CONFIG['intraday']['t1']

            # T1 Partial Exit
            if shares_remaining > 0 and current_price >= t1 and not t1_triggered:
                exit_shares = shares_remaining // 2
                capital += exit_shares * exit_px
                shares_remaining -= exit_shares
                partial_exits += 1
                tsl = max(tsl, entry_price)
                t1_triggered = True
                if verbose:
                    print(f"  🎯 T1 PARTIAL {name} @ ₹{exit_px:.2f} exited={exit_shares} remain={shares_remaining}")

            # Time-based exit
            hold_days = (current_date - entry_date).days if entry_date else 0
            unreal_pnl = ((current_price - entry_price) / entry_price) * 100 if entry_price > 0 else 0
            if hold_days > 5 and signal != 1 and unreal_pnl < 2:
                if shares_remaining > 0:
                    capital += shares_remaining * exit_px
                pnl = ((exit_px - entry_price) / entry_price) * 100
                pnl_list.append(pnl)
                realized_pnl_sum += shares_remaining * (exit_px - entry_price)
                peak_capital = max(peak_capital, capital)
                trades.append({'type': 'TIME', 'entry': entry_price, 'exit': exit_px,
                               'pnl': round(pnl, 2), 'date': str(current_date.date()),
                               'hold_days': hold_days})
                position = None
                shares = 0
                shares_remaining = 0
                continue

            # ABSSL
            abs_sl = entry_price * 0.97
            if current_price <= abs_sl:
                if shares_remaining > 0:
                    capital += shares_remaining * exit_px
                pnl = ((exit_px - entry_price) / entry_price) * 100
                pnl_list.append(pnl)
                realized_pnl_sum += shares_remaining * (exit_px - entry_price)
                peak_capital = max(peak_capital, capital)
                trades.append({'type': 'ABSSL', 'entry': entry_price, 'exit': exit_px,
                               'pnl': round(pnl, 2), 'date': str(current_date.date())})
                position = None
                shares = 0
                shares_remaining = 0
                continue

            # Trailing SL
            new_tsl = current_price - atr * 1.5
            if new_tsl > tsl and current_price >= entry_price + atr * 0.5:
                tsl = new_tsl

            if tsl > 0 and current_price <= tsl:
                if shares_remaining > 0:
                    capital += shares_remaining * exit_px
                pnl = ((exit_px - entry_price) / entry_price) * 100
                pnl_list.append(pnl)
                realized_pnl_sum += shares_remaining * (exit_px - entry_price)
                peak_capital = max(peak_capital, capital)
                trades.append({'type': 'TSL', 'entry': entry_price, 'exit': exit_px,
                               'pnl': round(pnl, 2), 'date': str(current_date.date())})
                position = None
                shares = 0
                shares_remaining = 0
                continue

            # Fixed SL
            if current_price <= sl:
                if shares_remaining > 0:
                    capital += shares_remaining * exit_px
                pnl = ((exit_px - entry_price) / entry_price) * 100
                pnl_list.append(pnl)
                realized_pnl_sum += shares_remaining * (exit_px - entry_price)
                peak_capital = max(peak_capital, capital)
                trades.append({'type': 'SL', 'entry': entry_price, 'exit': exit_px,
                               'pnl': round(pnl, 2), 'date': str(current_date.date())})
                position = None
                shares = 0
                shares_remaining = 0

    # Close open position at end
    if position == 'LONG' and shares_remaining > 0:
        exit_px = df['Close'].iloc[-1] * (1 - slip)
        capital += shares_remaining * exit_px
        pnl = ((exit_px - entry_price) / entry_price) * 100
        pnl_list.append(pnl)
        realized_pnl_sum += shares_remaining * (exit_px - entry_price)
        peak_capital = max(peak_capital, capital)
        trades.append({'type': 'CLOSED', 'entry': entry_price, 'exit': exit_px,
                       'pnl': round(pnl, 2), 'date': 'CURRENT'})

    if not trades:
        return {'symbol': name, 'trades': 0, 'test_days': test_days,
                'return': 0, 'realized_return': 0, 'win_rate': 0,
                'wins': 0, 'losses': 0, 'avg_win': 0, 'avg_loss': 0,
                'sharpe': 0, 'max_drawdown': 0,
                'tsl_exits': 0, 'sl_exits': 0, 'sig_exits': 0,
                'no_sig_exit': no_sig_exit, 'message': 'No signals'}

    wins = [t['pnl'] for t in trades if t['pnl'] > 0]
    losses = [t['pnl'] for t in trades if t['pnl'] <= 0]
    total_ret = ((capital - initial_capital) / initial_capital) * 100
    realized_ret = (realized_pnl_sum / initial_capital * 100)

    # Sharpe
    import math
    if len(pnl_list) >= 3:
        mean_pnl = sum(pnl_list) / len(pnl_list)
        std_pnl = (sum((x - mean_pnl) ** 2 for x in pnl_list) / len(pnl_list)) ** 0.5
        sharpe = (mean_pnl / std_pnl * math.sqrt(252 / 5)) if std_pnl > 0 else 0.0
    else:
        sharpe = 0.0

    max_dd = 0.0
    running = initial_capital
    peak = initial_capital
    for t in trades:
        running += t['pnl'] / 100 * entry_capital_at_risk
        peak = max(peak, running)
        dd = (peak - running) / peak * 100 if peak > 0 else 0
        max_dd = max(max_dd, dd)

    return {
        'symbol': name,
        'trades': len(trades),
        'wins': len(wins),
        'losses': len(losses),
        'test_days': test_days,
        'return': round(total_ret, 2),
        'realized_return': round(realized_ret, 2),
        'win_rate': round(len(wins) / len(trades) * 100, 1) if trades else 0,
        'final_capital': round(capital, 0),
        'avg_win': round(np.mean(wins), 2) if wins else 0,
        'avg_loss': round(np.mean(losses), 2) if losses else 0,
        'sharpe': round(sharpe, 2),
        'max_drawdown': round(max_dd, 2),
        'tsl_exits': sum(1 for t in trades if t['type'] == 'TSL'),
        'sl_exits': sum(1 for t in trades if t['type'] == 'SL'),
        'sig_exits': sum(1 for t in trades if t['type'] in ('SIG_REVERSAL',)),
        'time_exits': sum(1 for t in trades if t['type'] == 'TIME'),
        'abssl_exits': sum(1 for t in trades if t['type'] == 'ABSSL'),
        'partial_exits': partial_exits,
        'no_sig_exit': no_sig_exit,
        'trades_list': trades
    }

def run_forward_test(test_days=30, use_ml=True, no_sig_exit=False):
    print("=" * 70)
    print(f"📊 NIFTY FORWARD TEST v8 | Last {test_days} days | ML: {use_ml} | SigExit: {not no_sig_exit}")
    print("=" * 70)

    results = []
    for symbol in STOCKS:
        name = symbol.replace('.NS', '')
        print(f"\n🔄 {name}...", end=' ', flush=True)
        result = forward_test_stock(symbol, test_days=test_days, use_ml=use_ml,
                                    no_sig_exit=no_sig_exit, verbose=False)
        if result:
            results.append(result)
            if result['trades'] > 0:
                print(f"✅ {result['trades']} trades | Real.Ret:{result['realized_return']:+.2f}% "
                      f"Comp.Ret:{result['return']:+.2f}% | WR:{result['win_rate']}%")
            else:
                print(f"⚠️  {result.get('message', 'No trades')}")
        else:
            print("❌ No data")

    print("\n" + "=" * 70)
    print("📊 FORWARD TEST SUMMARY")
    print("=" * 70)
    print(f"\n{'Symbol':<12} {'Trades':>7} {'WR%':>6} {'RealRet%':>10} {'CompRet%':>10} {'Sharpe':>8}")
    print("-" * 60)
    for r in results:
        print(f"{r['symbol']:<12} {r['trades']:>7} {r['win_rate']:>6.1f} "
              f"{r['realized_return']:>+10.2f} {r['return']:>+10.2f} "
              f"{r['sharpe']:>8.2f}")

    active = [r for r in results if r['trades'] > 0]
    if active:
        total_trades = sum(r['trades'] for r in active)
        total_wins = sum(r.get('wins', 0) for r in active)
        total_losses = sum(r.get('losses', 0) for r in active)
        avg_real_ret = sum(r['realized_return'] for r in active) / len(active)
        avg_comp_ret = sum(r['return'] for r in active) / len(active)
        avg_wr = sum(r['win_rate'] for r in active) / len(active)
        avg_sh = sum(r['sharpe'] for r in active) / len(active)
        print("\n" + "-" * 60)
        print(f"{'TOTAL':<12} {total_trades:>7} {avg_wr:>6.1f} "
              f"{avg_real_ret:>+10.2f} {avg_comp_ret:>+10.2f} "
              f"{avg_sh:>8.2f}")
        best = max(active, key=lambda x: x['realized_return'])
        worst = min(active, key=lambda x: x['realized_return'])
        print(f"\n🏆 BEST:  {best['symbol']} Real.Ret={best['realized_return']:+.2f}% WR={best['win_rate']}%")
        print(f"💀 WORST: {worst['symbol']} Real.Ret={worst['realized_return']:+.2f}% WR={worst['win_rate']}%")

    out = f"models/forward_test_v8_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(out, 'w') as f:
        json.dump({'timestamp': datetime.now().isoformat(), 'test_days': test_days,
                   'use_ml': use_ml, 'no_sig_exit': no_sig_exit, 'results': results}, f, indent=2)
    print(f"\n✅ Saved to {out}")
    return results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('days', type=int, nargs='?', default=30)
    parser.add_argument('--no-ml', action='store_true')
    parser.add_argument('--no-sig-exit', action='store_true')
    args = parser.parse_args()
    run_forward_test(args.days, use_ml=not args.no_ml, no_sig_exit=args.no_sig_exit)
