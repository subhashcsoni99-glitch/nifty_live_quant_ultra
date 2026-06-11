#!/usr/bin/env python3
"""
NIFTY Intraday Backtest v1 — Same Day Entry + Exit
===================================================

Tests EMA(9,21) crossover strategy on 5-min candles:
- Entry: EMA9 × EMA21 crossover confirmed by hourly RSI
- Exit: SL or T1 or 3:15 PM sharp square-off
- No overnight positions

Usage:
  python3 intraday_backtest.py                    # all GOOD_STOCKS, last 5 days
  python3 intraday_backtest.py SBIN,TCS          # specific stocks
  python3 intraday_backtest.py --days 10          # 10 days backtest
"""

import sys
import os
import warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import yfinance as yf
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# ─── Config ────────────────────────────────────────────────────────────────
SQUARE_OFF_HOUR = 15
SQUARE_OFF_MIN  = 15
OPEN_RANGE_MINUTES = 15
EMA_FAST = 9
EMA_SLOW = 21
RSI_PERIOD = 14
# Intraday levels: SL=1×ATR, T1=2×ATR, T2=4×ATR (RR=2:1, need 33% WR)
SL_MULT = 1.0   # 1× hourly ATR — tight stop
T1_MULT = 2.0   # 2× hourly ATR — primary target (2:1 RR)
T2_MULT = 4.0   # 4× hourly ATR — big move target (4:1 RR)
MIN_TRADES = 5   # minimum trades for qualification

# ─── Helpers ───────────────────────────────────────────────────────────────
def calc_ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def calc_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def get_15min_candles(symbol, days=5):
    """Fetch 15-min candles for last N days."""
    try:
        tk = yf.Ticker(f"{symbol.upper()}.NS")
        df = tk.history(period=f"{days}d", interval='15m')
        if df.empty or len(df) < 20:
            return None
        df = df.copy()
        if df.index.tz is not None:
            df = df.tz_localize(None)
        return df
    except:
        return None

def get_hourly_atr(symbol, days=5):
    """Compute hourly ATR(14) for a symbol."""
    try:
        tk = yf.Ticker(f"{symbol.upper()}.NS")
        df = tk.history(period=f"{days}d", interval='1h')
        if df.empty or len(df) < 15:
            return None
        high = df['High']; low = df['Low']; close = df['Close']
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        h_atr = tr.rolling(14).mean().iloc[-1]
        if h_atr <= 0 or h_atr > df['Close'].iloc[-1] * 0.05:
            return None
        return h_atr
    except:
        return None

# ─── Intraday Backtest Single Stock ──────────────────────────────────────
def backtest_intraday(symbol, days=5):
    """Backtest intraday EMA crossover for one stock.
    
    Returns dict with: trades count, win rate, avg return, qualified
    """
    df15 = get_15min_candles(symbol, days=days)
    if df15 is None or len(df15) < 40:
        return None

    df = df15.copy()
    df['ema9']  = calc_ema(df['Close'], EMA_FAST)
    df['ema21'] = calc_ema(df['Close'], EMA_SLOW)
    df['ema9_prev']  = df['ema9'].shift(1)
    df['ema21_prev'] = df['ema21'].shift(1)

    # Get hourly ATR (use last available)
    h_atr = get_hourly_atr(symbol, days=min(days, 10))
    if h_atr is None:
        h_atr = df['Close'].iloc[-1] * 0.01  # fallback 1%

    per_hr = round(h_atr * 0.75, 1)

    # Add hour/minute for filtering (15-min candles)
    df['hour'] = df.index.hour
    df['minute'] = df.index.minute
    df['date'] = df.index.date

    # Keep only 9:15-15:15 IST (15-min candles: 09:15, 09:30, ... 15:15)
    # 15:30 candle is close-only — square off before it opens
    market_mask = ((df['hour'] == 9) & (df['minute'] >= 15)) | \
                  (df['hour'].between(10, 14)) | \
                  ((df['hour'] == 15) & (df['minute'] == 0))
    df = df[market_mask]

    # Get hourly RSI (for trend confirmation)
    tk_h = yf.Ticker(f"{symbol.upper()}.NS")
    df_h = tk_h.history(period=f"{days}d", interval='1h')
    if df_h is not None and len(df_h) >= RSI_PERIOD:
        df_h = df_h.tz_localize(None) if df_h.index.tz else df_h
        df_h['rsi_h'] = calc_rsi(df_h['Close'], RSI_PERIOD)
        # Map hourly RSI to 5min timestamps
        df['rsi_h'] = df.index.map(lambda t: df_h['rsi_h'].asof(t) if t in df_h.index else 50)

    # ── Backtest Loop ────────────────────────────────────────────────────
    trades = []
    position = None
    entry_price = None
    entry_time = None
    entry_sl = None
    entry_t1 = None
    entry_t2 = None
    tsl = None   # v27: trailing SL activated after T1 hit

    for i in range(20, len(df)):
        row = df.iloc[i]
        price = row['Close']
        hour = row['hour']
        minute = row['minute']
        dt = df.index[i]

        # Skip outside market hours (9:15-15:00)
        if hour < 9 or (hour == 9 and minute < 15) or hour > 15:
            continue
            continue

        # ── Square-off check ───────────────────────────────────────────
            if position is not None:
                # Close at current price
                pnl = ((price - entry_price) / entry_price) * 100
                trades.append({'pnl': round(pnl, 2), 'type': 'SQUARE_OFF',
                               'entry': entry_price, 'exit': price,
                               'hour': f"{hour}:{minute:02d}"})
                position = None; tsl = None
            continue

        # ── EMA Crossover ─────────────────────────────────────────────
        ema9_c = row['ema9']
        ema21_c = row['ema21']
        ema9_p = row['ema9_prev']
        ema21_p = row['ema21_prev']
        bull_cross = (ema9_p <= ema21_p) and (ema9_c > ema21_c)
        bear_cross = (ema9_p >= ema21_p) and (ema9_c < ema21_c)

        rsi_h = row.get('rsi_h', 50)

        # ── Entry ─────────────────────────────────────────────────────
        if position is None:
            if bull_cross and rsi_h < 55:
                position = 'LONG'
                entry_price = price * 1.001  # slight slippage
                entry_time = dt
                entry_sl = round(entry_price - h_atr * SL_MULT, 2)
                entry_t1 = round(entry_price + h_atr * T1_MULT, 2)
                entry_t2 = round(entry_price + h_atr * T2_MULT, 2)
                tsl = None  # reset trailing SL on new entry
            elif bear_cross and rsi_h > 45:
                position = 'SHORT'
                entry_price = price * 0.999
                entry_time = dt
                entry_sl = round(entry_price + h_atr * SL_MULT, 2)
                entry_t1 = round(entry_price - h_atr * T1_MULT, 2)
                entry_t2 = round(entry_price - h_atr * T2_MULT, 2)

        # ── In Position ───────────────────────────────────────────────
        elif position == 'LONG':
            # Update trailing SL if active (after T1 hit)
            if tsl is not None:
                tsl = max(tsl, price - h_atr * 1.5)  # trail up, never down

            # Trailing SL hit (after T1)
            if tsl is not None and price <= tsl:
                pnl = ((tsl - entry_price) / entry_price) * 100
                trades.append({'pnl': round(pnl, 2), 'type': 'TSL',
                               'entry': entry_price, 'exit': tsl})
                position = None; tsl = None
            # SL hit
            elif price <= entry_sl:
                pnl = ((entry_sl - entry_price) / entry_price) * 100
                trades.append({'pnl': round(pnl, 2), 'type': 'SL',
                               'entry': entry_price, 'exit': entry_sl})
                position = None; tsl = None
            # T2 hit first (bigger target)
            elif price >= entry_t2:
                pnl = ((entry_t2 - entry_price) / entry_price) * 100
                trades.append({'pnl': round(pnl, 2), 'type': 'T2',
                               'entry': entry_price, 'exit': entry_t2})
                position = None; tsl = None
            # T1 hit — activate trailing SL
            elif price >= entry_t1:
                pnl = ((entry_t1 - entry_price) / entry_price) * 100
                trades.append({'pnl': round(pnl, 2), 'type': 'T1',
                               'entry': entry_price, 'exit': entry_t1})
                tsl = entry_t1  # lock T1 as new floor

        elif position == 'SHORT':
            # Update trailing SL if active
            if tsl is not None:
                tsl = min(tsl, price + h_atr * 1.5)  # trail down, never up

            # Trailing SL hit (after T1)
            if tsl is not None and price >= tsl:
                pnl = ((entry_price - tsl) / entry_price) * 100
                trades.append({'pnl': round(pnl, 2), 'type': 'TSL',
                               'entry': entry_price, 'exit': tsl})
                position = None; tsl = None
            # SL hit
            elif price >= entry_sl:
                pnl = ((entry_price - entry_sl) / entry_price) * 100
                trades.append({'pnl': round(pnl, 2), 'type': 'SL',
                               'entry': entry_price, 'exit': entry_sl})
                position = None; tsl = None
            # T2 hit
            elif price <= entry_t2:
                pnl = ((entry_price - entry_t2) / entry_price) * 100
                trades.append({'pnl': round(pnl, 2), 'type': 'T2',
                               'entry': entry_price, 'exit': entry_t2})
                position = None; tsl = None
            # T1 hit — activate trailing SL
            elif price <= entry_t1:
                pnl = ((entry_price - entry_t1) / entry_price) * 100
                trades.append({'pnl': round(pnl, 2), 'type': 'T1',
                               'entry': entry_price, 'exit': entry_t1})
                tsl = entry_t1  # lock T1 as new floor

    return {
        'symbol': symbol,
        'trades': len(trades),
        'wins': sum(1 for t in trades if t['pnl'] > 0),
        'losses': sum(1 for t in trades if t['pnl'] <= 0),
        'win_rate': round(sum(1 for t in trades if t['pnl'] > 0) / max(1, len(trades)) * 100, 1),
        'avg_return': round(sum(t['pnl'] for t in trades) / max(1, len(trades)), 2),
        'total_return': round(sum(t['pnl'] for t in trades), 2),
        'qualified': len(trades) >= MIN_TRADES,
        'sl_count': sum(1 for t in trades if t['type'] == 'SL'),
        't1_count': sum(1 for t in trades if t['type'] == 'T1'),
        't2_count': sum(1 for t in trades if t['type'] == 'T2'),
        'tsl_count': sum(1 for t in trades if t['type'] == 'TSL'),
        'sq_count': sum(1 for t in trades if t['type'] == 'SQUARE_OFF'),
        'h_atr': round(h_atr, 2),
        'per_hr': per_hr,
    }

# ─── Main ──────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    from nifty_core import GOOD_STOCKS

    days = 5
    stocks = GOOD_STOCKS

    if '--days' in sys.argv:
        idx = sys.argv.index('--days')
        try: days = int(sys.argv[idx+1])
        except: days = 5

    if len(sys.argv) > 1 and sys.argv[1] != '--days':
        stocks = [s.strip().upper() for s in sys.argv[1].split(',')]

    print(f"📊 INTRADAY BACKTEST | EMA({EMA_FAST},{EMA_SLOW}) | {days} days | {len(stocks)} stocks")
    print(f"   SL={SL_MULT}×ATR | T1={T1_MULT}×ATR | Square-off: 3:15 PM")
    print(f"   Entry: EMA crossover + hourly RSI confirm")
    print("=" * 70)

    results = []
    for sym in stocks:
        try:
            r = backtest_intraday(sym, days=days)
            if r and r['trades'] > 0:
                results.append(r)
                q = "✅" if r['qualified'] else "⚠️"
                print(f"  {q} {sym:12s} | trds={r['trades']:2d} WR={r['win_rate']:5.1f}% "
                      f"Avg={r['avg_return']:+6.2f}% Tot={r['total_return']:+7.2f}% "
                      f"SL={r['sl_count']} T1={r['t1_count']} SQ={r['sq_count']}")
            else:
                print(f"  ⚠️  {sym:12s} | no trades")
        except Exception as e:
            print(f"  ❌ {sym:12s} | error: {e}")

    if results:
        active = [r for r in results if r['trades'] >= 3]  # at least 3 trades
        print("=" * 70)
        print(f"AVG      | {sum(r['trades'] for r in active):3d} trds | "
              f"WR={sum(r['win_rate'] for r in active)/max(1,len(active)):5.1f}% | "
              f"AvgRet={sum(r['avg_return'] for r in active)/max(1,len(active)):+.2f}% | "
              f"TotRet={sum(r['total_return'] for r in active):+.2f}%")
        q_results = [r for r in active if r['qualified']]
        print(f"Qualified: {len(q_results)}/{len(active)}")
        best = max(active, key=lambda x: x['avg_return']) if active else None
        worst = min(active, key=lambda x: x['avg_return']) if active else None
        if best: print(f"🏆 BEST:  {best['symbol']} {best['avg_return']:+.2f}% WR={best['win_rate']}% ({best['trades']} trades)")
        if worst: print(f"💀 WORST: {worst['symbol']} {worst['avg_return']:+.2f}% WR={worst['win_rate']}% ({worst['trades']} trades)")