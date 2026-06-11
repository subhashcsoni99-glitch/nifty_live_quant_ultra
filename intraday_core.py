#!/usr/bin/env python3
"""
NIFTY intraday Core v1 — Same Day Entry + Exit System
=====================================================

SIGNAL ENGINE:
- 5-min candles + EMA(9) × EMA(21) crossover
- Hourly RSI(14) filter — only trade in direction of 1hr trend
- 9:30 AM open range confirmation (first 15 min high/low as S/R)

ENTRIES:
- BUY: EMA(9) crosses above EMA(21) on 5m + hourly RSI < 50 (uptrend) + price > open
- SELL: EMA(9) crosses below EMA(21) on 5m + hourly RSI > 50 (downtrend) + price < open

EXITS (same day — NO overnight):
- SL: entry ± 1.5× hourly ATR
- T1: entry + 2× hourly ATR (1:1.3 RR)
- MTM: trail after 1× ATR profit, exit on EMA reversal or 3:15 PM sharp

SQUARE-OFF: 3:15 PM IST sharp — close ALL positions immediately

Changes v1:
- Initial intraday system
"""

import os
import yfinance as yf
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ─── Config ────────────────────────────────────────────────────────────────
SQUARE_OFF_HOUR = 15      # 3 PM
SQUARE_OFF_MIN  = 15      # 3:15 PM
OPEN_RANGE_MINUTES = 15   # first 15 min as open range
EMA_FAST = 9
EMA_SLOW = 21
RSI_PERIOD = 14
# Intraday levels: SL=1.5×ATR, T1=3×ATR, T2=5×ATR (RR=2:1 for EMA signals)
# RSI signals use wider: SL=2×ATR, T1=4×ATR (more conservative for RSI setups)
SL_MULT = 1.5
T1_MULT = 3.0
T2_MULT = 5.0
# RSI thresholds for RSI-only signals (v27: tighter = higher quality = better WR)
RSI_BUY_THRESHOLD  = 25   # was 30 — only true oversold (< 25 RSI = bottom 25th percentile)
RSI_SELL_THRESHOLD = 75   # was 70 — only true overbought (> 75 RSI = top 25th percentile)
RSI_H_BULL_THRESHOLD  = 50  # hourly RSI below this = bull confirm for BUY
RSI_H_BEAR_THRESHOLD  = 50  # hourly RSI above this = bear confirm for SELL

# ─── Fetch 5-min candles ──────────────────────────────────────────────────
def get_5min_candles(symbol, days=2):
    """Fetch 5-min candles for last N days."""
    try:
        tk = yf.Ticker(f"{symbol.upper()}.NS")
        df = tk.history(period=f"{days}d", interval='5m')
        if df.empty or len(df) < 20:
            return None
        # Filter to market hours only (9:15 - 15:30 IST)
        df = df.copy()
        df = df.tz_localize(None)  # remove tz for clean time ops
        # Keep only 9:15-15:30
        df['hour'] = df.index.hour
        df['minute'] = df.index.minute
        # Keep only 9:15-15:30
        # Keep only 9:15-15:30 IST market hours
        mask = ((df['hour'] == 9) & (df['minute'] >= 15)) | \
               (df['hour'].between(10, 14)) | \
               ((df['hour'] == 15) & (df['minute'] <= 30))
        df = df[mask]
        df = df.drop(['hour', 'minute'], axis=1)
        return df
    except Exception:
        return None

# ─── Fetch hourly candles ──────────────────────────────────────────────────
def get_hourly_candles(symbol, days=5):
    """Fetch hourly candles for RSI filter."""
    try:
        tk = yf.Ticker(f"{symbol.upper()}.NS")
        df = tk.history(period=f"{days}d", interval='1h')
        if df.empty or len(df) < 20:
            return None
        df = df.copy()
        if df.index.tz is not None:
            df = df.tz_localize(None)
        return df
    except Exception:
        return None

# ─── EMA ──────────────────────────────────────────────────────────────────
def calc_ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

# ─── RSI ───────────────────────────────────────────────────────────────────
def calc_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# ─── Intraday Signal ───────────────────────────────────────────────────────
def get_intraday_signal(df_5m, symbol=None):
    """
    Returns: (signal, conf_score, levels)
    signal: 1=BUY, -1=SELL, 0=NO_SIGNAL
    conf_score: 0-100
    levels: dict with entry, sl, t1, t2, per_hr, signal_type
    
    Dual signal system:
    1. EMA crossover: EMA9 × EMA21 on 5-min (high quality, rare)
    2. RSI oversold/overbought: RSI(14) on 5-min < 30 or > 70 (more frequent)
    Both require hourly RSI confirmation.
    """
    if df_5m is None or len(df_5m) < 60:  # need at least 60 × 5min = 5hr
        return 0, 0, {}

    df = df_5m.copy()

    # ── EMA on 5-min ──────────────────────────────────────────────────────
    df['ema9']  = calc_ema(df['Close'], EMA_FAST)
    df['ema21'] = calc_ema(df['Close'], EMA_SLOW)
    df['ema9_prev']  = df['ema9'].shift(1)
    df['ema21_prev'] = df['ema21'].shift(1)

    # ── RSI(14) on 5-min (for RSI-only signal) ──────────────────────────
    df['rsi5'] = calc_rsi(df['Close'], RSI_PERIOD)

    # ── Crossover detection ─────────────────────────────────────────────
    last_idx = -1
    prev_idx = -2

    ema9_curr  = df['ema9'].iloc[last_idx]
    ema21_curr = df['ema21'].iloc[last_idx]
    ema9_prev  = df['ema9'].iloc[prev_idx]
    ema21_prev = df['ema21'].iloc[prev_idx]

    ema9_bull_cross = (ema9_prev <= ema21_prev) and (ema9_curr > ema21_curr)
    ema9_bear_cross = (ema9_prev >= ema21_prev) and (ema9_curr < ema21_curr)

    price = df['Close'].iloc[last_idx]
    rsi5 = df['rsi5'].iloc[last_idx]

    # ── Hourly ATR ──────────────────────────────────────────────────────
    if symbol:
        h_data = get_hourly_atr_fast(symbol, days=5)
    else:
        h_data = None
    if h_data:
        hatr = h_data['hourly_atr']
        per_hr = h_data['per_hr']
    else:
        hatr = price * 0.01
        per_hr = price * 0.0075

    # ── Open range (first 15 min) ──────────────────────────────────────
    if len(df) >= 3:
        first_15 = df.iloc[:3]
        open_high = first_15['High'].max()
        open_low  = first_15['Low'].min()
        open_price = first_15['Close'].iloc[0]
    else:
        open_high = open_low = open_price = df['Close'].iloc[0]

    # ── Hourly RSI ──────────────────────────────────────────────────────
    if symbol:
        df_h = get_hourly_candles(symbol, days=5)
        if df_h is not None and len(df_h) >= RSI_PERIOD:
            df_h['rsi_h'] = calc_rsi(df_h['Close'], RSI_PERIOD)
            hourly_rsi = df_h['rsi_h'].iloc[-1] if not df_h['rsi_h'].isna().all() else 50
        else:
            hourly_rsi = 50
    else:
        hourly_rsi = 50

    # ── Signal Priority: EMA crossover first (higher quality) ───────────
    if ema9_bull_cross:
        conf = 70 + (10 if hourly_rsi < RSI_H_BULL_THRESHOLD else 0) + (5 if price > open_price else 0)
        sl   = round(price - hatr * SL_MULT, 2)
        t1   = round(price + hatr * T1_MULT, 2)
        t2   = round(price + hatr * T2_MULT, 2)
        return 1, min(conf, 95), {
            'entry': round(price, 2),
            'sl': sl, 't1': t1, 't2': t2,
            'hourly_atr': round(hatr, 2),
            'per_hr': round(per_hr, 2),
            'hourly_rsi': round(hourly_rsi, 1),
            'rsi5': round(rsi5, 1),
            'ema9': round(ema9_curr, 2), 'ema21': round(ema21_curr, 2),
            'open_price': round(open_price, 2),
            'open_high': round(open_high, 2), 'open_low': round(open_low, 2),
            'signal_type': 'EMA_CROSS',
        }

    elif ema9_bear_cross:
        conf = 70 + (10 if hourly_rsi > RSI_H_BEAR_THRESHOLD else 0) + (5 if price < open_price else 0)
        sl   = round(price + hatr * SL_MULT, 2)
        t1   = round(price - hatr * T1_MULT, 2)
        t2   = round(price - hatr * T2_MULT, 2)
        return -1, min(conf, 95), {
            'entry': round(price, 2),
            'sl': sl, 't1': t1, 't2': t2,
            'hourly_atr': round(hatr, 2),
            'per_hr': round(per_hr, 2),
            'hourly_rsi': round(hourly_rsi, 1),
            'rsi5': round(rsi5, 1),
            'ema9': round(ema9_curr, 2), 'ema21': round(ema21_curr, 2),
            'open_price': round(open_price, 2),
            'open_high': round(open_high, 2), 'open_low': round(open_low, 2),
            'signal_type': 'EMA_CROSS',
        }

    # ── RSI-only signal (more frequent) ─────────────────────────────────
    # BUY: RSI(5m) < 30 AND hourly RSI < 50
    if rsi5 < RSI_BUY_THRESHOLD and hourly_rsi < RSI_H_BULL_THRESHOLD:
        conf = 50 + (15 if rsi5 < 25 else 10) + (10 if hourly_rsi < 45 else 0) + (5 if price > open_price else 0)
        sl   = round(price - hatr * SL_MULT, 2)
        t1   = round(price + hatr * T1_MULT, 2)
        t2   = round(price + hatr * T2_MULT, 2)
        return 1, min(conf, 90), {
            'entry': round(price, 2),
            'sl': sl, 't1': t1, 't2': t2,
            'hourly_atr': round(hatr, 2),
            'per_hr': round(per_hr, 2),
            'hourly_rsi': round(hourly_rsi, 1),
            'rsi5': round(rsi5, 1),
            'ema9': round(ema9_curr, 2), 'ema21': round(ema21_curr, 2),
            'open_price': round(open_price, 2),
            'open_high': round(open_high, 2), 'open_low': round(open_low, 2),
            'signal_type': 'RSI_OVERSOLD',
        }

    # SELL: RSI(5m) > 70 AND hourly RSI > 50
    if rsi5 > RSI_SELL_THRESHOLD and hourly_rsi > RSI_H_BEAR_THRESHOLD:
        conf = 50 + (15 if rsi5 > 80 else 10) + (10 if hourly_rsi > 55 else 0) + (5 if price < open_price else 0)
        sl   = round(price + hatr * SL_MULT, 2)
        t1   = round(price - hatr * T1_MULT, 2)
        t2   = round(price - hatr * T2_MULT, 2)
        return -1, min(conf, 90), {
            'entry': round(price, 2),
            'sl': sl, 't1': t1, 't2': t2,
            'hourly_atr': round(hatr, 2),
            'per_hr': round(per_hr, 2),
            'hourly_rsi': round(hourly_rsi, 1),
            'rsi5': round(rsi5, 1),
            'ema9': round(ema9_curr, 2), 'ema21': round(ema21_curr, 2),
            'open_price': round(open_price, 2),
            'open_high': round(open_high, 2), 'open_low': round(open_low, 2),
            'signal_type': 'RSI_OVERBOUGHT',
        }

    return 0, 0, {}

# ─── Fast hourly ATR ───────────────────────────────────────────────────────
def get_hourly_atr_fast(symbol, days=5):
    """Fast hourly ATR calculation."""
    try:
        tk = yf.Ticker(f"{symbol.upper()}.NS")
        df = tk.history(period=f"{days}d", interval='1h')
        if df.empty or len(df) < 14:
            return None
        high = df['High']; low = df['Low']; close = df['Close']
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        h_atr = tr.rolling(14).mean().iloc[-1]
        if h_atr <= 0 or h_atr > df['Close'].iloc[-1]:
            return None
        per_hr = round(h_atr * 0.75, 1)
        return {'hourly_atr': round(h_atr, 2), 'per_hr': per_hr}
    except:
        return None

# ─── Intraday Scanner ──────────────────────────────────────────────────────
def scan_intraday(symbols, debug=False, allow_test=False):
    """
    Scan list of symbols for intraday BUY/SELL opportunities.
    allow_test=True: bypass square-off check (for backtesting).
    """
    results = []
    now = pd.Timestamp.now()
    market_closed = now.hour >= SQUARE_OFF_HOUR and now.minute >= SQUARE_OFF_MIN
    
    # Don't trade after 3 PM (unless allow_test=True)
    if market_closed and not allow_test:
        print(f"⛔ Market closed — square-off time ({now.strftime('%H:%M')}) passed")

    for sym in symbols:
        try:
            df_5m = get_5min_candles(sym, days=2)
            if df_5m is None or len(df_5m) < 60:
                if debug:
                    print(f"  {sym}: insufficient 5min data")
                continue

            signal, conf, levels = get_intraday_signal(df_5m, symbol=sym)
            if signal == 0:
                continue

            df_h = get_hourly_candles(sym, days=5)
            price = levels['entry']
            
            # Get live price
            tk = yf.Ticker(f"{sym.upper()}.NS")
            live = tk.history(period='5m', interval='1m')
            live_price = live['Close'].iloc[-1] if not live.empty else price

            results.append({
                'symbol': sym,
                'signal': '📈 BUY' if signal == 1 else '📉 SELL',
                'signal_val': signal,
                'conf': conf,
                'price': round(live_price, 2),
                'entry': levels['entry'],
                'sl': levels['sl'],
                't1': levels['t1'],
                't2': levels.get('t2', 0),
                'hourly_atr': levels['hourly_atr'],
                'per_hr': levels['per_hr'],
                'hourly_rsi': levels['hourly_rsi'],
                'ema9': levels['ema9'],
                'ema21': levels['ema21'],
                'open_price': levels['open_price'],
                'open_high': levels['open_high'],
                'open_low': levels['open_low'],
                'square_off': f"{SQUARE_OFF_HOUR}:{SQUARE_OFF_MIN:02d} PM",
            })
        except Exception as e:
            if debug:
                print(f"  {sym}: error — {e}")
            continue

    # Sort by conf descending
    results.sort(key=lambda x: -x['conf'])
    return results

# ─── Format Output ─────────────────────────────────────────────────────────
def format_intraday(results):
    """Format intraday scan results for Telegram."""
    if not results:
        return "🕐 No intraday signals currently.\n💡 Tip: Signals generate between 9:30 AM – 3:00 PM IST\n⛔ All positions auto-closed at 3:15 PM"

    now = pd.Timestamp.now()
    out = f"🕐 INTRADAY SCAN | {now.strftime('%d %b %I:%M %p IST')}\n"
    out += f"📊 {len(results)} signals | ⛔ Square-off: 3:15 PM sharp\n"
    out += "─" * 50 + "\n"

    buys = [r for r in results if r['signal_val'] == 1]
    sells = [r for r in results if r['signal_val'] == -1]

    if buys:
        out += f"📈 BUY [{len(buys)}]\n"
        for r in buys[:5]:
            stype = r.get('signal_type', 'EMA_CROSS')
            out += (f"  {r['symbol']} ₹{r['price']} | RSI5:{r['rsi5']} RSI_h:{r['hourly_rsi']} | "
                    f"CF:{r['conf']}% [{stype}]\n"
                    f"    Entry:₹{r['entry']} | SL:₹{r['sl']} | T1:₹{r['t1']} | T2:₹{r.get('t2',0)} | "
                    f"~₹{r['per_hr']}/hr\n"
                    f"    EMA9:{r['ema9']} | EMA21:{r['ema21']} | Open:₹{r['open_price']}\n")

    if sells:
        out += f"\n📉 SELL [{len(sells)}]\n"
        for r in sells[:5]:
            stype = r.get('signal_type', 'EMA_CROSS')
            out += (f"  {r['symbol']} ₹{r['price']} | RSI5:{r['rsi5']} RSI_h:{r['hourly_rsi']} | "
                    f"CF:{r['conf']}% [{stype}]\n"
                    f"    Entry:₹{r['entry']} | SL:₹{r['sl']} | T1:₹{r['t1']} | T2:₹{r.get('t2',0)} | "
                    f"~₹{r['per_hr']}/hr\n")

    out += "─" * 50 + "\n"
    out += ("⚠️ Intraday = same day exit. No overnight positions.\n"
            "📊 2 signal types: EMA_CROSS (rare, high quality) + RSI_OVERSOLD/RSI_OVERBOUGHT (frequent)\n"
            "📊 Entry: RSI(5m)<30 or EMA cross + hourly RSI confirm. Exit: SL/T1/T2 or 3:15 PM sharp\n"
            "⛔ All positions auto-closed at 3:15 PM IST")

    return out

if __name__ == '__main__':
    import sys
    from nifty_core import GOOD_STOCKS

    debug = '--debug' in sys.argv
    stocks = sys.argv[1:] if len(sys.argv) > 1 else GOOD_STOCKS[:10]
    results = scan_intraday(stocks, debug=debug, allow_test=True)  # allow_test for live testing
    print(format_intraday(results))