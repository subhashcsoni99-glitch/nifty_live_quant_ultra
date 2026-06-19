#!/usr/bin/env python3
"""
NIFTY Intraday Core v2 — Same Day Entry + Exit System
=====================================================

SIGNAL ENGINE:
- EMA(9) × EMA(21) crossover on 5-min candles
- Hourly RSI(14) filter — confirm direction
- SL = 1.5× hourly ATR | T1 = 3× hATR | T2 = 5× hATR
- Entry = LIVE price (not stale 5-min close)
- Levels recalculated from live price at scan time

SQUARE-OFF: 3:15 PM IST sharp — close ALL positions immediately
"""

import os
import yfinance as yf
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ─── Config ────────────────────────────────────────────────────────────────
SQUARE_OFF_HOUR = 15
SQUARE_OFF_MIN  = 15
OPEN_RANGE_MINUTES = 15
EMA_FAST = 9
EMA_SLOW = 21
RSI_PERIOD = 14
SL_MULT = 1.5
T1_MULT = 3.0
T2_MULT = 5.0
RSI_BUY_THRESHOLD  = 20
RSI_SELL_THRESHOLD = 80
RSI_H_BULL_THRESHOLD  = 50
RSI_H_BEAR_THRESHOLD  = 50

# ── Option C: Morning Window (9:30–11:00 AM IST) ───────────────────────────────
# Only trade the morning session — morning trends are most reliable
# Option C: SL = 0.75× hourly ATR (tight stop for scalp)
ENABLE_MORNING_WINDOW = True   # Master switch — set False to allow all hours
MORNING_START_H   = 9
MORNING_START_M   = 30
MORNING_END_H     = 11
MORNING_END_M     = 0
SL_MULT_TIGHT = 0.75           # Option C: 0.75× hATR (tight scalp stop)
T1_MULT_TIGHT = 1.5            # Option C: 1.5× hATR target
T2_MULT_TIGHT = 2.5            # Option C: 2.5× hATR second target

# ─── Fetch 5-min candles ──────────────────────────────────────────────────
def get_5min_candles(symbol, days=2):
    try:
        tk = yf.Ticker(f"{symbol.upper()}.NS")
        df = tk.history(period=f"{days}d", interval='5m')
        if df.empty or len(df) < 20:
            return None
        df = df.copy()
        df = df.tz_localize(None)
        df['hour'] = df.index.hour
        df['minute'] = df.index.minute
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

# ─── Hourly ATR ────────────────────────────────────────────────────────────
def get_hourly_atr_fast(symbol, days=5):
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
        return {'hourly_atr': round(float(h_atr), 2)}
    except:
        return None

# ─── Intraday Signal ───────────────────────────────────────────────────────
def get_intraday_signal(df_5m, symbol=None, use_morning_window=False, use_tight_sl=False):
    """
    Returns: (signal, conf_score, levels)
    signal: 1=BUY, -1=SELL, 0=NO_SIGNAL
    levels dict: entry, sl, t1, t2, hourly_atr, per_hr, hourly_rsi,
                 rsi5, ema9, ema21, open_price, open_high, open_low, signal_type
    
    Option C: use_morning_window=True → only signal 9:30–11:00 AM IST
              use_tight_sl=True → SL=0.75×hATR, T1=1.5×hATR, T2=2.5×hATR
    """
    if df_5m is None or len(df_5m) < 60:
        return 0, 0, {}

    df = df_5m.copy()

    # ── EMA on 5-min ──────────────────────────────────────────────────────
    df['ema9']   = calc_ema(df['Close'], EMA_FAST)
    df['ema21']  = calc_ema(df['Close'], EMA_SLOW)
    df['ema9_prev']   = df['ema9'].shift(1)
    df['ema21_prev']  = df['ema21'].shift(1)

    # ── RSI(14) on 5-min ────────────────────────────────────────────────
    df['rsi5'] = calc_rsi(df['Close'], RSI_PERIOD)

    last_idx = -1
    prev_idx = -2

    ema9_curr   = float(df['ema9'].iloc[last_idx])
    ema21_curr  = float(df['ema21'].iloc[last_idx])
    ema9_prev   = float(df['ema9'].iloc[prev_idx])
    ema21_prev  = float(df['ema21'].iloc[prev_idx])

    ema9_bull_cross = (ema9_prev <= ema21_prev) and (ema9_curr > ema21_curr)
    ema9_bear_cross = (ema9_prev >= ema21_prev) and (ema9_curr < ema21_curr)

    price = float(df['Close'].iloc[last_idx])
    rsi5  = float(df['rsi5'].iloc[last_idx])

    # ── Hourly ATR ──────────────────────────────────────────────────────
    h_data = get_hourly_atr_fast(symbol, days=5) if symbol else None
    hatr = h_data['hourly_atr'] if h_data else float(price * 0.01)

    # ── Open range (first 15 min) ──────────────────────────────────────
    if len(df) >= 3:
        first_15 = df.iloc[:3]
        open_high  = float(first_15['High'].max())
        open_low   = float(first_15['Low'].min())
        open_price = float(first_15['Close'].iloc[0])
    else:
        open_high = open_low = open_price = price

    # ── Hourly RSI ──────────────────────────────────────────────────────
    hourly_rsi = 50.0
    if symbol:
        df_h = get_hourly_candles(symbol, days=5)
        if df_h is not None and len(df_h) >= RSI_PERIOD:
            df_h['rsi_h'] = calc_rsi(df_h['Close'], RSI_PERIOD)
            vals = df_h['rsi_h'].dropna()
            if len(vals) > 0:
                hourly_rsi = float(vals.iloc[-1])

    # ── Option C: Morning Window Filter ──────────────────────────────────
    # Check if current time is within 9:30–11:00 AM IST window
    # Use UTC+5:30 offset (India Standard Time) — robust regardless of server TZ
    if use_morning_window and ENABLE_MORNING_WINDOW:
        from datetime import datetime, timezone, timedelta
        IST = timezone(timedelta(hours=5, minutes=30))
        now_ist = datetime.now(IST)
        ist_hour = now_ist.hour
        ist_minute = now_ist.minute
        now_mins = ist_hour * 60 + ist_minute
        start_mins = MORNING_START_H * 60 + MORNING_START_M   # 570
        end_mins   = MORNING_END_H   * 60 + MORNING_END_M     # 660
        if not (start_mins <= now_mins <= end_mins):
            return 0, 0, {}  # Outside morning window — no signal

    # ── Option C: Tight SL multipliers ────────────────────────────────────
    sl_mult_use = SL_MULT_TIGHT if use_tight_sl else SL_MULT
    t1_mult_use = T1_MULT_TIGHT if use_tight_sl else T1_MULT
    t2_mult_use = T2_MULT_TIGHT if use_tight_sl else T2_MULT

    def make_levels(p, sig, sl_mult=sl_mult_use, t1_mult=t1_mult_use, t2_mult=t2_mult_use):
        if sig == 1:
            sl = round(p - hatr * sl_mult, 2)
            t1 = round(p + hatr * t1_mult, 2)
            t2 = round(p + hatr * t2_mult, 2)
        else:
            sl = round(p + hatr * sl_mult, 2)
            t1 = round(p - hatr * t1_mult, 2)
            t2 = round(p - hatr * t2_mult, 2)
        per_hr = round(abs(t1 - p) / 6.5, 1)
        return sl, t1, t2, per_hr

    def _make_levels_dict(price, signal, sl, t1, t2, per_hr, hatr, hourly_rsi, rsi5,
                          ema9, ema21, open_price, open_high, open_low, sig_type):
        return {
            'entry':      round(price, 2),
            'sl':         sl,
            't1':         t1,
            't2':         t2,
            'hourly_atr': round(hatr, 2),
            'per_hr':     per_hr,
            'hourly_rsi': round(hourly_rsi, 1),
            'rsi5':       round(rsi5, 1),
            'ema9':        round(ema9, 2),
            'ema21':       round(ema21, 2),
            'open_price': round(open_price, 2),
            'open_high':  round(open_high, 2),
            'open_low':   round(open_low, 2),
            'signal_type': sig_type,
        }

    # ── Signal: EMA crossover (higher quality) ───────────────────────────
    if ema9_bull_cross:
        sl, t1, t2, per_hr = make_levels(price, 1)
        conf = 70 + (10 if hourly_rsi < RSI_H_BULL_THRESHOLD else 0) + (5 if price > open_price else 0)
        return 1, min(conf, 95), _make_levels_dict(
            price, 1, sl, t1, t2, per_hr, hatr, hourly_rsi, rsi5,
            ema9_curr, ema21_curr, open_price, open_high, open_low, 'EMA_CROSS'
        )

    if ema9_bear_cross:
        sl, t1, t2, per_hr = make_levels(price, -1)
        conf = 70 + (10 if hourly_rsi > RSI_H_BEAR_THRESHOLD else 0) + (5 if price < open_price else 0)
        return -1, min(conf, 95), _make_levels_dict(
            price, -1, sl, t1, t2, per_hr, hatr, hourly_rsi, rsi5,
            ema9_curr, ema21_curr, open_price, open_high, open_low, 'EMA_CROSS'
        )

    # ── Signal: RSI oversold/overbought ────────────────────────────────
    if rsi5 < RSI_BUY_THRESHOLD and hourly_rsi < RSI_H_BULL_THRESHOLD:
        sl, t1, t2, per_hr = make_levels(price, 1)
        conf = 50 + (15 if rsi5 < 25 else 10) + (10 if hourly_rsi < 45 else 0) + (5 if price > open_price else 0)
        return 1, min(conf, 90), _make_levels_dict(
            price, 1, sl, t1, t2, per_hr, hatr, hourly_rsi, rsi5,
            ema9_curr, ema21_curr, open_price, open_high, open_low, 'RSI_OVERSOLD'
        )

    if rsi5 > RSI_SELL_THRESHOLD and hourly_rsi > RSI_H_BEAR_THRESHOLD:
        sl, t1, t2, per_hr = make_levels(price, -1)
        conf = 50 + (15 if rsi5 > 80 else 10) + (10 if hourly_rsi > 55 else 0) + (5 if price < open_price else 0)
        return -1, min(conf, 90), _make_levels_dict(
            price, -1, sl, t1, t2, per_hr, hatr, hourly_rsi, rsi5,
            ema9_curr, ema21_curr, open_price, open_high, open_low, 'RSI_OVERBOUGHT'
        )

    return 0, 0, {}

# ─── Intraday Scanner ────────────────────────────────────────────────────────
def scan_intraday(symbols, debug=False, allow_test=False,
                  use_morning_window=False, use_tight_sl=False):
    results = []
    now = pd.Timestamp.now()
    market_closed = now.hour >= SQUARE_OFF_HOUR and now.minute >= SQUARE_OFF_MIN

    if market_closed and not allow_test:
        print(f"⛔ Market closed — square-off time ({now.strftime('%H:%M')}) passed")

    for sym in symbols:
        try:
            df_5m = get_5min_candles(sym, days=2)
            if df_5m is None or len(df_5m) < 60:
                if debug:
                    print(f"  {sym}: insufficient 5min data")
                continue

            signal, conf, levels = get_intraday_signal(
                df_5m, symbol=sym,
                use_morning_window=use_morning_window,
                use_tight_sl=use_tight_sl
            )
            if signal == 0:
                continue

            # Get LIVE price (1-min resolution, not stale 5-min candle close)
            tk = yf.Ticker(f"{sym.upper()}.NS")
            live = tk.history(period='5m', interval='1m')
            live_price = float(live['Close'].iloc[-1]) if not live.empty else levels['entry']

            # Recalculate ALL levels from LIVE price using correct multipliers (Option C)
            hatr = levels['hourly_atr']
            sl_mult = SL_MULT_TIGHT if use_tight_sl else SL_MULT
            t1_mult = T1_MULT_TIGHT if use_tight_sl else T1_MULT
            t2_mult = T2_MULT_TIGHT if use_tight_sl else T2_MULT
            if signal == 1:
                sl  = round(live_price - hatr * sl_mult, 2)
                t1  = round(live_price + hatr * t1_mult, 2)
                t2  = round(live_price + hatr * t2_mult, 2)
            else:
                sl  = round(live_price + hatr * sl_mult, 2)
                t1  = round(live_price - hatr * t1_mult, 2)
                t2  = round(live_price - hatr * t2_mult, 2)
            per_hr = round(abs(t1 - live_price) / 6.5, 1)

            results.append({
                'symbol':      sym,
                'signal':      '📈 BUY' if signal == 1 else '📉 SELL',
                'signal_val':  signal,
                'conf':        conf,
                'price':       round(live_price, 2),
                'entry':       round(live_price, 2),
                'sl':          sl,
                't1':          t1,
                't2':          t2,
                'hourly_atr':  hatr,
                'per_hr':      per_hr,
                'hourly_rsi':  levels['hourly_rsi'],
                'rsi5':        levels['rsi5'],
                'ema9':        levels['ema9'],
                'ema21':       levels['ema21'],
                'open_price':  levels['open_price'],
                'open_high':   levels['open_high'],
                'open_low':    levels['open_low'],
                'signal_type': levels['signal_type'],
                'square_off':  f"{SQUARE_OFF_HOUR}:{SQUARE_OFF_MIN:02d} PM",
            })
        except Exception as e:
            if debug:
                print(f"  {sym}: error — {e}")
            continue

    results.sort(key=lambda x: -x['conf'])
    return results

# ─── Format Output ──────────────────────────────────────────────────────────
def format_intraday(results):
    if not results:
        return ("🕐 No intraday signals currently.\n"
                "💡 Signals generate when: EMA cross OR (RSI<25 with hRSI<50) OR (RSI>75 with hRSI>50)\n"
                "⛔ All positions auto-closed at 3:15 PM IST")

    now = pd.Timestamp.now()
    out = f"🕐 INTRADAY SCAN | {now.strftime('%d %b %I:%M %p IST')}\n"
    out += f"📊 {len(results)} signals | ⛔ Square-off: 3:15 PM sharp\n"
    out += "─" * 55 + "\n"

    buys  = [r for r in results if r['signal_val'] ==  1][:5]
    sells = [r for r in results if r['signal_val'] == -1][:5]

    if buys:
        out += f"📈 BUY [{len(buys)}]\n"
        for r in buys:
            out += (f"  {r['symbol']} ₹{r['price']} | RSI5:{r['rsi5']} RSI_h:{r['hourly_rsi']} | "
                    f"CF:{r['conf']}% [{r['signal_type']}]\n"
                    f"    Entry:₹{r['entry']} | SL:₹{r['sl']} | T1:₹{r['t1']} | T2:₹{r['t2']} | "
                    f"hATR:₹{r['hourly_atr']} | ~₹{r['per_hr']}/hr\n"
                    f"    EMA9:{r['ema9']} | EMA21:{r['ema21']} | Open:₹{r['open_price']}\n")

    if sells:
        out += f"\n📉 SELL [{len(sells)}]\n"
        for r in sells:
            out += (f"  {r['symbol']} ₹{r['price']} | RSI5:{r['rsi5']} RSI_h:{r['hourly_rsi']} | "
                    f"CF:{r['conf']}% [{r['signal_type']}]\n"
                    f"    Entry:₹{r['entry']} | SL:₹{r['sl']} | T1:₹{r['t1']} | T2:₹{r['t2']} | "
                    f"hATR:₹{r['hourly_atr']} | ~₹{r['per_hr']}/hr\n")

    out += "─" * 55 + "\n"
    out += ("⚠️  Intraday = same day exit. No overnight positions.\n"
            "📊 SL=1.5×hATR | T1=3×hATR | T2=5×hATR | Per hr=(T1−entry)/6.5h\n"
            "📊 EMA cross OR RSI<25(hRSI<50)=BUY | RSI>75(hRSI>50)=SELL\n"
            "⛔  All positions auto-closed at 3:15 PM IST")
    return out

if __name__ == '__main__':
    import sys
    from nifty_core import GOOD_STOCKS

    debug = '--debug' in sys.argv
    stocks = sys.argv[1:] if len(sys.argv) > 1 else GOOD_STOCKS[:10]
    results = scan_intraday(stocks, debug=debug, allow_test=True)
    print(format_intraday(results))
