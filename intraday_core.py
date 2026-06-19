#!/usr/bin/env python3
"""
NIFTY Intraday Core v34 — Same Day Entry + Exit System
======================================================

SIGNAL ENGINE:
- EMA(9) × EMA(21) crossover on 5-min candles
- Hourly RSI(14) filter — confirm direction
- Entry = today's 9:15 AM OPEN (NOT stale 5-min close)  ← v34 FIX
- Targets scale with TIME ZONE (morning=full, midday=reduced, afternoon=tight) ← v34 FIX
- SL = 1.5× hourly ATR (fixed)
- Square-off: 3:15 PM IST sharp

v34 ZONE TARGETS:
  🌅 Morning  (9:30–11AM): T1=3×hATR  T2=5×hATR  (6.5h left)
  ☀️ Midday   (11AM–1PM): T1=2×hATR  T2=3.5×hATR (4h left)
  🌆 Afternoon (1PM–3PM):  T1=1.5×hATR T2=2.5×hATR (2h left)

SQUARE-OFF: 3:15 PM IST — ALL positions closed immediately
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

# ── v34 ADAPTIVE TARGETS — scale with time of day ────────────────────────────
# Entry = today's 9:15 AM open (fixed, not live price)
# SL: always 1.5× hATR (fixed risk)
# T1/T2: shrinks as day progresses — less time = smaller targets
# Zone thresholds based on IST (UTC+5:30)
SL_MULT           = 1.5   # SL always 1.5× hATR
T1_MULT_MORNING   = 3.0   # 9:30–11:00 AM:  full targets
T2_MULT_MORNING   = 5.0
T1_MULT_MIDDAY    = 2.0   # 11:00 AM–1:00 PM: reduced
T2_MULT_MIDDAY    = 3.5
T1_MULT_AFTERNOON = 1.5  # 1:00 PM–3:00 PM:  tight
T2_MULT_AFTERNOON = 2.5

RSI_BUY_THRESHOLD  = 20
RSI_SELL_THRESHOLD = 80
RSI_H_BULL_THRESHOLD  = 50
RSI_H_BEAR_THRESHOLD  = 50

# ── Option C: Morning Window (9:30–11:00 AM IST) ───────────────────────────────
ENABLE_MORNING_WINDOW = True
MORNING_START_H, MORNING_START_M = 9, 30
MORNING_END_H,   MORNING_END_M   = 11, 0
SL_MULT_TIGHT  = 0.75   # Option C tight SL
T1_MULT_TIGHT = 1.5    # Option C target
T2_MULT_TIGHT = 2.5    # Option C second target

# ─── Helpers ─────────────────────────────────────────────────────────────────
def _ist_now():
    from datetime import datetime, timezone, timedelta
    IST = timezone(timedelta(hours=5, minutes=30))
    return datetime.now(IST)

def _zone_multipliers():
    """Return (t1_mult, t2_mult, zone_label) based on current IST time."""
    now_ist = _ist_now()
    now_mins = now_ist.hour * 60 + now_ist.minute
    if now_mins < 11 * 60:
        return T1_MULT_MORNING,   T2_MULT_MORNING,   '🌅 MORNING'
    elif now_mins < 13 * 60:
        return T1_MULT_MIDDAY,    T2_MULT_MIDDAY,    '☀️ MIDDAY'
    else:
        return T1_MULT_AFTERNOON, T2_MULT_AFTERNOON, '🌆 AFTERNOON'

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
    except:
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
    except:
        return None

# ─── EMA / RSI ─────────────────────────────────────────────────────────────
def calc_ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def calc_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / (loss + 1e-10)
    return 100 - (100 / (1 + rs))

# ─── Hourly ATR ─────────────────────────────────────────────────────────────
def get_hourly_atr_fast(symbol, days=5):
    try:
        tk = yf.Ticker(f"{symbol.upper()}.NS")
        df = tk.history(period=f"{days}d", interval='1h')
        if df.empty or len(df) < 14:
            return None
        h, l, c = df['High'], df['Low'], df['Close']
        tr = pd.concat([h - l, abs(h - c.shift(1)), abs(l - c.shift(1))], axis=1).max(axis=1)
        hatr = tr.rolling(14).mean().iloc[-1]
        cp = c.iloc[-1]
        if hatr <= 0 or hatr > cp:
            return None
        return {'hourly_atr': round(float(hatr), 2)}
    except:
        return None

# ─── Intraday Signal ─────────────────────────────────────────────────────────
def get_intraday_signal(df_5m, symbol=None,
                        use_morning_window=False, use_tight_sl=False):
    """
    Returns: (signal, conf_score, levels_dict)

    v34 key changes:
    - Entry = today's 9:15 AM open (from daily candle), NOT stale 5-min close
    - T1/T2 multipliers adapt to zone (morning/midday/afternoon)
    - per_hr = T1_distance / actual hours remaining

    levels dict keys: entry, entry_type, sl, t1, t2, zone, t1_mult, t2_mult,
                      hatr, hatr_pct, per_hr, remaining_h, hourly_rsi, rsi5,
                      ema9, ema21, open_price, open_high, open_low, signal_type
    """
    if df_5m is None or len(df_5m) < 60:
        return 0, 0, {}

    df = df_5m.copy()
    df['ema9']  = calc_ema(df['Close'], EMA_FAST)
    df['ema21'] = calc_ema(df['Close'], EMA_SLOW)
    df['rsi5']  = calc_rsi(df['Close'], RSI_PERIOD)

    last_idx = -1
    prev_idx = -2

    ema9_curr  = float(df['ema9'].iloc[last_idx])
    ema21_curr = float(df['ema21'].iloc[last_idx])
    ema9_prev  = float(df['ema9'].iloc[prev_idx])
    ema21_prev = float(df['ema21'].iloc[prev_idx])

    bull_cross = (ema9_prev <= ema21_prev) and (ema9_curr > ema21_curr)
    bear_cross = (ema9_prev >= ema21_prev) and (ema9_curr < ema21_curr)

    live_price = float(df['Close'].iloc[last_idx])
    rsi5 = float(df['rsi5'].iloc[last_idx])

    # ── Hourly ATR ────────────────────────────────────────────────────────
    h_data = get_hourly_atr_fast(symbol, days=5) if symbol else None
    hatr = h_data['hourly_atr'] if h_data else float(live_price * 0.01)

    # ── First 15-min open range ─────────────────────────────────────────
    first_15 = df.iloc[:3] if len(df) >= 3 else df
    open_high  = float(first_15['High'].max())
    open_low   = float(first_15['Low'].min())
    open_price = float(first_15['Close'].iloc[0])

    # ── v34: Get TODAY's 9:15 AM open from daily candle ─────────────────
    today_open = live_price
    try:
        tk_d = yf.Ticker(f"{symbol.upper()}.NS")
        df_d = tk_d.history(period='2d')
        if df_d is not None and len(df_d) >= 2:
            today_open = float(df_d['Open'].iloc[-1])   # today's 9:15 AM open
    except:
        pass

    # ── v34: Use today's open as entry, not live price ──────────────────
    entry_price = today_open

    # ── Hourly RSI ───────────────────────────────────────────────────────
    hourly_rsi = 50.0
    if symbol:
        df_h = get_hourly_candles(symbol, days=5)
        if df_h is not None and len(df_h) >= RSI_PERIOD:
            vals = calc_rsi(df_h['Close'], RSI_PERIOD).dropna()
            if len(vals) > 0:
                hourly_rsi = float(vals.iloc[-1])

    # ── Option C: Morning Window filter ───────────────────────────────────
    if use_morning_window and ENABLE_MORNING_WINDOW:
        now_ist = _ist_now()
        now_mins = now_ist.hour * 60 + now_ist.minute
        start_mins = MORNING_START_H * 60 + MORNING_START_M
        end_mins   = MORNING_END_H   * 60 + MORNING_END_M
        if not (start_mins <= now_mins <= end_mins):
            return 0, 0, {}

    # ── Zone-based multipliers (v34) ─────────────────────────────────────
    t1_mult, t2_mult, zone_label = _zone_multipliers()

    # Option C tight overrides
    if use_tight_sl:
        sl_mult, t1_mult, t2_mult = SL_MULT_TIGHT, T1_MULT_TIGHT, T2_MULT_TIGHT
    else:
        sl_mult = SL_MULT

    # ── Build levels dict helper ─────────────────────────────────────────
    def _mk(sig, sl, t1, t2):
        """Build full levels dict with per_hr, % distances, remaining time."""
        now_ist = _ist_now()
        now_mins = now_ist.hour * 60 + now_ist.minute
        rem_h = max(0.5, (15 * 60 + 15 - now_mins) / 60.0)
        closed = now_mins > 15 * 60 + 30  # past 3:30 PM IST
        per_h = 0 if closed else (round(abs(t1 - entry_price) / rem_h, 2) if rem_h > 0 else 0)
        sl_p  = round(abs(sl  - entry_price) / entry_price * 100, 2)
        t1_p  = round(abs(t1  - entry_price) / entry_price * 100, 2)
        t2_p  = round(abs(t2  - entry_price) / entry_price * 100, 2)
        hp    = round(hatr / entry_price * 100, 2)
        return {
            'entry':      round(entry_price, 2),
            'entry_type': '📌 TODAY_OPEN',
            'sl':         sl,
            't1':         t1,
            't2':         t2,
            'sl_pct':     sl_p,
            't1_pct':     t1_p,
            't2_pct':     t2_p,
            'zone':       zone_label,
            't1_mult':    t1_mult,
            't2_mult':    t2_mult,
            'hatr':       round(hatr, 2),
            'hatr_pct':   hp,
            'per_hr':     per_h,
            'remaining_h': 0.0 if closed else round(rem_h, 1),
            'hourly_rsi': round(hourly_rsi, 1),
            'rsi5':       round(rsi5, 1),
            'ema9':        round(ema9_curr, 2),
            'ema21':       round(ema21_curr, 2),
            'open_price': round(open_price, 2),
            'open_high':  round(open_high, 2),
            'open_low':   round(open_low, 2),
            'signal_type': '',
        }

    def _calc(sig):
        if sig == 1:
            return (round(entry_price - hatr * sl_mult, 2),
                    round(entry_price + hatr * t1_mult, 2),
                    round(entry_price + hatr * t2_mult, 2))
        else:
            return (round(entry_price + hatr * sl_mult, 2),
                    round(entry_price - hatr * t1_mult, 2),
                    round(entry_price - hatr * t2_mult, 2))

    # ── EMA Crossover signals (high quality) ─────────────────────────────
    if bull_cross:
        sl, t1, t2 = _calc(1)
        conf = 70 + (10 if hourly_rsi < RSI_H_BULL_THRESHOLD else 0) \
                   + (5 if entry_price >= open_price else 0)
        d = _mk(1, sl, t1, t2)
        d['signal_type'] = 'EMA_CROSS'
        return 1, min(conf, 95), d

    if bear_cross:
        sl, t1, t2 = _calc(-1)
        conf = 70 + (10 if hourly_rsi > RSI_H_BEAR_THRESHOLD else 0) \
                   + (5 if entry_price <= open_price else 0)
        d = _mk(-1, sl, t1, t2)
        d['signal_type'] = 'EMA_CROSS'
        return -1, min(conf, 95), d

    # ── RSI oversold/overbought signals ───────────────────────────────────
    if rsi5 < RSI_BUY_THRESHOLD and hourly_rsi < RSI_H_BULL_THRESHOLD:
        sl, t1, t2 = _calc(1)
        conf = 50 + (15 if rsi5 < 25 else 10) + (10 if hourly_rsi < 45 else 0) \
                   + (5 if entry_price >= open_price else 0)
        d = _mk(1, sl, t1, t2)
        d['signal_type'] = 'RSI_OVERSOLD'
        return 1, min(conf, 90), d

    if rsi5 > RSI_SELL_THRESHOLD and hourly_rsi > RSI_H_BEAR_THRESHOLD:
        sl, t1, t2 = _calc(-1)
        conf = 50 + (15 if rsi5 > 80 else 10) + (10 if hourly_rsi > 55 else 0) \
                   + (5 if entry_price <= open_price else 0)
        d = _mk(-1, sl, t1, t2)
        d['signal_type'] = 'RSI_OVERBOUGHT'
        return -1, min(conf, 90), d

    return 0, 0, {}

# ─── Intraday Scanner ────────────────────────────────────────────────────────
def scan_intraday(symbols, debug=False, allow_test=False,
                  use_morning_window=False, use_tight_sl=False):
    """
    v34: entry = today_open, targets = zone-adaptive.
    Returns list of signal dicts sorted by conf descending.
    """
    results = []
    now = pd.Timestamp.now()
    market_closed = (now.hour >= SQUARE_OFF_HOUR and now.minute >= SQUARE_OFF_MIN)

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

            # Live price only for display reference (not for entry)
            tk = yf.Ticker(f"{sym.upper()}.NS")
            live = tk.history(period='1d', interval='1m')
            live_price = float(live['Close'].iloc[-1]) if not live.empty else levels['entry']

            results.append({
                'symbol':      sym,
                'signal':      '📈 BUY' if signal == 1 else '📉 SELL',
                'signal_val':  signal,
                'conf':        conf,
                'live_price':  round(live_price, 2),
                'entry':       levels['entry'],
                'entry_type':  levels['entry_type'],
                'sl':          levels['sl'],
                't1':          levels['t1'],
                't2':          levels['t2'],
                'sl_pct':      levels['sl_pct'],
                't1_pct':      levels['t1_pct'],
                't2_pct':      levels['t2_pct'],
                'zone':        levels['zone'],
                'hatr':        levels['hatr'],
                'hatr_pct':    levels['hatr_pct'],
                'per_hr':      levels['per_hr'],
                'remaining_h': levels['remaining_h'],
                'hourly_rsi':  levels['hourly_rsi'],
                'rsi5':        levels['rsi5'],
                'ema9':        levels['ema9'],
                'ema21':       levels['ema21'],
                'signal_type': levels['signal_type'],
                'square_off':  '3:15 PM',
            })
        except Exception as e:
            if debug:
                print(f"  {sym}: error — {e}")
            continue

    results.sort(key=lambda x: -x['conf'])
    return results

# ─── Format Output ──────────────────────────────────────────────────────────
def format_intraday(results):
    """v34: Shows zone, today_open entry, per_hr, % targets, remaining hours."""
    if not results:
        return ("🕐 No intraday signals.\n"
                "💡 Signals: EMA cross OR (RSI<20+hRSI<50)=BUY | (RSI>80+hRSI>50)=SELL\n"
                "⛔ All positions auto-closed at 3:15 PM IST")

    now = pd.Timestamp.now()
    z = results[0]['zone']
    out = f"🕐 INTRADAY SCAN v34 | {now.strftime('%d %b %I:%M %p IST')}\n"
    out += f"📊 {len(results)} signals | {z} zone | ⛔ Square-off: 3:15 PM\n"
    out += "─" * 60 + "\n"

    buys  = [r for r in results if r['signal_val'] ==  1][:5]
    sells = [r for r in results if r['signal_val'] == -1][:5]

    if buys:
        out += f"📈 BUY [{len(buys)}]\n"
        for r in buys:
            out += (f"  {r['symbol']} {r['entry_type']} ₹{r['entry']} | Live:₹{r['live_price']}\n"
                    f"    SL:₹{r['sl']}({r['sl_pct']}%) | T1:₹{r['t1']}({r['t1_pct']}%) | "
                    f"T2:₹{r['t2']}({r['t2_pct']}%)\n"
                    f"    hATR:₹{r['hatr']}({r['hatr_pct']}%) | ~₹{r['per_hr']}/hr | "
                    f"{r['remaining_h']}h left\n"
                    f"    RSI5:{r['rsi5']} | hRSI:{r['hourly_rsi']} | EMA9:{r['ema9']} | "
                    f"EMA21:{r['ema21']} | CF:{r['conf']}%\n")

    if sells:
        out += f"\n📉 SELL [{len(sells)}]\n"
        for r in sells:
            out += (f"  {r['symbol']} {r['entry_type']} ₹{r['entry']} | Live:₹{r['live_price']}\n"
                    f"    SL:₹{r['sl']}({r['sl_pct']}%) | T1:₹{r['t1']}({r['t1_pct']}%) | "
                    f"T2:₹{r['t2']}({r['t2_pct']}%)\n"
                    f"    hATR:₹{r['hatr']}({r['hatr_pct']}%) | ~₹{r['per_hr']}/hr | "
                    f"{r['remaining_h']}h left\n"
                    f"    RSI5:{r['rsi5']} | hRSI:{r['hourly_rsi']} | EMA9:{r['ema9']} | "
                    f"EMA21:{r['ema21']} | CF:{r['conf']}%\n")

    out += "─" * 60 + "\n"
    out += ("⚠️  v34: Entry = today's 9:15 AM open (not live/stale price).\n"
            "🌅 MORNING(9:30-11AM): T1=3× hATR | T2=5× hATR\n"
            "☀️ MIDDAY(11AM-1PM):   T1=2× hATR | T2=3.5× hATR\n"
            "🌆 AFTERNOON(1-3PM):   T1=1.5× hATR | T2=2.5× hATR\n"
            "⛔ All positions closed at 3:15 PM IST. No overnight holds.\n"
            "📊 Option C (--tight): SL=0.75× hATR | T1=1.5× | T2=2.5×")
    return out

# ─── CLI ────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import sys
    from nifty_core import GOOD_STOCKS

    debug = '--debug' in sys.argv
    morning = '--morning' in sys.argv
    tight   = '--tight'   in sys.argv
    stocks  = [s for s in sys.argv[1:] if not s.startswith('--')] or GOOD_STOCKS[:10]

    results = scan_intraday(stocks, debug=debug, allow_test=True,
                            use_morning_window=morning, use_tight_sl=tight)
    print(format_intraday(results))
