#!/usr/bin/env python3
"""
NIFTY Intraday Core v36 DYNAMIC — Same Day Entry + Exit System
===============================================================

SIGNAL ENGINE (3 signal paths):
  1. EMA(9) × EMA(21) crossover — morning quality signal
  2. RSI oversold/overbought + hourly RSI confirm
  3. Price break of day high/low (AFTER 1PM) — afternoon path

Entry = today's 9:15 AM OPEN (NOT stale 5-min close)
Targets scale DYNAMICALLY with REMAINING TIME + VWAP DISTANCE  ← v36
  T1 = hATR × remaining_frac × 3.0 × vol_mult × zone_bias
  T2 = hATR × remaining_frac × 5.0 × vol_mult × zone_bias
  remaining_frac = rem_h / 5.75  (1.0 at open → ~0 at 3:15 PM)
  vol_mult: HIGH(vwap>1.5×ATR)=1.3 | NORMAL=1.0 | LOW=0.75
  zone_bias: 🌅BULL=1.25 | ☀️NEUTRAL=1.0 | 🌆BEAR=0.80

Multi-timeframe RSI boost: +10 conf if 15-min + hourly RSI both agree
SL = 1.5× hourly ATR (fixed)
Square-off: 3:15 PM IST sharp

Zone now controls BIAS only (bull/neutral/bear directional lean), NOT hardcoded multiples.
"""

import os
import yfinance as yf
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ─── Config ─────────────────────────────────────────────────────────────────
SQUARE_OFF_HOUR, SQUARE_OFF_MIN = 15, 15
OPEN_RANGE_MINUTES = 15
EMA_FAST, EMA_SLOW = 9, 21
RSI_PERIOD = 14
FULL_DAY_HOURS = 5.75   # 9:30 AM → 3:15 PM

# ── Zone bias multipliers (v36) ──────────────────────────────────────────────
# Zone now controls BIAS only — directional lean for bull/neutral/bear sessions.
# BUY zones get higher T1/T2 (uptrend persistence).
# SELL zones get tighter T1/T2 (downtrend reversal).
SL_MULT = 1.5   # SL always fixed hATR

ZONE_BIAS = {
    'bull':    1.25,   # morning bull trend — larger targets
    'neutral': 1.00,   # midday — standard
    'bear':    0.80,   # afternoon / bear — tighter
}

# Volatility mode multiplier — price distance from VWAP amplifies/shrinks targets
VOL_HIGH_MULT   = 1.30   # |price - vwap| > 1.5× hATR  → big move, extend targets
VOL_NORMAL_MULT = 1.00   # |price - vwap| > 0.75× hATR
VOL_LOW_MULT    = 0.75   # |price - vwap| < 0.75× hATR  → choppy / ranging

RSI_BUY_THRESHOLD  = 20
RSI_SELL_THRESHOLD = 80
RSI_H_BULL_THRESHOLD  = 50
RSI_H_BEAR_THRESHOLD  = 50
RSI_15M_BULL_THRESHOLD = 40   # 15-min RSI must be < 40 for BUY confirm
RSI_15M_BEAR_THRESHOLD = 60   # 15-min RSI must be > 60 for SELL confirm

# ── Option C: Morning Window ─────────────────────────────────────────────────
ENABLE_MORNING_WINDOW = True
MORNING_START_H, MORNING_START_M = 9, 30
MORNING_END_H,   MORNING_END_M   = 11, 0
SL_MULT_TIGHT, T1_MULT_TIGHT, T2_MULT_TIGHT = 0.75, 1.5, 2.5

# ─── Helpers ─────────────────────────────────────────────────────────────────
def _ist_now():
    from datetime import datetime, timezone, timedelta
    return datetime.now(timezone(timedelta(hours=5, minutes=30)))


def _zone_bias(now_mins=None):
    """Return (bias_mult, zone_label, zone_key) based on current IST time."""
    if now_mins is None:
        now_mins = _ist_now().hour * 60 + _ist_now().minute
    if now_mins < 11 * 60:
        return ZONE_BIAS['bull'], '🌅 MORNING',   'bull'
    elif now_mins < 13 * 60:
        return ZONE_BIAS['neutral'], '☀️ MIDDAY',  'neutral'
    else:
        return ZONE_BIAS['bear'], '🌆 AFTERNOON',  'bear'


def calc_vwap(df_day):
    """Compute VWAP from a day's 5-min candles. Requires Volume column."""
    if df_day is None or 'Volume' not in df_day.columns or df_day['Volume'].sum() == 0:
        return None
    tp = (df_day['High'] + df_day['Low'] + df_day['Close']) / 3.0
    cum_pv = (tp * df_day['Volume']).cumsum()
    cum_v  = df_day['Volume'].cumsum()
    vwap   = cum_pv / cum_v
    return float(vwap.iloc[-1])


def _dynamic_mults(hatr, price, vwap, zone_bias_mult, use_tight_sl=False):
    """
    v36 DYNAMIC T1/T2 formula:
      T1 = hATR × remaining_frac × 3.0 × vol_mult × zone_bias
      T2 = hATR × remaining_frac × 5.0 × vol_mult × zone_bias

    remaining_frac = rem_h / FULL_DAY_HOURS  →  1.0 at open, ~0 at 3:15 PM
    vol_mult       = based on |price - vwap| / hATR  →  adapts to vol regime
    zone_bias      = bull/neutral/bear directional lean
    """
    if use_tight_sl:
        return SL_MULT_TIGHT, T1_MULT_TIGHT, T2_MULT_TIGHT, 'tight'

    now_mins = _ist_now().hour * 60 + _ist_now().minute
    rem_h = max(0.5, (15 * 60 + 15 - now_mins) / 60.0)
    rem_frac = rem_h / FULL_DAY_HOURS   # 1.0 at open → ~0 at close

    # ── Volatility mode from VWAP distance ──────────────────────────────
    if vwap and vwap > 0 and hatr > 0:
        vwap_dist_atr = abs(price - vwap) / hatr   # ATR-normalised distance
    else:
        vwap_dist_atr = 1.0   # neutral when no VWAP

    if   vwap_dist_atr > 1.5: vol_key = 'HIGH';   vol_mult = VOL_HIGH_MULT
    elif vwap_dist_atr > 0.75: vol_key = 'NORMAL'; vol_mult = VOL_NORMAL_MULT
    else:                       vol_key = 'LOW';    vol_mult = VOL_LOW_MULT

    # ── Base multiples scaled by remaining time ─────────────────────────
    # At full day left:  base_t1 = 3.0×,  base_t2 = 5.0×
    # At 50% day left:   base_t1 = 1.5×,  base_t2 = 2.5×
    base_t1 = round(3.0 * rem_frac, 3)   # e.g. 3.0 at open, 1.3 at 2 PM, 0.5 at 3 PM
    base_t2 = round(5.0 * rem_frac, 3)

    t1_mult = round(base_t1 * vol_mult * zone_bias_mult, 3)
    t2_mult = round(base_t2 * vol_mult * zone_bias_mult, 3)

    # Clamp: don't let targets become absurd
    t1_mult = max(0.5,  min(t1_mult, 4.5))
    t2_mult = max(0.8,  min(t2_mult, 8.0))

    mode_tag = f"{vol_key}_rem{rem_frac:.1f}"
    return SL_MULT, t1_mult, t2_mult, mode_tag


# ─── Fetch candles ───────────────────────────────────────────────────────────
def get_5min_candles(symbol, days=2):
    try:
        tk = yf.Ticker(f"{symbol.upper()}.NS")
        df = tk.history(period=f"{days}d", interval='5m')
        if df.empty or len(df) < 20:
            return None
        df = df.copy().tz_localize(None)
        df = df[(df.index.hour >= 9) & (df.index.hour < 16)]
        return df
    except:
        return None


def get_15min_candles(symbol, days=2):
    """v35: 15-min candles for multi-timeframe confirmation."""
    try:
        tk = yf.Ticker(f"{symbol.upper()}.NS")
        df = tk.history(period=f"{days}d", interval='15m')
        if df.empty or len(df) < 10:
            return None
        return df.copy().tz_localize(None)
    except:
        return None


def get_hourly_candles(symbol, days=5):
    try:
        tk = yf.Ticker(f"{symbol.upper()}.NS")
        df = tk.history(period=f"{days}d", interval='1h')
        if df.empty or len(df) < 20:
            return None
        return df.copy().tz_localize(None)
    except:
        return None


# ─── Indicators ─────────────────────────────────────────────────────────────
def calc_ema(series, period):
    return series.ewm(span=period, adjust=False).mean()


def calc_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / (loss + 1e-10)
    return 100 - (100 / (1 + rs))


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


def _get_rsi15_and_day_levels(symbol, df_5m):
    """v35: Compute 15-min RSI + day high/low for afternoon breakout signals."""
    df_15 = get_15min_candles(symbol)
    if df_15 is None:
        rsi15 = 50.0
    else:
        rsi15 = float(calc_rsi(df_15['Close'], RSI_PERIOD).iloc[-1])

    today = pd.Timestamp('today').date()
    df_today = df_5m[df_5m.index.date == today]
    if len(df_today) < 3:
        df_today = df_5m
    day_high = float(df_today['High'].max())
    day_low  = float(df_today['Low'].min())
    return rsi15, day_high, day_low


# ─── Intraday Signal ─────────────────────────────────────────────────────────
def get_intraday_signal(df_5m, symbol=None,
                        use_morning_window=False, use_tight_sl=False):
    """v36 DYNAMIC: 3 signal paths + dynamic T1/T2 from remaining time + VWAP."""
    if df_5m is None or len(df_5m) < 60:
        return 0, 0, {}

    df = df_5m.copy()
    df['ema9']  = calc_ema(df['Close'], EMA_FAST)
    df['ema21'] = calc_ema(df['Close'], EMA_SLOW)
    df['rsi5']  = calc_rsi(df['Close'], RSI_PERIOD)

    ema9_curr  = float(df['ema9'].iloc[-1])
    ema21_curr = float(df['ema21'].iloc[-1])
    ema9_prev  = float(df['ema9'].iloc[-2])
    ema21_prev = float(df['ema21'].iloc[-2])
    bull_cross = (ema9_prev <= ema21_prev) and (ema9_curr > ema21_curr)
    bear_cross = (ema9_prev >= ema21_prev) and (ema9_curr < ema21_curr)

    live_price = float(df['Close'].iloc[-1])
    rsi5       = float(df['rsi5'].iloc[-1])

    # ── Hourly ATR ─────────────────────────────────────────────────────────
    h_data = get_hourly_atr_fast(symbol, days=5) if symbol else None
    hatr = h_data['hourly_atr'] if h_data else float(live_price * 0.01)

    # ── First 15-min open range ───────────────────────────────────────────
    first_15 = df.iloc[:3] if len(df) >= 3 else df
    open_high  = float(first_15['High'].max())
    open_low   = float(first_15['Low'].min())
    open_price = float(first_15['Close'].iloc[0])

    # ── Today's open from daily candle ───────────────────────────────────
    today_open = live_price
    try:
        df_d = yf.Ticker(f"{symbol.upper()}.NS").history(period='2d')
        if df_d is not None and len(df_d) >= 2:
            today_open = float(df_d['Open'].iloc[-1])
    except:
        pass
    entry_price = today_open

    # ── Hourly RSI ────────────────────────────────────────────────────────
    hourly_rsi = 50.0
    if symbol:
        df_h = get_hourly_candles(symbol, days=5)
        if df_h is not None and len(df_h) >= RSI_PERIOD:
            vals = calc_rsi(df_h['Close'], RSI_PERIOD).dropna()
            if len(vals) > 0:
                hourly_rsi = float(vals.iloc[-1])

    # ── v35: 15-min RSI + day high/low ───────────────────────────────────
    rsi15, day_high, day_low = _get_rsi15_and_day_levels(symbol, df_5m)

    # ── Morning window filter ─────────────────────────────────────────────
    if use_morning_window and ENABLE_MORNING_WINDOW:
        now_ist = _ist_now()
        now_mins = now_ist.hour * 60 + now_ist.minute
        start_mins = MORNING_START_H * 60 + MORNING_START_M
        end_mins   = MORNING_END_H   * 60 + MORNING_END_M
        if not (start_mins <= now_mins <= end_mins):
            return 0, 0, {}

    # ── v36: Zone bias + Dynamic multipliers ──────────────────────────────
    now_ist  = _ist_now()
    now_mins = now_ist.hour * 60 + now_ist.minute

    # Today's 5-min candles for VWAP
    today = pd.Timestamp('today').date()
    df_today_5m = df_5m[df_5m.index.date == today] if df_5m is not None else None
    vwap = calc_vwap(df_today_5m)

    zone_bias_mult, zone_label, zone_key = _zone_bias(now_mins)
    sl_mult, t1_mult, t2_mult, vol_mode = _dynamic_mults(
        hatr, live_price, vwap,
        zone_bias_mult=zone_bias_mult,
        use_tight_sl=use_tight_sl
    )

    # ── Levels builder ────────────────────────────────────────────────────
    def _mk(sig, sl, t1, t2):
        closed = now_mins > 15 * 60 + 30
        rem_h = max(0.5, (15 * 60 + 15 - now_mins) / 60.0)
        per_h = 0 if closed else (round(abs(t1 - entry_price) / rem_h, 2) if rem_h > 0 else 0)
        sl_p = round(abs(sl - entry_price) / entry_price * 100, 2)
        t1_p = round(abs(t1 - entry_price) / entry_price * 100, 2)
        t2_p = round(abs(t2 - entry_price) / entry_price * 100, 2)
        hp   = round(hatr / entry_price * 100, 2)
        vwap_dist_atr  = round(abs(live_price - vwap) / hatr, 2) if (vwap and hatr > 0) else None
        vwap_dist_pct  = round(abs(live_price - vwap) / live_price * 100, 2) if vwap else None
        return {
            'entry': entry_price, 'entry_type': '📌 TODAY_OPEN',
            'sl': sl, 't1': t1, 't2': t2,
            'sl_pct': sl_p, 't1_pct': t1_p, 't2_pct': t2_p,
            'zone': zone_label, 'zone_key': zone_key,     # bull / neutral / bear
            't1_mult': t1_mult, 't2_mult': t2_mult,         # dynamic multipliers
            'hatr': round(hatr, 2), 'hatr_pct': hp,
            'per_hr': per_h, 'remaining_h': 0.0 if closed else round(rem_h, 1),
            'hourly_rsi': round(hourly_rsi, 1), 'rsi5': round(rsi5, 1),
            'rsi15': round(rsi15, 1),
            'day_high': round(day_high, 2), 'day_low': round(day_low, 2),
            'vwap': round(vwap, 2) if vwap else None,
            'vwap_dist_atr': vwap_dist_atr,   # price-vwap in ATR units
            'vwap_dist_pct': vwap_dist_pct,   # price-vwap in %
            'vol_mode': vol_mode,              # HIGH_remX / NORMAL_remX / LOW_remX
            'ema9': round(ema9_curr, 2), 'ema21': round(ema21_curr, 2),
            'open_price': round(open_price, 2), 'signal_type': '',
            '_closed': closed,
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

    def _conf_boost(base_conf, sig):
        """v35: Multi-timeframe RSI boost — +10 if 15-min + hourly RSI agree."""
        boost = 0; reason = ''
        if sig == 1:   # BUY
            if rsi15 < RSI_15M_BULL_THRESHOLD and hourly_rsi < RSI_H_BULL_THRESHOLD:
                boost += 10; reason += '+rsi15+hRSI'
            elif rsi15 < RSI_15M_BULL_THRESHOLD:
                boost += 5;  reason += '+rsi15'
            elif hourly_rsi < RSI_H_BULL_THRESHOLD:
                boost += 5;  reason += '+hRSI'
        else:          # SELL
            if rsi15 > RSI_15M_BEAR_THRESHOLD and hourly_rsi > RSI_H_BEAR_THRESHOLD:
                boost += 10; reason += '+rsi15+hRSI'
            elif rsi15 > RSI_15M_BEAR_THRESHOLD:
                boost += 5;  reason += '+rsi15'
            elif hourly_rsi > RSI_H_BEAR_THRESHOLD:
                boost += 5;  reason += '+hRSI'
        return min(base_conf + boost, 95), reason

    # ── Signal Path 1: EMA Crossover ─────────────────────────────────────
    if bull_cross:
        sl, t1, t2 = _calc(1)
        conf, boost_reason = _conf_boost(70, 1)
        d = _mk(1, sl, t1, t2)
        d['signal_type'] = f'EMA_CROSS{(" ▲+"+boost_reason) if boost_reason else ""}'
        return 1, conf, d

    if bear_cross:
        sl, t1, t2 = _calc(-1)
        conf, boost_reason = _conf_boost(70, -1)
        d = _mk(-1, sl, t1, t2)
        d['signal_type'] = f'EMA_CROSS{(" ▼+"+boost_reason) if boost_reason else ""}'
        return -1, conf, d

    # ── Signal Path 2: RSI oversold/overbought ─────────────────────────────
    if rsi5 < RSI_BUY_THRESHOLD and hourly_rsi < RSI_H_BULL_THRESHOLD:
        sl, t1, t2 = _calc(1)
        conf, boost_reason = _conf_boost(50, 1)
        d = _mk(1, sl, t1, t2)
        d['signal_type'] = f'RSI_OVERSOLD{(" ▲+"+boost_reason) if boost_reason else ""}'
        return 1, conf, d

    if rsi5 > RSI_SELL_THRESHOLD and hourly_rsi > RSI_H_BEAR_THRESHOLD:
        sl, t1, t2 = _calc(-1)
        conf, boost_reason = _conf_boost(50, -1)
        d = _mk(-1, sl, t1, t2)
        d['signal_type'] = f'RSI_OVERBOUGHT{(" ▼+"+boost_reason) if boost_reason else ""}'
        return -1, conf, d

    # ── Signal Path 3: Afternoon breakout (v35) ────────────────────────────
    if now_mins >= 13 * 60:
        breakout_tolerance = hatr * 0.5
        if live_price >= day_high - breakout_tolerance and rsi15 < RSI_15M_BULL_THRESHOLD:
            sl, t1, t2 = _calc(1)
            conf, boost_reason = _conf_boost(60, 1)
            d = _mk(1, sl, t1, t2)
            d['signal_type'] = f'BREAK_HIGH{(" ▲+"+boost_reason) if boost_reason else ""}'
            return 1, conf, d
        if live_price <= day_low + breakout_tolerance and rsi15 > RSI_15M_BEAR_THRESHOLD:
            sl, t1, t2 = _calc(-1)
            conf, boost_reason = _conf_boost(60, -1)
            d = _mk(-1, sl, t1, t2)
            d['signal_type'] = f'BREAK_LOW{(" ▼+"+boost_reason) if boost_reason else ""}'
            return -1, conf, d

    return 0, 0, {}


# ─── Scanner ─────────────────────────────────────────────────────────────────
def scan_intraday(symbols, debug=False, allow_test=False,
                  use_morning_window=False, use_tight_sl=False):
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

            tk = yf.Ticker(f"{sym.upper()}.NS")
            live = tk.history(period='1d', interval='1m')
            live_price = float(live['Close'].iloc[-1]) if not live.empty else levels['entry']

            results.append({
                'symbol': sym,
                'signal': '📈 BUY' if signal == 1 else '📉 SELL',
                'signal_val': signal,
                'conf': conf,
                'live_price': round(live_price, 2),
                'entry': levels['entry'], 'entry_type': levels['entry_type'],
                'sl': levels['sl'], 't1': levels['t1'], 't2': levels['t2'],
                'sl_pct': levels['sl_pct'], 't1_pct': levels['t1_pct'], 't2_pct': levels['t2_pct'],
                'zone': levels['zone'], 'zone_key': levels.get('zone_key', levels['zone']),
                't1_mult': levels['t1_mult'], 't2_mult': levels['t2_mult'],
                'hatr': levels['hatr'], 'hatr_pct': levels['hatr_pct'],
                'vwap': levels.get('vwap'), 'vwap_dist_atr': levels.get('vwap_dist_atr'),
                'vwap_dist_pct': levels.get('vwap_dist_pct'),
                'vol_mode': levels.get('vol_mode', 'NORMAL'),
                'per_hr': levels['per_hr'], 'remaining_h': levels['remaining_h'],
                'hourly_rsi': levels['hourly_rsi'],
                'rsi5': levels['rsi5'],
                'rsi15': levels['rsi15'],
                'day_high': levels['day_high'], 'day_low': levels['day_low'],
                'ema9': levels['ema9'], 'ema21': levels['ema21'],
                'signal_type': levels['signal_type'],
                'square_off': '3:15 PM',
            })
        except Exception as e:
            if debug:
                print(f"  {sym}: error — {e}")
            continue

    results.sort(key=lambda x: -x['conf'])
    return results


# ─── Format Output ──────────────────────────────────────────────────────────
def format_intraday(results):
    """v36 DYNAMIC: Shows remaining_h, VWAP distance, vol_mode, dynamic T1/T2."""
    if not results:
        return ("🕐 No intraday signals.\n"
                "💡 Paths: EMA cross | RSI<20+hRSI<50=BUY | RSI>80+hRSI>50=SELL | "
                "BREAK_HIGH/LOW(1PM+)\n"
                "⛔ All positions closed at 3:15 PM IST.\n\n"
                "📊 v36 DYNAMIC T1/T2:\n"
                "  T1 = hATR × remaining_frac × 3.0 × vol_mult × zone_bias\n"
                "  remaining_frac = rem_h / 5.75  (1.0 at open → ~0 at 3:15 PM)\n"
                "  vol_mult: HIGH(>1.5×ATR from VWAP)=1.3 | NORMAL=1.0 | LOW=0.75\n"
                "  zone_bias: 🌅BULL=1.25 | ☀️NEUTRAL=1.0 | 🌆BEAR=0.80")

    now = pd.Timestamp.now()
    z   = results[0]['zone']
    vol = results[0].get('vol_mode', '?')
    out = (f"🕐 INTRADAY SCAN v36 DYNAMIC | {now.strftime('%d %b %I:%M %p IST')}\n"
           f"📊 {len(results)} signals | {z} zone [{vol}] | ⛔ 3:15 PM\n"
           f"{'─' * 60}\n")

    buys  = [r for r in results if r['signal_val'] ==  1][:5]
    sells = [r for r in results if r['signal_val'] == -1][:5]

    if buys:
        out += f"📈 BUY [{len(buys)}]\n"
        for r in buys:
            vw = f"VWAP:₹{r['vwap']}" if r.get('vwap') else "VWAP:N/A"
            vd = f"d₿:{r['vwap_dist_atr']}×ATR" if r.get('vwap_dist_atr') is not None else ""
            out += (f"  {r['symbol']} {r['entry_type']} ₹{r['entry']} | Live:₹{r['live_price']}\n"
                    f"    SL:₹{r['sl']}({r['sl_pct']}%) | "
                    f"T1:₹{r['t1']}({r['t1_pct']}%)(×{r['t1_mult']:.2f}) | "
                    f"T2:₹{r['t2']}({r['t2_pct']}%)(×{r['t2_mult']:.2f})\n"
                    f"    hATR:₹{r['hatr']}({r['hatr_pct']}%) | ~₹{r['per_hr']}/hr | "
                    f"{r['remaining_h']}h left | {vw} {vd}\n"
                    f"    RSI5:{r['rsi5']} | hRSI:{r['hourly_rsi']} | RSI15:{r['rsi15']} | "
                    f"DayH:₹{r['day_high']} | DayL:₹{r['day_low']}\n"
                    f"    EMA9:{r['ema9']} | EMA21:{r['ema21']} | CF:{r['conf']}% "
                    f"[{r['signal_type']}]\n")

    if sells:
        out += f"\n📉 SELL [{len(sells)}]\n"
        for r in sells:
            vw = f"VWAP:₹{r['vwap']}" if r.get('vwap') else "VWAP:N/A"
            vd = f"d₿:{r['vwap_dist_atr']}×ATR" if r.get('vwap_dist_atr') is not None else ""
            out += (f"  {r['symbol']} {r['entry_type']} ₹{r['entry']} | Live:₹{r['live_price']}\n"
                    f"    SL:₹{r['sl']}({r['sl_pct']}%) | "
                    f"T1:₹{r['t1']}({r['t1_pct']}%)(×{r['t1_mult']:.2f}) | "
                    f"T2:₹{r['t2']}({r['t2_pct']}%)(×{r['t2_mult']:.2f})\n"
                    f"    hATR:₹{r['hatr']}({r['hatr_pct']}%) | ~₹{r['per_hr']}/hr | "
                    f"{r['remaining_h']}h left | {vw} {vd}\n"
                    f"    RSI5:{r['rsi5']} | hRSI:{r['hourly_rsi']} | RSI15:{r['rsi15']} | "
                    f"DayH:₹{r['day_high']} | DayL:₹{r['day_low']}\n"
                    f"    EMA9:{r['ema9']} | EMA21:{r['ema21']} | CF:{r['conf']}% "
                    f"[{r['signal_type']}]\n")

    out += f"{'─' * 60}\n"
    out += ("⚠️  v36 DYNAMIC T1/T2:\n"
            "  T1 = hATR × remaining_frac × 3.0 × vol_mult × zone_bias\n"
            "  T2 = hATR × remaining_frac × 5.0 × vol_mult × zone_bias\n"
            "  remaining_frac = rem_h / 5.75  (1.0 at open → ~0 at 3:15 PM)\n"
            "  vol_mult: HIGH(>1.5×ATR from VWAP)=1.3 | NORMAL=1.0 | LOW=0.75\n"
            "  zone_bias: 🌅BULL=1.25 | ☀️NEUTRAL=1.0 | 🌆BEAR=0.80\n"
            "  Zone = directional lean only (not hardcoded multiples)\n"
            "⛔ All positions closed at 3:15 PM IST.")
    return out


# ─── Paper Trade Journal ────────────────────────────────────────────────────
JOURNAL_FILE = SCRIPT_DIR + '/journal.csv'


def journal_signal(r):
    """Write one signal to journal.csv (v36)."""
    import csv, datetime
    file_exists = os.path.exists(JOURNAL_FILE)
    with open(JOURNAL_FILE, 'a', newline='') as f:
        w = csv.writer(f)
        if not file_exists:
            w.writerow(['timestamp','symbol','signal','entry','sl','t1','t2',
                        't1_mult','t2_mult','zone','zone_key','vol_mode',
                        'vwap','vwap_dist_atr','per_hr','conf','signal_type',
                        'rsi5','rsi15','hourly_rsi','day_high','day_low','result','pnl_pct'])
        w.writerow([
            datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            r['symbol'], r['signal_val'], r['entry'], r['sl'], r['t1'], r['t2'],
            r['t1_mult'], r['t2_mult'], r['zone'], r.get('zone_key',''),
            r.get('vol_mode',''), r.get('vwap',''), r.get('vwap_dist_atr',''),
            r['per_hr'], r['conf'], r['signal_type'],
            r['rsi5'], r['rsi15'], r['hourly_rsi'],
            r['day_high'], r['day_low'], '', ''
        ])


def update_journal_result(symbol, result, pnl_pct):
    """Update journal entry with trade result."""
    try:
        rows = []
        with open(JOURNAL_FILE) as f:
            rows = list(csv.DictReader(f))
        for row in rows:
            if row['symbol'] == symbol and row['result'] == '':
                row['result']   = result
                row['pnl_pct']  = pnl_pct
        with open(JOURNAL_FILE, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader()
            w.writerows(rows)
    except:
        pass


# ─── CLI ────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import sys
    from nifty_core import GOOD_STOCKS

    debug   = '--debug'   in sys.argv
    morning = '--morning' in sys.argv
    tight   = '--tight'   in sys.argv
    stocks  = [s for s in sys.argv[1:] if not s.startswith('--')] or GOOD_STOCKS[:10]

    results = scan_intraday(stocks, debug=debug, allow_test=True,
                            use_morning_window=morning, use_tight_sl=tight)
    for r in results:
        journal_signal(r)
    print(format_intraday(results))
    if results:
        print(f"\n📝 {len(results)} signals logged to journal.csv")
