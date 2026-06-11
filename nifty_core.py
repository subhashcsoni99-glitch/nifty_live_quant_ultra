#!/usr/bin/env python3
"""
NIFTY Live Quant Ultra - Core Module v3
Single source of truth for: OHLC fetching, feature engineering, signal logic,
ATR levels, 9-stage AI pipeline, S/R, fundamental scoring.
All scripts import from here. ONE SOURCE OF TRUTH.

Changes v3:
- RSI buy_strict: 45 → 38 (true oversold, not mildly bearish)
- RSI sell_relaxed: 35 → 40 (block sell below RSI 40 — deeply oversold bounce zone)
- Label definition note added to docstring
"""
import os
import yfinance as yf
import numpy as np
import pandas as pd
import urllib.request
import json
import csv
import warnings
warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = SCRIPT_DIR + '/models'

# ─── Stock Lists ─────────────────────────────────────────────────────────────
EXCLUDED_STOCKS = {'TRENT', 'NATIONALUM', 'PIIND', 'JIOFIN'}

NIFTY50_STOCKS = [
    'ADANIENT', 'ADANIPORTS', 'APOLLOHOSP', 'ASIANPAINT', 'AXISBANK',
    'BAJAJFINSV', 'BAJFINANCE', 'BHEL', 'BPCL', 'BRITANNIA',
    'CIPLA', 'COALINDIA', 'COFORGE', 'DIVISLAB', 'DRREDDY',
    'EICHERMOT', 'GRASIM', 'HAL', 'HCLTECH', 'HDFCBANK',
    'HDFCLIFE', 'HINDALCO', 'HINDUNILVR', 'ICICIBANK', 'INDUSINDBK',
    'INFY', 'ITC', 'JSWSTEEL', 'KOTAKBANK', 'LT', 'M&M',
    'MARUTI', 'NESTLEIND', 'NMDC', 'NTPC', 'ONGC',
    'PCBL', 'RELIANCE', 'SBILIFE', 'SBIN', 'SHREECEM',
    'SUNPHARMA', 'TATACONSUM', 'TATASTEEL', 'TCS', 'TECHM',
    'TITAN', 'TRENT', 'ULTRACEMCO', 'WIPRO',
]

# NIFTY100 extra stocks (NIFTY50 already covers 50; extras = the remaining ~50)
# Set arithmetic removes any NIFTY50 overlap; EXCLUDED_STOCKS also filtered
NIFTY100_EXTRA = sorted({
    'SBICARD', 'ADANIPORTS', 'ASIANPAINT', 'BPCL', 'CIPLA', 'COALINDIA',
    'HDFCLIFE', 'MARUTI', 'M&MFIN', 'NESTLEIND', 'SHREECEM', 'JSWSTEEL',
    'ACC', 'AMBUJACEM', 'APOLLOHOSP', 'BANKBARODA', 'BERGEPAINT',
    'BHARTIARTL', 'BOSCHLTD', 'BRITANNIA', 'CANBK', 'CHOLAFIN',
    'COLPAL', 'DIVISLAB', 'DRREDDY', 'EICHERMOT', 'GAIL',
    'GODREJCP', 'HONAUT', 'HSCL', 'IBULHSGFIN',
    'INDIGOPNTS', 'JUBLFOOD', 'LICHSGFIN', 'MOTHERSON', 'NMDC',
    'NTPC', 'ONGC', 'PFC', 'PIIND', 'POWERGRID',
    'RECLTD', 'SUNTV', 'TATACONSUM', 'TATASTEEL',
    'ETERNAL', 'JSWENERGY', 'MANKIND', 'SIEMENS',
} - set(NIFTY50_STOCKS) - EXCLUDED_STOCKS)

NIFTY100_STOCKS = NIFTY50_STOCKS + NIFTY100_EXTRA
GOOD_STOCKS = [s for s in NIFTY50_STOCKS if s not in EXCLUDED_STOCKS]
SCANNABLE_STOCKS = [s for s in NIFTY100_STOCKS if s not in EXCLUDED_STOCKS]
DEFAULT_STOCKS = GOOD_STOCKS

# ─── ATR Multipliers ─────────────────────────────────────────────────────────
ATR_CONFIG = {
    # INTRADAY: default trading view (daily candles)
    'intraday': {'sl': 3.0, 't1': 2.0, 't2': 3.5},
    # INTRADAY_TIGHT: hourly-based tight levels for scalp/intraday
    # T1=0.75×ATR (~0.5-0.7% in 1hr), T2=1.5×ATR (~1-1.5% in 2-3hr), SL=1.5×ATR
    'intraday_tight': {'sl': 1.5, 't1': 0.75, 't2': 1.5},
    # SWING: wider SL, wider targets for multi-day holds
    'swing':    {'sl': 2.0, 't1': 3.0, 't2': 6.0},
    'period': 14,
}

# ─── RSI Thresholds ──────────────────────────────────────────────────────────
# IMPORTANT: RSI thresholds are calibrated for MEAN-REVERSION setups.
# The rule-based signal fires when RSI is low (<38) — expect price to bounce.
# The ML model independently predicts 5-day momentum direction (>±2% threshold).
# These two signals measure different things: mean-reversion vs momentum.
RSI_CONFIG = {
    'period': 14,
    'buy_strict':  38,   # RSI < 38 = oversold (v27: was 35, relaxed to generate more signals)
    'buy_relaxed': 65,   # allow BUY up to RSI 65 in strong uptrend
    'sell_strict': 50,   # RSI > 50 = overbought (v27: was 55, relaxed to generate more signals)
    'sell_relaxed': 36,  # block SELL below 36 (deeply oversold bounce zone)
}

# ─── Signal Thresholds ──────────────────────────────────────────────────────
SIGNAL_CONFIG = {
    'min_confirmations': 2,  # was 3 — need 2 conditions instead of 3 for signal
    'volume_spike': 0.8,
    'vol_spike_strong': 1.3,
    'momentum_zero': 0,
}

# ─── Sector Mapping ─────────────────────────────────────────────────────────
SECTORS = {
    'Banking':      ['SBIN', 'SBICARD', 'HDFCBANK', 'INDUSINDBK', 'KOTAKBANK', 'AXISBANK', 'ICICIBANK', 'HDFCLIFE'],
    'Steel':        ['HINDALCO', 'JSWSTEEL', 'TATASTEEL'],
    'Diversified':  ['RELIANCE', 'BAJFINANCE', 'BAJAJFINSV', 'GRASIM'],
    'IT':           ['TCS', 'HCLTECH', 'WIPRO', 'TECHM', 'INFY', 'COFORGE'],
    'FMCG':         ['ITC', 'HINDUNILVR', 'NESTLEIND', 'BRITANNIA', 'TITAN', 'TATACONSUM', 'GODREJCP'],
    'CapitalGoods': ['BHEL', 'LT', 'SIEMENS', 'HAL'],
    'Materials':    ['ULTRACEMCO', 'SHREECEM', 'PCBL'],
    'Auto':         ['M&M', 'MARUTI', 'EICHERMOT', 'TATAMOTORS'],
    'Pharma':       ['SUNPHARMA', 'SBILIFE', 'CIPLA', 'DRREDDY', 'DIVISLAB', 'APOLLOHOSP', 'MANKIND', 'ETERNAL'],
    'Power':        ['NTPC', 'ONGC', 'COALINDIA', 'NMDC', 'JSWENERGY'],
    'Infrastructure': ['ADANIENT', 'ADANIPORTS'],
    'OilGas':       ['BPCL'],
    'Paints':       ['ASIANPAINT'],
    'Retail':       ['TRENT'],
}
MAX_PER_SECTOR = 2

# ─── Fundamental Cache ──────────────────────────────────────────────────────
FUNDAMENTAL_CSV = MODEL_DIR + '/nifty100_fundamental.csv'
FUNDAMENTAL_MIN_SCORE = 30
_FUNDAMENTAL_CACHE = None

def load_fundamentals():
    """Load cached fundamental data (cached in memory after first call)."""
    global _FUNDAMENTAL_CACHE
    if _FUNDAMENTAL_CACHE is not None:
        return _FUNDAMENTAL_CACHE
    if not os.path.exists(FUNDAMENTAL_CSV):
        _FUNDAMENTAL_CACHE = {}
        return {}
    with open(FUNDAMENTAL_CSV) as f:
        reader = csv.DictReader(f)
        _FUNDAMENTAL_CACHE = {row['symbol']: row for row in reader}
    return _FUNDAMENTAL_CACHE

def get_fundamental_score(symbol):
    """Get fundamental score for a symbol. Returns 0 if no data."""
    data = load_fundamentals()
    row = data.get(symbol)
    if not row:
        return 0
    try:
        return int(row.get('fundamental_score') or 0)
    except:
        return 0

def filter_by_fundamentals(results, min_score=FUNDAMENTAL_MIN_SCORE):
    """Add fundamental data to each result. Flag if below threshold."""
    data = load_fundamentals()
    for r in results:
        row = data.get(r['symbol'])
        if row:
            r['fundamental_score'] = int(row.get('fundamental_score') or 0)
            r['pe'] = row.get('pe', '-')
            r['mcap'] = row.get('mcap', '-')
            r['div_yield'] = row.get('dividend_yield', '-')
        else:
            r['fundamental_score'] = 0
            r['pe'] = '-'
            r['mcap'] = '-'
            r['div_yield'] = '-'
        r['fundamental_ok'] = r['fundamental_score'] >= min_score
    return results

NIFTY_REGIME_CACHE = {'regime': 'NEUTRAL', 'score': 0, 'fetched_at': None}

def get_market_regime():
    """Get cached market regime. Fetch once per session, reuse for all stocks."""
    global NIFTY_REGIME_CACHE
    now = pd.Timestamp.now()
    cache_age = (now - NIFTY_REGIME_CACHE['fetched_at']).total_seconds() if NIFTY_REGIME_CACHE['fetched_at'] else 999999
    if NIFTY_REGIME_CACHE['fetched_at'] is None or cache_age > 300:
        try:
            nifty = yf.Ticker("^NSEI").history(period="30d")
            nifty_ma20 = nifty['Close'].rolling(20).mean().iloc[-1]
            nifty_ma50 = nifty['Close'].rolling(50).mean().iloc[-1]
            nifty_price = nifty['Close'].iloc[-1]
            if nifty_price > nifty_ma20 and nifty_price > nifty_ma50:
                regime = "BULLISH"; score = 15
            elif nifty_price < nifty_ma20 and nifty_price < nifty_ma50:
                regime = "BEARISH"; score = -15
            else:
                regime = "NEUTRAL"; score = 0
            NIFTY_REGIME_CACHE = {'regime': regime, 'score': score, 'fetched_at': now}
        except:
            NIFTY_REGIME_CACHE = {'regime': 'NEUTRAL', 'score': 0, 'fetched_at': now}
    return NIFTY_REGIME_CACHE

# ─── Price Fetching ──────────────────────────────────────────────────────────
def get_price(sym):
    try:
        req = urllib.request.Request(
            f"https://www.nseindia.com/api/quote-equity?symbol={sym}",
            headers={'User-Agent': 'Mozilla/5.0'}
        )
        with urllib.request.urlopen(req, timeout=8) as r:
            p = json.loads(r.read()).get('priceInfo', {})
            price = p.get('lastPrice', 0)
            prev = p.get('previousClose', 0)
            if price > 0:
                return price, prev
    except:
        pass
    try:
        tk = yf.Ticker(f"{sym.upper()}.NS")
        hist = tk.history(period="5d")
        if len(hist) > 0:
            price = hist['Close'].iloc[-1]
            prev = hist['Close'].iloc[-2] if len(hist) > 1 else price
            return price, prev
    except:
        pass
    return None, None

def get_ohlc(sym, days=365):
    try:
        return yf.Ticker(f"{sym.upper()}.NS").history(period=f"{days}d")
    except:
        return None

# ─── Feature Engineering ────────────────────────────────────────────────────
def add_features(df):
    df = df.copy()
    for w in [5, 10, 20, 50, 100, 200]:
        df[f'ma{w}'] = df['Close'].rolling(w).mean()
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['rsi'] = 100 - (100 / (1 + gain / (loss + 1e-10)))
    ema12 = df['Close'].ewm(span=12).mean()
    ema26 = df['Close'].ewm(span=26).mean()
    df['macd'] = ema12 - ema26
    df['macd_sig'] = df['macd'].ewm(span=9).mean()
    tr = pd.concat([
        df['High'] - df['Low'],
        abs(df['High'] - df['Close'].shift()),
        abs(df['Low'] - df['Close'].shift())
    ], axis=1).max(axis=1)
    df['atr'] = tr.rolling(14).mean()
    df['vol_ma'] = df['Volume'].rolling(20).mean()
    df['vol_ratio'] = df['Volume'] / (df['vol_ma'] + 1)
    df['ret5'] = df['Close'].pct_change(5)
    return df

# ─── Support / Resistance ──────────────────────────────────────────────────
def calc_support_resistance(df, lookback=20):
    """Calculate lookback-bar S/R levels. Fixed 20-bar lookback."""
    low = df['Low'].iloc[-lookback:].min()
    high = df['High'].iloc[-lookback:].max()
    return {'support': round(low, 2), 'resistance': round(high, 2)}

# ─── Divergence Detection ────────────────────────────────────────────────────
def detect_divergence(df):
    rsi = df['rsi'].values
    price = df['Close'].values
    if len(rsi) < 50:
        return None
    lookback = 20
    recent_rsi = rsi[-lookback:]
    recent_price = price[-lookback:]
    rsi_trough = np.min(recent_rsi)
    price_trough_idx = np.argmin(recent_price)
    price_peak_idx = np.argmax(recent_price)
    if price_trough_idx > 5:
        prev_price_low = np.min(recent_price[:price_trough_idx])
        prev_rsi_low = np.min(recent_rsi[:price_trough_idx])
        if recent_price[price_trough_idx] < prev_price_low and rsi_trough > prev_rsi_low:
            return "BULLISH"
    if price_peak_idx > 5:
        prev_price_high = np.max(recent_price[:price_peak_idx])
        prev_rsi_high = np.max(recent_rsi[:price_peak_idx])
        if recent_price[price_peak_idx] > prev_price_high and recent_rsi[price_peak_idx] < prev_rsi_high:
            return "BEARISH"
    return None

# ─── ATR-based Levels ────────────────────────────────────────────────────────
def calc_levels(price, atr, mode='intraday'):
    mults = ATR_CONFIG.get(mode, ATR_CONFIG['intraday'])
    return {
        'sl': round(price - atr * mults['sl'], 2),
        't1': round(price + atr * mults['t1'], 2),
        't2': round(price + atr * mults['t2'], 2),
    }

def calc_levels_hourly(price, hourly_atr, sig='BUY'):
    """Hourly scalp levels: SL=1×ATR, T1=0.5×ATR, T2=1×ATR — tight daily targets."""
    if sig == 'SELL':
        return {
            'sl': round(price + hourly_atr * 1.0, 0),
            't1': round(price - hourly_atr * 0.5, 0),
            't2': round(price - hourly_atr * 1.0, 0),
        }
    else:
        return {
            'sl': round(price - hourly_atr * 1.0, 0),
            't1': round(price + hourly_atr * 0.5, 0),
            't2': round(price + hourly_atr * 1.0, 0),
        }

def get_hourly_atr_and_pivot(symbol, price):
    """Fetch last 20 hourly candles, compute hourly ATR(14) and pivot-based levels."""
    try:
        tk = yf.Ticker(f"{symbol.upper()}.NS")
        df_h = tk.history(period='3d', interval='1h')
        if df_h.empty or len(df_h) < 10:
            return None
        high = df_h['High']; low = df_h['Low']; close = df_h['Close']
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        h_atr = tr.rolling(14).mean().iloc[-1]
        if h_atr <= 0 or h_atr > price:
            return None
        last_high = high.iloc[-1]; last_low = low.iloc[-1]
        pivot = (last_high + last_low + close.iloc[-1]) / 3
        r1 = 2 * pivot - last_low
        s1 = 2 * pivot - last_high
        r2 = pivot + (last_high - last_low)
        s2 = pivot - (last_high - last_low)
        per_hr = round(h_atr * 0.75, 1)
        return {
            'hourly_atr': round(h_atr, 2),
            'pivot': round(pivot, 2),
            'r1': round(r1, 2), 's1': round(s1, 2),
            'r2': round(r2, 2), 's2': round(s2, 2),
            'per_hr': per_hr,
            'pivot_dist_pct': round((price - pivot) / pivot * 100, 2),
        }
    except:
        return None

# ─── Core Signal Engine ──────────────────────────────────────────────────────
def get_signal(df, i):
    """
    Returns (signal_val, meta_dict, []).
    signal_val: 1=BUY, -1=SELL, 0=RANGE
    Signal logic: MEAN-REVERSION + SHORT-TERM MOMENTUM.
    BUY: RSI < 35 (oversold) + MA5>MA20 OR ret5>0 → +1 momentum bonus
    SELL: RSI > 55 (overbought) + MA5<MA20 AND ret5<0 → +1 momentum bonus
    """
    if i < 200:
        return 0, {'signal': 'RANGE', 'buy_cnt': 0, 'sell_cnt': 0, 'divergence': None, 'reasons': []}, []

    row = df.iloc[i]
    pv = row['Close']
    ma5  = row['ma5']
    ma20 = row['ma20']
    ma50 = row['ma50']
    ma200 = row['ma200']
    rsi = row['rsi']
    macd = row['macd']
    macd_sig = row['macd_sig']
    vol_ratio = row['vol_ratio']
    ret5 = row['ret5']

    # Momentum (v24: short-term trend filter — don't buy falling knives)
    c_ma5_above_ma20 = ma5 > ma20 if not (pd.isna(ma5) or pd.isna(ma20)) else False
    c_ret5_positive  = ret5 > 0
    c_short_term_bull = c_ma5_above_ma20 or c_ret5_positive
    c_short_term_bear = not c_short_term_bull

    c_price_ma20 = pv > ma20
    c_price_ma50 = pv > ma50
    c_ma50_ma200 = ma50 > ma200
    c_rsi_buy = rsi < RSI_CONFIG['buy_strict']   # 35: only true oversold
    c_rsi_sell = rsi > RSI_CONFIG['sell_strict']  # 55: overbought zone
    c_macd = macd > macd_sig
    c_vol = vol_ratio > SIGNAL_CONFIG['volume_spike']
    c_mom = ret5 > SIGNAL_CONFIG['momentum_zero']
    # v24: short-term momentum for BUY (don't buy falling knives — require uptick)
    c_ma5_above_ma20 = (not pd.isna(ma5) and not pd.isna(ma20) and ma5 > ma20)
    c_ret5_positive = ret5 > 0

    buy_cnt = sum([c_price_ma20, c_price_ma50, c_ma50_ma200, c_rsi_buy, c_macd, c_vol, c_mom])
    # Bonus: if RSI < 35 AND short-term uptick (MA5>MA20 OR ret5>0) → extra +1
    if c_rsi_buy and (c_ma5_above_ma20 or c_ret5_positive):
        buy_cnt += 1

    sell_cnt = sum([not c_price_ma20, not c_price_ma50, not c_ma50_ma200, c_rsi_sell, not c_macd, c_vol, not c_mom])
    # Bonus: if RSI > 55 AND price below MA20 → extra +1
    if c_rsi_sell and not c_price_ma20:
        sell_cnt += 1

    divergence = detect_divergence(df.iloc[:i+1])
    if divergence == "BULLISH":
        buy_cnt += 2

    # RSI Guards
    # Block SELL when RSI < 38 (deeply oversold — don't short bounce)
    if rsi < RSI_CONFIG['sell_relaxed']:
        sell_cnt = 0
    # RSI 35-38 with price > MA20: oversold bounce zone — don't sell into recovery
    elif rsi < RSI_CONFIG['buy_strict'] and c_price_ma20:
        sell_cnt = 0

    reasons = build_reasons(c_price_ma20, c_price_ma50, c_ma50_ma200, c_rsi_buy, c_rsi_sell,
                            c_macd, c_vol, c_mom, divergence, 'BUY')
    reasons_sell = build_reasons(c_price_ma20, c_price_ma50, c_ma50_ma200, c_rsi_buy, c_rsi_sell,
                                 c_macd, c_vol, c_mom, divergence, 'SELL')

    if buy_cnt >= SIGNAL_CONFIG['min_confirmations']:
        return 1, {'signal': 'BUY', 'buy_cnt': buy_cnt, 'sell_cnt': sell_cnt,
                   'divergence': divergence, 'reasons': reasons}, []
    elif sell_cnt >= SIGNAL_CONFIG['min_confirmations']:
        return -1, {'signal': 'SELL', 'buy_cnt': buy_cnt, 'sell_cnt': sell_cnt,
                    'divergence': divergence, 'reasons': reasons_sell}, []
    else:
        return 0, {'signal': 'RANGE', 'buy_cnt': buy_cnt, 'sell_cnt': sell_cnt,
                   'divergence': divergence, 'reasons': []}, []

def build_reasons(c_price_ma20, c_price_ma50, c_ma50_ma200, c_rsi_buy, c_rsi_sell,
                  c_macd, c_vol, c_mom, divergence, signal):
    reasons = []
    if signal == 'BUY':
        if c_price_ma20: reasons.append("Price > MA20")
        else: reasons.append("Price < MA20")
        if c_price_ma50: reasons.append("Price > MA50")
        if c_ma50_ma200: reasons.append("MA50 > MA200")
        if c_rsi_buy: reasons.append(f"RSI < {RSI_CONFIG['buy_strict']} (oversold)")
        else: reasons.append(f"RSI > {RSI_CONFIG['buy_strict']}")
        if c_macd: reasons.append("MACD > Signal")
        else: reasons.append("MACD < Signal")
        if c_vol: reasons.append("Vol above avg")
        if c_mom: reasons.append("+ve Momentum")
        else: reasons.append("-ve Momentum")
    else:
        if not c_price_ma20: reasons.append("Price < MA20")
        else: reasons.append("Price > MA20")
        if not c_price_ma50: reasons.append("Price < MA50")
        if not c_ma50_ma200: reasons.append("MA50 < MA200")
        if c_rsi_sell: reasons.append(f"RSI > {RSI_CONFIG['sell_strict']} (overbought)")
        elif c_rsi_buy: reasons.append(f"RSI < {RSI_CONFIG['buy_strict']} (oversold)")
        if not c_macd: reasons.append("MACD < Signal")
        if c_vol: reasons.append("Vol above avg")
        if not c_mom: reasons.append("-ve Momentum")
    if divergence == "BULLISH":
        reasons.append("🟢 RSI BULLISH DIVERGENCE")
    elif divergence == "BEARISH":
        reasons.append("🔴 RSI BEARISH DIVERGENCE ⚠️")
    return reasons

# ─── Position Sizing ─────────────────────────────────────────────────────────
def calc_position_size(capital, price, sl_distance, risk_pct=0.01):
    risk_amount = capital * risk_pct
    shares = int(risk_amount / sl_distance) if sl_distance > 0 else 0
    if shares < 1:
        shares = 1
    return {
        'risk_amount': round(risk_amount, 0),
        'shares': shares,
        'position_value': round(shares * price, 0),
    }

# ─── Sector Diversification ──────────────────────────────────────────────────
def get_sector(name):
    for sector, stocks in SECTORS.items():
        if name in stocks:
            return sector
    return 'Other'

def check_sector_limit(sector, sector_counts, max_per_sector=MAX_PER_SECTOR):
    return sector_counts.get(sector, 0) >= max_per_sector

# ─── ML Feature Building ──────────────────────────────────────────────────────
def build_ml_features(df, idx=None):
    """
    Build 34-feature array for ML inference.
    NB: ML model labels = 5-day return >+2% (UP) / <-2% (DOWN) / else NEUTRAL.
    This is a MOMENTUM predictor, not a mean-reversion signal.
    The rule-based signal fires on oversold RSI (mean-reversion bounce).
    Use ML as a momentum filter: only take BUY when ML also says UP.
    """
    if idx is None:
        last = df.iloc[-1]
    else:
        last = df.iloc[idx]
    c = last.get('Close', 0)
    closes = df['Close'].values
    n = len(closes)
    f_ma = [last.get(f'ma{w}', c) for w in [5, 10, 20, 50, 100, 200]]
    f_tech = [last.get('rsi', 50), last.get('macd', 0), last.get('macd_sig', 0),
              last.get('atr', 1), last.get('vol_ratio', 1), last.get('ret5', 0)]
    f_ratio = [
        c / (last.get('ma20', c) + 1e-10), c / (last.get('ma50', c) + 1e-10),
        last.get('ma20', c) / (last.get('ma50', c) + 1e-10),
        last.get('ma50', c) / (last.get('ma200', c) + 1e-10),
        last.get('macd', 0) - last.get('macd_sig', 0),
        last.get('rsi', 50) * last.get('vol_ratio', 1) / 100,
    ]
    atr_val = last.get('atr', 1) + 1
    f_atr = [
        (c - last.get('ma20', c)) / atr_val,
        (c - last.get('ma50', c)) / atr_val,
        last.get('macd', 0) / (c + 1e-10),
        last.get('vol_ratio', 1) - 1,
    ]
    f_ret = [
        (c - closes[max(0, n-6)]) / (closes[max(0, n-6)] + 1e-10) if n >= 6 else 0,
        (c - closes[max(0, n-11)]) / (closes[max(0, n-11)] + 1e-10) if n >= 11 else 0,
        (c - closes[max(0, n-21)]) / (closes[max(0, n-21)] + 1e-10) if n >= 21 else 0,
        last.get('rsi', 50) / 100,
        last.get('ret5', 0) * 10,
        df['Close'].iloc[-5:].mean() / (df['Close'].iloc[-20:].mean() + 1e-10) if n >= 20 else 1,
    ]
    f_vol = [
        df['Close'].iloc[-20:].std() / (df['Close'].iloc[-20:].mean() + 1e-10) if n >= 20 else 0,
        df['Volume'].iloc[-20:].mean() / (df['Volume'].iloc[-50:].mean() + 1e-10) if n >= 50 else 1,
        df['Close'].iloc[-3:].mean() / (df['Close'].iloc[-10:].mean() + 1e-10) if n >= 10 else 1,
        df['Close'].iloc[-5:].mean() / (df['Close'].iloc[-30:].mean() + 1e-10) if n >= 30 else 1,
        last.get('vol_ratio', 1) * (c / (last.get('ma20', c) + 1e-10)),
        (last.get('macd', 0) / (last.get('atr', 1) + 1e-10)) if last.get('atr', 0) > 0 else 0,
    ]
    all_f = f_ma + f_tech + f_ratio + f_atr + f_ret + f_vol
    all_f = [0 if (np.isnan(x) or np.isinf(x)) else x for x in all_f]
    return np.array([all_f])

# ─── 9-Stage AI Opinion Pipeline ───────────────────────────────────────────────
def ai_opinion_pipeline(symbol, price, rsi, macd, macd_sig, atr, vol_ratio, ret5, df):
    """
    Full 9-stage AI opinion pipeline.
    Stages: Market Regime → News Sentiment → Scanner → Validator → Options → Risk → Execution → Replay → Learning
    """
    from datetime import datetime

    # Stage 1: Market Regime (cached — one fetch per scan session)
    mkt = get_market_regime()
    market_regime = mkt['regime']
    regime_score = mkt['score']

    # Stage 2: News Sentiment (proxy via ret5)
    if ret5 > 0.02: news_score = 30
    elif ret5 > 0: news_score = 15
    elif ret5 < -0.02: news_score = -30
    else: news_score = -5

    # Stage 3: Stock Scanner (setup quality)
    setup_score = 0
    if rsi < 40: setup_score += 25
    elif rsi < RSI_CONFIG['buy_strict']: setup_score += 15
    elif rsi > 55: setup_score -= 20
    if macd > macd_sig: setup_score += 20
    else: setup_score -= 15
    if vol_ratio > 1.5: setup_score += 15
    elif vol_ratio > 1.2: setup_score += 8
    if not (np.isnan(df["ma20"].iloc[-1]) or np.isnan(df["ma50"].iloc[-1]) or np.isnan(df["ma200"].iloc[-1])):
        ma_bonus = sum(df["Close"].iloc[-1] > df[f"ma{w}"].iloc[-1] for w in [20, 50, 200]) * 10
        setup_score += ma_bonus

    # Stage 4: Trade Validator (S/R + volume)
    support = df['Low'].rolling(20).min().iloc[-1]
    resistance = df['High'].rolling(20).max().iloc[-1]
    range_pct = (resistance - support) / price * 100 if price > 0 else 0
    if price <= support * 1.02 and vol_ratio > 1.2:
        validator = "CONFIRMED"; validator_bonus = 15
    elif price >= resistance * 0.98 and vol_ratio > 1.3:
        validator = "RESISTANCE_BREAK"; validator_bonus = 10
    elif range_pct < 3:
        validator = "TIGHT_RANGE"; validator_bonus = -10
    else:
        validator = "CAUTION"; validator_bonus = 0

    # Stage 5: Options Flow (proxy via RSI)
    if rsi < 40: options_signal = "BEARISH_SENTIMENT"; options_bias = -10
    elif rsi > 55: options_signal = "BULLISH_SENTIMENT"; options_bias = 10
    else: options_signal = "NEUTRAL"; options_bias = 0

    # Stage 6: Risk Manager (ATR-based)
    daily_range_pct = atr / price * 100 if price > 0 else 0
    if daily_range_pct > 4: risk_level = "HIGH_VOLATILITY"; risk_multiplier = 0.5
    elif daily_range_pct > 2.5: risk_level = "MEDIUM"; risk_multiplier = 1.0
    else: risk_level = "LOW"; risk_multiplier = 1.5
    sl_distance = atr * ATR_CONFIG['swing']['sl']
    t1_distance = atr * ATR_CONFIG['swing']['t1']
    risk_per_trade = 100000 * 0.01
    position_size = max(1, int(risk_per_trade / sl_distance)) if sl_distance > 0 else 1
    rr_ratio = round(t1_distance / sl_distance, 2) if sl_distance > 0 else 0

    # Stage 7: Execution Timing
    hour = (datetime.now().hour + 5) % 24
    if 9.5 <= hour <= 10.5: exec_timing = "OPEN_RUSH"; exec_type = "LIMIT"
    elif 10.5 <= hour <= 12: exec_timing = "TREND_CONFIRM"; exec_type = "MARKET"
    elif 12 <= hour <= 14.5: exec_timing = "MID_SESSION"; exec_type = "MARKET"
    elif 14.5 <= hour <= 15: exec_timing = "CLOSE_CATCH"; exec_type = "LIMIT"
    else: exec_timing = "AFTER_HOURS"; exec_type = "LIMIT_OPEN"

    # Stage 8: Trade Replay
    recent_ret = df['Close'].pct_change(5).iloc[-5:].mean() * 100
    if abs(recent_ret) > 5: replay_match = 75; replay_note = "HIGH_VOLATILITY"
    elif abs(recent_ret) > 3: replay_match = 55; replay_note = "MODERATE"
    else: replay_match = 35; replay_note = "LOW_ACTIVITY"

    # Stage 9: Learning (RSI-trend alignment correction)
    correction_factor = 1.0
    if (rsi < 50 and ret5 > 0) or (rsi > 50 and ret5 < 0): correction_factor = 0.9
    if rsi > 55 or rsi < 35: correction_factor = 0.85

    # Compute total
    scores = {
        'market_regime': regime_score,
        'news_sentiment': news_score,
        'stock_scanner': setup_score,
        'trade_validator': validator_bonus,
        'options_flow': options_bias,
        'execution': 0,
        'trade_replay': replay_match - 40,
        'learning': (correction_factor - 1) * 50,
    }
    total = sum(scores.values())
    total = max(-100, min(100, total))

    if total >= 20: outlook = "BULLISH"; confidence = "HIGH" if total >= 45 else "MEDIUM"
    elif total <= -20: outlook = "BEARISH"; confidence = "HIGH" if total <= -45 else "MEDIUM"
    else: outlook = "NEUTRAL"; confidence = "MEDIUM" if abs(total) < 15 else "LOW"

    return {
        'outlook': outlook,
        'confidence': confidence,
        'total_score': round(total, 1),
        'stages': {
            '1_market_regime': {'value': market_regime, 'score': int(regime_score)},
            '2_news_sentiment': {'score': int(news_score)},
            '3_stock_scanner': {'score': int(setup_score)},
            '4_trade_validator': {'value': validator, 'score': int(validator_bonus),
                                   'support': float(round(support, 2)), 'resistance': float(round(resistance, 2))},
            '5_options_flow': {'value': options_signal, 'score': int(options_bias)},
            '6_risk_manager': {'value': risk_level,
                               'sl': float(round(price - sl_distance, 2)),
                               't1': float(round(price + t1_distance, 2)),
                               't2': float(round(price + t1_distance * 2, 2)),
                               'rr_ratio': float(rr_ratio),
                               'position_size': int(position_size)},
            '7_execution': {'timing': exec_timing, 'type': exec_type},
            '8_trade_replay': {'match_pct': int(replay_match), 'note': replay_note},
            '9_learning': {'correction_factor': float(correction_factor), 'score': float(scores['learning'])},
        }
    }

# ─── Signal Age Tracking ────────────────────────────────────────────────────────
# Track the most recent date each stock fired a BUY/SELL signal (for stale detection)
_SIGNAL_AGE_CACHE = {}

def update_signal_age(sym, signal, price, t1):
    """Update the signal age cache. Returns days since last signal."""
    import time as _time
    key = sym
    now = _time.time()
    if signal in ('BUY', 'SELL') and price > 0 and t1 > 0:
        _SIGNAL_AGE_CACHE[key] = {
            'signal': signal,
            'signal_price': price,
            'signal_t1': t1,
            'timestamp': now,
        }
    if key in _SIGNAL_AGE_CACHE:
        age_secs = now - _SIGNAL_AGE_CACHE[key]['timestamp']
        return age_secs / 86400.0  # days
    return None

def get_signal_age_days(sym):
    """Return days since signal was first recorded, or None if no signal on record."""
    import time as _time
    if sym not in _SIGNAL_AGE_CACHE:
        return None
    return (_time.time() - _SIGNAL_AGE_CACHE[sym]['timestamp']) / 86400.0

def check_level_alignment(ai_t1, signal_t1):
    """Check AI vs Signal T1 level alignment.
    Returns (aligned, gap_pct, status).
    - aligned: True if within 3%
    - gap_pct: absolute % gap between AI_T1 and Signal_T1
    - status: 'ALIGNED' | 'WARN' (3-6% gap) | 'MISMATCH' (>6% gap) | 'CONFLICT' (opposite direction)
    """
    if ai_t1 is None or signal_t1 is None or signal_t1 == 0:
        return True, 0.0, 'ALIGNED'
    gap = abs(ai_t1 - signal_t1) / signal_t1 * 100
    if gap <= 3:
        return True, gap, 'ALIGNED'
    elif gap <= 6:
        return False, gap, 'WARN'
    else:
        return False, gap, 'MISMATCH'
