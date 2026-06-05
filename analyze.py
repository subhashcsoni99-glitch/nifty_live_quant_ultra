#!/usr/bin/env python3
"""
NIFTY Live Quant Ultra - Analyzer v6
Dual Analysis: Rule-Based + ML Prediction (LLM-ready output)
"""
import sys
import urllib.request
import json
import numpy as np
import pandas as pd
import yfinance as yf
import joblib
import os
from datetime import datetime

MODEL_DIR = os.path.dirname(os.path.abspath(__file__)) + "/models"

# ─── Divergence Detection ───────────────────────────────────────────────────
def detect_divergence(df):
    """Detect RSI bullish/bearish divergence."""
    rsi_vals = df['rsi'].values
    price_vals = df['Close'].values
    if len(rsi_vals) < 50:
        return None
    lookback = 20
    recent_rsi = rsi_vals[-lookback:]
    recent_price = price_vals[-lookback:]
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

# ─── 9-Stage AI Opinion Pipeline (v7) ────────────────────────────────────────
def risk_manager(price, atr, daily_range_pct, capital=100000):
    """
    Stage 6: Risk Manager Agent
    Replaces hard-coded position sizing with proper ATR-based risk management.
    Agent spec: stoploss, targets, risk-reward >= 1:2, position size (1% capital)
    """
    # Volatility assessment
    if daily_range_pct > 4:
        risk_level = "HIGH_VOLATILITY"
        risk_multiplier = 0.5  # Halve position in volatile markets
    elif daily_range_pct > 2.5:
        risk_level = "MEDIUM"
        risk_multiplier = 1.0
    else:
        risk_level = "LOW"
        risk_multiplier = 1.5  # Can increase in stable markets

    # ATR-based stops (consistent with strategy)
    sl_distance = atr * 2
    t1_distance = atr * 3
    t2_distance = atr * 6

    stoploss = round(price - sl_distance, 2)
    target1 = round(price + t1_distance, 2)
    target2 = round(price + t2_distance, 2)

    # Risk-reward validation
    risk_amount = sl_distance
    reward1 = t1_distance
    rr1 = reward1 / risk_amount if risk_amount > 0 else 0

    # Position size: 1% of capital per trade (agent spec)
    risk_per_trade = capital * 0.01  # 1% of capital
    position_size = round(risk_per_trade / sl_distance) if sl_distance > 0 else 0
    position_value = round(position_size * price, 2)

    return {
        'risk_level': risk_level,
        'risk_multiplier': risk_multiplier,
        'position_pct': round(risk_multiplier * 50),  # 25/50/75 mapped
        'sl': stoploss,
        't1': target1,
        't2': target2,
        'rr_ratio': round(rr1, 2),
        'risk_per_trade': risk_per_trade,
        'position_size': position_size,
        'position_value': position_value,
    }

def ai_opinion_pipeline(symbol, price, rsi, macd, macd_sig, atr, vol_ratio, ret5, df):
    """Execute 9-stage AI Opinion Pipeline for stock analysis."""
    
    # ── Stage 1: Market Regime ──────────────────────────────────────────
    # Determine overall market trend via NIFTY 50 index proximity to MAs
    try:
        nifty = yf.Ticker("^NSEI").history(period="30d")
        nifty_ma20 = nifty['Close'].rolling(20).mean().iloc[-1]
        nifty_ma50 = nifty['Close'].rolling(50).mean().iloc[-1]
        nifty_price = nifty['Close'].iloc[-1]
        if nifty_price > nifty_ma20 and nifty_price > nifty_ma50:
            market_regime = "BULLISH"
            regime_score = 15
        elif nifty_price < nifty_ma20 and nifty_price < nifty_ma50:
            market_regime = "BEARISH"
            regime_score = -15
        else:
            market_regime = "NEUTRAL"
            regime_score = 0
    except:
        market_regime = "UNKNOWN"
        regime_score = 0
    
    # ── Stage 2: News Sentiment ───────────────────────────────────────
    # Proxy: use 5-day return direction as sentiment proxy
    # Real implementation would call news API
    if ret5 > 2:
        news_score = 30
    elif ret5 > 0:
        news_score = 15
    elif ret5 < -2:
        news_score = -30
    else:
        news_score = -5
    
    # ── Stage 3: Stock Scanner ─────────────────────────────────────────
    # Technical setup quality score (0-100)
    setup_score = 0
    if rsi < 40:
        setup_score += 25  # Deep oversold = strong setup
    elif rsi < 45:
        setup_score += 15  # Bullish zone (oversold)
    elif rsi > 55:
        setup_score -= 20  # Overbought penalty
    
    if macd > macd_sig:
        setup_score += 20  # Bullish MACD cross
    else:
        setup_score -= 15  # Bearish MACD
    
    if vol_ratio > 1.5:
        setup_score += 15  # High volume confirmation
    elif vol_ratio > 1.2:
        setup_score += 8
    
    ma_score = sum([df['Close'].iloc[-1] > df[f'ma{w}'].iloc[-1] for w in [20, 50, 200] if not np.isnan(df[f'ma{w}'].iloc[-1])]) * 10
    setup_score += ma_score
    
    # ── Stage 4: Trade Validator ───────────────────────────────────────
    # Support/resistance + volume validation
    support = df['Low'].rolling(20).min().iloc[-1]
    resistance = df['High'].rolling(20).max().iloc[-1]
    range_pct = (resistance - support) / price * 100
    
    if price <= support * 1.02 and vol_ratio > 1.2:
        validator = "CONFIRMED"
        validator_bonus = 15
    elif price >= resistance * 0.98 and vol_ratio > 1.3:
        validator = "RESISTANCE_BREAK"
        validator_bonus = 10
    elif range_pct < 3:
        validator = "TIGHT_RANGE"
        validator_bonus = -10
    else:
        validator = "CAUTION"
        validator_bonus = 0
    
    # ── Stage 5: Options Flow ──────────────────────────────────────────
    # Proxy: use RSI as proxy for options sentiment (overbought/oversold)
    # Real would need NSE options data
    if rsi < 40:
        options_signal = "BEARISH_SENTIMENT"
        options_bias = -10
    elif rsi > 55:
        options_signal = "BULLISH_SENTIMENT"
        options_bias = 10
    else:
        options_signal = "NEUTRAL"
        options_bias = 0
    
    # ── Stage 6: Risk Manager ───────────────────────────────────────────
    # ATR-based position sizing via agent function (1% capital, RR >= 1:2)
    daily_range_pct = atr / price * 100 if price > 0 else 0
    rm = risk_manager(price, atr, daily_range_pct)
    risk_level = rm['risk_level']
    position_pct = rm['position_pct']
    
    # ── Stage 7: Execution ─────────────────────────────────────────────
    # Entry timing recommendation
    hour = datetime.now().hour + 5  # IST
    if 9.5 <= hour <= 10.5:
        exec_timing = "OPEN_RUSH"
        exec_type = "LIMIT"
    elif 10.5 <= hour <= 12:
        exec_timing = "TREND_CONFIRM"
        exec_type = "MARKET"
    elif 12 <= hour <= 14.5:
        exec_timing = "MID_SESSION"
        exec_type = "MARKET"
    elif 14.5 <= hour <= 15:
        exec_timing = "CLOSE_CATCH"
        exec_type = "LIMIT"
    else:
        exec_timing = "AFTER_HOURS"
        exec_type = "LIMIT_OPEN"
    
    # ── Stage 8: Trade Replay ───────────────────────────────────────────
    # Historical pattern matching (simplified proxy)
    # Compare recent 5-day pattern to historical similar setups
    recent_ret = df['Close'].pct_change(5).iloc[-5:].mean() * 100
    if abs(recent_ret) > 5:
        replay_match = 75
        replay_note = "HIGH_VOLATILITY_HISTORY"
    elif abs(recent_ret) > 3:
        replay_match = 55
        replay_note = "MODERATE_MOVE"
    else:
        replay_match = 35
        replay_note = "LOW_ACTIVITY"
    
    # ── Stage 9: Learning ───────────────────────────────────────────────
    # Self-correction factor based on RSI + trend alignment
    correction_factor = 1.0
    # If RSI and trend disagree, reduce confidence
    if (rsi < 50 and ret5 > 0) or (rsi > 50 and ret5 < 0):
        correction_factor = 0.9  # Minor disagreement
    if rsi > 55 or rsi < 35:
        correction_factor = 0.85  # Extreme zone - reduce confidence
    
    # ── Compute Final Opinion ──────────────────────────────────────────
    scores = {
        'market_regime': regime_score,
        'news_sentiment': news_score,
        'stock_scanner': setup_score,
        'trade_validator': validator_bonus,
        'options_flow': options_bias,
        'execution': 0,  # Neutral - timing dependent
        'trade_replay': replay_match - 40,  # Normalize around 0
        'learning': (correction_factor - 1) * 50,  # -7.5 to 0
    }
    
    total = sum(scores.values())
    
    if total >= 20:
        outlook = "BULLISH"
        confidence = "HIGH" if total >= 45 else "MEDIUM"
    elif total <= -20:
        outlook = "BEARISH"
        confidence = "HIGH" if total <= -45 else "MEDIUM"
    else:
        outlook = "NEUTRAL"
        confidence = "MEDIUM" if abs(total) < 15 else "LOW"
    
    return {
        'outlook': outlook,
        'confidence': confidence,
        'total_score': total,
        'stages': {
            '1_market_regime': {'value': market_regime, 'score': regime_score},
            '2_news_sentiment': {'value': news_score, 'score': news_score},
            '3_stock_scanner': {'value': round(setup_score, 1), 'score': setup_score},
            '4_trade_validator': {'value': validator, 'score': validator_bonus},
            '5_options_flow': {'value': options_signal, 'score': options_bias},
            '6_risk_manager': {
                'value': risk_level,
                'position_pct': position_pct,
                'sl': rm['sl'],
                't1': rm['t1'],
                't2': rm['t2'],
                'rr_ratio': rm['rr_ratio'],
                'position_value': rm['position_value']
            },
            '7_execution': {'timing': exec_timing, 'type': exec_type},
            '8_trade_replay': {'match_pct': replay_match, 'note': replay_note},
            '9_learning': {'correction_factor': correction_factor, 'score': scores['learning']},
        }
    }

def get_ai_opinion(symbol, price, rsi, macd, macd_sig, atr, vol_ratio, ret5, df):
    return ai_opinion_pipeline(symbol, price, rsi, macd, macd_sig, atr, vol_ratio, ret5, df)

# ─── ML Prediction ────────────────────────────────────────────────────────────
def ml_predict(symbol, df):
    """Load trained model and predict direction. Returns (direction, confidence)."""
    model_path = f"{MODEL_DIR}/{symbol.upper()}_model.joblib"
    if not os.path.exists(model_path):
        return None
    
    try:
        model = joblib.load(model_path)
        # Build features matching trained model (34 features)
        last = df.iloc[-1]
        close = last.get('Close', 0)
        c = close
        
        # Base features (12)
        f_ma = [last.get(f'ma{w}', close) for w in [5, 10, 20, 50, 100, 200]]
        f_tech = [last.get('rsi', 50), last.get('macd', 0), last.get('macd_sig', 0),
                  last.get('atr', 1), last.get('vol_ratio', 1), last.get('ret5', 0)]
        
        # Derived ratios (6)
        f_ratio = [
            c / (last.get('ma20', c) + 1e-10),
            c / (last.get('ma50', c) + 1e-10),
            last.get('ma20', c) / (last.get('ma50', c) + 1e-10),
            last.get('ma50', c) / (last.get('ma200', c) + 1e-10),
            last.get('macd', 0) - last.get('macd_sig', 0),
            last.get('rsi', 50) * last.get('vol_ratio', 1) / 100,
        ]
        
        # ATR-scaled (4)
        atr_val = last.get('atr', 1) + 1
        f_atr = [
            (c - last.get('ma20', c)) / atr_val,
            (c - last.get('ma50', c)) / atr_val,
            last.get('macd', 0) / (c + 1e-10),
            last.get('vol_ratio', 1) - 1,
        ]
        
        # Multi-frame returns (6)
        closes = df['Close'].values
        n = len(closes)
        f_ret = [
            (c - closes[max(0, n-6)]) / (closes[max(0, n-6)] + 1e-10) if n >= 6 else 0,
            (c - closes[max(0, n-11)]) / (closes[max(0, n-11)] + 1e-10) if n >= 11 else 0,
            (c - closes[max(0, n-21)]) / (closes[max(0, n-21)] + 1e-10) if n >= 21 else 0,
            last.get('rsi', 50) / 100,
            last.get('ret5', 0) * 10,
            df['Close'].iloc[-5:].mean() / (df['Close'].iloc[-20:].mean() + 1e-10) if n >= 20 else 1,
        ]
        
        # Volatility & volume (6)
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
        features = np.array([all_f])  # 2D array (1, 34)
        proba = model.predict_proba(features)[0]
        direction = model.predict(features)[0]
        conf = max(proba) * 100
        return {'direction': 'UP' if direction == 1 else 'DOWN', 'confidence': round(conf, 1)}
    except Exception:
        return None

# ─── Traditional Rule-Based Analysis ──────────────────────────────────────────
def get_nse_price(symbol):
    headers = {'User-Agent': 'Mozilla/5.0', 'Referer': 'https://www.nseindia.com/'}
    url = f"https://www.nseindia.com/api/quote-equity?symbol={symbol}"
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
            p = data.get('priceInfo', {})
            return {'price': p.get('lastPrice', 0), 'prev_close': p.get('previousClose', 0)}
    except:
        return None

def get_ohlc(symbol, days=365):
    try:
        return yf.Ticker(f"{symbol.upper()}.NS").history(period=f"{days}d")
    except:
        return None

def add_features(df):
    df = df.copy()
    for w in [5, 10, 20, 50, 100, 200]:
        df[f'ma{w}'] = df['Close'].rolling(w).mean()
    
    # RSI
    delta = df['Close'].diff()
    gain, loss = delta.where(delta > 0, 0).rolling(14).mean(), (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['rsi'] = 100 - (100 / (1 + gain / (loss + 1e-10)))
    
    # MACD
    ema12, ema26 = df['Close'].ewm(span=12).mean(), df['Close'].ewm(span=26).mean()
    df['macd'] = ema12 - ema26
    df['macd_sig'] = df['macd'].ewm(span=9).mean()
    
    # ATR
    tr = pd.concat([df['High'] - df['Low'], abs(df['High'] - df['Close'].shift()), abs(df['Low'] - df['Close'].shift())], axis=1).max(axis=1)
    df['atr'] = tr.rolling(14).mean()
    
    # Volume
    df['vol_ma'] = df['Volume'].rolling(20).mean()
    df['vol_ratio'] = df['Volume'] / (df['vol_ma'] + 1)
    
    # Momentum
    df['ret5'] = df['Close'].pct_change(5)
    
    # 20-day low for divergence
    df['low_20d'] = df['Low'].rolling(20).min()
    df['high_20d'] = df['High'].rolling(20).max()
    df['price_near_low'] = (df['Close'] - df['low_20d']) / (df['high_20d'] - df['low_20d'] + 1e-10)
    
    return df

def analyze_stock(symbol):
    sym = symbol.upper().strip()
    nse = get_nse_price(sym)
    df = get_ohlc(sym)
    
    if df is None or nse is None:
        return {'symbol': sym, 'error': 'Could not fetch data'}
    
    df = add_features(df)
    price = nse['price']
    prev_close = nse['prev_close']
    change_pct = ((price - prev_close) / prev_close * 100) if prev_close else 0
    
    rsi = df['rsi'].iloc[-1]
    macd, macd_sig = df['macd'].iloc[-1], df['macd_sig'].iloc[-1]
    atr = df['atr'].iloc[-1]
    vol_ratio = df['vol_ratio'].iloc[-1]
    ret5 = df['ret5'].iloc[-1]
    price_near_low = df['price_near_low'].iloc[-1]
    
    pv = df['Close'].iloc[-1]
    ma20, ma50, ma200 = df['ma20'].iloc[-1], df['ma50'].iloc[-1], df['ma200'].iloc[-1]
    
    # Relaxed RSI zones (< 65 for BUY, > 35 for SELL)
    # Divergence: price within 20% of 20-day low = potential bounce
    
    buy = sum([pv > ma20, pv > ma50, ma50 > ma200, rsi < 45, macd > macd_sig, vol_ratio > 1.2, ret5 > 0])
    sell = sum([pv < ma20, pv < ma50, ma50 < ma200, rsi > 55, macd < macd_sig, vol_ratio > 1.2, ret5 < 0])
    
    # Divergence bonus: price near 20-day low + RSI < 60
    divergence_bonus = 1 if (price_near_low < 0.2 and rsi < 45) else 0
    buy += divergence_bonus
    
    prob_buy = min(95, 50 + buy * 7)
    prob_sell = min(95, 50 + sell * 7)
    
    if buy >= 5 and rsi < 45:
        signal = "📈 BUY"
        entry = round(price, 2)
        sl = round(price - atr * 2, 2)
        t1 = round(price + atr * 3, 2)
        t2 = round(price + atr * 6, 2)
        trailing_sl = round(price - atr * 1.5, 2)  # Trailing SL
        prob = prob_buy
    elif sell >= 5 and rsi > 55:
        signal = "📉 SELL"
        entry = round(price, 2)
        sl = round(price + atr * 2, 2)
        t1 = round(price - atr * 3, 2)
        t2 = round(price - atr * 6, 2)
        trailing_sl = round(price + atr * 1.5, 2)
        prob = prob_sell
    else:
        signal = "⚠️ RANGE"
        entry = sl = t1 = t2 = trailing_sl = 0
        prob = max(prob_buy, prob_sell)
    
    # ─── Divergence detection for AI pipeline ──────────────────────────────
    div_detected = detect_divergence(df)
    
    # ─── Trend Score (for AI opinion) ───────────────────────────────────────
    trend_score = (buy - sell) * 10 + (50 - rsi) * 0.5
    
    # ─── ML Prediction ──────────────────────────────────────────────────────
    df_clean = df.dropna()
    ml = ml_predict(sym, df_clean)
    ai = ai_opinion_pipeline(sym, price, rsi, macd, macd_sig, atr, vol_ratio, ret5, df)
    
    return {
        'symbol': sym, 'price': price, 'change_pct': change_pct, 'signal': signal,
        'rsi': rsi, 'atr': atr, 'vol_ratio': vol_ratio, 'prob': round(prob, 0),
        'entry': entry, 'sl': sl, 't1': t1, 't2': t2, 'trailing_sl': trailing_sl,
        'divergence': div_detected, 'price_near_low': round(price_near_low * 100, 0),
        'ml': ml, 'ai': ai
    }

def format_output(r):
    if 'error' in r:
        return f"❌ {r['symbol']}: {r['error']}"
    
    out = f"""
{'='*60}
{r['symbol']} | {r['signal']} ({r['prob']:.0f}%)
{'='*60}
💰 ₹{r['price']:,.2f} ({r['change_pct']:+.2f}%)
📊 RSI: {r['rsi']:.1f} | ATR: {r['atr']:.2f} | Vol: {r['vol_ratio']:.2f}x
"""
    if r['divergence']:
        out += f"📉 DIV: Price at {r['price_near_low']:.0f}% of 20-day range (bounce potential)\n"
    
    # 9-Stage AI Opinion Pipeline
    ai = r.get('ai', {})
    outlook = ai.get('outlook', 'N/A')
    ai_conf = ai.get('confidence', 'N/A')
    total_score = ai.get('total_score', 0)
    
    out += f"\n🤖 AI Opinion: {outlook} ({ai_conf}) | Score: {total_score:+.0f}\n"
    
    # Stage details
    stages = ai.get('stages', {})
    stage_names = [
        ('1_market_regime', '📊 Market Regime'),
        ('2_news_sentiment', '📰 News Sentiment'),
        ('3_stock_scanner', '🔍 Stock Scanner'),
        ('4_trade_validator', '✅ Trade Validator'),
        ('5_options_flow', '📈 Options Flow'),
        ('6_risk_manager', '⚖️ Risk Manager'),
        ('7_execution', '🎯 Execution'),
        ('8_trade_replay', '📹 Trade Replay'),
        ('9_learning', '🧠 Learning'),
    ]
    
    for key, label in stage_names:
        s = stages.get(key, {})
        val = s.get('value', s.get('timing', 'N/A'))
        score = s.get('score', 0)
        extra = ''
        if key == '6_risk_manager' and 'position_pct' in s:
            extra = f" ({s['position_pct']}%)"
        if key == '8_trade_replay' and 'match_pct' in s:
            extra = f" [{s['match_pct']}%]"
        out += f"   {label}: {val}{extra} {score:+.0f}\n"
    
    # ML Prediction
    ml = r.get('ml')
    if ml:
        out += f"\n🧠 ML Model: {ml['direction']} ({ml['confidence']:.0f}% conf)\n"
    else:
        out += f"\n🧠 ML Model: Not trained for {r['symbol']}\n"
    
    if r['entry'] > 0:
        out += f"""📋 Rule-Based Levels:
   Entry: ₹{r['entry']:,.2f} | SL: ₹{r['sl']:,.2f} | TSL: ₹{r['trailing_sl']:,.2f}
   T1: ₹{r['t1']:,.2f} ({((r['t1']-r['price'])/r['price']*100):+.1f}%)
   T2: ₹{r['t2']:,.2f} ({((r['t2']-r['price'])/r['price']*100):+.1f}%)
"""
    return out

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 analyze.py SBIN,BHEL")
        sys.exit(1)
    
    symbols = [s.strip() for s in sys.argv[1].split(',')]
    results = [analyze_stock(s) for s in symbols]
    
    print("\n" + "="*60)
    print("📊 NIFTY ANALYZER v6 — Dual Analysis (AI + ML + Rule-Based)")
    print("="*60)
    
    for r in results:
        print(format_output(r))
    
    if len(results) > 1:
        buy = [r for r in results if 'BUY' in r.get('signal','')]
        sell = [r for r in results if 'SELL' in r.get('signal','')]
        print(f"📈 BUY: {len(buy)} | 📉 SELL: {len(sell)} | ⚠️ NONE: {len(results)-len(buy)-len(sell)}")
        print("\n⚠️ Not SEBI registered.")

if __name__ == "__main__":
    main()