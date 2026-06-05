#!/usr/bin/env python3
"""
NIFTY Scanner v19 - Unified Rule-Based + AI (9-stage) + ML

v18 changes (9.5/10 review suggestions applied):
  1. NEG_HIST: rr < -2% AND wr < 40% → demote (was wr < 35; tightened AND threshold)
     → Cat B candidates: BPCL (45%WR/-2.0%), AXISBANK (38%/-0.8%), etc.
  2. BEAR_DIV in BULL = 🔴 CONTRARIAN (honest counter-trend label, stock stays in C2)
  3. Tiered stale: <4d=FRESH, 4-7d=⚠️ STALE, 8+d=💀 STALE_CRITICAL
  4. Regime-signal CF: regime fit 15% weight in confluence score
  5. Cat B relaxed: wr >= 35%, rr >= -2% (was wr >= 40%, rr >= 0)

v19 changes (8.5→9.5/10 review suggestions applied):
  1. 🏆 TOP_PICK: Cat A/B stocks with positive RR + CF>=8 → 🏆 tag + sorted first
     → Cat A/B blocks now sorted by RR desc (positive RR bubbles to top)
  2. Cat A-SHORT HEDGE: AI_BEARISH + SELL in BULL regime → Cat A- + 🛡️ HEDGE tag
     → HDFCBANK short moves from Cat C1 → Cat A- (appropriate hedge classification)
  3. Cat C1 renamed "SIGNAL + AI AGREE (SHORT)" — clear that shorts live here
  4. TOP LONG: sorted by (_starred, CF desc) — 🏆 TOP_PICKs appear first with 🏆 badge
  5. TOP SHORT: shows 🛡️ HEDGE count, each hedge short labeled 🛡️
"""
import sys
import os
import warnings
import json
import logging
import time as _time
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import yfinance as yf
import numpy as np
import pandas as pd
from datetime import datetime

from nifty_core import (
    NIFTY50_STOCKS, NIFTY100_STOCKS,
    EXCLUDED_STOCKS, GOOD_STOCKS, SCANNABLE_STOCKS, DEFAULT_STOCKS,
    ATR_CONFIG, RSI_CONFIG, SIGNAL_CONFIG,
    SECTORS, MAX_PER_SECTOR,
    get_price, get_ohlc, add_features,
    detect_divergence, calc_levels, calc_levels_hourly, get_hourly_atr_and_pivot, calc_position_size,
    calc_support_resistance,
    get_sector, check_sector_limit, build_ml_features,
    get_signal as core_get_signal,
    ai_opinion_pipeline,
    filter_by_fundamentals, get_fundamental_score,
    check_level_alignment, update_signal_age, get_signal_age_days,
)

# ─── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(SCRIPT_DIR, 'models')

# ─── ML Prediction (v17: sklearn version check + graceful failure) ───────────
ML_LOG = logging.getLogger('nifty_ml')
ML_LOG.setLevel(logging.WARNING)
_SKLEARN_VERSION = None
_RECOMMENDED_RETRAIN = False  # set True when version mismatch found

def _get_sklearn_version():
    """Cache sklearn version to avoid repeated imports."""
    global _SKLEARN_VERSION
    if _SKLEARN_VERSION is None:
        import sklearn
        _SKLEARN_VERSION = sklearn.__version__
    return _SKLEARN_VERSION

def get_ml_prediction(sym, df, auto_retrain=False):
    """Get ML prediction with version checking and optional auto-retrain.
    
    Returns:
        dict with direction/confidence/_ml_fail_reason/_needs_retrain
    """
    model_path = os.path.join(MODEL_DIR, f"{sym.upper()}_model.joblib")
    if not os.path.exists(model_path):
        return {'direction': None, 'confidence': 0, '_ml_fail_reason': 'NO_MODEL'}
    try:
        import joblib
        model = joblib.load(model_path)
        
        # Verify sklearn version matches (model trained with different version → fail fast)
        model_version = getattr(model, 'sklearn_version', None)
        current_version = _get_sklearn_version()
        
        if model_version and model_version != current_version:
            ML_LOG.warning(f"[{sym}] sklearn version mismatch: model={model_version} current={current_version}")
            # Model is incompatible — auto-retrain if requested
            if auto_retrain:
                _retrain_model(sym)
                # Reload after retrain
                try:
                    model = joblib.load(model_path)
                except Exception:
                    return {'direction': None, 'confidence': 0, 
                            '_ml_fail_reason': f'sklearn:{model_version}(retrain_failed)',
                            '_needs_retrain': True}
            else:
                return {'direction': None, 'confidence': 0,
                        '_ml_fail_reason': f'sklearn:{model_version}',
                        '_needs_retrain': True}
        
        features = build_ml_features(df)
        proba = model.predict_proba(features)[0]
        direction = model.predict(features)[0]
        conf = max(proba) * 100
        return {'direction': 'UP' if direction == 1 else 'DOWN', 'confidence': round(conf, 1)}
    except Exception as e:
        ML_LOG.warning(f"[{sym}] ML prediction failed: {e}")
        return {'direction': None, 'confidence': 0,
                '_ml_fail_reason': str(e)[:60], '_needs_retrain': True}


def _retrain_model(sym):
    """Retrain a single stock model. Called when version mismatch detected."""
    import subprocess
    try:
        ML_LOG.warning(f"[{sym}] Retraining model (version mismatch)...")
        result = subprocess.run(
            ['python3', 'train.py', sym],
            capture_output=True, text=True, timeout=300,
            cwd=os.path.dirname(os.path.abspath(__file__))
        )
        if result.returncode == 0:
            ML_LOG.warning(f"[{sym}] Retrain OK: {result.stdout.strip().split(chr(10))[-1]}")
        else:
            ML_LOG.warning(f"[{sym}] Retrain FAILED: {result.stderr.strip()[:100]}")
    except Exception as e:
        ML_LOG.warning(f"[{sym}] Retrain subprocess error: {e}")

# ─── Backtest Stats Cache (v17: proper time.time() TTL) ──────────────────────
_STATS_CACHE = None
_STATS_LOADED_AT = 0  # v18fix: was None → reload logic never fired
_STATS_TTL = 3600  # 1 hour

def _get_stats(symbol):
    """Load stats from most recent backtest JSON. Cache expires after _STATS_TTL seconds."""
    global _STATS_CACHE, _STATS_LOADED_AT
    now = _time.time()
    if _STATS_CACHE is None or (_STATS_LOADED_AT and (now - _STATS_LOADED_AT) > _STATS_TTL):
        _STATS_CACHE = {}
        try:
            files = sorted(
                [f for f in os.listdir(MODEL_DIR)
                 if f.startswith('backtest_v') and f.endswith('.json')],
                key=lambda x: os.path.getmtime(os.path.join(MODEL_DIR, x)),
                reverse=True
            )
            for fname in files[:3]:
                try:
                    with open(os.path.join(MODEL_DIR, fname)) as f:
                        data = json.load(f)
                    for r in data.get('results', []):
                        sym = r['symbol']
                        if sym not in _STATS_CACHE:
                            _STATS_CACHE[sym] = {
                                'win_rate': r.get('win_rate', 0),
                                'realized_return': r.get('realized_return', 0),
                                'sharpe': r.get('sharpe', 0),
                                'max_drawdown': r.get('max_drawdown', 999),
                            }
                except:
                    pass
        except:
            pass
        _STATS_LOADED_AT = now
    return _STATS_CACHE.get(symbol, {
        'win_rate': 0, 'realized_return': 0, 'sharpe': 0, 'max_drawdown': 999
    })

# ─── Confluence Scoring (1-10) ──────────────────────────────────────────────────
# v18: NEG_HIST now requires rr < -2% AND wr < 35% (was rr < 0 AND wr <= 50%)
# Cat B threshold: wr >= 35%, rr >= -2% (was wr >= 40%, rr >= 0)
_NEG_HIST_RR_THRESHOLD = -2.0
_NEG_HIST_WR_THRESHOLD = 40  # v18fix: was 35 — COALINDIA wr=35.5/rr=-4.5 was slipping through as Cat B

def _is_neg_hist(stats):
    """NEG_HIST: rr < -2% AND wr < 40% — poor return OR weak win rate together"""
    rr = stats.get('realized_return', 0)
    wr = stats.get('win_rate', 0)
    return rr < _NEG_HIST_RR_THRESHOLD and wr < _NEG_HIST_WR_THRESHOLD


def _is_poor_history(stats):
    """POOR_HIST: rr < -2% OR wr < 25% — Cat A fails these gates"""
    rr = stats.get('realized_return', 0)
    wr = stats.get('win_rate', 0)
    return rr < -2.0 or wr < 25


# ─── Regime-Signal Coherence ──────────────────────────────────────────────────
def _regime_coherence(signal, divergence, regime):
    """Compute regime fit for a signal.
    Returns: score 0.0-1.0, label string, is_contrarian bool
    
    BULL regime:  BUY + no divergence = coherent (1.0)
                  BUY + BEAR_DIV = contrarian (0.0) → labeled 🔴 CONTRARIAN
                  SELL = hedge (0.6)
    BEAR regime: SELL + no divergence = coherent (1.0)
                 SELL + BULL_DIV = contrarian (0.0) → labeled 🔴 CONTRARIAN
                 BUY = hedge (0.6)
    RANGE regime: any trending signal = low fit (0.3)
    """
    if regime == 'BULLISH':
        if signal == 'BUY' and divergence != 'BEARISH':
            return 1.0, 'ALIGNED', False
        elif signal == 'BUY' and divergence == 'BEARISH':
            return 0.0, '🔴 CONTRARIAN', True
        elif signal == 'SELL':
            return 0.6, 'HEDGE', False
    elif regime == 'BEARISH':
        if signal == 'SELL' and divergence != 'BULLISH':
            return 1.0, 'ALIGNED', False
        elif signal == 'SELL' and divergence == 'BULLISH':
            return 0.0, '🔴 CONTRARIAN', True
        elif signal == 'BUY':
            return 0.6, 'HEDGE', False
    else:  # RANGE
        return 0.3, 'RANGE_BOUND', False
    return 0.5, 'NEUTRAL', False


def _stale_tier(age_days):
    """Return stale tier label and confluence penalty multiplier.
    Fresh: < 4 days → no penalty
    Stale: 4-7 days → 10% penalty, tag ⏰ STALE
    Critical: 8+ days → 30% penalty, tag 💀 STALE_CRITICAL
    """
    if age_days < 4:
        return 'FRESH', 0.0
    elif age_days < 8:
        return 'STALE', 0.10
    else:
        return 'CRITICAL', 0.30


# ─── Confluence Scoring v18 (tiered stale + regime fit + NEG_HIST grace) ──────
def calc_confluence_score(r, regime='BULLISH'):
    """Calculate 1-10 confluence score for a stock.
    v18 components: Signal_Conf×0.20 + AI_Conf×0.25 + WR×0.20 + Level_Align×0.10 + Age×0.10 + RegimeFit×0.15
    NEG_HIST grace: rr >= -2% passes, only rr < -2% triggers 30% penalty.
    """
    sig_conf = r.get('prob', 50) / 100.0  # 0-1
    ai = r.get('ai') or {}
    ml = r.get('ml') or {}
    ai_score = ai.get('total_score', 0)  # -100 to 100 → normalize to 0-1
    ai_conf = (ai_score + 100) / 200.0   # 0-1
    sig = r['signal']
    ai_dir = ai.get('outlook', 'NEUTRAL')
    if sig == 'BUY' and ai_dir == 'BULLISH': ai_dir_score = 1.0
    elif sig == 'SELL' and ai_dir == 'BEARISH': ai_dir_score = 1.0
    elif sig == 'BUY' and ai_dir == 'BEARISH': ai_dir_score = 0.0
    elif sig == 'SELL' and ai_dir == 'BULLISH': ai_dir_score = 0.0
    else: ai_dir_score = 0.5
    ai_final = ai_conf * 0.5 + ai_dir_score * 0.5
    stats = r.get('_stats', {})
    wr = stats.get('win_rate', 0) / 100.0  # 0-1
    align = r.get('_level_align', 'ALIGNED')
    align_score = 1.0 if align == 'ALIGNED' else (0.5 if align == 'WARN' else 0.0)
    # Age tiered penalty
    age_days = r.get('signal_age_days', 0)
    age_tier, age_penalty = _stale_tier(age_days)
    age_score = max(0, 1.0 - age_penalty)  # 1.0 for fresh, 0.9 for stale, 0.7 for critical
    # Regime fit bonus
    div = r.get('divergence')
    _, regime_label, is_contrarian = _regime_coherence(sig, div, regime)
    regime_score = 1.0 if not is_contrarian else 0.3  # contrarian gets big penalty here
    score = (sig_conf * 0.20 + ai_final * 0.25 + wr * 0.20
             + align_score * 0.10 + age_score * 0.10 + regime_score * 0.15)
    # NEG_HIST penalty only for rr < -2% (mild negatives pass)
    neg_hist = stats.get('realized_return', 0) < _NEG_HIST_RR_THRESHOLD
    if neg_hist:
        score *= 0.7  # 30% penalty for truly bad history
    return round(score * 10, 1), age_tier, regime_label  # v18 returns tuple

# ─── Categorization (v17: quality filters + Cat A- + WATCHLIST + stale flag) ──
def _categorize(results, regime='BULLISH'):
    """Categorize results into Cat A/A-/B/C1/C2/D/WATCHLIST.

    Cat A:   Triple confirmed (Signal + AI_BULL + ML_UP) + poor/good history (rr >= -2%, wr >= 25%)
    Cat A-:  2-of-3 confirmed + profitable but level mismatch
    Cat B:   AI HIGH/MEDIUM conviction, no ML UP + rr >= -2% + wr >= 35% + no severe NEG_HIST
    Cat C1:  Signal + AI agree (outlook matches signal direction), not Cat A/A-
    Cat C2:  Signal only, AI neutral, or NEG_HIST / BEAR_DIV in BULL regime
    Cat D:   ML signal only (ml_up for BUY or ml_down for SELL), AI neutral
    WATCHLIST: RANGE-bound stocks

    Quality gates (v18):
    - NEG_HIST: rr < -2% AND wr < 35% → ⚠️ NEG_HIST tag + demote to C2
    - BEAR_DIV in BULL → 🔴 CONTRARIAN tag (stocks remain in C2, not suppressed)
    - Level alignment: >3% AI_T1 vs Signal_T1 → UNCONFIRMED tag
    - Stale tiers: <4d=FRESH, 4-7d=⏰ STALE, 8+d=💀 STALE_CRITICAL
    - Cat B: wr >= 35%, rr >= -2% (from wr >= 40%, rr >= 0)
    """
    cat_a, cat_a_minus, cat_b, cat_c1, cat_c2, cat_d, watchlist = [], [], [], [], [], [], []
    for r in results:
        ai = r.get('ai') or {}
        ml = r.get('ml') or {}
        ai_dir = ai.get('outlook', 'NEUTRAL')
        ai_conf = ai.get('confidence', 'LOW')
        ai_t1 = ai.get('stages', {}).get('6_risk_manager', {}).get('t1')
        ml_dir = (ml.get('direction', None) if ml else None)
        ml_fail = ml.get('_ml_fail_reason') if ml else None
        div = r.get('divergence')
        sig = r['signal']
        stats = r.get('_stats', {})
        r['_stats'] = stats
        rr = stats.get('realized_return', 0)
        wr = stats.get('win_rate', 0)
        no_history = (wr == 0 and rr == 0)
        neg_hist_severe = _is_neg_hist(stats)   # rr < -2% AND wr < 35%
        poor_hist = _is_poor_history(stats)    # rr < -2% OR wr < 25%

        # Level alignment check
        sig_t1 = r.get('t1')
        align_ok, gap_pct, align_status = True, 0.0, 'ALIGNED'
        if sig in ('BUY', 'SELL') and ai_t1 and sig_t1 and sig_t1 > 0:
            align_ok, gap_pct, align_status = check_level_alignment(ai_t1, sig_t1)
        r['_level_align'] = align_status
        r['_level_gap_pct'] = round(gap_pct, 2)

        # Stale tiers
        age_days = r.get('signal_age_days', 0)
        age_tier, age_penalty = _stale_tier(age_days)
        is_stale = age_tier in ('STALE', 'CRITICAL')

        # Confluence score (v18: returns CF, age_tier, regime_label)
        r['_confluence'], _, r['_regime_label'] = calc_confluence_score(r, regime)

        def _tag_all(r_obj):
            if neg_hist_severe: _add_tag(r_obj, '⚠️ NEG_HIST')
            if age_tier == 'STALE': _add_tag(r_obj, '⏰ STALE')
            if age_tier == 'CRITICAL': _add_tag(r_obj, '💀 STALE_CRITICAL')
            if not align_ok: _add_tag(r_obj, f'⚠️ LVL_{align_status}')
            if r_obj.get('_regime_label') == '🔴 CONTRARIAN':
                _add_tag(r_obj, '🔴 CONTRARIAN')

        if sig == 'RANGE':
            _add_tag(r, '📋 RANGE')
            if age_days > 1:
                _add_tag(r, '⏰ STALE')
            watchlist.append(r)
            continue

        if sig == 'BUY':
            ml_up = ml_dir == 'UP'
            ai_bull = ai_dir == 'BULLISH'

            if div == 'BEARISH':
                # BEAR_DIV in BULL = counter-trend, honestly labeled
                _, regime_lbl, is_contra = _regime_coherence(sig, div, regime)
                if is_contra:
                    _add_tag(r, 'BEAR_DIV')
                    _add_tag(r, '🔴 CONTRARIAN')
                else:
                    _add_tag(r, 'BEAR_DIV')
                _tag_all(r)
                cat_c2.append(r)
            elif ai_bull and ml_up:
                # Cat A: triple confirmed + history not poor
                if no_history:
                    _add_tag(r, '⚠️ NO_BACKTEST')
                    _tag_all(r)
                    cat_c2.append(r)
                elif poor_hist:
                    _add_tag(r, '⚠️ POOR_HIST')
                    _tag_all(r)
                    cat_c2.append(r)
                else:
                    _tag_all(r)
                    if not align_ok:
                        _add_tag(r, 'UNCONFIRMED')
                        cat_a_minus.append(r)
                    else:
                        cat_a.append(r)
            elif ai_bull and (ai_conf in ('HIGH', 'MEDIUM') or ml is None):
                if no_history:
                    _add_tag(r, '⚠️ NO_BACKTEST')
                # v18: NEG_HIST only if rr < -2% AND wr < 35% (severe)
                # Cat B threshold: wr >= 35%, rr >= -2% (relaxed from wr >= 40%, rr >= 0)
                if neg_hist_severe:
                    _add_tag(r, '⚠️ NEG_HIST')
                    _tag_all(r)
                    cat_c2.append(r)
                elif wr > 0 and wr < 35:
                    _add_tag(r, f'⚠️ WR_LOW({wr:.0f}%)')
                    _tag_all(r)
                    cat_c2.append(r)
                else:
                    _tag_all(r)
                    if not align_ok:
                        _add_tag(r, 'UNCONFIRMED')
                    cat_b.append(r)
            elif ml_up:
                if ai_dir == 'NEUTRAL':
                    cat_d.append(r)
                else:
                    _add_tag(r, 'AI_CONTRADICT')
                    _tag_all(r)
                    cat_c2.append(r)
            elif ai_dir == 'NEUTRAL':
                if no_history:
                    _add_tag(r, '⚠️ NO_BACKTEST')
                _tag_all(r)
                cat_c2.append(r)
            else:
                _add_tag(r, 'AI_CONTRADICT')
                _tag_all(r)
                cat_c2.append(r)

        elif sig == 'SELL':
            ml_down = ml_dir == 'DOWN'
            ai_bear = ai_dir == 'BEARISH'
            ai_bull = ai_dir == 'BULLISH'
            # Regime fit for shorts
            _, rlabel, is_hedge = _regime_coherence(sig, div, regime)

            if ml_down:
                if ai_bull:
                    _add_tag(r, 'AI_CONTRADICT')
                    _tag_all(r)
                    cat_c2.append(r)
                elif poor_hist:
                    _add_tag(r, '⚠️ POOR_HIST')
                    _tag_all(r)
                    cat_d.append(r)
                else:
                    _tag_all(r)
                    if not align_ok:
                        _add_tag(r, 'UNCONFIRMED')
                        cat_a_minus.append(r)
                    else:
                        cat_a.append(r)  # Cat A SHORT: Signal+SELL+ML_DOWN+AI not bullish
            elif ai_bear:
                if no_history:
                    _add_tag(r, '⚠️ NO_BACKTEST')
                if neg_hist_severe:
                    _add_tag(r, '⚠️ NEG_HIST')
                    _tag_all(r)
                    cat_c2.append(r)
                elif wr > 0 and wr < 35:
                    _add_tag(r, f'⚠️ WR_LOW({wr:.0f}%)')
                    _tag_all(r)
                    cat_c2.append(r)
                else:
                    _tag_all(r)
                    if not align_ok:
                        _add_tag(r, 'UNCONFIRMED')
                    # v19: AI_BEARISH + SELL in BULL regime → Cat A-SHORT (HEDGE tier)
                    if regime == 'BULLISH':
                        _add_tag(r, '🛡️ HEDGE')
                        cat_a_minus.append(r)  # A- is where we put hedge shorts
                    else:
                        cat_c1.append(r)
            else:
                if no_history:
                    _add_tag(r, '⚠️ NO_BACKTEST')
                _tag_all(r)
                cat_c2.append(r)

    # ── v19: Star Cat A/B stocks with positive RR as ⭐ TOP_PICK ───────────────
    for r in cat_a + cat_b:
        stats = r.get('_stats', {})
        rr = stats.get('realized_return', 0)
        cf = r.get('_confluence', 0)
        if rr > 0 and cf >= 8.0:
            r['_starred'] = True
            _add_tag(r, '⭐ TOP_PICK')

    return cat_a, cat_a_minus, cat_b, cat_c1, cat_c2, cat_d, watchlist

def _add_tag(r, tag):
    if 'tags' not in r:
        r['tags'] = []
    if tag not in r['tags']:
        r['tags'].append(tag)

# ─── Level Mode Helpers ─────────────────────────────────────────────────────
def get_level_modes_extended(price, atr):
    """Return all level sets for a stock."""
    return {
        'tight':   calc_levels(price, atr, mode='intraday_tight'),
        'regular': calc_levels(price, atr, mode='intraday'),
        'swing':   calc_levels(price, atr, mode='swing'),
    }

# ─── Analyze Single Stock ──────────────────────────────────────────────────
def analyze(sym, use_ai=False, use_trailing=False, fundamental_filter=False, level_mode='intraday', auto_retrain=False):
    price, prev = get_price(sym)
    df = get_ohlc(sym)
    if df is None or price is None or len(df) < 200:
        return None

    df = add_features(df)
    pv = df['Close'].iloc[-1]
    atr = df['atr'].iloc[-1]
    if pd.isna(atr) or atr == 0:
        atr = price * 0.02

    rsi = df['rsi'].iloc[-1]
    macd = df['macd'].iloc[-1]
    macd_sig = df['macd_sig'].iloc[-1]
    vol_ratio = df['vol_ratio'].iloc[-1]
    ret5 = df['ret5'].iloc[-1]
    change_pct = ((price - prev) / prev * 100) if prev and prev > 0 else 0

    sig_val, meta, _ = core_get_signal(df, len(df) - 1)
    signal = meta['signal']
    divergence = meta.get('divergence')

    levels = get_level_modes_extended(price, atr)
    sr = calc_support_resistance(df)
    pos = calc_position_size(100000, price, atr * ATR_CONFIG[level_mode]['sl'])
    pos_pct = round((pos['position_value'] / 100000) * 100, 1)

    prob_buy = min(95, 50 + meta['buy_cnt'] * 8)
    prob_sell = min(95, 50 + meta['sell_cnt'] * 8)
    prob = prob_buy if signal == 'BUY' else (prob_sell if signal == 'SELL' else max(prob_buy, prob_sell))
    reasons = meta.get('reasons', [])

    ai = None
    ml = None
    hourly = None
    if use_ai:
        ai = ai_opinion_pipeline(sym, price, rsi, macd, macd_sig, atr, vol_ratio, ret5, df)
        ml = get_ml_prediction(sym, df, auto_retrain=auto_retrain)
        hourly = get_hourly_atr_and_pivot(sym, price)

    # Track signal age (for stale detection)
    sig_val_for_age = sig_val
    signal_age_days = get_signal_age_days(sym)
    if sig_val_for_age in (1, -1):
        t1_for_age = levels['regular']['t1']
        update_signal_age(sym, signal, price, t1_for_age)
        signal_age_days = get_signal_age_days(sym)

    # Load backtest stats (v18: needed for NEG_HIST and WR threshold checks)
    stats = _get_stats(sym)

    return {
        'symbol': sym, 'price': price, 'prev': prev,
        'rsi': round(rsi, 1), 'change': round(change_pct, 2),
        'signal': signal, 'prob': prob,
        'sl_tight': levels['tight']['sl'], 't1_tight': levels['tight']['t1'], 't2_tight': levels['tight']['t2'],
        'sl_intraday': levels['regular']['sl'], 't1_intraday': levels['regular']['t1'], 't2_intraday': levels['regular']['t2'],
        'sl_swing': levels['swing']['sl'], 't1_swing': levels['swing']['t1'], 't2_swing': levels['swing']['t2'],
        'support': sr['support'], 'resistance': sr['resistance'],
        'buy_cnt': meta['buy_cnt'], 'sell_cnt': meta['sell_cnt'],
        'divergence': divergence, 'reasons': reasons,
        'atr': round(atr, 2), 'vol_ratio': round(vol_ratio, 2), 'ret5': round(ret5 * 100, 2),
        'pos_size': pos['shares'], 'pos_value': pos['position_value'], 'pos_pct': pos_pct,
        'ai': ai, 'ml': ml, 'hourly': hourly, 'tags': [],
        '_stats': stats,
        'signal_age_days': round(signal_age_days, 1) if signal_age_days else 0.0,
        'warnings': ['BEARISH_DIVERGENCE'] if divergence == 'BEARISH' else [],
    }

# ─── CLI Args ──────────────────────────────────────────────────────────────
def parse_args():
    stocks = DEFAULT_STOCKS
    use_ai = False
    use_trailing = False
    sector_cap = False
    fundamental_filter = False
    output_format = 'default'
    index_override = None
    level_mode = 'intraday'
    top_n = None
    auto_retrain = False  # v17
    filter_neg_hist = False  # v17: hide stocks with negative backtest history
    backtest_first = False    # v17: run backtest before scan to refresh stats
    conversation_label = None   # passed to Telegram header

    args = sys.argv[1:]
    i = 0
    while i < len(args):
        arg = args[i]
        if arg == '--ai':
            use_ai = True; i += 1
        elif arg == '--trailing':
            use_trailing = True; i += 1
        elif arg == '--sector-cap':
            sector_cap = True; i += 1
        elif arg == '--fundamentals':
            fundamental_filter = True; i += 1
        elif arg == '--filter-neg-hist':  # v17: filter negative-history stocks from output
            filter_neg_hist = True; i += 1
        elif arg == '--backtest-first':  # v17: run backtest before scan
            backtest_first = True; i += 1
        elif arg == '--auto-retrain':  # v17: retrain ML models on sklearn version mismatch
            auto_retrain = True; i += 1
        elif arg in ('--intraday', '--swing', '--tight'):
            level_mode = 'intraday_tight' if arg == '--tight' else ('intraday' if arg == '--intraday' else 'swing'); i += 1
        elif arg == '--conversation':
            if i + 1 < len(args):
                conversation_label = args[i + 1]; i += 2
            else: i += 1
        elif arg == '--default-mode':
            if i + 1 < len(args):
                if args[i+1].lower() in ('intraday', 'swing', 'intraday_tight'):
                    level_mode = args[i+1].lower()
                i += 2
            else: i += 1
        elif arg == '--top':
            if i + 1 < len(args):
                try: top_n = int(args[i + 1])
                except: top_n = 3
                i += 2
            else: i += 1
        elif arg == '--json':
            output_format = 'json'; i += 1
        elif arg == '--format':
            if i + 1 < len(args):
                fmt = args[i + 1].lower()
                if fmt in ('telegram', 'tg'):
                    output_format = 'telegram'
                elif fmt == 'json':
                    output_format = 'json'
                elif fmt == 'table':
                    output_format = 'table'
                i += 2
            else: i += 1
        elif arg == '--index':
            if i + 1 < len(args):
                idx = args[i + 1].lower()
                if idx == 'nifty50': index_override = NIFTY50_STOCKS
                elif idx == 'nifty100': index_override = NIFTY100_STOCKS
                i += 2
            else: i += 1
        elif arg == '--symbols':
            if i + 1 < len(args):
                stocks = [s.strip().upper() for s in args[i + 1].split(',')]; i += 2
            else: i += 1
        elif arg.startswith('--'):
            i += 1
        else:
            stocks = [s.strip().upper() for s in arg.split(',')]; i += 1

    if index_override:
        stocks = index_override

    return stocks, use_ai, use_trailing, sector_cap, fundamental_filter, output_format, level_mode, top_n, auto_retrain, filter_neg_hist, backtest_first, conversation_label

# ─── Telegram Format (v17: Cat C split, SWING-first in BULLISH, tags shown) ──
def format_telegram(results, today, top_n=None, conversation_label=None):
    buy = [r for r in results if r['signal'] == 'BUY']
    sell = [r for r in results if r['signal'] == 'SELL']
    range_list = [r for r in results if r['signal'] == 'RANGE']

    # Market regime (computed before _categorize so it can be passed)
    bullish_count = sum(1 for r in results if (r.get('ai') or {}).get('outlook') == 'BULLISH')
    bear_count = sum(1 for r in results if (r.get('ai') or {}).get('outlook') == 'BEARISH')
    regime = "BULLISH" if bullish_count > bear_count else ("BEARISH" if bear_count > bullish_count else "NEUTRAL")

    cat_a, cat_a_minus, cat_b, cat_c1, cat_c2, cat_d, watchlist = _categorize(results, regime=regime)

    long_a=sum(1 for r in cat_a if r['signal']=='BUY'); short_a=sum(1 for r in cat_a if r['signal']=='SELL')
    long_a_m=sum(1 for r in cat_a_minus if r['signal']=='BUY'); short_a_m=sum(1 for r in cat_a_minus if r['signal']=='SELL')
    long_b=sum(1 for r in cat_b if r['signal']=='BUY'); short_b=sum(1 for r in cat_b if r['signal']=='SELL')
    long_c1=sum(1 for r in cat_c1 if r['signal']=='BUY'); short_c1=sum(1 for r in cat_c1 if r['signal']=='SELL')
    long_c2=sum(1 for r in cat_c2 if r['signal']=='BUY'); short_c2=sum(1 for r in cat_c2 if r['signal']=='SELL')

    conv_tag = f"📋 {conversation_label} | " if conversation_label else ""
    out = f"{conv_tag}🗓️ {today}\n"
    out += f"📊 Regime: {'🟢' if regime=='BULLISH' else '🔴' if regime=='BEARISH' else '🟡'} {regime} (📈{bullish_count} | 📉{bear_count})\n"
    c2_total = long_c2 + short_c2
    out += f"📦 NIFTY50: {len(results)} | CatA📈{long_a}/📉{short_a} | CatA-📈{long_a_m}/📉{short_a_m} | CatB🤖{long_b}/📉{short_b} | CatC1📈{long_c1}/📉{short_c1} | CatC2📊{c2_total} | CatD📉 | WL📋{len(watchlist)}\n\n"

    def _inv(price, sl, t1, t2):
        return {
            'sl': round(price + (price - sl), 0),
            't1': round(price - (t1 - price), 0),
            't2': round(price - (t2 - price), 0),
        }

    def _round_lev(d):
        return {k: round(v, 0) for k, v in d.items()}

    def fmt_levels_hr(r, sig):
        """Compute hourly ATR + pivot-based levels for display."""
        h = r.get('hourly')
        if not h:
            # Fallback to tight/intraday levels
            tk_sl=r.get('sl_tight'); tk_t1=r.get('t1_tight'); tk_t2=r.get('t2_tight')
            id_sl=r.get('sl_intraday',tk_sl); id_t1=r.get('t1_intraday',tk_t1); id_t2=r.get('t2_intraday',tk_t2)
            sw_sl=r.get('sl_swing'); sw_t1=r.get('t1_swing'); sw_t2=r.get('t2_swing')
            if sig == 'SELL':
                return _round_lev(_inv(r['price'], id_sl, id_t1, id_t2)), _round_lev(_inv(r['price'], sw_sl, sw_t1, sw_t2))
            return {'sl': round(id_sl,1), 't1': round(id_t1,0), 't2': round(id_t2,0)}, {'sl': round(sw_sl,1), 't1': round(sw_t1,0), 't2': round(sw_t2,0)}
        price = r['price']
        h_atr = h['hourly_atr']
        per_hr = h['per_hr']
        if sig == 'SELL':
            hr_l = {
                'sl': round(price + h_atr * 1.0, 0),
                't1': round(price - h_atr * 0.5, 0),
                't2': round(price - h_atr * 1.0, 0),
            }
        else:
            hr_l = {
                'sl': round(price - h_atr * 1.0, 0),
                't1': round(price + h_atr * 0.5, 0),
                't2': round(price + h_atr * 1.0, 0),
            }
        # SWING levels
        sw_sl=r.get('sl_swing'); sw_t1=r.get('t1_swing'); sw_t2=r.get('t2_swing')
        if sig == 'SELL':
            sw_l = _round_lev(_inv(r['price'], sw_sl, sw_t1, sw_t2))
        else:
            sw_l = {'sl': round(sw_sl,1), 't1': round(sw_t1,0), 't2': round(sw_t2,0)}
        return hr_l, sw_l

    def fmt_stock_short(r):
        stats = r.get('_stats', {})
        sig = r['signal']
        dir_icon = "📈" if sig == 'BUY' else "📉"
        rr = stats.get('realized_return', 0)
        wr = stats.get('win_rate', 0)
        tags = r.get('tags', [])
        confluence = r.get('_confluence', 0)
        rsi = r.get('rsi', 0)
        hr_l, sw_l = fmt_levels_hr(r, sig)
        h = r.get('hourly') or {}
        per_hr = h.get('per_hr', 0)
        # BEARISH: HOURLY-first
        # BULLISH: SWING first, HOURLY second
        if regime == 'BEARISH':
            tline = f"  {dir_icon} {r['symbol']} ₹{r['price']:,.0f} | RSI:{rsi:.0f} | Conf:{r['prob']}% | RR:{rr:+.0f}% WR:{wr:.0f}% | CF:{confluence}/10\n"
            tline += f"     💠 SL:{hr_l['sl']:.0f} T1:{hr_l['t1']:.0f} T2:{hr_l['t2']:.0f} | ~₹{per_hr}/hr\n"
            tline += f"     🎯 SL:{sw_l['sl']:.0f} T1:{sw_l['t1']:.0f} T2:{sw_l['t2']:.0f}"
        else:
            tline = f"  {dir_icon} {r['symbol']} ₹{r['price']:,.0f} | RSI:{rsi:.0f} | Conf:{r['prob']}% | RR:{rr:+.0f}% WR:{wr:.0f}% | CF:{confluence}/10\n"
            tline += f"     🎯 SL:{sw_l['sl']:.0f} T1:{sw_l['t1']:.0f} T2:{sw_l['t2']:.0f}\n"
            tline += f"     💠 SL:{hr_l['sl']:.0f} T1:{hr_l['t1']:.0f} T2:{hr_l['t2']:.0f} | ~₹{per_hr}/hr"
        if tags:
            tag_str = " ".join(tags)
            tline += f" [{tag_str}]"
        return tline

    def fmt_cat_block(label, cat_list, max_items=10, sort_key=None):
        if not cat_list:
            return f"{label} [0]: —\n\n"
        # v19: Cat A/B sorted by RR desc (positive RR stocks first → TOP_PICKs bubble up)
        if sort_key == 'rr_desc':
            sorted_s = sorted(cat_list, key=lambda x: ( -x.get('_stats',{}).get('realized_return',0), -x.get('_confluence',0) ))[:max_items]
        else:
            sorted_s = sorted(cat_list, key=lambda x: -x.get('prob', 0))[:max_items]
        lines = [fmt_stock_short(r) for r in sorted_s]
        # v19: show HEDGE label if this block has shorts
        has_short = any(r['signal'] == 'SELL' for r in cat_list)
        label_out = label
        if has_short and 'A-' in label:
            label_out = label.replace('2-of-3 CONFIRMED (level mismatch)', '2-of-3 CONFIRMED + HEDGE')
        return f"{label_out} [{len(cat_list)}]\n" + "\n".join(lines) + "\n\n"

    # Regime context note
    if regime == 'BULLISH' and short_a > 0:
        out += "ℹ️ Cat A SHORTs in BULLISH = sector-specific bet (hedge against rotation). SWING targets shown first.\n\n"

    out += f"{'─'*60}\n\n"
    # v19: Cat A/B sorted by RR desc so ⭐ TOP_PICKs (positive RR + high CF) bubble to top
    out += fmt_cat_block("📈 ✅ Cat A — HIGHEST QUALITY", cat_a, sort_key='rr_desc')
    out += fmt_cat_block("📈 ⚠️ Cat A- — 2-of-3 CONFIRMED + HEDGE", cat_a_minus, sort_key='rr_desc')
    out += fmt_cat_block("🤖 Cat B — AI CONFIRMED", cat_b, sort_key='rr_desc')
    out += fmt_cat_block("📊 Cat C1 — SIGNAL + AI AGREE (SHORT)", cat_c1)
    out += fmt_cat_block("📊 Cat C2 — SIGNAL ONLY", cat_c2)
    out += fmt_cat_block("📋 WATCHLIST — RANGE BOUND", watchlist)

    # v19: ⭐ TOP_PICK logic — positive RR + CF>=8.0 gets starred, sorted first in top list
    # (set in _categorize; we just use it here for display)
    top = top_n if top_n else 3
    out += f"{'─'*60}\n"
    short_count = len([r for r in results if r['signal'] == 'SELL'])
    out += f"📋 SUMMARY: Total:{len(results)} | 📈LONG:{len(buy)} | 📉SHORT:{short_count} | ➡️RANGE:{len(range_list)}\n\n"

    def fmt_top_pick(r, sig):
        stats = r.get('_stats', {})
        rr = stats.get('realized_return', 0)
        wr = stats.get('win_rate', 0)
        tags = r.get('tags', [])
        confluence = r.get('_confluence', 0)
        rsi = r.get('rsi', 0)
        hr_l, sw_l = fmt_levels_hr(r, sig)
        h = r.get('hourly') or {}
        per_hr = h.get('per_hr', 0)
        if regime == 'BEARISH':
            tline = (f"  {('📈' if sig=='BUY' else '📉')} {r['symbol']} ₹{r['price']:,.0f} | RSI:{rsi:.0f} | RR:{rr:+.0f}% WR:{wr:.0f}% | CF:{confluence}/10\n"
                     f"     💠 SL:{hr_l['sl']:.0f} T1:{hr_l['t1']:.0f} T2:{hr_l['t2']:.0f} | ~₹{per_hr}/hr\n"
                     f"     🎯 SL:{sw_l['sl']:.0f} T1:{sw_l['t1']:.0f} T2:{sw_l['t2']:.0f}")
        else:
            tline = (f"  {('📈' if sig=='BUY' else '📉')} {r['symbol']} ₹{r['price']:,.0f} | RSI:{rsi:.0f} | RR:{rr:+.0f}% WR:{wr:.0f}% | CF:{confluence}/10\n"
                     f"     🎯 SL:{sw_l['sl']:.0f} T1:{sw_l['t1']:.0f} T2:{sw_l['t2']:.0f}\n"
                     f"     💠 SL:{hr_l['sl']:.0f} T1:{hr_l['t1']:.0f} T2:{hr_l['t2']:.0f} | ~₹{per_hr}/hr")
        if tags:
            tline += " " + " ".join(tags)
        return tline

    # v19: TOP LONG — ⭐ TOP_PICK (positive RR + CF>=8) bubbles to top, then by CF
    all_longs = sorted(buy, key=lambda x: ( -x.get('_starred', False), -x.get('_confluence', 0) ))
    top_buy = all_longs[:top]
    starred_buy = [r for r in top_buy if r.get('_starred')]
    if top_buy:
        label = f"🏆 TOP 📈 LONG ({len(top_buy)}"
        if starred_buy:
            label += f" — ⭐ {len(starred_buy)} TOP_PICK{'S' if len(starred_buy)>1 else ''}"
        out += label + ")\n"
        for r in top_buy:
            out += fmt_top_pick(r, 'BUY') + "\n"  # ⭐ tag already in fmt_top_pick output
    else:
        out += "🏆 TOP 📈 LONG: none\n"

    # v19: TOP SHORT — sorted by CF desc. Label 🛡️ HEDGE only if not already in tags
    all_sells = sorted(sell, key=lambda x: -x.get('_confluence', 0))
    top_sell = all_sells[:top]
    hedge_sells = [r for r in top_sell if '🛡️ HEDGE' in r.get('tags', [])]
    if top_sell:
        label = f"\n💀 TOP 📉 SHORT ({len(top_sell)}"
        if hedge_sells:
            label += f" — 🛡️ {len(hedge_sells)} HEDGE (BULL-regime shorts)"
        out += label + ")\n"
        for r in top_sell:
            # Don't double-add 🛡️ HEDGE — fmt_top_pick already includes it via tags
            out += fmt_top_pick(r, 'SELL') + "\n"
    else:
        out += "\n💀 TOP 📉 SHORT: none\n"

    out += "\n⚠️ Not SEBI registered. Validate before trading."
    return out

# ─── JSON Format ────────────────────────────────────────────────────────────
def format_json(results, today):
    buy = [r for r in results if r['signal'] == 'BUY']
    sell = [r for r in results if r['signal'] == 'SELL']
    range_list = [r for r in results if r['signal'] == 'RANGE']
    # Compute regime
    bullish_count = sum(1 for r in results if (r.get('ai') or {}).get('outlook') == 'BULLISH')
    bear_count = sum(1 for r in results if (r.get('ai') or {}).get('outlook') == 'BEARISH')
    regime = "BULLISH" if bullish_count > bear_count else ("BEARISH" if bear_count > bullish_count else "NEUTRAL")
    cat_a, cat_a_minus, cat_b, cat_c1, cat_c2, cat_d, watchlist = _categorize(results, regime=regime)

    def sanitize(r):
        ai = r.get('ai') or {}
        ml = r.get('ml') or {}
        stats = r.get('_stats', {})
        return {
            'symbol': r['symbol'],
            'price': float(r['price']),
            'change': float(r['change']),
            'rsi': float(r['rsi']),
            'signal': r['signal'],
            'prob': float(r['prob']),
            'sl': float(r['sl']),
            't1': float(r['t1']),
            't2': float(r['t2']),
            'support': float(r['support']),
            'resistance': float(r['resistance']),
            'divergence': r.get('divergence'),
            'atr': float(r['atr']),
            'vol_ratio': float(r['vol_ratio']),
            'ret5': float(r['ret5']),
            'pos_pct': float(r['pos_pct']),
            'ai_outlook': ai.get('outlook'),
            'ai_confidence': ai.get('confidence'),
            'ai_score': float(ai.get('total_score')) if ai.get('total_score') is not None else None,
            'ai_t1': float(ai.get('stages', {}).get('6_risk_manager', {}).get('t1')) if ai else None,
            'ml_direction': ml.get('direction') if ml else None,
            'ml_confidence': float(ml.get('confidence')) if ml and ml.get('confidence') else None,
            'ml_fail_reason': ml.get('_ml_fail_reason') if ml else None,
            'win_rate': float(stats.get('win_rate', 0)),
            'realized_return': float(stats.get('realized_return', 0)),
            'sharpe': float(stats.get('sharpe', 0)),
            'confluence_score': float(r.get('_confluence', 0)),
            'level_align': r.get('_level_align', 'ALIGNED'),
            'level_gap_pct': float(r.get('_level_gap_pct', 0)),
            'signal_age_days': float(r.get('signal_age_days', 0)),
            'regime_label': r.get('_regime_label', ''),
            'warnings': r.get('warnings', []),
            'tags': r.get('tags', []),
        }

    return json.dumps({
        'timestamp': today,
        'regime': regime,
        'bullish_count': bullish_count,
        'bearish_count': bear_count,
        'total': len(results),
        'buy_count': len(buy),
        'sell_count': len(sell),
        'range_count': len(range_list),
        'categories': {
            'cat_a': [sanitize(r) for r in cat_a],
            'cat_a_minus': [sanitize(r) for r in cat_a_minus],
            'cat_b': [sanitize(r) for r in cat_b],
            'cat_c1': [sanitize(r) for r in cat_c1],
            'cat_c2': [sanitize(r) for r in cat_c2],
            'cat_d': [sanitize(r) for r in cat_d],
            'watchlist': [sanitize(r) for r in watchlist],
        },
        'buy': [sanitize(r) for r in sorted(buy, key=lambda x: -x['prob'])],
        'sell': [sanitize(r) for r in sorted(sell, key=lambda x: -x['prob'])],
        'range': [sanitize(r) for r in range_list],
    }, indent=2)

# ─── Main ───────────────────────────────────────────────────────────────────
def main():
    (stocks, use_ai, use_trailing, sector_cap, fundamental_filter,
     output_format, level_mode, top_n, auto_retrain,
     filter_neg_hist, backtest_first, conversation_label) = parse_args()
    today = datetime.now().strftime("%d %b %Y %I:%M %p IST")

    # ── Backtest first (optional) ────────────────────────────────────────
    if backtest_first and use_ai:
        print("\n🔄 Running backtest to refresh stats before scan...")
        import subprocess, sys as _sys
        result = subprocess.run(
            ['python3', 'backtest.py', '--all', '--years', '3'],
            capture_output=True, text=True, timeout=600,
            cwd=os.path.dirname(os.path.abspath(__file__))
        )
        if result.returncode == 0:
            # Print last 20 lines of backtest output
            lines = result.stdout.strip().split('\n')
            print('\n'.join(lines[-20:]))
        else:
            print(f"Backtest warning: {result.stderr.strip()[:200]}")
        print()

    sector_counts = {}
    results = []

    for sym in stocks:
        r = analyze(sym, use_ai=use_ai, use_trailing=use_trailing,
                    fundamental_filter=fundamental_filter,
                    level_mode=level_mode, auto_retrain=auto_retrain)
        if r and r['price'] and r['price'] > 0:
            if filter_neg_hist:
                stats = r.get('_stats', {})
                if stats.get('realized_return', 0) < 0:
                    continue  # skip historically money-losing stocks
            if sector_cap and r['signal'] in ('BUY', 'SELL'):
                sector = get_sector(sym)
                if check_sector_limit(sector, sector_counts, MAX_PER_SECTOR):
                    r['_sector_skipped'] = sector
                else:
                    sector_counts[sector] = sector_counts.get(sector, 0) + 1
            results.append(r)

    if fundamental_filter:
        results = filter_by_fundamentals(results)
        results = [r for r in results if r.get('fundamental_ok', True)]

    if output_format == 'json':
        print(format_json(results, today))
        return

    if output_format == 'telegram' and use_ai:
        print(format_telegram(results, today, top_n=top_n, conversation_label=conversation_label))
        return

    if output_format == 'table':
        from table_format import run_table_format
        print(run_table_format(results, today))
        return

    # Default verbose output
    print("=" * 70)
    tags = []
    if use_ai: tags.append("+AI/ML")
    if use_trailing: tags.append("+TrailingSL")
    if sector_cap: tags.append("+SectorCap")
    if fundamental_filter: tags.append("+Fundamentals")
    if top_n: tags.append(f"--top{top_n}")
    tag = f" ({', '.join(tags)})" if tags else ""
    print(f"📊 NIFTY SCANNER v17{tag} | {today}")
    print("=" * 70)

    buy = sorted([r for r in results if r['signal'] == 'BUY'], key=lambda x: -x['prob'])
    sell = sorted([r for r in results if r['signal'] == 'SELL'], key=lambda x: -x['prob'])

    def print_stock(r, label="BUY"):
        sec_tag = f" [SKIP:{r.get('_sector_skipped')}]" if '_sector_skipped' in r else ""
        div_tag = f" [{r['divergence']}]" if r['divergence'] else ""
        tags = r.get('tags', [])
        tag_str = f" {' '.join(tags)}" if tags else ""
        ai = r.get('ai') or {}
        ml = r.get('ml') or {}
        print(f"  {r['symbol']} ₹{r['price']:,.2f} | {r['prob']}% | RSI:{r['rsi']} | {r['change']:+.2f}%{div_tag}{sec_tag}{tag_str}")
        print(f"    Entry:₹{r['price']} SL:₹{r['sl']} T1:₹{r['t1']} T2:₹{r['t2']} | S:₹{r['support']} R:₹{r['resistance']}")
        print(f"    Conf:{r['buy_cnt' if label=='BUY' else 'sell_cnt']}/7 | Vol:{r['vol_ratio']}x | Mom:{r['ret5']:+.1f}% | Pos:{r['pos_pct']}%")
        if fundamental_filter:
            print(f"    Fundamentals: Score:{r.get('fundamental_score',0)} | PE:{r.get('pe','-')} | MCap:{r.get('mcap','-')} | Div:{r.get('div_yield','-')}")
        if use_ai and ai:
            print(f"    🤖 AI: {ai.get('outlook')}({ai.get('confidence')}) Score:{ai.get('total_score')}")
            v = ai.get('stages', {}).get('4_trade_validator', {})
            print(f"       Validator: {v.get('value')} | S:₹{v.get('support')} R:₹{v.get('resistance')}")
            e = ai.get('stages', {}).get('7_execution', {})
            print(f"       Exec: {e.get('timing')} ({e.get('type')})")
        if use_ai and ml:
            fail = ml.get('_ml_fail_reason', '')
            if fail:
                print(f"    🧠 ML: {fail}")
            else:
                print(f"    🧠 ML: {ml.get('direction')}({ml.get('confidence')}%)")
        if r.get('warnings'):
            print(f"    ⚠️  {r['warnings']}")

    print(f"\n📈 BUY ({len(buy)})")
    print("-" * 70)
    for r in buy:
        print_stock(r, "BUY")

    print(f"\n📉 SELL ({len(sell)})")
    print("-" * 70)
    for r in sell:
        print_stock(r, "SELL")

    total_filtered = len(results)
    skipped_sector = sum(1 for r in results if '_sector_skipped' in r)
    skipped_fund = sum(1 for r in results if not r.get('fundamental_ok', True)) if fundamental_filter else 0
    print(f"\n📊 SUMMARY: Total={total_filtered} | BUY={len(buy)} | SELL={len(sell)} | RANGE={total_filtered-len(buy)-len(sell)}")
    if sector_cap: print(f"   Sector filtered: {skipped_sector}")
    if fundamental_filter: print(f"   Fundamental filtered: {skipped_fund}")
    print("⚠️ Not SEBI registered. Validate before trading.")

if __name__ == "__main__":
    main()