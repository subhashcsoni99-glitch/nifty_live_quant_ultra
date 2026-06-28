#!/usr/bin/env python3
"""
NIFTY Scanner v52 - Unified Rule-Based + AI (9-stage) + ML

v25 UPGRADE (achieving 9.5/10 rating):
  1. WR badge: 🟢 >50% | 🟡 40-50% | 🔴 <40% on every stock line
  2. LOW_WINRATE tag: ⚠️ LOW_WINRATE for WR < 40% stocks
  3. Raised Cat A/B WR thresholds: Cat A requires WR >= 45%, Cat B >= 40%
  4. TOP_PICK requires WR >= 45% (was CF>=8 + RR>0 only)
  5. Backtest stale threshold: > 3 days (was 7)
  6. Position sizing warning: ⚠️ OVERSIZE(n%) if pos > 5% of capital
  7. WR distribution summary: 📈 WR Quality: 🟢 | 🟡 | 🔴 | ❓
  8. Cat D excluded from TOP_PICKS
  9. Position sizing recommendation in footer
  10. Quality gates explanation in footer

v24 changes (7.2→9.5/10 review):
  1. BEAR_REGIME_SL_FACTOR=1.2 (was 0.8 — wider stops in bear markets)
  2. train.py --index nifty100: trains all NIFTY100 models
  3. backtest.py/forward_test.py: get_signal delegated to nifty_core
  4. Auto sector: M&M MARUTI EICHERMOT TATAMOTORS added
  5. NIFTY100_STOCKS deduplicated via set arithmetic
  6. --debug flag: confluence component breakdown per stock
  7. forward_test.py v9: signal via nifty_core + --index nifty100
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

from nifty_categorize import (
    categorize_results as _categorize,
    get_primary_trigger as _get_primary_trigger,
    _wr_badge, _stale_tier, _is_neg_hist, _is_poor_history,
    _regime_coherence, calc_confluence_score, _add_tag,
    _pos_size_warning,
)
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

# ─── Market Hours (NEW-1/NEW-2) ────────────────────────────────────────────
# NSE market hours: 9:00 AM – 3:30 PM IST
MARKET_OPEN_HOUR,   MARKET_OPEN_MIN   = 9,  0    # market open
MARKET_CLOSE_HOUR,  MARKET_CLOSE_MIN  = 15, 30   # NEW-2: configurable (handle early closes)
_MARKET_OPEN_MINS  = MARKET_OPEN_HOUR * 60 + MARKET_OPEN_MIN     # 540
_MARKET_CLOSE_MINS = MARKET_CLOSE_HOUR * 60 + MARKET_CLOSE_MIN   # 930

def _is_market_open():
    """NEW-1: Check if NSE is currently open. Returns (bool, reason_str)."""
    from datetime import datetime as _dt, timezone, timedelta
    IST = timezone(timedelta(hours=5, minutes=30))
    now_ist = _dt.now(IST)
    now_mins = now_ist.hour * 60 + now_ist.minute
    if now_mins < _MARKET_OPEN_MINS:
        return False, "pre-market"
    elif now_mins >= _MARKET_CLOSE_MINS:
        return False, "post-market"
    else:
        return True, "live"


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
_STATS_TTL = 300  # NEW-3: was 3600 (1hr) → 300s (5min) for always-fresh stats

def _get_stats(symbol):
    """Load stats from most recent backtest JSON. Cache expires after _STATS_TTL seconds."""
    global _STATS_CACHE, _STATS_LOADED_AT
    now = _time.time()
    if _STATS_CACHE is None or (_STATS_LOADED_AT and (now - _STATS_LOADED_AT) > _STATS_TTL):
        _STATS_CACHE = {}
        try:
            all_files = sorted(
                [f for f in os.listdir(MODEL_DIR)
                 if f.startswith('backtest_v') and f.endswith('.json')],
                key=lambda x: os.path.getmtime(os.path.join(MODEL_DIR, x)),
                reverse=True
            )
            for fname in all_files:
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
        'win_rate': 0, 'realized_return': 0, 'sharpe': 0, 'max_drawdown': 999,
        'no_backtest': True,
    })

# ─── Confluence Scoring (1-10) ──────────────────────────────────────────────────
# v18: NEG_HIST now requires rr < -2% AND wr < 35% (was rr < 0 AND wr <= 50%)
# Cat B threshold: wr >= 35%, rr >= -2% (was wr >= 40%, rr >= 0)
_NEG_HIST_RR_THRESHOLD = -2.0
_NEG_HIST_WR_THRESHOLD = 40  # v18fix: was 35 — COALINDIA wr=35.5/rr=-4.5 was slipping through as Cat B

# ── v35: ADX FILTER DEFAULT + LOWER WR GATE ───────────────────────────────
_MIN_WR_CAT_A = 45        # v36 P1-E: was 25, raised to 45 (top 10-15% by WR — cleaner TOP_PICKs)
_MIN_WR_CAT_B = 35        # v35: keep at 35
_MIN_WR_TOP_PICK = 40     # TOP_PICK still requires WR >= 40%
_BACKTEST_STALE_DAYS = 3   # warn if backtest > 3 days old
_LOW_WR_WARNING = 35       # tag stocks with WR < 35% as LOW_WINRATE
_POS_SIZE_WARNING_PCT = 10  # warn if position > 10% of capital

# ── v35: ADX filter ON by default (Option A = recommended) ─────────────────
# Override with --no-adx flag to disable
_DEFAULT_ADX_ENABLED = True  # v35: ADX>25 enabled by default
def get_level_modes_extended(price, atr, atr5=None, entry_price=None, signal=None, today_open=None):
    """Return all level sets for a stock.
    v49 FIX: Targets use CURRENT PRICE as entry (not today's open) so they're always achievable.
             Today's open only used for regime detection, not target calculation.
    v47: today_open was used as entry for tight intraday targets — but this caused targets
         to be below current price when stock had already moved up from open (impossible targets).
         FIX: use current `price` as intraday entry so T1/T2 are always reachable from now.
    v43: tight mode: T1=ATR(5), T2/T3=ATR(14)
    v45: BUY/SELL directional levels (targets below entry for SELL)
    """
    if atr5 is None:
        atr5 = atr  # fallback
    # v49 FIX: use current price as intraday entry (always reachable)
    # v50 FIX: T1=0.5×ATR5 (tight scalp), T2=1.0×ATR14 (moderate), T3=1.5×ATR14 (full)
    # v43: tight mode: T1=ATR(5), T2/T3=ATR(14) — now with different multipliers
    intra_entry = price
    swing_entry = entry_price if entry_price is not None else price
    scalp_t1 = atr5 * 0.5    # v50: T1 very tight — half ATR(5)
    scalp_t2 = atr * 1.0     # v50: T2 moderate — full ATR(14)
    scalp_t3 = atr * 1.5     # v50: T3 full — 1.5× ATR(14)
    if signal == 'SELL':
        tight_levels = {
            'sl': round(intra_entry + atr * 1.5, 2),
            't1': round(intra_entry - scalp_t1, 2),    # v50: 0.5×ATR5 — tightest target
            't2': round(intra_entry - scalp_t2, 2),    # v50: 1.0×ATR14 — moderate target
            't3': round(intra_entry - scalp_t3, 2),    # v50: 1.5×ATR14 — full target
        }
        swing_levels = {
            'sl': round(swing_entry + atr * 1.5, 2),
            't1': round(swing_entry - scalp_t1, 2),
            't2': round(swing_entry - scalp_t2, 2),
        }
    else:
        tight_levels = {
            'sl': round(intra_entry - atr * 1.5, 2),
            't1': round(intra_entry + scalp_t1, 2),    # v50: 0.5×ATR5 — tight scalp
            't2': round(intra_entry + scalp_t2, 2),    # v50: 1.0×ATR14 — moderate
            't3': round(intra_entry + scalp_t3, 2),    # v50: 1.5×ATR14 — full
        }
        swing_levels = {
            'sl': round(swing_entry - atr * 1.5, 2),
            't1': round(swing_entry + scalp_t1, 2),
            't2': round(swing_entry + scalp_t2, 2),
        }
    return {
        'tight':   tight_levels,
        'regular': calc_levels(price, atr, mode='intraday'),
        'swing':   swing_levels,
    }

# ─── Analyze Single Stock ──────────────────────────────────────────────────
def analyze(sym, use_ai=False, use_trailing=False, fundamental_filter=False, level_mode='intraday', auto_retrain=False, momentum_mode=False, high_conviction_mode=False):
    price, prev = get_price(sym)
    df = get_ohlc(sym)
    if df is None or price is None or len(df) < 200:
        return None

    # v47: fetch today's open to use as intraday entry (not stale prev_close)
    today_open = None
    try:
        # get_ohlc() is daily timeframe — today row has the actual day open
        df_today = df[df.index.date == pd.Timestamp('today').date()]
        if len(df_today) >= 1:
            today_open = float(df_today['Open'].iloc[0])
    except Exception:
        today_open = None

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
    atr5 = float(df['atr5'].iloc[-1]) if 'atr5' in df.columns and not pd.isna(df['atr5'].iloc[-1]) else atr
    change_pct = ((price - prev) / prev * 100) if prev and prev > 0 else 0

    # v44: market condition from ADX + MA
    adx = float(df['adx'].iloc[-1]) if 'adx' in df.columns and not pd.isna(df['adx'].iloc[-1]) else 0
    di_plus = float(df['adx_di_plus'].iloc[-1]) if 'adx_di_plus' in df.columns and not pd.isna(df['adx_di_plus'].iloc[-1]) else 0
    di_minus = float(df['adx_di_minus'].iloc[-1]) if 'adx_di_minus' in df.columns and not pd.isna(df['adx_di_minus'].iloc[-1]) else 0
    ma20 = float(df['ma20'].iloc[-1]) if 'ma20' in df.columns and not pd.isna(df['ma20'].iloc[-1]) else 0
    ma50 = float(df['ma50'].iloc[-1]) if 'ma50' in df.columns and not pd.isna(df['ma50'].iloc[-1]) else 0
    ma100 = float(df['ma100'].iloc[-1]) if 'ma100' in df.columns and not pd.isna(df['ma100'].iloc[-1]) else 0
    # Market condition
    if adx < 20:   condition = 'FLAT'
    elif adx < 25: condition = 'CHOPPY'
    elif di_plus > di_minus and price > ma20 and price > ma50: condition = 'BULL'
    elif di_minus > di_plus and price < ma20 and price < ma50: condition = 'BEAR'
    elif di_plus > di_minus: condition = 'BULL_PULLBACK'
    elif di_minus > di_plus: condition = 'BEAR_RALLY'
    else: condition = 'NEUTRAL'

    sig_val, meta, _ = core_get_signal(df, len(df) - 1, momentum_mode=momentum_mode, high_conviction_mode=high_conviction_mode)
    signal = meta['signal']
    divergence = meta.get('divergence')

    # v44: prev_close used as baseline | v47: today_open used for tight intraday targets (gap-aware entry)
    # v45: pass signal for directional SELL targets | v47: pass today_open for live intraday entry
    levels = get_level_modes_extended(price, atr, atr5=atr5, entry_price=prev, signal=signal, today_open=today_open)
    if _OPTION_C_ACTIVE:
        levels['swing'] = {'sl': round(price - atr * 0.75, 0), 't1': round(price + atr * 1.5, 0), 't2': round(price + atr * 2.5, 0)}
    sr = calc_support_resistance(df)
    pos = calc_position_size(100000, price, atr * ATR_CONFIG[level_mode]['sl'])
    pos_pct = round((pos['position_value'] / 100000) * 100, 1)

    prob_buy  = min(95, 45 + meta['buy_cnt']  * 10)   # v44: wider spread, base 45→85% at cnt=4, 95% at cnt=5
    prob_sell = min(95, 45 + meta['sell_cnt'] * 10)
    prob = prob_buy if signal == 'BUY' else (prob_sell if signal == 'SELL' else max(prob_buy, prob_sell))
    reasons = meta.get('reasons', [])

    ai = None
    ml = None
    hourly = None
    if use_ai:
        ai = ai_opinion_pipeline(sym, price, rsi, macd, macd_sig, atr, vol_ratio, ret5, df, momentum_mode=momentum_mode)
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

    # ── WR Gate: only show BUY/SELL for high-conviction stocks ──
    # v52: HC mode + backtest data shows 55% is statistically achievable with 20+ trades
    # (75% needs 15/15 wins — mathematically nearly impossible for a mean-reversion strategy)
    WR_HIGH_THRESHOLD = 55 if high_conviction_mode else 75
    stats_wr = stats.get('win_rate', 0)
    wr_gate_tags = []
    if signal in ('BUY', 'SELL') and stats_wr < WR_HIGH_THRESHOLD:
        wr_gate_tags.append(f'⚠️ WR_LOW({stats_wr:.0f}%<{WR_HIGH_THRESHOLD}%)')
        signal = 'RANGE'
        prob = max(prob_buy, prob_sell)  # downgrade to neutral confidence

    return {
        'symbol': sym, 'price': price, 'prev': prev,
        'rsi': round(rsi, 1), 'change': round(change_pct, 2),
        'signal': signal, 'prob': prob,
        'condition': condition, 'adx': round(adx, 1),
        'di_plus': round(di_plus, 1), 'di_minus': round(di_minus, 1),
        'sl_tight': levels['tight']['sl'], 't1_tight': levels['tight']['t1'], 't2_tight': levels['tight']['t2'], 't3_tight': levels['tight']['t3'],
        'sl_intraday': levels['regular']['sl'], 't1_intraday': levels['regular']['t1'], 't2_intraday': levels['regular']['t2'],
        'sl_swing': levels['swing']['sl'], 't1_swing': levels['swing']['t1'], 't2_swing': levels['swing']['t2'],
        'support': sr['support'], 'resistance': sr['resistance'],
        'buy_cnt': meta['buy_cnt'], 'sell_cnt': meta['sell_cnt'],
        'divergence': divergence, 'reasons': reasons,
        'atr': round(atr, 2), 'atr5': round(atr5, 2), 'vol_ratio': round(vol_ratio, 2), 'ret5': round(ret5 * 100, 2),
        'pos_size': pos['shares'], 'pos_value': pos['position_value'], 'pos_pct': pos_pct,
        'ai': ai, 'ml': ml, 'hourly': hourly, 'tags': wr_gate_tags,
        '_stats': stats,
        'today_open': today_open,   # v47: today's open for gap-aware intraday targets
        'signal_age_days': round(signal_age_days, 1) if signal_age_days else 0.0,
        'warnings': ['BEARISH_DIVERGENCE'] if divergence == 'BEARISH' else [],
        'primary_trigger': _get_primary_trigger(meta, rsi, divergence, signal),
    }

# ─── Global Option Flags ─────────────────────────────────────────────────
_OPTION_C_ACTIVE = False  # v34: True when --C passed (morning window + tight SL)

# ─── CLI Args ──────────────────────────────────────────────────────────────
def parse_args():
    _MAX_POS_PCT_OVERRIDE = None  # v38 P1-6
    stocks = DEFAULT_STOCKS
    use_ai = False
    use_trailing = False
    momentum_mode = False   # v34: Option B — RSI>70/<30 momentum mode
    high_conviction_mode = False  # v52: HC signal mode
    adx_filter = True     # v34: Option A — ADX>25 filter (default ON)
    sector_cap = False
    fundamental_filter = False
    output_format = 'default'
    index_override = None
    level_mode = 'swing'
    top_n = None
    auto_retrain = False  # v17
    filter_neg_hist = False  # v17: hide stocks with negative backtest history
    backtest_first = False    # v17: run backtest before scan to refresh stats
    conversation_label = None   # passed to Telegram header
    debug_mode = False         # v22: print confluence component breakdown per stock
    wait_morning = False  # v35: sleep until 9:40 AM IST before scanning
    stream_output = False  # BUG-3: per-stock output, no buffering

    args = sys.argv[1:]
    i = 0
    while i < len(args):
        arg = args[i]
        if arg == '--ai':
            use_ai = True; i += 1
        elif arg == '--trailing':
            use_trailing = True; i += 1
        elif arg.upper() in ('A', 'B', 'C', 'A,B', 'B,C', 'A,C', 'A,B,C'):
            opts = arg.upper().split(',')
            if 'A' in opts: adx_filter = True
            else: adx_filter = False
            if 'B' in opts: momentum_mode = True
            if 'C' in opts:
                global _OPTION_C_ACTIVE
                _OPTION_C_ACTIVE = True
            i += 1
        elif arg == '--no-adx-filter':
            adx_filter = False; i += 1
        elif arg == '--momentum-mode':
            momentum_mode = True; i += 1
        elif arg == '--hc':   high_conviction_mode = True; i += 1  # v52
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
        elif arg == '--debug':
            debug_mode = True; i += 1
        elif arg == '--strict':
            _MIN_WR_CAT_A = 40
            _MIN_WR_CAT_B = 35
            i += 1
        elif arg == '--max-pos-pct':
            if i+1 < len(args):
                try:
                    _MAX_POS_PCT_OVERRIDE = float(args[i+1])
                    print(f"  Position cap: {_MAX_POS_PCT_OVERRIDE}%")
                except:
                    print(f"Invalid --max-pos-pct value: {args[i+1]}")
                i += 2
            else:
                i += 1
        elif arg == '--wait-morning':
            wait_morning = True; i += 1
        elif arg == '--stream':
            stream_output = True; i += 1   # BUG-3: per-stock output, no buffering
        elif arg.startswith('--'):
            i += 1
        else:
            stocks = [s.strip().upper() for s in arg.split(',')]; i += 1

    if index_override:
        stocks = index_override

    return stocks, use_ai, use_trailing, momentum_mode, high_conviction_mode, sector_cap, fundamental_filter, output_format, level_mode, top_n, auto_retrain, filter_neg_hist, backtest_first, conversation_label, debug_mode, wait_morning, _MAX_POS_PCT_OVERRIDE, stream_output

# ─── Telegram Format (v17: Cat C split, SWING-first in BULLISH, tags shown) ──
def format_telegram(results, today, top_n=None, conversation_label=None, max_pos_pct=None, level_mode='swing', choppy=False):
    # NEW-1: suppress BUY/SELL signals when market is closed
    market_open, market_reason = _is_market_open()
    # v49: choppy → show only T1 target (scalp), not T2/T3
    # trending → show full T1/T2/T3 as designed
    # sideways → show T1/T2, skip T3
    # Override the display mode so nested fmt functions use the right level
    if choppy:
        level_mode = 'intraday_tight'   # choppy = scalp only (T1)
    # else keep the user-specified level_mode (swing/tight/intraday)
    top = top_n if top_n else 3
    buy = [r for r in results if r['signal'] == 'BUY']
    sell = [r for r in results if r['signal'] == 'SELL']
    range_list = [r for r in results if r['signal'] == 'RANGE']

    # v34: 4-tier Index Regime (NIFTY50 MA-based) + A/D breadth + trading hours
    # P1-2 Fix: use ADX to confirm BEARISH — price below MA5+MA20 AND ADX>20 required
    # v49 FIX: Regime uses TODAY'S OPEN as intraday baseline (not MA20/MA50)
    # Also count stocks from YESTERDAY CLOSE (not today's open)
    try:
        import yfinance as yf
        idx = yf.Ticker("^NSEI")
        df_i = idx.history(period="5d", interval="1d")
        if len(df_i) >= 2:
            price_now = float(df_i["Close"].iloc[-1])
            price_prev = float(df_i["Close"].iloc[-2])
            ma5 = float(df_i["Close"].tail(5).mean())
            ma20 = float(df_i["Close"].tail(20).mean()) if len(df_i) >= 20 else float(df_i["Close"].mean())
            # v49: Get today's open from intraday data for regime baseline
            idx_intra = yf.Ticker("^NSEI")
            df_intra = idx_intra.history(period="5d", interval="5m")
            today_start = pd.Timestamp.now().normalize()  # today's midnight UTC
            df_today_intra = df_intra[df_intra.index >= today_start]
            if len(df_today_intra) > 0:
                today_open = float(df_today_intra["Open"].iloc[0])
            else:
                today_open = float(df_i["Open"].iloc[-1])  # fallback to daily open
            yest_close = price_prev  # yesterday's close
            nifty_df = add_features(df_i)
            adx_val = float(nifty_df['adx'].iloc[-1]) if 'adx' in nifty_df.columns and not pd.isna(nifty_df['adx'].iloc[-1]) else 25
            # v49: Count stocks from YESTERDAY CLOSE (correct for intraday)
            adv, dec = 0, 0
            for sym in NIFTY50_STOCKS:
                try:
                    t = yf.Ticker(sym + ".NS")
                    h = t.history(period="2d", interval="1d")
                    if len(h) >= 2:
                        # Count from yesterday's close
                        chg = float(h["Close"].iloc[-1]) - float(h["Close"].iloc[-2])
                        if chg > 0: adv += 1
                        elif chg < 0: dec += 1
                except: pass
        else:
            price_now, ma5, ma20, adx_val, today_open, yest_close, adv, dec = 23999, 24000, 23600, 25, 23999, 23850, 0, 0
    except Exception:
        price_now, ma5, ma20, adx_val, today_open, yest_close, adv, dec = 23999, 24000, 23600, 25, 23999, 23850, 0, 0
    # v49 INTRADAY REGIME: use today's open as primary baseline
    intraday_chg = price_now - today_open
    intraday_pct = (intraday_chg / today_open * 100) if today_open > 0 else 0
    # Secondary: from yesterday close
    from_yest = price_now - yest_close
    from_yest_pct = (from_yest / yest_close * 100) if yest_close > 0 else 0
    near_ma = abs(price_now - ma20) / ma20 * 100 < 1.0 if ma20 > 0 else False
    choppy = adx_val < 20 and near_ma
    # v49: 5-tier intraday regime based on actual price movement
    if choppy:
        regime = "CHOPPY"; regime_icon = "🔶"
    elif intraday_pct >= 0.5:
        regime = "BULLISH"; regime_icon = "🟢"
    elif intraday_pct <= -0.5:
        regime = "BEARISH"; regime_icon = "🔴"
    elif intraday_pct >= 0.2:
        regime = "SLIGHTLY_BULLISH"; regime_icon = "🟢"
    elif intraday_pct <= -0.2:
        regime = "SLIGHTLY_BEARISH"; regime_icon = "🔴"
    else:
        regime = "NEUTRAL"; regime_icon = "⚪"
    breadth = f" ({adv}↑/{dec}↓)" if (adv + dec) > 0 else ""
    bullish_count = sum(1 for r in results if (r.get('ai') or {}).get('outlook') == 'BULLISH')
    bear_count = sum(1 for r in results if (r.get('ai') or {}).get('outlook') == 'BEARISH')
    market_open, market_reason = _is_market_open()
    session_icon = "🟢 LIVE" if market_open else f"⚫ {market_reason.upper()}"

    cat_a, cat_a_minus, cat_b, cat_c1, cat_c2, cat_c2a, cat_c2b, cat_d, watchlist, bear_div_shorts, short_qualified = _categorize(results, regime=regime)
    short_q = len(short_qualified)  # P2: SHORT count for header

    long_a=sum(1 for r in cat_a if r['signal']=='BUY'); short_a=sum(1 for r in cat_a if r['signal']=='SELL')
    long_a_m=sum(1 for r in cat_a_minus if r['signal']=='BUY'); short_a_m=sum(1 for r in cat_a_minus if r['signal']=='SELL')
    long_b=sum(1 for r in cat_b if r['signal']=='BUY'); short_b=sum(1 for r in cat_b if r['signal']=='SELL')
    long_c1=sum(1 for r in cat_c1 if r['signal']=='BUY'); short_c1=sum(1 for r in cat_c1 if r['signal']=='SELL')
    long_c2=sum(1 for r in cat_c2 if r['signal']=='BUY'); short_c2=sum(1 for r in cat_c2 if r['signal']=='SELL')

    conv_tag = f"📋 {conversation_label} | " if conversation_label else ""
    out = f"{conv_tag}🗓️ {today}\n"
    out += f"📊 Regime: {regime_icon}{regime}{breadth} | NIFTY {price_now:.0f}({intraday_chg:+.0f},{intraday_pct:+.2f}%) | NIFTY50:{len(results)} {session_icon}\n"
    if choppy:
        out += f"🔶 CHOPPY MARKET — ADX:{adx_val:.0f}<20 + Price within 1% of MA20 → No clear trend, manage positions tightly\n"
    if not market_open:
        out += f"\n⚠️  POST-MARKET — signals shown for reference only. Not for live trading.\n"
    c2_total = long_c2 + short_c2
    out += f"📦 NIFTY50: {len(results)} stocks | CatA📈{long_a}/📉{short_a} | CatA-📈{long_a_m}/📉{short_a_m} | CatB🤖{long_b}/📉{short_b} | CatC1📈{long_c1}/📉{short_c1} | CatC2📊{c2_total} | CatD📉 | WL📋{len(watchlist)}\n"
    if short_q > 0:
        out += f"📉 SHORT-qualified: {short_q} stocks\n"
    # ── ML coverage + backtest freshness ───────────────────────────────
    import os as _os
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
    ml_covered = sum(1 for r in results if not (r.get('ml') or {}).get('_ml_fail_reason') == 'NO_MODEL')
    ml_missing = len(results) - ml_covered
    if ml_missing > 0:
        out += f"⚠️ ML: {ml_covered}/{len(results)} stocks have models | {ml_missing} missing → train.py --index nifty100\n"
    bt_files = [f for f in _os.listdir(model_dir) if f.startswith('backtest_v')]
    if bt_files:
        latest_bt = max(bt_files, key=lambda f: _os.path.getmtime(_os.path.join(model_dir, f)))
        bt_mtime = _os.path.getmtime(_os.path.join(model_dir, latest_bt))
        age_hrs = (datetime.now().timestamp() - bt_mtime) / 3600
        age_days = age_hrs / 24
        stale = "\u26a0\ufe0f" if age_days >= _BACKTEST_STALE_DAYS else "\u2705\ufe0f"
        age_str = f"{age_hrs:.1f}h" if age_hrs < 24 else f"{age_days:.0f}d"
        out += f"{stale} Backtest: {latest_bt} ({age_str})"
        if age_hrs > 2:
            out += " \u26a0\ufe0f(>2h \u2014 RR may drift)"
        out += "\n"
    all_wr = [r.get('_stats',{}).get('win_rate',0) for r in results]
    green_wr = sum(1 for w in all_wr if w >= 50)
    yellow_wr = sum(1 for w in all_wr if 40 <= w < 50)
    red_wr = sum(1 for w in all_wr if 0 < w < 40)
    no_data = sum(1 for w in all_wr if w == 0)
    out += f"📈 WR Quality: 🟢{green_wr} | 🟡{yellow_wr} | 🔴{red_wr} | ❓{no_data}\n"
    out += "\n"

    def _inv(price, sl, t1, t2):
        return {
            'sl': round(price + (price - sl), 0),
            't1': round(price - (t1 - price), 0),
            't2': round(price - (t2 - price), 0),
        }

    def _round_lev(d):
        return {k: round(v, 0) for k, v in d.items()}

    def _qty_rs10k(price, sl, sig='BUY', mode='swing'):
        """Position size for ~Rs 10,000 risk at given SL distance.
        P0-1 Fix: mode-aware — swing uses swing_SL (1.5×), intraday uses intraday_SL (3×).
        SHORT: SL is ABOVE entry price, so abs(price - sl) is correct."""
        sl_dist = abs(price - sl)
        if sl_dist < 0.5:
            sl_dist = price * 0.01   # fallback: 1% of price
        return max(1, int(10000 / sl_dist))

    def fmt_levels_hr(r, sig, tight=False, atr5=None, choppy=False):
        """Compute intraday + swing levels for display.
        v47: tight mode uses today_open as entry (not prev_close) for achievable targets.
        Stored tight levels (sl_tight/t1_tight/t2_tight) already have correct directional values.
        """
        h = r.get('hourly')
        price = r['price']
        prev = r.get('prev', price)
        atr = r.get('atr', price * 0.02)
        if atr5 is None:
            atr5 = r.get('atr5', atr)

        # Non-hourly path: use pre-stored tight/intraday/swing levels directly
        # v49: choppy → only T1, sideways → T1+T2, trending → T1+T2+T3
        if choppy and hr_l.get('t3') is not None:
            hr_l['t3'] = None
            tk_sl=r.get('sl_tight'); tk_t1=r.get('t1_tight'); tk_t2=r.get('t2_tight'); tk_t3=r.get('t3_tight')
            sw_sl=r.get('sl_swing'); sw_t1=r.get('t1_swing'); sw_t2=r.get('t2_swing')
            if tight:
                # tight mode: use stored tight levels (directional from get_level_modes_extended)
                hr_l = {'sl':round(tk_sl,0),'t1':round(tk_t1,0),'t2':round(tk_t2,0),'t3':round(tk_t3,0) if tk_t3 else None}
            else:
                # regular intraday: use stored intraday levels
                id_sl=r.get('sl_intraday',tk_sl); id_t1=r.get('t1_intraday',tk_t1); id_t2=r.get('t2_intraday',tk_t2)
                hr_l = {'sl':round(id_sl,0),'t1':round(id_t1,0),'t2':round(id_t2,0),'t3':None}
            sw_l = {'sl':round(sw_sl,0),'t1':round(sw_t1,0),'t2':round(sw_t2,0)}
            return hr_l, sw_l, atr, atr5

        # Hourly path: compute dynamically from entry
        # v49 FIX: use current price as entry (not today_open) so targets are always reachable
        if tight:
            entry_t = price  # v49: was today_open — caused impossible targets
            if sig == 'SELL':
                hr_l = {
                    'sl': round(entry_t + atr * 1.5, 0),
                    't1': round(entry_t - atr5 * 0.5, 0),   # v50: 0.5×ATR5 tight scalp
                    't2': round(entry_t - atr * 1.0, 0),
                    't3': round(entry_t - atr * 1.5, 0),
                }
            else:
                hr_l = {
                    'sl': round(entry_t - atr * 1.5, 0),
                    't1': round(entry_t + atr5 * 0.5, 0),   # v50: 0.5×ATR5 tight scalp
                    't2': round(entry_t + atr * 1.0, 0),
                    't3': round(entry_t + atr * 1.5, 0),
                }
        else:
            ic = ATR_CONFIG['intraday']
            if sig == 'SELL':
                hr_l = {'sl':round(price+atr*ic['sl'],0),'t1':round(price-atr*ic['t1'],0),'t2':round(price-atr*ic['t2'],0),'t3':None}
            else:
                hr_l = {'sl':round(price-atr*ic['sl'],0),'t1':round(price+atr*ic['t1'],0),'t2':round(price+atr*ic['t2'],0),'t3':None}

        # Swing levels always from stored values (prev_close anchored, multi-day)
        sw_sl=r.get('sl_swing'); sw_t1=r.get('t1_swing'); sw_t2=r.get('t2_swing')
        sw_l = {'sl':round(sw_sl,0),'t1':round(sw_t1,0),'t2':round(sw_t2,0)}
        return hr_l, sw_l, atr, atr5


    def fmt_stock_short(r, show_mode=None):
        """Format a stock line with levels filtered by show_mode ('intraday'|'swing').
        v38 P0-4: show only the relevant level set per --mode flag."""
        stats = r.get('_stats', {})
        sig = r['signal']
        dir_icon = "📈" if sig == 'BUY' else "📉"
        rr = stats.get('realized_return', 0)
        wr = stats.get('win_rate', 0)
        tags = r.get('tags', [])
        confluence = r.get('_confluence', 0)
        rsi = r.get('rsi', 0)
        hr_l, sw_l, atr_val, atr5_val = fmt_levels_hr(r, sig, tight=(level_mode == 'intraday_tight'), atr5=r.get('atr5'), choppy=choppy)
        price = r['price']
        mode = show_mode or level_mode
        age_days = r.get('signal_age_days', 0)
        # P0-1 Fix: WR badge + swing-based position sizing (not intraday)
        # Swing SL = price - atr*1.5 → gives realistic 4-10% pos_pct
        wr_badge = _wr_badge(wr)
        swing_sl_dist = abs(price - r.get('sl_swing', price * 0.015))
        if swing_sl_dist < 0.5:
            swing_sl_dist = price * 0.01
        swing_qty = max(1, int(10000 / swing_sl_dist))
        swing_pos_pct = round((swing_qty * price) / 100000 * 100, 1)
        pos_warn = None   # P0-1 fix: swing positions naturally 50-500% of capital (fixed-risk sizing).
                          # HIGH_POS warning removed — qty_10k is correct; qty is what matters.
        pos_tag = ''
        age_icon = ''
        if age_days >= 8: age_icon = ' 💀'
        elif age_days >= 4: age_icon = ' ⚠️'
        wr_info = f"{wr_badge}WR:{wr:.0f}%"
        # v38 P0-4: show ONE level set based on mode
        # v43: tight mode shows T1<T2<T3 (T1=ATR5, T2=ATR14, T3=ATR14 full)
        # v44: market condition
        cond = r.get('condition', '')
        cond_icon = ''
        if cond == 'BULL':        cond_icon = '📈BULL'
        elif cond == 'BEAR':       cond_icon = '📉BEAR'
        elif cond == 'BULL_PULLBACK': cond_icon = '📈BULL_PULL'
        elif cond == 'BEAR_RALLY': cond_icon = '📉BEAR_RALLY'
        elif cond == 'CHOPPY':     cond_icon = '⚠️CHOPPY'
        elif cond == 'FLAT':       cond_icon = '➖FLAT'
        else:                      cond_icon = f'🔄{cond}' if cond else ''

        def _fmt_tight(lvl, mode):
            # v50: T1=primary target(★), T2/T3=secondary
            t1_str = f"T1:★{lvl['t1']:.0f}"
            if lvl.get('t3') is None or mode == 'swing':
                return f"SL:{lvl['sl']:.0f} {t1_str} T2:{lvl['t2']:.0f}"
            return f"SL:{lvl['sl']:.0f} {t1_str} T2:{lvl['t2']:.0f} → T3:{lvl['t3']:.0f}"

        prev = r.get('prev', r['price'])
        if sig == 'SELL':
            qty = _qty_rs10k(price, hr_l['sl'], sig='SELL', mode=mode)
            if mode == 'swing':
                tline = f"  {dir_icon} {r['symbol']} ₹{r['price']:,.0f}({prev:,.0f}) | RSI:{rsi:.0f} {cond_icon} | Conf:{r['prob']}% | RR:{rr:+.0f}% {wr_info} | CF:{confluence}/10{age_icon}{pos_tag}\n"
                tline += f"     🎯 {_fmt_tight(sw_l, mode)} | ~₹{atr_val*ATR_CONFIG['swing']['t1']/24:.1f}/hr | Qty:{qty}"
            else:
                tline = f"  {dir_icon} {r['symbol']} ₹{r['price']:,.0f}({prev:,.0f}) | RSI:{rsi:.0f} {cond_icon} | Conf:{r['prob']}% | RR:{rr:+.0f}% {wr_info} | CF:{confluence}/10{age_icon}{pos_tag}\n"
                tline += f"     💠 {_fmt_tight(hr_l, mode)} | ~₹{atr_val*ATR_CONFIG['swing']['t1']/24:.1f}/hr | Qty:{qty}"
        else:
            qty = _qty_rs10k(price, hr_l['sl'], mode=mode)
            if mode == 'swing':
                tline = f"  {dir_icon} {r['symbol']} ₹{r['price']:,.0f}({prev:,.0f}) | RSI:{rsi:.0f} {cond_icon} | Conf:{r['prob']}% | RR:{rr:+.0f}% {wr_info} | CF:{confluence}/10{age_icon}{pos_tag}\n"
                tline += f"     🎯 {_fmt_tight(sw_l, mode)} | ~₹{atr_val*ATR_CONFIG['swing']['t1']/24:.1f}/hr | Qty:{qty}"
            else:
                tline = f"  {dir_icon} {r['symbol']} ₹{r['price']:,.0f}({prev:,.0f}) | RSI:{rsi:.0f} {cond_icon} | Conf:{r['prob']}% | RR:{rr:+.0f}% {wr_info} | CF:{confluence}/10{age_icon}{pos_tag}\n"
                tline += f"     💠 {_fmt_tight(hr_l, mode)} | ~₹{atr_val*ATR_CONFIG['swing']['t1']/24:.1f}/hr | Qty:{qty}"
        if tags:
            tag_str = " ".join(tags)
            tline += f" [{tag_str}]"
        return tline

    def fmt_cat_block(label, cat_list, max_items=10, sort_key=None, max_pos_pct=None):
        if not cat_list:
            return f"{label} [0]: \xe2\x80\x94\n\n", []
        if sort_key == 'rr_desc':
            sorted_s = sorted(cat_list, key=lambda x: ( -x.get('_stats',{}).get('realized_return',0), -x.get('_confluence',0) ))[:max_items]
        else:
            sorted_s = sorted(cat_list, key=lambda x: -x.get('prob', 0))[:max_items]
        lines = [fmt_stock_short(r, show_mode=level_mode) for r in sorted_s]
        has_short = any(r['signal'] == 'SELL' for r in cat_list)
        if has_short and 'A-' in label and 'HEDGE' not in label:
            label_out = label + " + HEDGE"
        else:
            label_out = label
        result_lines = [f"{label_out} [{len(cat_list)}]\n"] + lines + ["\n"]
        oversized_lines = []
        if max_pos_pct is not None:
            normal_lines, oversized_lines = [], []
            for l in result_lines:
                if 'OVERSIZE' in l:
                    oversized_lines.append(l)
                else:
                    normal_lines.append(l)
            result_lines = normal_lines
        return "".join(result_lines), oversized_lines



    def fmt_top_pick(r, sig):
        """Format a TOP_PICK line — one level set based on mode, with Qty.
        v38 P0-4: respects --mode flag; P1-6: uses corrected qty from --max-pos-pct."""
        stats = r.get('_stats', {})
        rr = stats.get('realized_return', 0)
        wr = stats.get('win_rate', 0)
        tags = r.get('tags', [])
        confluence = r.get('_confluence', 0)
        rsi = r.get('rsi', 0)
        hr_l, sw_l, atr_val, atr5_val = fmt_levels_hr(r, sig, tight=(level_mode == 'intraday_tight'), atr5=r.get('atr5'), choppy=choppy)
        price = r['price']
        age_days = r.get('signal_age_days', 0)
        # P0-1 Fix: swing-based position sizing (realistic 4-10%, not intraday 15-40%)
        swing_sl_dist = abs(price - r.get('sl_swing', price * 0.015))
        if swing_sl_dist < 0.5:
            swing_sl_dist = price * 0.01
        swing_qty_for_pos = max(1, int(10000 / swing_sl_dist))
        swing_pos_pct = round((swing_qty_for_pos * price) / 100000 * 100, 1)
        pos_warn = None
        pos_tag = ''
        age_icon = ''
        if age_days >= 8: age_icon = ' 💀'
        elif age_days >= 4: age_icon = ' ⚠️'
        wr_badge = _wr_badge(wr)
        wr_info = f"{wr_badge}WR:{wr:.0f}%"
        # v44: market condition
        cond = r.get('condition', '')
        cond_icon = ''
        if cond == 'BULL':        cond_icon = '📈BULL'
        elif cond == 'BEAR':       cond_icon = '📉BEAR'
        elif cond == 'BULL_PULLBACK': cond_icon = '📈BULL_PULL'
        elif cond == 'BEAR_RALLY': cond_icon = '📉BEAR_RALLY'
        elif cond == 'CHOPPY':     cond_icon = '⚠️CHOPPY'
        elif cond == 'FLAT':       cond_icon = '➖FLAT'
        else:                      cond_icon = f'🔄{cond}' if cond else ''
        mode = level_mode
        if sig == 'SELL':
            qty = _qty_rs10k(price, hr_l['sl'], sig='SELL', mode=mode)
            lvl = sw_l if mode == 'swing' else hr_l
            icon = '💠' if mode != 'swing' else '🎯'
            def _fmt(lvl, mode):
                t1_str = f"T1:★{lvl['t1']:.0f}"
                if lvl.get('t3') is None or mode == 'swing':
                    return f"SL:{lvl['sl']:.0f} {t1_str} T2:{lvl['t2']:.0f}"
                return f"SL:{lvl['sl']:.0f} {t1_str} T2:{lvl['t2']:.0f} → T3:{lvl['t3']:.0f}"
            prev = r.get('prev', r['price'])
            tline = (f"  📉 {r['symbol']} ₹{r['price']:,.0f}({prev:,.0f}) | RSI:{rsi:.0f} {cond_icon} | RR:{rr:+.0f}% {wr_info} | CF:{confluence}/10{age_icon}{pos_tag}\n"
                     f"     {icon} {_fmt(lvl, mode)} | ~₹{atr_val*ATR_CONFIG['swing']['t1']/24:.1f}/hr | Qty:{qty}")
        else:
            qty = _qty_rs10k(price, hr_l['sl'], mode=mode)
            lvl = sw_l if mode == 'swing' else hr_l
            icon = '💠' if mode != 'swing' else '🎯'
            def _fmt(lvl, mode):
                t1_str = f"T1:★{lvl['t1']:.0f}"
                if lvl.get('t3') is None or mode == 'swing':
                    return f"SL:{lvl['sl']:.0f} {t1_str} T2:{lvl['t2']:.0f}"
                return f"SL:{lvl['sl']:.0f} {t1_str} T2:{lvl['t2']:.0f} → T3:{lvl['t3']:.0f}"
            prev = r.get('prev', r['price'])
            tline = (f"  📈 {r['symbol']} ₹{r['price']:,.0f}({prev:,.0f}) | RSI:{rsi:.0f} {cond_icon} | RR:{rr:+.0f}% {wr_info} | CF:{confluence}/10{age_icon}{pos_tag}\n"
                     f"     {icon} {_fmt(lvl, mode)} | ~₹{atr_val*ATR_CONFIG['swing']['t1']/24:.1f}/hr | Qty:{qty}")
        if tags:
            tline += " " + " ".join(tags)
        return tline

    def _top_sort(lst):
        return sorted(lst, key=lambda x: (
            -x.get('_confluence', 0),
            -x.get('prob', 0),
            -x.get('_stats', {}).get('win_rate', 0),
        ))
    top_buys   = _top_sort([r for r in buy  if r.get('_starred')])[:top]
    top_shorts = _top_sort([r for r in sell if r.get('_starred')])[:top]

    if top_buys:
        g = sum(1 for r in top_buys if r.get('_stats', {}).get('win_rate', 0) >= 50)
        y = sum(1 for r in top_buys if 35 <= r.get('_stats', {}).get('win_rate', 0) < 50)
        out += f"🏆⭐ TOP BUYS  [{len(top_buys)}/{len([r for r in buy if r.get('_starred')])}] 📈BUY ⭐\n"
        out += "━" * 60 + "\n"
        for r in top_buys:
            out += fmt_top_pick(r, 'BUY') + "\n\n"
    else:
        out += f"🏆⭐ TOP BUYS  [0]: no BUY signals meet ⭐ criteria (WR>=55%+CF>=7.4+AI_Conf>=90%)\n"

    if top_shorts:
        g = sum(1 for r in top_shorts if r.get('_stats', {}).get('win_rate', 0) >= 50)
        y = sum(1 for r in top_shorts if 35 <= r.get('_stats', {}).get('win_rate', 0) < 50)
        out += f"🏆⭐ TOP SHORTS [{len(top_shorts)}/{len([r for r in sell if r.get('_starred')])}] 📉SHORT ⭐\n"
        out += "━" * 60 + "\n"
        for r in top_shorts:
            out += fmt_top_pick(r, 'SELL') + "\n\n"
    else:
        out += f"🏆⭐ TOP SHORTS [0]: no SELL signals meet ⭐ criteria (WR>=55%+CF>=7.4+AI_Conf>=60% (MEDIUM/HIGH)+TRENDING)\n"

    # ── Quality tier blocks (v38 P0-4/P0-5/P1-5/P1-6) ──────────────────────────
    # v38 P1-6: oversized separation (use max_pos_pct from parse_args)
    oversized_sections = []
    def safe_fmt_cat(label, cat_list, sort_key=None):
        result, ov = fmt_cat_block(label, cat_list, sort_key=sort_key, max_pos_pct=max_pos_pct)
        if ov:
            oversized_sections.extend(ov)
        return result

    out += safe_fmt_cat("📈 ✅ Cat A — HIGHEST QUALITY", cat_a, sort_key='rr_desc')
    out += safe_fmt_cat("📈 ⚠️ Cat A- — 2-of-3 CONFIRMED + HEDGE", cat_a_minus, sort_key='rr_desc')
    out += safe_fmt_cat("🤖 Cat B — AI CONFIRMED", cat_b, sort_key='rr_desc')
    out += safe_fmt_cat("📊 Cat C1 — SIGNAL + AI AGREE (SHORT)", cat_c1)

    # S1 Fix: Cat C2 split into C2a (EDGE) + C2b (NO EDGE) + total Cat C2
    # C2a: WR>=40% OR Ret>0% + has backtest — meaningful candidates
    # C2b: poor history, no backtest, low WR — informational only
    def _fmt_c2_compact(label, stocks, max_show=15):
        if stocks:
            sorted_s = sorted(stocks, key=lambda x: -x.get('prob', 0))
            out_str = f"{label} [{len(sorted_s)}]\n"
            for r in sorted_s[:max_show]:
                stats = r.get('_stats', {})
                tags = r.get('tags', [])
                tag_str = " ".join(tags) if tags else ""
                rr = stats.get('realized_return', 0)
                wr = stats.get('win_rate', 0)
                rr_str = f"RR:{rr:+.0f}%" if rr != 0 else ""
                wr_str = f"WR:{wr:.0f}%" if wr > 0 else ""
                meta = " | ".join(filter(None, [rr_str, wr_str, tag_str]))
                sig_icon = "📈" if r['signal'] == 'BUY' else "📉"
                out_str += f"  {sig_icon} {r['symbol']:15} RSI:{r['rsi']:.0f} | {meta}\n"
            if len(sorted_s) > max_show:
                out_str += f"  ... +{len(sorted_s)-max_show} more\n"
            out_str += "\n"
            return out_str
        return f"{label} [0]: —\n\n\n"

    out += _fmt_c2_compact("📊 Cat C2a — EDGE CANDIDATES (WR>=40% OR Ret>0%)", cat_c2a)
    out += _fmt_c2_compact("📊 Cat C2b — NO EDGE (poor/no history)", cat_c2b)
    out += _fmt_c2_compact("📊 Cat C2 — UNCONFIRMED [total]", cat_c2)
    out += safe_fmt_cat("🧠 Cat D — ML_CONFLICT SHORT", cat_d)

    # P1-1: BEAR_DIV SHORT — momentum shorts from bearish divergence
    # Show only if RSI 60-85 (valid momentum zone) — sorted by RSI desc
    if bear_div_shorts:
        sorted_bds = sorted(bear_div_shorts, key=lambda x: -x.get('rsi', 0))
        out += f"🐻 BEAR_DIV SHORT [{len(sorted_bds)}]\n"
        for r in sorted_bds[:10]:
            out += fmt_stock_short(r, show_mode=level_mode) + "\n"
        if len(sorted_bds) > 10:
            out += f"  ... +{len(sorted_bds)-10} more\n"
        out += "\n"

    # v38 P1-5: Filter WL — RSI 30-42 (buy zone) or RSI 58-70 (sell zone)
    # P1-1: Exclude BEAR_DIV stocks (shown in BEAR_DIV SHORT section above)
    def _wl_filter(r):
        if r.get('divergence') == 'BEARISH' and r.get('signal') == 'BUY':
            return False   # P1-1: BEAR_DIV BUY → Cat C2, not WL
        rsi = r.get('rsi', 50)
        if 30 <= rsi <= 42 or 58 <= rsi <= 70:
            return True
        h = r.get('hourly')
        if h:
            atr = r.get('atr', r['price'] * 0.02)
            sup = r['price'] - atr * 2.0
            res = r['price'] + atr * 2.0
            if abs(r['price'] - sup) / r['price'] < 0.03 or abs(r['price'] - res) / r['price'] < 0.03:
                return True
        return False
    wl_filtered = [r for r in watchlist if _wl_filter(r)]
    wl_note = f" (showing {len(wl_filtered)}/{len(watchlist)})" if wl_filtered else ""
    out += safe_fmt_cat(f"📋 WATCHLIST — RANGE BOUND{wl_note}", wl_filtered)

    # v38 P1-6: Append oversized section
    if oversized_sections:
        out += "\n" + "="*60 + "\n"
        out += "⚠️ OVERSIZED POSITIONS (reduce qty to stay within --max-pos-pct)\n\n"
        for line in oversized_sections:
            out += line + "\n"

    out += "\n⚠️ Not SEBI registered. Validate before trading."
    if choppy:
        out += "\n🔶 CHOPPY MARKET — ADX:" + str(round(adx_val)) + " <20 + Price near MA20 → Avoid new positions. Close shorts on bounces, exits on breakdowns."
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
    cat_a, cat_a_minus, cat_b, cat_c1, cat_c2, cat_c2a, cat_c2b, cat_d, watchlist, bear_div_shorts, short_qualified = _categorize(results, regime=regime)

    def sanitize(r):
        ai = r.get('ai') or {}
        ml = r.get('ml') or {}
        stats = r.get('_stats', {})
        # v36 Fix #3: qty for Rs10k risk (use intraday SL)
        price = float(r['price'])
        sl = float(r['sl'])
        qty_10k = max(1, int(10000 / max(abs(price - sl), 0.01)))
        return {
            'symbol': r['symbol'],
            'price': price,
            'change': float(r['change']),
            'rsi': float(r['rsi']),
            'signal': r['signal'],
            'prob': float(r['prob']),
            'sl': sl,
            't1': float(r['t1']),
            't2': float(r['t2']),
            'support': float(r['support']),
            'resistance': float(r['resistance']),
            'divergence': r.get('divergence'),
            'atr': float(r['atr']),
            'vol_ratio': float(r['vol_ratio']),
            'ret5': float(r['ret5']),
            'pos_pct': float(r['pos_pct']), 'max_pos_pct': float(max_pos_pct) if max_pos_pct else None,
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
            'qty_10k': qty_10k,  # v36: shares for ~Rs10k risk
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
    (stocks, use_ai, use_trailing, momentum_mode, high_conviction_mode, sector_cap, fundamental_filter,
     output_format, level_mode, top_n, auto_retrain,
     filter_neg_hist, backtest_first, conversation_label, debug_mode,
     wait_morning, max_pos_pct, stream_output) = parse_args()

    # ── v35: --wait-morning — sleep until 9:40 AM IST before scanning ────
    if wait_morning:
        from datetime import datetime as _dt, timezone, timedelta
        IST = timezone(timedelta(hours=5, minutes=30))
        while True:
            now_ist = _dt.now(IST)
            market_open, market_reason = _is_market_open()
            if not market_open:
                print(f"Market {market_reason} ({now_ist.strftime('%I:%M %p IST')}), skipping wait.")
                break
            now_ist = _dt.now(IST)
            now_mins = now_ist.hour * 60 + now_ist.minute
            target_mins = _MARKET_OPEN_MINS + 40   # 9:40 AM IST
            if now_mins >= target_mins:
                print(f"Market open ({now_ist.strftime('%I:%M %p IST')}), starting scan.")
                break
            wait_secs = (target_mins - now_mins) * 60
            print(f"Waiting... ({now_ist.strftime('%I:%M %p IST')}) sleeping {wait_secs}s")
            import time
            time.sleep(min(wait_secs, 300))

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

    # BUG-3: ThreadPoolExecutor — parallel scan, no buffering
    import concurrent.futures, sys as _sys
    _done = 0
    _total = len(stocks)

    def _process_one(sym):
        """Analyze one stock, return (sym, result_dict or None)."""
        r = analyze(sym, use_ai=use_ai, use_trailing=use_trailing,
                    fundamental_filter=fundamental_filter,
                    level_mode=level_mode, auto_retrain=auto_retrain,
                    momentum_mode=momentum_mode,
                    high_conviction_mode=high_conviction_mode)
        return sym, r

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as _executor:
        _futures = {_executor.submit(_process_one, sym): sym for sym in stocks}
        for _fut in concurrent.futures.as_completed(_futures):
            _sym, r = _fut.result()
            _done += 1
            if r and r.get('price') and r.get('price') > 0:
                if filter_neg_hist:
                    stats = r.get('_stats', {})
                    if stats.get('realized_return', 0) < 0:
                        _futures[_fut]  # consumed
                        if stream_output:
                            print(f"\r  [SKIP-] {_sym}: negative history", end='', flush=True)
                        continue
                if sector_cap and r['signal'] in ('BUY', 'SELL'):
                    sector = get_sector(_sym)
                    if check_sector_limit(sector, sector_counts, MAX_PER_SECTOR):
                        r['_sector_skipped'] = sector
                    else:
                        sector_counts[sector] = sector_counts.get(sector, 0) + 1
                results.append(r)
                if stream_output:
                    sig = r['signal']
                    rsi = r.get('rsi', 0)
                    cf = r.get('_confluence', 0)
                    tags = r.get('tags', [])
                    icon = '📈' if sig == 'BUY' else ('📉' if sig == 'SELL' else '📋')
                    tag_str = ' '.join([t for t in tags if '⚠️' in t or 'TOP_PICK' in t or '⭐' in t])
                    print(f"\r  {icon} {_sym}: ₹{r['price']:.0f} {sig} RSI:{rsi:.0f} CF:{cf:.1f}/10 {tag_str}", end='', flush=True)
            else:
                if stream_output:
                    print(f"\r  [SKIP ] {_sym}: no data", end='', flush=True)
            # Progress indicator every 5 stocks
            if _done % 5 == 0 or _done == _total:
                print(f"\r  Scanned {_done}/{_total} stocks...", end='', flush=True)
                _sys.stdout.flush()

    if stream_output:
        print(f"\r  Scan complete — {len(results)} results.{" "*10}", flush=True)

    if fundamental_filter:
        results = filter_by_fundamentals(results)
        results = [r for r in results if r.get('fundamental_ok', True)]

    if output_format == 'json':
        print(format_json(results, today))
        return

    if output_format == 'telegram' and use_ai:
        print(format_telegram(results, today, top_n=top_n, conversation_label=conversation_label, max_pos_pct=max_pos_pct, level_mode=level_mode))
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
    print(f"📊 NIFTY SCANNER v22{tag} | {today}")
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
    # ── ML coverage + backtest freshness warnings ───────────────────────
    if use_ai:
        import os as _os
        models = [f for f in _os.listdir(model_dir) if f.endswith('_model.joblib')]
        total = len(stocks)
        covered = sum(1 for r in results if not r.get('ml', {}).get('_ml_fail_reason') == 'NO_MODEL')
        missing = total - covered
        if missing > 0:
            print(f"  ⚠️  ML: {covered}/{total} stocks have models | {missing} NO_MODEL → run: train.py --index nifty100")
    # Backtest freshness check
    bt_files = [f for f in os.listdir(model_dir) if f.startswith('backtest_v')]
    if bt_files:
        latest = max(bt_files, key=lambda f: os.path.getmtime(os.path.join(model_dir, f)))
        age_days = (datetime.now() - datetime.fromtimestamp(os.path.getmtime(os.path.join(model_dir, latest)))).days
        if age_days >= 7:
            print(f"  ⚠️  Backtest stale: {latest} ({age_days}d old) → run: scan.py --backtest-first")
        else:
            print(f"  ✅ Backtest: {latest} ({age_days}d old)")
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
        # ── Debug: confluence component breakdown ──────────────────────
        if debug_mode:
            stats = r.get('_stats', {})
            ai_score = (ai.get('total_score', 0) + 100) / 200.0 if ai else 0.5
            sig_conf = r.get('prob', 50) / 100.0
            wr = stats.get('win_rate', 0) / 100.0
            align = r.get('_level_align', 'ALIGNED')
            align_s = 1.0 if align == 'ALIGNED' else (0.5 if align == 'WARN' else 0.0)
            age_days = r.get('signal_age_days', 0)
            age_tier, age_pen = _stale_tier(age_days)
            age_s = max(0, 1.0 - age_pen)
            rr = stats.get('realized_return', 0)
            regime = r.get('_regime_label', 'NEUTRAL')
            neg = '⚠️NEG' if rr < -2.0 else ('⚠️LOW' if rr < 0 else '✅')
            print(f"    🔍 CF-DEBUG: sig_conf={sig_conf:.2f}×0.20 | ai={ai_score:.2f}×0.25 | wr={wr:.2f}×0.20 | align={align_s:.2f}×0.10 | age={age_s:.2f}×0.10 | regime=1.0×0.15 | neg_hist={neg}(rr={rr:.1f}%)")
            print(f"       → CF={r.get('_confluence', 0)}/10 | RR={rr:+.1f}% WR={stats.get('win_rate',0):.0f}% Trades={stats.get('total_trades',0)} | regime={regime} | age={age_days}d({age_tier})")
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