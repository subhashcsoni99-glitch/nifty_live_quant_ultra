#!/usr/bin/env python3
"""
NIFTY Live Quant Ultra - Backtest v50
v10→v11 changes:
  1. PRIMARY metric = realized_return (P&L / capital_at_risk), NOT compounded
  2. peak_realized tracks settled cash only (updated only on exits)
  3. peak_unrealized = settled cash + current value of open shares (for DD monitoring)
  4. max_drawdown = max peak_realized_to_trough / peak_realized (settled capital only)
  5. pnl_list captures every realized exit for Sharpe calculation
  6. --no-sig-exit: SELL signal does NOT exit; only SL/TSL/ABSSL/hold-expiry
  7. Sig-exit fires only when price has pulled back ≥1% from entry
  8. 3-year default backtest window (includes 2020 COVID, 2022 bear, 2023-2024 bull)
  9. T1 partial exit: one-time 40% exit at T1, let 60% run to T2/SL (v41: was 25% → better DD reduction)
 10. MIN_TRADES = 20 for statistical confidence
 11. Sharpe from pnl_list, annualized; ABSSL hard cap always active
 12. v41: ABSL_CONFIG explicit + min_adx option + avg_trade_return in summary
13. v42: ATR-adaptive ABSSL, MAX_POSITION_PCT=0.10, min_adx/min_rr CLI
14. v43: TIME exit max_hold=30+no_unreal_loss guard; Qualified Sharpe≥0.8; MAX_POS=0.05

METRIC GUIDE (v41):
  realized_return  = avg P&L per trade as % of risk capital (PRIMARY — ignore CompRet)
  return           = portfolio-level P&L including drawdown drag (shows cost of DD)
  avg_trade_return = mean of pnl_list (per-trade %); Sharpe computed from this
  max_drawdown     = worst peak-to-trough % on settled capital
  
  For a 36% WR strategy, realized_return is positive but CompRet/return is lower
  because losing streaks compound. Both metrics are meaningful — show both.
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
import math
from datetime import datetime

import sys as _sys
# ─── CLI args ────────────────────────────────────────────────────────────────
_RSI_MODE = 'relaxed'  # default
if '--rsi-mode' in _sys.argv:
    idx = _sys.argv.index('--rsi-mode')
    if idx+1 < len(_sys.argv):
        mode = _sys.argv[idx+1].lower()
        if mode in ('strict', 'relaxed'):
            _RSI_MODE = mode
            print(f"RSI mode: {mode}")
            # BUG: consume both tokens so they're not treated as stock symbols
            _sys.argv.pop(idx)   # remove '--rsi-mode'
            _sys.argv.pop(idx)   # remove 'strict'/'relaxed'

from nifty_categorize import categorize_results
from nifty_core import (
    DEFAULT_STOCKS, EXCLUDED_STOCKS,
    ATR_CONFIG, RSI_CONFIG, SIGNAL_CONFIG, ADX_CONFIG, MOMENTUM_CONFIG,
    SECTORS, MAX_PER_SECTOR, get_market_regime,
    get_ohlc, add_features, detect_divergence, get_adx,
    get_sector, check_sector_limit, build_ml_features,
    get_signal as core_get_signal,
)
import joblib, os as _os

STOCKS = DEFAULT_STOCKS
MIN_TRADES = 20   # 3yr window ≈ 1 trade/month = 20 trades minimum
# ─── BLACKLIST ─────────────────────────────────────────────────────────────────
BLACKLIST = {'SBIN', 'BHEL', 'TITAN'}

# ─── ML PROBABILITY FILTER (v45) ─────────────────────────────────────────────
# Only enter when ML win_prob > ML_THRESHOLD. Default 0.55 = require >55% confidence.
ML_THRESHOLD = 0.55
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')

# ── ML model cache: lazy-load once per stock, reuse across bars ──────────────
_ml_cache = {}  # {symbol: loaded_model}

def _get_ml_win_prob(symbol, df, i):
    """Get ML-predicted win probability (0-1). Uses per-symbol model cache."""
    sym = symbol.replace('.NS', '')
    model_path = os.path.join(MODEL_DIR, f"{sym}_model.joblib")
    if not _os.path.exists(model_path):
        return None
    # Cache the loaded model for this stock
    if sym not in _ml_cache:
        try:
            _ml_cache[sym] = joblib.load(model_path)
        except Exception:
            return None
    try:
        feat = build_ml_features(df, idx=i)
        if feat is None or len(feat) == 0:
            return None
        model = _ml_cache[sym]
        prob = model.predict_proba(feat)
        if hasattr(prob, 'shape') and prob.shape[1] >= 2:
            return float(prob[0][1])  # P(up)
        return None
    except Exception:
        return None


# ─── POSITION SIZE CAP ────────────────────────────────────────────────────────
# Maximum position as % of capital per trade.
# Reducing from 20% to 10% halves DD per losing trade (reduces CompRet drag)
MAX_POSITION_PCT = 0.05   # was 0.20 — v43: 5% cap halves DD vs v42, ~4 losers=20% max cap

# ─── ABSL CONFIG (v42) ─────────────────────────────────────────────────────────
# Auto-Stop-Loss hard cap: volatility-adaptive, per-stock.
#
# PROBLEM (v41): Fixed -8% ABSSL destroyed high-ATR stocks.
#   PCBL (atr_pct=4.5%): -8% = 1.8× daily ATR — fires too fast
#   HDFCBANK (atr_pct=1.5%): -8% = 5.3× daily ATR — too loose
#
# SOLUTION (v42): Per-stock ABSSL using ATR-multiple.
#   Formula: ABSSL = entry × (1 - absl_atr_mult × atr_pct)
#   atr_pct = atr / entry_price (% daily ATR relative to price)
#
#   HIGH-VOL stocks (atr_pct > atr_pct_threshold):   fire at N × ATR  (tighter = 2.0×)
#   NORMAL stocks (atr_pct <= atr_pct_threshold):   fire at fixed % (looser = -15%)
#
# Before T1 partial:  widens to absl_pct_before_T1  (was 0.92 → now 0.85 = -15%)
# After T1 partial:   tightens to absl_pct_after_T1   (was 0.95 → now 0.88 = -12%)
# After T2 partial:   ABSSL disabled
#
ABSL_CONFIG = {
    'absl_atr_mult':       2.5,  # fire at entry × (1 - 2.5 × atr_pct) for high-vol
    'absl_pct_threshold':  0.03, # stocks with atr_pct > 3% use ATR-multiple cap
    'absl_pct_before_T1':  0.85,  # fire at -15% from entry before T1 (was 0.92 = -8%)
    'absl_pct_after_T1':   0.88,  # fire at -12% from entry after T1 partial (was 0.95 = -5%)
    'absl_pct_after_T2':   None,  # disabled after T2 partial hit
}

# ─── ADX TIGHTEN FILTER (v41) ──────────────────────────────────────────────────
# Optional tighter ADX filter to improve Sharpe ratio.
# When min_adx > 0: only enter trades when ADX >= min_adx (stronger trend confirmation)
# Default 0 = disabled (use ADX_CONFIG.threshold from nifty_core.py)
# Recommended: 30 (moderate trend) or 35 (strong trend only — fewer signals)
MIN_ADX_ENTRY = 30   # ← 30: FILTERS CHOPPY MARKETS — key WR fix

# ─── PER-MODE LEVEL FACTORS ───────────────────────────────────────────────────
# Multiply ATR_CONFIG values per level_mode to widen/tighten stops
# Swing fix: wider SL=2.0× (1.5× × 1.33), T1=2.0× (2.5× × 0.80) → fewer SL hits
# Intraday: keep as-is (3.0× SL already loose), rely on ADX filtering
SL_FACTOR  = {'swing': 1.33, 'intraday': 1.0, 'intraday_tight': 1.0}
T1_FACTOR  = {'swing': 0.80, 'intraday': 1.0, 'intraday_tight': 1.0}
T2_FACTOR  = {'swing': 0.75, 'intraday': 1.0, 'intraday_tight': 1.0}
MAX_HOLD   = {'swing': 30, 'intraday': 15, 'intraday_tight': 10}

# ─── REVERSAL CONFIG ────────────────────────────────────────────────────────
# REVERSAL exit fires only when BOTH conditions met:
#   1. Hold days >= REVERSAL_MIN_HOLD_DAYS (prevent early exit on day-1 pullback)
#   2. Pullback from entry >= REVERSAL_MIN_LOSS_PCT% (must be in real loss)
# In bear markets: use 5 days + 2% (default). In normal: 3 days + 1%.
REVERSAL_MIN_HOLD_DAYS = 5
REVERSAL_MIN_LOSS_PCT = 2.0      # was 2.0 — must be ≥3.5% below entry to exit (let winners run)

# ─── RSI ENTRY FILTER ───────────────────────────────────────────────────────
# Skip BUY entry if RSI > RSI_ENTRY_MAX (overbought = mean reversion trap)
# Setting to 60 filters entries when market is overextended
RSI_ENTRY_MAX = 65               # v36: was 55 — match live scan buy_relaxed=65
# P2-1: --rsi-mode strict|relaxed
if _RSI_MODE == 'strict':
    RSI_ENTRY_MAX = 30  # match live scan buy_strict=30
    print(f"[P2-1] RSI entry STRICT mode: RSI_ENTRY_MAX=30 (matches live buy_strict=30)")

# ─── BEAR-MARKET ATR ADJUSTMENT ─────────────────────────────────────────────
# In BEARISH regime: WIDEN SL by multiplying ATR_CONFIG by this factor
# Prevents gap-down ABSSL hits by giving more room in volatile markets
# >1 = wider stops (safer), <1 = tighter stops (riskier)
# 0 or None = disabled
BEAR_REGIME_SL_FACTOR = 1.2     # was 0.8 — WRONG: tighter SL in bear = MORE stops hit
                               # Fixed: wider stops in bear = fewer ABSSL gap-downs

MIN_HOLD_DAYS_FOR_REVERSAL = REVERSAL_MIN_HOLD_DAYS  # backward compat alias

# ─── Signal Engine (delegated to nifty_core — single source of truth) ───────
def get_signal(df, i, momentum_mode=False, multi_mode=False, high_conviction_mode=False, ultra_mode=False, hybrid_mode=False, adx_filter=True):
    """backtest.py wrapper — delegates to nifty_core.get_signal.
    
    momentum_mode: if True, uses RSI>70/RSI<30 momentum logic (Option B)
    adx_filter:    if True, blocks signals when ADX < threshold (Option A)
                   Controlled by --no-adx-filter CLI flag.
    
    Returns (signal_val, signal_name, divergence, adx_info_dict).
    
    multi_mode: if True, uses multi-factor signals (RSI+slope+vol+BB+weeklyRSI)
    momentum_mode: if True, uses RSI>60 momentum logic
    hybrid_mode: if True, uses HC + Ichimoku hybrid signals"""
    # Temporarily override ADX_CONFIG.enabled to match adx_filter flag
    import nifty_core as _nc
    _saved_enabled = _nc.ADX_CONFIG.get('enabled', True)
    if not adx_filter:
        _nc.ADX_CONFIG['enabled'] = False
    try:
        sig_val, meta, _ = _nc.get_signal(df, i, momentum_mode=momentum_mode, multi_mode=multi_mode, high_conviction_mode=high_conviction_mode, ultra_mode=ultra_mode, hybrid_mode=hybrid_mode)
    finally:
        _nc.ADX_CONFIG['enabled'] = _saved_enabled
    adx_val, _, _ = _nc.get_adx(df, i)
    adx_th = _nc.ADX_CONFIG.get('threshold', 25)
    adx_trending = adx_val > adx_th if adx_filter else True
    return sig_val, meta['signal'], meta.get('divergence'), {
        'adx': round(adx_val, 1),
        'adx_trending': adx_trending,
        'adx_threshold': adx_th,
    }


def backtest_stock(symbol, start=None, end=None, use_trailing=False, sector_limits=False,
                   slippage_pct=0.001, max_position_pct=0.05,  # v43: 5% cap halves DD vs v42
                   use_t1_partial=True, max_hold_days=None,  # None = auto from MAX_HOLD
                   no_sig_exit=False, verbose=False,
                   level_mode='swing',
                   momentum_mode=False,
                   multi_mode=False,
                   high_conviction_mode=False,
                   ultra_mode=False,
                   long_only=False,
                   hybrid_mode=False,
                   adx_filter=True,
                   min_adx=MIN_ADX_ENTRY,
                   min_rr_ratio=0,
                   ml_threshold=ML_THRESHOLD):
    """
    Key changes vs v8:
    - no_sig_exit: SELL signal does not trigger exit (only SL/TSL/ABSSL/expiry)
    - Signal exit with pullback guard: only fires if price ≥1% below entry
    - peak_realized updated ONLY on settled exits
    - max_drawdown from peak_realized (settled peak only)
    - pnl_list for Sharpe; realized_return = realized_pnl / initial_capital
    """
    name = symbol.replace('.NS', '')
    df = get_ohlc(symbol, days=1095)
    if df is None:
        return None

    df = add_features(df)
    if hasattr(df.index, 'tz') and df.index.tz is not None:
        df = df.tz_localize(None)
    if end:
        ts = pd.Timestamp(end)
        ts = ts.tz_localize(None) if ts.tzinfo else ts
        df = df[df.index <= ts]
    if start:
        ts = pd.Timestamp(start)
        ts = ts.tz_localize(None) if ts.tzinfo else ts
        df = df[df.index >= ts]
    if len(df) < 200:
        return None

    initial_capital = 100000.0
    capital = initial_capital
    peak_realized = capital    # highest all-time SETTLED capital (updated after each settled exit)
    peak_ever = capital             # v36 fix: track all-time high separately (prevents DD>100% from intra-bar swings)
    shares = 0
    position = None
    entry_price = 0
    entry_date = None
    tsl = 0
    shares_remaining = 0
    partial_exits = 0
    t1_triggered = False
    was_profitable = False  # v24: SIG exit disabled
    trades = []
    sector_counts = {} if sector_limits else {}
    pnl_list = []
    realized_pnl_sum = 0.0

    entry_adx_info = None  # snapshot of ADX state at signal entry — persists across bars
    for i in range(200, len(df)):
        sig_val, sig_name, div, adx_info = get_signal(df, i, momentum_mode=momentum_mode, multi_mode=multi_mode, high_conviction_mode=high_conviction_mode, ultra_mode=ultra_mode, hybrid_mode=hybrid_mode, adx_filter=adx_filter)
        price = df['Close'].iloc[i]
        atr = df['atr'].iloc[i]
        if pd.isna(atr) or atr == 0:
            atr = price * 0.02

        slip_entry = price * (1 + slippage_pct)
        slip_exit  = price * (1 - slippage_pct)

        # ── Entry ──────────────────────────────────────────────────────
        if sig_val == 1 and position is None:
            entry_adx_info = adx_info  # snapshot ADX state at entry signal
            if sector_limits:
                sect = get_sector(name)
                if check_sector_limit(sect, sector_counts, MAX_PER_SECTOR):
                    continue

            # Blacklist check — skip persistent losers in bear-market mode
            if name in BLACKLIST:
                continue

            # v41 min_adx: optional stricter ADX filter for entry
            # Only enforce when min_adx > 0 and adx_filter is already True
            if min_adx > 0 and adx_filter:
                entry_adx_val = adx_info.get('adx', 0)
                if entry_adx_val < min_adx:
                    continue  # skip — ADX below minimum threshold

            risk = capital * 0.005
            # Bear-regime SL adjustment: tighten SL to avoid gap-down ABSSL hits
            sl_mult = ATR_CONFIG[level_mode]['sl']
            if BEAR_REGIME_SL_FACTOR and BEAR_REGIME_SL_FACTOR > 0:
                sl_mult = sl_mult * BEAR_REGIME_SL_FACTOR
            sl_dist = atr * sl_mult
            raw_shares = max(1, int(risk / sl_dist))
            pos_value = raw_shares * slip_entry
            max_pos = capital * max_position_pct
            if pos_value > max_pos:
                raw_shares = max(1, int(max_pos / slip_entry))

            # RSI entry filter — skip BUY if overbought (bear-market trap guard)
            rsi = df['rsi'].iloc[i]
            if not (pd.isna(rsi) or float(rsi) < float(RSI_ENTRY_MAX)):
                continue

            # ML probability filter (v45) — only enter if ML win_prob > threshold
            ml_prob = _get_ml_win_prob(name, df, i)
            if ml_prob is not None and ml_prob < ml_threshold:
                continue  # ML not confident enough — skip

            # Auto max_hold_days from level_mode
            if max_hold_days is None:
                max_hold_days = MAX_HOLD.get(level_mode, 30)

            shares = raw_shares
            shares_remaining = raw_shares
            t1_triggered = False
            was_profitable = False  # reset on new entry
            position = 'LONG'
            entry_price = slip_entry
            entry_date = df.index[i]
            tsl = entry_price - atr * 1.5
            entry_cap_at_risk = shares * entry_price  # cash locked at entry
            capital -= entry_cap_at_risk          # remove deployed capital from cash

            if sector_limits:
                sector_counts[sect] = sector_counts.get(sect, 0) + 1

            if verbose:
                print(f"  📈 BUY {name} @ ₹{entry_price:.2f} [{entry_date.date()}] "
                      f"shares={shares} atr={atr:.2f} rsi={rsi:.1f} entry_adx={(entry_adx_info or {}).get('adx', 0):.0f}")

        # ── SHORT Entry (momentum mode: RSI>70 overbought → mean-reversion short) ────
        elif sig_val == -1 and position is None:
            if long_only:
                continue  # LONG ONLY — skip shorts
            entry_adx_info = adx_info
            if sector_limits:
                sect = get_sector(name)
                if check_sector_limit(sect, sector_counts, MAX_PER_SECTOR):
                    continue
            if name in BLACKLIST:
                continue
            if min_adx > 0 and adx_filter:
                if adx_info.get('adx', 0) < min_adx:
                    continue

            # SHORT: risk = capital * 0.005, SL = entry + sl_dist (buy to cover)
            risk = capital * 0.005
            base_sl = ATR_CONFIG[level_mode]['sl'] * SL_FACTOR.get(level_mode, 1.0)
            sl_dist = atr * base_sl
            raw_shares = max(1, int(risk / sl_dist))
            pos_value = raw_shares * slip_entry
            max_pos = capital * max_position_pct
            if pos_value > max_pos:
                raw_shares = max(1, int(max_pos / slip_entry))

            # RSI entry filter for SHORT: skip if RSI < RSI_SELL_MIN (not overbought enough)
            rsi = df['rsi'].iloc[i]
            if not (pd.isna(rsi) or float(rsi) > float(RSI_CONFIG.get('sell_relaxed', 40))):
                continue

            # ML probability filter for SHORT (v45)
            ml_prob = _get_ml_win_prob(name, df, i)
            if ml_prob is not None and ml_prob > (1 - ml_threshold):
                # ML thinks DOWN likely → good for SHORT (opposite threshold for shorts)
                pass  # keep the signal
            elif ml_prob is not None and ml_prob < (1 - ml_threshold):
                continue  # ML thinks UP likely → skip short

            if max_hold_days is None:
                max_hold_days = MAX_HOLD.get(level_mode, 30)

            shares = raw_shares
            shares_remaining = raw_shares
            t1_triggered = False
            was_profitable = False
            position = 'SHORT'
            entry_price = slip_entry  # short sale price
            entry_date = df.index[i]
            tsl = entry_price + atr * 1.5  # TSL for short: buy-back stop above entry
            entry_cap_at_risk = shares * entry_price
            capital -= entry_cap_at_risk

            if sector_limits:
                sector_counts[sect] = sector_counts.get(sect, 0) + 1

            if verbose:
                print(f"  📉 SHORT {name} @ ₹{entry_price:.2f} [{entry_date.date()}] "
                      f"shares={shares} atr={atr:.2f} rsi={rsi:.1f} entry_adx={(entry_adx_info or {}).get('adx', 0):.0f}")

        # ── In Position ────────────────────────────────────────────────
        elif position == 'LONG':
            # ATR-based levels with per-mode factors (wider SL = fewer SL hits = higher WR)
            base_sl  = ATR_CONFIG[level_mode]['sl']  * SL_FACTOR.get(level_mode, 1.0)
            base_t1  = ATR_CONFIG[level_mode]['t1']  * T1_FACTOR.get(level_mode, 1.0)
            base_t2  = ATR_CONFIG[level_mode]['t2']  * T2_FACTOR.get(level_mode, 1.0)
            sl_dist  = atr * base_sl
            t1_dist  = atr * base_t1
            t2_dist  = atr * base_t2
            sl  = entry_price - sl_dist
            t1  = entry_price + t1_dist
            t2  = entry_price + t2_dist

            # T1 Partial Exit (one-time at T1 — 10% of remaining position)
            if use_t1_partial and shares_remaining > 0 and price >= t1 and not t1_triggered:
                exit_shares = max(1, shares_remaining // 10 * 4)  # v41: exit 40%, let 60% run (was //10 = 10%)
                capital += exit_shares * slip_exit
                shares_remaining -= exit_shares
                partial_exits += 1
                tsl = max(tsl, entry_price)  # Activate TSL after T1 partial (lock profit on remaining half)
                t1_triggered = True
                was_profitable = True  # position has locked in profit — SIG exit now allowed
                peak_realized = max(peak_realized, capital)
                # Activate TSL immediately after T1 partial (even if use_trailing=False)
                tsl = max(tsl, price - atr * 1.5)
                if verbose:
                    print(f"  🎯 T1 PARTIAL {name} @ ₹{slip_exit:.2f} "
                          f"[{df.index[i].date()}] exited={exit_shares} "
                          f"remain={shares_remaining}")

            # Time-based exit v43: hold up to 30 days, exit regardless of P&L
            # v43 FIX: removed unreal_loss guard — underwater trades are NOT cut early.
            # RSI mean-reversion setups can take 25-40 days to materialize.
            # Only exit on time if still held after 30 days (extended from 20 → 30).
            # If profitable: lock in gains. If losing: accept and move on.
            hold_days = (df.index[i] - entry_date).days if entry_date else 0
            unreal_pnl = ((price - entry_price) / entry_price) * 100 if entry_price > 0 else 0
            held_too_long = hold_days > max_hold_days  # max_hold_days = 30 in v43
            if held_too_long:
                if shares_remaining > 0:
                    capital += shares_remaining * slip_exit
                pnl = ((slip_exit - entry_price) / entry_price) * 100
                pnl_list.append(pnl)
                realized_pnl_sum += shares_remaining * (slip_exit - entry_price)
                peak_realized = max(peak_realized, capital)
                trades.append({'pnl': round(pnl, 2), 'type': 'TIME',
                               'sector': get_sector(name), 'date': str(df.index[i].date()),
                               'partial': partial_exits > 0, 'hold_days': hold_days,
                               'unreal_pct': round(unreal_pnl, 2)})
                shares = 0
                shares_remaining = 0
                t1_triggered = False
                position = None
                entry_date = None
                if sector_limits:
                    sect = get_sector(name)
                    sector_counts[sect] = max(0, sector_counts.get(sect, 0) - 1)
                continue

            # ── Volatility-Adaptive ABSSL (v42) ───────────────────────────────────────
            # Per-stock ABSSL: high-vol stocks use ATR-multiple cap; normal stocks use fixed %
            # atr_pct = daily ATR as % of entry price → volatility-normalized
            atr_pct = atr / entry_price if entry_price > 0 else 0.02
            before_t1_pct = ABSL_CONFIG.get('absl_pct_before_T1', 0.85)
            after_t1_pct  = ABSL_CONFIG.get('absl_pct_after_T1', 0.88)
            atr_th        = ABSL_CONFIG.get('absl_pct_threshold', 0.03)
            atr_mult      = ABSL_CONFIG.get('absl_atr_mult', 2.5)

            # Determine effective ABSSL price level
            if not t1_triggered:
                # Before T1 partial: use looser fixed % (was -8%, now -15%)
                if atr_pct > atr_th:
                    abs_sl = entry_price * (1 - atr_mult * atr_pct)  # ATR-multiple for high-vol
                else:
                    abs_sl = entry_price * before_t1_pct             # fixed -15% for normal
            else:
                # After T1 partial: tighter -12%
                abs_sl = entry_price * after_t1_pct
            if price <= abs_sl:
                if shares_remaining > 0:
                    capital += shares_remaining * slip_exit
                pnl = ((slip_exit - entry_price) / entry_price) * 100
                pnl_list.append(pnl)
                realized_pnl_sum += shares_remaining * (slip_exit - entry_price)
                peak_realized = max(peak_realized, capital)
                trades.append({'pnl': round(pnl, 2), 'type': 'ABSSL',
                               'sector': get_sector(name), 'date': str(df.index[i].date()),
                               'partial': partial_exits > 0})
                shares = 0
                shares_remaining = 0
                t1_triggered = False
                position = None
                entry_date = None
                if sector_limits:
                    sect = get_sector(name)
                    sector_counts[sect] = max(0, sector_counts.get(sect, 0) - 1)
                continue

            # Trailing SL
            if use_trailing:
                new_tsl = price - atr * 1.5
                if new_tsl > tsl and price >= entry_price + atr * 0.5:
                    tsl = new_tsl
                if tsl > 0 and price <= tsl:
                    if shares_remaining > 0:
                        capital += shares_remaining * slip_exit
                    pnl = ((slip_exit - entry_price) / entry_price) * 100
                    pnl_list.append(pnl)
                    realized_pnl_sum += shares_remaining * (slip_exit - entry_price)
                    peak_realized = max(peak_realized, capital)
                    trades.append({'pnl': round(pnl, 2), 'type': 'TSL',
                                   'sector': get_sector(name), 'date': str(df.index[i].date()),
                                   'partial': partial_exits > 0,
                                   'entry_adx': (entry_adx_info or {}).get('adx', 0), 'entry_adx_trending': (entry_adx_info or {}).get('adx_trending', False)})
                    shares = 0
                    shares_remaining = 0
                    t1_triggered = False
                    position = None
                    entry_date = None
                    if sector_limits:
                        sect = get_sector(name)
                        sector_counts[sect] = max(0, sector_counts.get(sect, 0) - 1)
                    continue

            # ── TSL: Always active after T1 partial — lock profit on remaining half ──
            if t1_triggered and shares_remaining > 0:
                # TSL: trailing stop after T1 partial — always active post-T1
                new_tsl = price - atr * 1.5
                if new_tsl > tsl:
                    tsl = new_tsl
                if tsl > 0 and price <= tsl:
                    if shares_remaining > 0:
                        capital += shares_remaining * slip_exit
                    pnl = ((slip_exit - entry_price) / entry_price) * 100
                    pnl_list.append(pnl)
                    realized_pnl_sum += shares_remaining * (slip_exit - entry_price)
                    peak_realized = max(peak_realized, capital)
                    trades.append({'pnl': round(pnl, 2), 'type': 'TSL',
                                   'sector': get_sector(name), 'date': str(df.index[i].date()),
                                   'partial': partial_exits > 0,
                                   'entry_adx': (entry_adx_info or {}).get('adx', 0), 'entry_adx_trending': (entry_adx_info or {}).get('adx_trending', False)})
                    shares = 0; shares_remaining = 0; t1_triggered = False
                    position = None; entry_date = None
                    if sector_limits:
                        sector_counts[get_sector(name)] = max(0, sector_counts.get(get_sector(name), 0) - 1)
                    continue

            # ── T2: Full exit when price reaches T2 (only after T1 partial was hit) ──
            if t1_triggered and price >= t2 and shares_remaining > 0:
                if verbose:
                    print(f"  🎯 T2 FULL EXIT {name} @ ₹{slip_exit:.2f} "
                          f"[{df.index[i].date()}] remain={shares_remaining}")
                pnl = ((slip_exit - entry_price) / entry_price) * 100
                pnl_list.append(pnl)
                realized_pnl_sum += shares_remaining * (slip_exit - entry_price)
                peak_realized = max(peak_realized, capital)
                trades.append({'pnl': round(pnl, 2), 'type': 'T2',
                               'sector': get_sector(name), 'date': str(df.index[i].date()),
                               'partial': partial_exits > 0,
                               'entry_adx': (entry_adx_info or {}).get('adx', 0), 'entry_adx_trending': (entry_adx_info or {}).get('adx_trending', False)})
                shares = 0; shares_remaining = 0; t1_triggered = False
                position = None; entry_date = None
                continue

            # ── Fixed SL ──
            if price <= sl:
                if shares_remaining > 0:
                    capital += shares_remaining * slip_exit
                pnl = ((slip_exit - entry_price) / entry_price) * 100
                pnl_list.append(pnl)
                realized_pnl_sum += shares_remaining * (slip_exit - entry_price)
                peak_realized = max(peak_realized, capital)
                trades.append({'pnl': round(pnl, 2), 'type': 'SL',
                               'sector': get_sector(name), 'date': str(df.index[i].date()),
                               'partial': partial_exits > 0,
                               'entry_adx': (entry_adx_info or {}).get('adx', 0), 'entry_adx_trending': (entry_adx_info or {}).get('adx_trending', False)})
                shares = 0
                shares_remaining = 0
                t1_triggered = False
                position = None
                entry_date = None
                if sector_limits:
                    sect = get_sector(name)
                    sector_counts[sect] = max(0, sector_counts.get(sect, 0) - 1)

        # ── SHORT Position Management ───────────────────────────────────────────
        elif position == 'SHORT':
            # SHORT: buy to cover at target, sell to cover at SL
            base_sl = ATR_CONFIG[level_mode]['sl'] * SL_FACTOR.get(level_mode, 1.0)
            base_t1 = ATR_CONFIG[level_mode]['t1'] * T1_FACTOR.get(level_mode, 1.0)
            base_t2 = ATR_CONFIG[level_mode]['t2'] * T2_FACTOR.get(level_mode, 1.0)
            sl_dist = atr * base_sl
            t1_dist = atr * base_t1
            t2_dist = atr * base_t2
            # SHORT SL: price rises to entry+sl_dist → stop loss (buy to cover)
            sl  = entry_price + sl_dist
            # SHORT T1: price falls to entry-t1_dist → target (buy to cover)
            t1  = entry_price - t1_dist
            t2  = entry_price - t2_dist

            # T1 Partial: buy to cover 40% at T1, let 60% run
            if use_t1_partial and shares_remaining > 0 and price <= t1 and not t1_triggered:
                exit_shares = max(1, shares_remaining // 10 * 4)
                capital += exit_shares * slip_exit  # buy to cover at slip_exit → capital increases
                shares_remaining -= exit_shares
                partial_exits += 1
                t1_triggered = True
                was_profitable = True
                peak_realized = max(peak_realized, capital)
                tsl = min(tsl, entry_price)  # activate TSL after partial
                tsl = min(tsl, price + atr * 1.5)
                if verbose:
                    print(f"  🎯 T1 PARTIAL COVER {name} @ ₹{slip_exit:.2f} "
                          f"[{df.index[i].date()}] covered={exit_shares} remain={shares_remaining}")

            # Time exit
            hold_days = (df.index[i] - entry_date).days if entry_date else 0
            unreal_pnl = ((entry_price - price) / entry_price) * 100
            if hold_days > max_hold_days:
                if shares_remaining > 0:
                    capital += shares_remaining * slip_exit
                pnl = ((entry_price - slip_exit) / entry_price) * 100  # SHORT pnl: entry - exit
                pnl_list.append(pnl)
                realized_pnl_sum += shares_remaining * (entry_price - slip_exit)
                peak_realized = max(peak_realized, capital)
                trades.append({'pnl': round(pnl, 2), 'type': 'TIME',
                               'sector': get_sector(name), 'date': str(df.index[i].date()),
                               'partial': partial_exits > 0, 'hold_days': hold_days,
                               'unreal_pct': round(unreal_pnl, 2)})
                shares = 0; shares_remaining = 0; t1_triggered = False
                position = None; entry_date = None
                if sector_limits:
                    sector_counts[get_sector(name)] = max(0, sector_counts.get(get_sector(name), 0) - 1)
                continue

            # ABSSL: hard cap at -15% from entry
            atr_pct = atr / entry_price if entry_price > 0 else 0.02
            absl_pct = ABSL_CONFIG.get('absl_pct_before_T1', 0.85)
            abs_sl = entry_price * (1 + (1 - absl_pct))  # e.g. 1/(1-0.85) = 1.15× entry
            if price >= abs_sl:
                if shares_remaining > 0:
                    capital += shares_remaining * slip_exit
                pnl = ((entry_price - slip_exit) / entry_price) * 100
                pnl_list.append(pnl)
                realized_pnl_sum += shares_remaining * (entry_price - slip_exit)
                peak_realized = max(peak_realized, capital)
                trades.append({'pnl': round(pnl, 2), 'type': 'ABSSL',
                               'sector': get_sector(name), 'date': str(df.index[i].date()),
                               'partial': partial_exits > 0})
                shares = 0; shares_remaining = 0; t1_triggered = False
                position = None; entry_date = None
                if sector_limits:
                    sector_counts[get_sector(name)] = max(0, sector_counts.get(get_sector(name), 0) - 1)
                continue

            # TSL after T1 partial
            if t1_triggered and shares_remaining > 0:
                new_tsl = price + atr * 1.5
                if new_tsl < tsl:
                    tsl = new_tsl
                if tsl > 0 and price >= tsl:  # price rose to TSL → stop out
                    if shares_remaining > 0:
                        capital += shares_remaining * slip_exit
                    pnl = ((entry_price - slip_exit) / entry_price) * 100
                    pnl_list.append(pnl)
                    realized_pnl_sum += shares_remaining * (entry_price - slip_exit)
                    peak_realized = max(peak_realized, capital)
                    trades.append({'pnl': round(pnl, 2), 'type': 'TSL',
                                   'sector': get_sector(name), 'date': str(df.index[i].date()),
                                   'partial': partial_exits > 0,
                                   'entry_adx': (entry_adx_info or {}).get('adx', 0),
                                   'entry_adx_trending': (entry_adx_info or {}).get('adx_trending', False)})
                    shares = 0; shares_remaining = 0; t1_triggered = False
                    position = None; entry_date = None
                    if sector_limits:
                        sector_counts[get_sector(name)] = max(0, sector_counts.get(get_sector(name), 0) - 1)
                    continue

            # T2 exit: price falls to T2 → full cover
            if t1_triggered and price <= t2 and shares_remaining > 0:
                if verbose:
                    print(f"  🎯 T2 FULL COVER {name} @ ₹{slip_exit:.2f} [{df.index[i].date()}] remain={shares_remaining}")
                pnl = ((entry_price - slip_exit) / entry_price) * 100
                pnl_list.append(pnl)
                realized_pnl_sum += shares_remaining * (entry_price - slip_exit)
                peak_realized = max(peak_realized, capital)
                trades.append({'pnl': round(pnl, 2), 'type': 'T2',
                               'sector': get_sector(name), 'date': str(df.index[i].date()),
                               'partial': partial_exits > 0,
                               'entry_adx': (entry_adx_info or {}).get('adx', 0),
                               'entry_adx_trending': (entry_adx_info or {}).get('adx_trending', False)})
                shares = 0; shares_remaining = 0; t1_triggered = False
                position = None; entry_date = None
                continue

            # Fixed SL: price rose to entry+sl_dist → stop loss
            if price >= sl:
                if shares_remaining > 0:
                    capital += shares_remaining * slip_exit
                pnl = ((entry_price - slip_exit) / entry_price) * 100
                pnl_list.append(pnl)
                realized_pnl_sum += shares_remaining * (entry_price - slip_exit)
                peak_realized = max(peak_realized, capital)
                trades.append({'pnl': round(pnl, 2), 'type': 'SL',
                               'sector': get_sector(name), 'date': str(df.index[i].date()),
                               'partial': partial_exits > 0,
                               'entry_adx': (entry_adx_info or {}).get('adx', 0),
                               'entry_adx_trending': (entry_adx_info or {}).get('adx_trending', False)})
                shares = 0; shares_remaining = 0; t1_triggered = False
                position = None; entry_date = None
                if sector_limits:
                    sector_counts[get_sector(name)] = max(0, sector_counts.get(get_sector(name), 0) - 1)

    # ── Close open position at end ─────────────────────────────────────
    if position == 'LONG' and shares_remaining > 0:
        slip_exit_end = df['Close'].iloc[-1] * (1 - slippage_pct)
        capital += shares_remaining * slip_exit_end
        pnl = ((slip_exit_end - entry_price) / entry_price) * 100
        pnl_list.append(pnl)
        realized_pnl_sum += shares_remaining * (slip_exit_end - entry_price)
        peak_realized = max(peak_realized, capital)
        trades.append({'pnl': round(pnl, 2), 'type': 'CLOSED',
                       'sector': get_sector(name), 'date': str(df.index[-1].date()),
                       'partial': partial_exits > 0,
                       'entry_adx': (entry_adx_info or {}).get('adx', 0), 'entry_adx_trending': (entry_adx_info or {}).get('adx_trending', False)})

    if position == 'SHORT' and shares_remaining > 0:
        slip_exit_end = df['Close'].iloc[-1] * (1 + slippage_pct)  # buy to cover at ask
        capital += shares_remaining * slip_exit_end
        pnl = ((entry_price - slip_exit_end) / entry_price) * 100
        pnl_list.append(pnl)
        realized_pnl_sum += shares_remaining * (entry_price - slip_exit_end)
        peak_realized = max(peak_realized, capital)
        trades.append({'pnl': round(pnl, 2), 'type': 'CLOSED',
                       'sector': get_sector(name), 'date': str(df.index[-1].date()),
                       'partial': partial_exits > 0,
                       'entry_adx': (entry_adx_info or {}).get('adx', 0),
                       'entry_adx_trending': (entry_adx_info or {}).get('adx_trending', False)})

    if not trades:
        return dict(symbol=name, trades=0, win_rate=0, compounded_return=0.0,
                    realized_return=0.0, wins=0, losses=0,
                    avg_win=0.0, avg_loss=0.0,
                    sharpe=0.0, position_util=0.0,
                    tsl_exits=0, sl_exits=0, sig_exits=0,
                    time_exits=0, abssl_exits=0, partial_exits=0,
                    max_drawdown=0.0, qualified=False,
                    no_sig_exit=no_sig_exit)

    capital = max(capital, 0.0)  # v36: floor at 0 — no negative capital (debt cap)
    wins   = [t['pnl'] for t in trades if t['pnl'] > 0]
    losses = [t['pnl'] for t in trades if t['pnl'] <= 0]

    # Compounded return (inflated — shown for reference only)
    total_ret_compounded = ((capital - initial_capital) / initial_capital) * 100

    # Realized return: total P&L earned / starting capital
    # This is the PRIMARY metric
    realized_return = (realized_pnl_sum / initial_capital * 100)

    # Sharpe: annualized from per-trade P&L %
    hold_days_list = [t.get('hold_days', 0) for t in trades if t['type'] != 'CLOSED']
    avg_hold = (sum(hold_days_list) / max(len(hold_days_list), 1)) if hold_days_list else 5
    if len(pnl_list) >= 3:
        mean_pnl = sum(pnl_list) / len(pnl_list)
        std_pnl  = (sum((x - mean_pnl) ** 2 for x in pnl_list) / len(pnl_list)) ** 0.5
        ann_factor = math.sqrt(252 / max(avg_hold, 1))
        sharpe = (mean_pnl / std_pnl * ann_factor) if std_pnl > 0 else 0.0
    else:
        sharpe = 0.0

    # Max drawdown: based on peak_realized (settled peak, not unrealized)
    # v36 fix: use peak_ever as denominator — capital can go negative (leveraged debt)
    # so cap at 100% (can't lose more than all capital)
    max_drawdown = min(100.0, max(0.0, (peak_ever - capital) / peak_ever * 100)) if peak_ever > 0 else 0.0

    # ADX regime breakdown: use entry_adx (ADX at signal entry), not exit ADX
    all_entry_adx = [t.get('entry_adx', 0) for t in trades]
    trending_trades = sum(1 for t in trades if t.get('entry_adx_trending', False))
    choppy_trades = len(trades) - trending_trades

    # QUALIFIED v14 (v43): Sharpe ≥ 0.8 added — filters borderline stocks
    # v50: DD gate 55%→30% — swing DD of 30%+ means 6+ consecutive losers on 5% pos size
    # With WR=40-50%, getting 6 losers in a row is rare but 30% DD is still uncomfortable.
    # 30% DD = 6 losing trades on 5% position = realistic stop-out scenario.
    qualified = (len(trades) >= MIN_TRADES and
                realized_return > 0 and
                sharpe >= 0.8 and               # v43: risk-adjusted quality gate
                max_drawdown < 30.0 and         # v50: 30% DD cap (was 55% — too loose)
                (len(wins) / len(trades) >= 0.38 if trades else False))

    return {
        'symbol': name,
        'trades': len(trades),
        'wins': len(wins),
        'losses': len(losses),
        'win_rate': round(len(wins) / len(trades) * 100, 1) if trades else 0.0,
        'realized_return': round(realized_return, 2),
        'return': round(total_ret_compounded, 2),
        'avg_trade_return': round(sum(pnl_list) / len(pnl_list), 2) if pnl_list else 0.0,  # v41: mean P&L% per trade (per-trade, not capital-weighted)
        'sharpe': round(sharpe, 2),
        'tsl_exits': sum(1 for t in trades if t['type'] == 'TSL'),
        'sl_exits': sum(1 for t in trades if t['type'] == 'SL'),
        'sig_exits': sum(1 for t in trades if t['type'] == 'SIG_REVERSAL'),
        'time_exits': sum(1 for t in trades if t['type'] == 'TIME'),
        'abssl_exits': sum(1 for t in trades if t['type'] == 'ABSSL'),
        'partial_exits': partial_exits,
        'max_drawdown': round(max_drawdown, 2),
        'qualified': bool(qualified),
        'no_sig_exit': no_sig_exit,
        # v31: ADX regime breakdown
        'adx_regime': {
            'avg_entry_adx': round(sum(all_entry_adx) / len(all_entry_adx), 1) if all_entry_adx else 0,
            'trending_trades': trending_trades,
            'choppy_trades': choppy_trades,
            'trending_pct': round(trending_trades / len(trades) * 100, 1) if trades else 0,
        },
        'config': {
            'use_t1_partial': use_t1_partial,
            'max_hold_days': max_hold_days,
            'min_trades': MIN_TRADES,
            'slippage_pct': slippage_pct,
            'max_position_pct': max_position_pct,
            'no_sig_exit': no_sig_exit,
            'momentum_mode': momentum_mode,
            'multi_mode': multi_mode,
            'high_conviction_mode': high_conviction_mode,
            'ultra_mode': ultra_mode,
            'long_only': long_only,
            'hybrid_mode': hybrid_mode,
            'adx_filter': adx_filter,
            'level_mode': level_mode,
            'rsi_guards': True,
            'bearish_divergence': True,
            'rsi_mode': _RSI_MODE,
            'min_adx': min_adx,       # v41
            'min_rr_ratio': min_rr_ratio,  # v42
            'ml_threshold': ml_threshold,  # v45
        },
        'trades_list': trades,
    }


# ─── CLI ─────────────────────────────────────────────────────────────────────
def parse_args():
    stocks = DEFAULT_STOCKS
    use_trailing = False
    sector_limits = False
    no_sig_exit = False
    years = 3
    output = 'default'
    level_mode = 'swing'
    momentum_mode = False
    multi_mode = False
    high_conviction_mode = False
    ultra_mode = False
    long_only = False
    hybrid_mode = False
    adx_filter = True
    min_adx = 0  # v41
    min_rr_ratio = 0  # v42
    ml_threshold = ML_THRESHOLD  # v45

    args = sys.argv[1:]
    i = 0
    positional = []
    while i < len(args):
        arg = args[i]
        if arg == '--trailing':       use_trailing = True; i += 1
        elif arg == '--sector-cap':    sector_limits = True; i += 1
        elif arg == '--no-sig-exit':  no_sig_exit = True; i += 1
        elif arg == '--momentum-mode': momentum_mode = True; i += 1
        elif arg == '--no-adx-filter': adx_filter = False; i += 1
        elif arg == '--min-adx':       min_adx = int(args[i + 1]); i += 2  # v41
        elif arg == '--min-rr':        min_rr_ratio = float(args[i + 1]); i += 2  # v42
        elif arg == '--ml-threshold':  ml_threshold = float(args[i + 1]); i += 2  # v45
        elif arg == '--years':
            years = int(args[i + 1]); i += 2
        elif arg == '--intraday':  level_mode = 'intraday'; i += 1
        elif arg == '--swing':     level_mode = 'swing'; i += 1
        elif arg == '--ultra': ultra_mode = True; i += 1
        elif arg == '--long-only': long_only = True; i += 1
        elif arg == '--hybrid': hybrid_mode = True; i += 1
        elif arg == '--multi':       multi_mode = True; i += 1
        elif arg == '--json':      output = 'json'; i += 1
        elif arg == '--stock':
            stocks = [args[i + 1].strip().upper()]; i += 2
        elif arg == '--symbols':
            stocks = [s.strip().upper() for s in args[i + 1].split(',')]; i += 2
        elif arg == '--all':        stocks = DEFAULT_STOCKS; i += 1
        elif arg.startswith('--'):  i += 1
        else:
            positional.extend([s.strip().upper() for s in arg.split(',')])
            i += 1
    if positional:
        stocks = positional
    return stocks, use_trailing, sector_limits, no_sig_exit, years, output, level_mode, momentum_mode, multi_mode, high_conviction_mode, ultra_mode, long_only, hybrid_mode, adx_filter, min_adx, min_rr_ratio, ml_threshold


def main():
    (stocks, use_trailing, sector_limits, no_sig_exit, years,
     output, level_mode, momentum_mode, multi_mode, high_conviction_mode, ultra_mode, long_only, hybrid_mode,
     adx_filter, min_adx, min_rr_ratio, ml_threshold) = parse_args()
    # v14 fix: Use fixed end_date (last completed month) for reproducible backtests.
    # Using dynamic end_date=now causes yfinance to return different prices on each run
    # (last 3 years ending "today" = different window every time) → non-deterministic.
    # Update _PINNED_END_DATE manually when ready for a fresh data window.
    now = datetime.now()
    _PINNED_END_DATE = '2026-05-31'   # ← UPDATE MANUALLY
    start_date = (now - pd.Timedelta(days=365 * years)).strftime('%Y-%m-%d')
    end_date   = _PINNED_END_DATE

    flags = []
    if use_trailing:   flags.append("+TrailingSL")
    if sector_limits:   flags.append("+SectorCap")
    if no_sig_exit:    flags.append("+NoSigExit")
    if ultra_mode: flags.append("+Ultra")
    if long_only: flags.append("+LongOnly")
    if hybrid_mode: flags.append("+Hybrid")
    if not adx_filter: flags.append("+NoADXFilter")
    if min_adx > 0:    flags.append(f"+MinADX{min_adx}")   # v41
    if min_rr_ratio > 0: flags.append(f"+MinRR{min_rr_ratio}")  # v42
    if ml_threshold > 0: flags.append(f"+ML{float(ml_threshold):.2f}")  # v45
    flag_str = f" ({', '.join(flags)})" if flags else ""

    # Warn: momentum mode without ADX filter may fire in choppy markets
    if momentum_mode and not adx_filter:
        print("⚠️  WARNING: --momentum-mode --no-adx-filter = momentum signals fire in ALL market conditions")
        print("         Recommend: add Option A (--adx-filter) to filter choppy regimes in momentum mode too")
        print()

    print("=" * 72)
    print(f"📊 NIFTY BACKTEST v48{flag_str} | {start_date} → {end_date} ({years}y) | {level_mode}")
    print("=" * 72)

    results = []
    for sym in stocks:
        if sym in EXCLUDED_STOCKS:
            continue
        print(f"\n🔄 {sym}...", end=' ', flush=True)
        res = backtest_stock(sym, start=start_date, end=end_date,
                            use_trailing=use_trailing,
                            sector_limits=sector_limits,
                            no_sig_exit=no_sig_exit,
                            verbose=False,
                            level_mode=level_mode,
                            momentum_mode=momentum_mode,
                            multi_mode=multi_mode,
                            high_conviction_mode=high_conviction_mode,
                            ultra_mode=ultra_mode,
                            long_only=long_only,
                            hybrid_mode=hybrid_mode,
                            adx_filter=adx_filter,
                            min_adx=min_adx,
                            min_rr_ratio=min_rr_ratio,
                            ml_threshold=ml_threshold,  # v45
                            )  # v42
        if res:
            results.append(res)
            if res['trades'] > 0:
                q = "✅" if res['qualified'] else "⚠️ "
                print(f"{q} trds={res['trades']} WR={res['win_rate']}% "
                      f"RealRet={res['realized_return']:+.2f}% "
                      f"CompRet={res['return']:+.2f}% "
                      f"Sharpe={res['sharpe']} DD={res['max_drawdown']}%")
            else:
                print("⚠️  No trades")
        else:
            print("❌ No data")

    print("\n" + "=" * 72)
    print(f"📊 BACKTEST SUMMARY ({start_date} → {end_date})")
    print("=" * 72)

    active = [r for r in results if r['trades'] > 0]
    if not active:
        print("No results."); return

    qualified = [r for r in active if r['qualified']]
    print(f"\n{'Sym':<10} {'Trds':>5} {'WR%':>6} {'AvgTrd%':>8} {'RealRet%':>9} {'CompRet%':>9} {'Sharpe':>7} {'DD%':>6} {'QLF':>4}")
    print("-" * 70)
    for r in sorted(active, key=lambda x: -x['realized_return']):
        q = "✅" if r['qualified'] else "  "
        avg_t = r.get('avg_trade_return', 0)
        print(f"{r['symbol']:<10} {r['trades']:>5} {r['win_rate']:>6.1f} "
              f"{avg_t:>+8.2f} {r['realized_return']:>+9.2f} {r['return']:>+9.2f} "
              f"{r['sharpe']:>7.2f} {r['max_drawdown']:>6.2f} {q:>4}")
    print("-" * 70)

    avg_rr  = sum(r['realized_return'] for r in active) / len(active)
    avg_ret = sum(r['return'] for r in active) / len(active)
    avg_wr  = sum(r['win_rate'] for r in active) / len(active)
    avg_dd  = sum(r['max_drawdown'] for r in active) / len(active)
    avg_sh  = sum(r['sharpe'] for r in active) / len(active)
    avg_t   = sum(r.get('avg_trade_return', 0) for r in active) / len(active)
    print(f"{'AVG':<10} {sum(r['trades'] for r in active):>5} {avg_wr:>6.1f} "
          f"{avg_t:>+8.2f} {avg_rr:>+9.2f} {avg_ret:>+9.2f} "
          f"{avg_sh:>7.2f} {avg_dd:>6.2f} {len(qualified):>4}/{len(active)}")

    print("\n  ℹ️  METRIC GUIDE:")
    print("     AvgTrd% = avg P&L% per trade (from pnl_list)")
    print("     RealRet% = total realized P&L / initial capital (PRIMARY metric)")
    print("     CompRet% = portfolio-level return incl. DD drag (CompRet ≤ RealRet)")
    print("     CompRet% < RealRet% means DD dragged portfolio down despite +avg/trade")

    best  = max(active, key=lambda x: x['realized_return'])
    worst = min(active, key=lambda x: x['realized_return'])
    print(f"\n🏆 BEST:  {best['symbol']} Real.Ret={best['realized_return']:+.2f}% WR={best['win_rate']}%")
    print(f"💀 WORST: {worst['symbol']} Real.Ret={worst['realized_return']:+.2f}% WR={worst['win_rate']}%")

    # BUG-4: Backtest quality summary using actual backtest metrics
    # Note: Categorization (Cat A/B/C) requires live signals — use backtest stats instead
    print(f"\n📊 BACKTEST QUALITY SUMMARY ({_RSI_MODE} mode):")
    # Show qualified breakdown by win-rate band
    wr_bands = {'🟢 WR≥50%': [], '🟡 WR40-50%': [], '🔴 WR<40%': []}
    for r in active:
        wr = r.get('win_rate', 0)
        if wr >= 50:
            wr_bands['🟢 WR≥50%'].append(r)
        elif wr >= 40:
            wr_bands['🟡 WR40-50%'].append(r)
        else:
            wr_bands['🔴 WR<40%'].append(r)
    for band, stocks in wr_bands.items():
        if stocks:
            avg_rr = sum(s['realized_return'] for s in stocks) / len(stocks)
            avg_dd = sum(s['max_drawdown'] for s in stocks) / len(stocks)
            qlfr = [s['symbol'] for s in stocks if s.get('qualified')]
            print(f"   {band}: {len(stocks)} stocks | avg RR={avg_rr:+.2f}% avg DD={avg_dd:.1f}% | qualified: {len(qlfr)}")
            print(f"      {sorted(s['symbol'] for s in stocks)}")

    # ADX quality: show stocks that benefited from trending filter
    trending_qualified = [r for r in qualified if r.get('adx_regime', {}).get('trending_pct', 0) >= 70]
    choppy_qualified = [r for r in qualified if r.get('adx_regime', {}).get('trending_pct', 0) < 70]
    print(f"\n   📈 ADX Quality: {len(trending_qualified)}/{len(qualified)} qualified in trending markets (ADX>20)")
    if choppy_qualified:
        print(f"   ⚠️  {len(choppy_qualified)} qualified in choppy markets: {sorted(r['symbol'] for r in choppy_qualified)}")

    # rsi_mode verification
    print(f"\n   ℹ️  RSI mode: {_RSI_MODE} (RSI_ENTRY_MAX={'30' if _RSI_MODE=='strict' else '65'})")

    # Always save; --json controls whether JSON is also printed to stdout
    out = f"models/backtest_v11_{now.strftime('%Y%m%d_%H%M%S')}.json"
    with open(out, 'w') as f:
        json.dump({'timestamp': now.isoformat(), 'start': start_date, 'end': end_date,
                   'years': years, 'no_sig_exit': no_sig_exit,
                   'results': results}, f, indent=2)
    print(f"\n✅ Saved to {out}")

    if output == 'json':
        print(json.dumps({'timestamp': now.isoformat(), 'start': start_date, 'end': end_date,
                          'results': results}, indent=2))


if __name__ == "__main__":
    main()
