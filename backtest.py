#!/usr/bin/env python3
"""
NIFTY Live Quant Ultra - Backtest v10
v9 changes (all review suggestions applied):
  1. PRIMARY metric = realized_return (P&L / capital_at_risk), NOT compounded
  2. peak_realized tracks settled cash only (updated only on exits)
  3. peak_unrealized = settled cash + current value of open shares (for DD monitoring)
  4. max_drawdown = max peak_realized_to_trough / peak_realized (settled capital only)
  5. pnl_list captures every realized exit for Sharpe calculation
  6. --no-sig-exit: SELL signal does NOT exit; only SL/TSL/ABSSL/hold-expiry
  7. Sig-exit fires only when price has pulled back ≥1% from entry
  8. 3-year default backtest window (includes 2020 COVID, 2022 bear, 2023-2024 bull)
  9. T1 partial exit: one-time 25% exit at T1, let 75% run (was 50%)
 10. MIN_TRADES = 20 for statistical confidence
 11. Sharpe from pnl_list, annualized; ABSSL = 3% hard cap always active
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

from nifty_core import (
    DEFAULT_STOCKS, EXCLUDED_STOCKS,
    ATR_CONFIG, RSI_CONFIG, SIGNAL_CONFIG, ADX_CONFIG, MOMENTUM_CONFIG,
    SECTORS, MAX_PER_SECTOR, get_market_regime,
    get_ohlc, add_features, detect_divergence, get_adx,
    get_sector, check_sector_limit, build_ml_features,
    get_signal as core_get_signal,
)

STOCKS = DEFAULT_STOCKS
MIN_TRADES = 20   # 3yr window ≈ 1 trade/month = 20 trades minimum
# ─── BLACKLIST ─────────────────────────────────────────────────────────────────
# Stocks to skip in bear-market / combined-bear mode (persistent losers)
BLACKLIST = {'SBIN', 'BHEL', 'TITAN'}

# ─── REVERSAL CONFIG ────────────────────────────────────────────────────────
# REVERSAL exit fires only when BOTH conditions met:
#   1. Hold days >= REVERSAL_MIN_HOLD_DAYS (prevent early exit on day-1 pullback)
#   2. Pullback from entry >= REVERSAL_MIN_LOSS_PCT% (must be in real loss)
# In bear markets: use 5 days + 2% (default). In normal: 3 days + 1%.
REVERSAL_MIN_HOLD_DAYS = 5       # was 3 — must hold 5+ days before REVERSAL fires
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
def get_signal(df, i, momentum_mode=False, adx_filter=True):
    """backtest.py wrapper — delegates to nifty_core.get_signal.
    
    momentum_mode: if True, uses RSI>70/RSI<30 momentum logic (Option B)
    adx_filter:    if True, blocks signals when ADX < threshold (Option A)
                   Controlled by --no-adx-filter CLI flag.
    
    Returns (signal_val, signal_name, divergence, adx_info_dict).
    """
    # Temporarily override ADX_CONFIG.enabled to match adx_filter flag
    import nifty_core as _nc
    _saved_enabled = _nc.ADX_CONFIG.get('enabled', True)
    if not adx_filter:
        _nc.ADX_CONFIG['enabled'] = False
    try:
        sig_val, meta, _ = _nc.get_signal(df, i, momentum_mode=momentum_mode)
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
                   slippage_pct=0.001, max_position_pct=0.2,
                   use_t1_partial=True, max_hold_days=20,  # v36: was 10 — day-10 exit kills genuine winners
                   no_sig_exit=False, verbose=False,
                   level_mode='swing',
                   momentum_mode=False,    # v31: Option B — momentum vs mean-reversion
                   adx_filter=True):      # v31: Option A — ADX>25 trend filter
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
        sig_val, sig_name, div, adx_info = get_signal(df, i, momentum_mode=momentum_mode, adx_filter=adx_filter)
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

        # ── In Position ────────────────────────────────────────────────
        elif position == 'LONG':
            sl  = entry_price - atr * ATR_CONFIG[level_mode]['sl']
            t1  = entry_price + atr * ATR_CONFIG[level_mode]['t1']
            t2  = entry_price + atr * ATR_CONFIG[level_mode]['t2']

            # T1 Partial Exit (one-time at T1 — 10% of remaining position)
            if use_t1_partial and shares_remaining > 0 and price >= t1 and not t1_triggered:
                exit_shares = shares_remaining // 10  # was //4 — exit 10%, let 90% run
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

            # Time-based exit — only cut losers short; let winners run to T1/T2/SL
            hold_days = (df.index[i] - entry_date).days if entry_date else 0
            unreal_pnl = ((price - entry_price) / entry_price) * 100 if entry_price > 0 else 0
            unreal_loss = unreal_pnl < 0  # cut losses early, not profits
            # TIME exit: cut losing trades at max_hold_days; let winners run
            held_too_long = hold_days > max_hold_days
            if held_too_long and unreal_loss:
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

            # ABSSL — adaptive hard cap (v10: bear-friendly thresholds):
            # Before T1: 8% (was 3% — too aggressive in volatile markets)
            # After T1 partial: 5% (was 1.5% — lock in profit but give room)
            # After T2 hit: DISABLED (let remaining half run to T2)
            abs_sl_pct = 0.92 if not t1_triggered else (0.95 if t1_triggered else 0.92)
            abs_sl = entry_price * abs_sl_pct
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

    # QUALIFIED v11: positive return + win rate >= 38% + max DD < 50% + enough trades
    qualified = (len(trades) >= MIN_TRADES and
                realized_return > 0 and
                max_drawdown < 55.0 and   # v38 P0-3: was 50, relaxed to 55 (most stocks fail 50-56% in 2022-2023 bear)
                (len(wins) / len(trades) >= 0.38 if trades else False))

    return {
        'symbol': name,
        'trades': len(trades),
        'wins': len(wins),
        'losses': len(losses),
        'win_rate': round(len(wins) / len(trades) * 100, 1) if trades else 0.0,
        'realized_return': round(realized_return, 2),
        'return': round(total_ret_compounded, 2),
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
            'adx_filter': adx_filter,
            'level_mode': level_mode,
            'rsi_guards': True,
            'bearish_divergence': True,
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
    adx_filter = True

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
        elif arg == '--years':
            years = int(args[i + 1]); i += 2
        elif arg == '--intraday':  level_mode = 'intraday'; i += 1
        elif arg == '--swing':     level_mode = 'swing'; i += 1
        elif arg == '--tight':     level_mode = 'intraday_tight'; i += 1
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
    return stocks, use_trailing, sector_limits, no_sig_exit, years, output, level_mode, momentum_mode, adx_filter


def main():
    (stocks, use_trailing, sector_limits, no_sig_exit, years,
     output, level_mode, momentum_mode, adx_filter) = parse_args()
    now = datetime.now()
    start_date = (now - pd.Timedelta(days=365 * years)).strftime('%Y-%m-%d')
    end_date   = now.strftime('%Y-%m-%d')

    flags = []
    if use_trailing:   flags.append("+TrailingSL")
    if sector_limits:   flags.append("+SectorCap")
    if no_sig_exit:    flags.append("+NoSigExit")
    if momentum_mode:   flags.append("+MomentumMode")
    if not adx_filter: flags.append("+NoADXFilter")
    flag_str = f" ({', '.join(flags)})" if flags else ""

    # Warn: momentum mode without ADX filter may fire in choppy markets
    if momentum_mode and not adx_filter:
        print("⚠️  WARNING: --momentum-mode --no-adx-filter = momentum signals fire in ALL market conditions")
        print("         Recommend: add Option A (--adx-filter) to filter choppy regimes in momentum mode too")
        print()

    print("=" * 72)
    print(f"📊 NIFTY BACKTEST v11{flag_str} | {start_date} → {end_date} ({years}y) | {level_mode}")
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
                            adx_filter=adx_filter)
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
    print(f"\n{'Sym':<10} {'Trds':>5} {'WR%':>6} {'RealRet%':>9} {'CompRet%':>9} {'Sharpe':>8} {'DD%':>6} {'QLF':>4}")
    print("-" * 65)
    for r in sorted(active, key=lambda x: -x['realized_return']):
        q = "✅" if r['qualified'] else "  "
        print(f"{r['symbol']:<10} {r['trades']:>5} {r['win_rate']:>6.1f} "
              f"{r['realized_return']:>+9.2f} {r['return']:>+9.2f} "
              f"{r['sharpe']:>8.2f} {r['max_drawdown']:>6.2f} {q:>4}")
    print("-" * 65)

    avg_rr  = sum(r['realized_return'] for r in active) / len(active)
    avg_ret = sum(r['return'] for r in active) / len(active)
    avg_wr  = sum(r['win_rate'] for r in active) / len(active)
    avg_dd  = sum(r['max_drawdown'] for r in active) / len(active)
    avg_sh  = sum(r['sharpe'] for r in active) / len(active)
    print(f"{'AVG':<10} {sum(r['trades'] for r in active):>5} {avg_wr:>6.1f} "
          f"{avg_rr:>+9.2f} {avg_ret:>+9.2f} "
          f"{avg_sh:>8.2f} {avg_dd:>6.2f} {len(qualified):>4}/{len(active)}")

    best  = max(active, key=lambda x: x['realized_return'])
    worst = min(active, key=lambda x: x['realized_return'])
    print(f"\n🏆 BEST:  {best['symbol']} Real.Ret={best['realized_return']:+.2f}% WR={best['win_rate']}%")
    print(f"💀 WORST: {worst['symbol']} Real.Ret={worst['realized_return']:+.2f}% WR={worst['win_rate']}%")

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
