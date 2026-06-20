#!/usr/bin/env python3
"""
nifty_categorize.py — Shared Stock Categorization Logic
======================================================
Single source of truth for stock categorization.
Imported by BOTH scan.py (live) and backtest.py (historical).

Ensures backtest and live scan use IDENTICAL categorization logic.

Exports:
    categorize_results() — main entry point (alias for _categorize)
    get_primary_trigger() — alias for _get_primary_trigger
    _is_neg_hist, _is_poor_history, _wr_badge,
    _regime_coherence, _stale_tier, calc_confluence_score
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from nifty_core import (
    RSI_CONFIG, SIGNAL_CONFIG, ADX_CONFIG, MOMENTUM_CONFIG,
    check_level_alignment,
)

# ─── Config (mirrored from scan.py for self-contained module) ─────────────────
_NEG_HIST_RR_THRESHOLD = -2.0
_NEG_HIST_WR_THRESHOLD = 40
_MIN_WR_CAT_A = 45        # Cat A: WR >= 45%
_MIN_WR_CAT_B = 35        # Cat B: WR >= 35% (v35)
_MIN_WR_TOP_PICK = 40     # TOP_PICK: WR >= 40%
_BACKTEST_STALE_DAYS = 3
_LOW_WR_WARNING = 35
_POS_SIZE_WARNING_PCT = 10


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _is_neg_hist(stats):
    """NEG_HIST: rr < -2% AND wr < 40% — poor return AND weak win rate together."""
    rr = stats.get('realized_return', 0)
    wr = stats.get('win_rate', 0)
    return rr < _NEG_HIST_RR_THRESHOLD and wr < _NEG_HIST_WR_THRESHOLD


def _is_poor_history(stats):
    """POOR_HIST: rr < -2% OR wr < 45% — aligned with _MIN_WR_CAT_A=45 (v37)."""
    rr = stats.get('realized_return', 0)
    wr = stats.get('win_rate', 0)
    return rr < -2.0 or wr < 45


def _wr_badge(wr):
    """Return WR color badge: 🟢 >50%, 🟡 40-50%, 🔴 <40%."""
    if wr >= 50:
        return '🟢'
    elif wr >= _LOW_WR_WARNING:
        return '🟡'
    else:
        return '🔴'


def _pos_size_warning(pos_pct):
    """Return position sizing warning if > 10% of capital."""
    if pos_pct > _POS_SIZE_WARNING_PCT:
        return f'⚠️ OVERSIZE({pos_pct:.0f}%)'
    return None


def _regime_coherence(signal, divergence, regime):
    """Compute regime fit for a signal.
    Returns: (score, label, is_contrarian)
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
    else:
        return 0.3, 'RANGE_BOUND', False
    return 0.5, 'NEUTRAL', False


def _stale_tier(age_days):
    """Return stale tier and penalty multiplier.
    Fresh: < 4 days → no penalty
    Stale: 4-7 days → 10% penalty
    Critical: 8+ days → 30% penalty
    """
    if age_days < 4:
        return 'FRESH', 0.0
    elif age_days < 8:
        return 'STALE', 0.10
    else:
        return 'CRITICAL', 0.30


def calc_confluence_score(r, regime='BULLISH'):
    """Calculate 1-10 confluence score for a stock.

    Components: Signal_Conf×0.20 + AI_Conf×0.25 + WR×0.20 +
                Level_Align×0.10 + Age×0.10 + RegimeFit×0.15

    NEG_HIST grace: rr >= -2% passes, only rr < -2% triggers 30% penalty.
    """
    sig_conf = r.get('prob', 50) / 100.0
    ai = r.get('ai') or {}
    ml = r.get('ml') or {}
    ai_score = ai.get('total_score', 0)
    ai_conf = (ai_score + 100) / 200.0
    sig = r['signal']
    ai_dir = ai.get('outlook', 'NEUTRAL')
    if sig == 'BUY' and ai_dir == 'BULLISH':
        ai_dir_score = 1.0
    elif sig == 'SELL' and ai_dir == 'BEARISH':
        ai_dir_score = 1.0
    elif sig == 'BUY' and ai_dir == 'BEARISH':
        ai_dir_score = 0.0
    elif sig == 'SELL' and ai_dir == 'BULLISH':
        ai_dir_score = 0.0
    else:
        ai_dir_score = 0.5
    ai_final = ai_conf * 0.5 + ai_dir_score * 0.5
    stats = r.get('_stats', {})
    wr = stats.get('win_rate', 0) / 100.0
    align = r.get('_level_align', 'ALIGNED')
    align_score = 1.0 if align == 'ALIGNED' else (0.5 if align == 'WARN' else 0.0)
    age_days = r.get('signal_age_days', 0)
    age_tier, age_penalty = _stale_tier(age_days)
    age_score = max(0, 1.0 - age_penalty)
    div = r.get('divergence')
    _, regime_label, is_contrarian = _regime_coherence(sig, div, regime)
    regime_score = 1.0 if not is_contrarian else 0.3
    score = (sig_conf * 0.20 + ai_final * 0.25 + wr * 0.20
             + align_score * 0.10 + age_score * 0.10 + regime_score * 0.15)
    neg_hist = stats.get('realized_return', 0) < _NEG_HIST_RR_THRESHOLD
    if neg_hist:
        score *= 0.7
    return round(score * 10, 1), age_tier, regime_label


def _add_tag(r, tag):
    if 'tags' not in r:
        r['tags'] = []
    if tag not in r['tags']:
        r['tags'].append(tag)


def get_primary_trigger(meta, rsi, divergence, signal):
    """v38 P2-3: Label the primary reason for the signal."""
    if signal == 'BUY':
        if rsi < 30:
            return 'MEAN_REVERSION'
        reasons = meta.get('reasons', [])
        for r in reasons:
            if 'MA200' in r:
                return 'BREAKOUT'
            if 'MA50' in r:
                return 'BREAKOUT'
            if 'MA20' in r:
                return 'BREAKOUT'
        if divergence == 'BULLISH':
            return 'DIVERGENCE_LONG'
        if meta.get('adx_trending'):
            return 'TREND_FOLLOW'
        return 'MEAN_REVERSION'
    elif signal == 'SELL':
        if rsi > 70:
            return 'MOMENTUM'
        if divergence == 'BEARISH':
            return 'DIVERGENCE_SHORT'
        reasons = meta.get('reasons', [])
        for r in reasons:
            if 'MA200' in r:
                return 'BREAKDOWN'
        if meta.get('adx_trending'):
            return 'TREND_FOLLOW'
        return 'MOMENTUM'
    return ''


# Alias for external use
categorize_results = None   # set after function is defined


def _categorize(results, regime='BULLISH'):
    """Categorize results into Cat A/A-/B/C1/C2/D/WATCHLIST.

    Single source of truth — used by BOTH scan.py (live) and backtest.py.

    Cat A:   Triple confirmed (Signal + AI_BULL + ML_UP) + wr >= 45% + not poor_hist
    Cat A-:  2-of-3 confirmed + profitable but level mismatch
    Cat B:   AI HIGH/MEDIUM + wr >= 35% + rr >= -2% + no severe NEG_HIST
    Cat C1:  Signal + AI agree (same direction) + not Cat A/A-
    Cat C2:  Signal only, AI neutral, NEG_HIST, or BEAR_DIV in BULL regime
    Cat D:   ML signal only + AI neutral
    WATCHLIST: RANGE-bound stocks
    """
    cat_a, cat_a_minus, cat_b, cat_c1, cat_c2, cat_d, watchlist = [], [], [], [], [], [], []
    for r in results:
        ai = r.get('ai') or {}
        ml = r.get('ml') or {}
        ai_dir = ai.get('outlook', 'NEUTRAL')
        ai_conf = ai.get('confidence', 'LOW')
        ai_t1 = ai.get('stages', {}).get('6_risk_manager', {}).get('t1')
        ml_dir = (ml.get('direction', None) if ml else None)
        div = r.get('divergence')
        sig = r['signal']
        stats = r.get('_stats', {})
        r['_stats'] = stats
        rr = stats.get('realized_return', 0)
        wr = stats.get('win_rate', 0)
        no_history = (wr == 0 and rr == 0)
        neg_hist_severe = _is_neg_hist(stats) or rr < -5.0
        poor_hist = _is_poor_history(stats)

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

        # Confluence score
        r['_confluence'], _, r['_regime_label'] = calc_confluence_score(r, regime)

        def _tag_all(r_obj):
            if neg_hist_severe:
                _add_tag(r_obj, '⚠️ NEG_HIST')
            if age_tier == 'STALE':
                _add_tag(r_obj, '⏰ STALE')
            if age_tier == 'CRITICAL':
                _add_tag(r_obj, '💀 STALE_CRITICAL')
            if not align_ok:
                _add_tag(r_obj, f'⚠️ LVL_{align_status}')
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
            s3 = ai.get('stages', {}).get('3_stock_scanner', {})
            adx_val = s3.get('adx', 0)
            adx_trending = s3.get('adx_trending', None)
            adx_ok = bool(adx_trending is True or (adx_trending is None and adx_val >= 25))

            # BEAR_DIV path: SHORT setup — promote/degrade based on regime
            if div == 'BEARISH':
                _, regime_lbl, is_contra = _regime_coherence(sig, div, regime)
                if is_contra:
                    _add_tag(r, 'BEAR_DIV')
                    _add_tag(r, '🔴 CONTRARIAN')
                else:
                    _add_tag(r, 'BEAR_DIV')
                # v38 P1-3: In BEARISH regime, BEAR_DIV + BUY = always Cat C2
                if poor_hist or no_history:
                    _tag_all(r)
                    cat_c2.append(r)
                elif sig == 'BUY':
                    _tag_all(r)
                    cat_c2.append(r)
                else:
                    _tag_all(r)
                    cat_c1.append(r)

            elif ai_bull and ml_up:
                # Cat A: triple confirmed
                rsi = r.get('rsi', 0)
                if rsi > RSI_CONFIG['buy_relaxed']:
                    _add_tag(r, '⚠️ OVERBOUGHT')
                    _tag_all(r)
                    cat_c2.append(r)
                elif not adx_ok:
                    _add_tag(r, '⚠️ LOW_ADX')
                    _tag_all(r)
                    cat_a_minus.append(r)
                elif no_history:
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
                wr_badge = _wr_badge(wr)
                if wr > 0 and wr < _LOW_WR_WARNING:
                    _add_tag(r, f'⚠️ LOW_WINRATE({wr:.0f}%)')
                if neg_hist_severe:
                    _add_tag(r, '⚠️ NEG_HIST')
                    _tag_all(r)
                    cat_c2.append(r)
                elif no_history:
                    _add_tag(r, '⚠️ NO_BACKTEST')
                    _tag_all(r)
                    cat_c2.append(r)
                elif wr > 0 and wr < _MIN_WR_CAT_B:
                    _add_tag(r, f'⚠️ WR_LOW({wr:.0f}%)')
                    _tag_all(r)
                    cat_c2.append(r)
                else:
                    ml_dir_ml = (ml or {}).get('direction', '')
                    if ml_dir_ml == 'DOWN' and wr < 50:
                        _add_tag(r, '⚠️ ML_CONTRADICT')
                        _tag_all(r)
                        cat_c2.append(r)
                    else:
                        if ml_dir_ml == 'DOWN':
                            _add_tag(r, '⚠️ ML_CONTRADICT')
                        _tag_all(r)
                        if not align_ok:
                            _add_tag(r, 'UNCONFIRMED')
                        cat_b.append(r)

            elif ml_up:
                if sig == 'SELL':
                    cat_c1.append(r)
                elif ai_dir == 'NEUTRAL':
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

            if ml_down:
                if ai_dir == 'BULLISH':
                    _add_tag(r, 'AI_CONTRADICT')
                    _tag_all(r)
                    cat_c2.append(r)
                elif poor_hist:
                    _add_tag(r, '⚠️ POOR_HIST')
                    _tag_all(r)
                    cat_c1.append(r)
                else:
                    _tag_all(r)
                    if not align_ok:
                        _add_tag(r, 'UNCONFIRMED')
                        cat_a_minus.append(r)
                    else:
                        cat_a.append(r)

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
                    if regime == 'BULLISH':
                        _add_tag(r, '🛡️ HEDGE')
                        cat_a_minus.append(r)
                    else:
                        cat_c1.append(r)

            else:
                if no_history:
                    _add_tag(r, '⚠️ NO_BACKTEST')
                _tag_all(r)
                cat_c2.append(r)

    # TOP_PICK selection
    for r in cat_a + cat_b:
        stats = r.get('_stats', {})
        rr = stats.get('realized_return', 0)
        wr = stats.get('win_rate', 0)
        cf = r.get('_confluence', 0)
        ml = r.get('ml') or {}
        ml_dir = ml.get('direction', '')
        if ml_dir == 'DOWN':
            continue
        if rr > 0 and cf >= 8.0 and wr >= _MIN_WR_TOP_PICK:
            r['_starred'] = True
            _add_tag(r, '⭐ TOP_PICK')

    return cat_a, cat_a_minus, cat_b, cat_c1, cat_c2, cat_d, watchlist


# Set the alias
categorize_results = _categorize
