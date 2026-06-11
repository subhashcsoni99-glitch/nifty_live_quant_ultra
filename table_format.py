#!/usr/bin/env python3
"""
NIFTY Scanner - Table format (like v17 example)
Includes: Price, Conf%, RSI, AI score, Entry/SL/T1/T2
Used when --format table is passed to scan.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scan import _categorize

def ai_score_emoji(score):
    if score is None: return ("—", "⚪")
    if score >= 80: return f"+{score:.0f}", "🟢"
    elif score >= 50: return f"+{score:.0f}", "🟡"
    else: return f"{score:.0f}", "🔴"

def run_table_format(results, today):
    """Table format using pre-computed results from scan.py"""

    buy = [r for r in results if r['signal'] == 'BUY']
    sell = [r for r in results if r['signal'] == 'SELL']

    bullish_count = sum(1 for r in results if (r.get('ai') or {}).get('outlook') == 'BULLISH')
    bear_count = sum(1 for r in results if (r.get('ai') or {}).get('outlook') == 'BEARISH')
    regime = "BULLISH" if bullish_count > bear_count else ("BEARISH" if bear_count > bullish_count else "NEUTRAL")
    regime_icon = "🟢" if regime == "BULLISH" else "🔴" if regime == "BEARISH" else "🟡"

    cat_a, cat_a_minus, cat_b, cat_c1, cat_c2, cat_d, watchlist = _categorize(results, regime=regime)

    def inv(price, sl, t1, t2):
        return {
            'sl': round(price + (price - sl), 0),
            't1': round(price - (t1 - price), 0),
            't2': round(price - (t2 - price), 0),
        }

    def round_lev(d):
        return {k: round(v, 0) for k, v in d.items()}

    def get_levels(r, sig):
        id_sl = r.get('sl_intraday', r['sl'])
        id_t1 = r.get('t1_intraday', r['t1'])
        id_t2 = r.get('t2_intraday', r['t2'])
        if sig == 'SELL':
            id_l = round_lev(inv(r['price'], id_sl, id_t1, id_t2))
        else:
            id_l = {'sl': round(id_sl, 1), 't1': round(id_t1, 0), 't2': round(id_t2, 0)}
        return id_l

    def fmt_cat_table(cat_list, label, emoji, sig, max_rows=10):
        if not cat_list:
            return f"\n{emoji} {label} [0]\n"
        sorted_s = sorted(cat_list, key=lambda x: (-x.get('prob', 0), -x.get('_confluence', 0)))[:max_rows]
        out = f"\n{emoji} {label} [{len(cat_list)}]\n"
        # v21: RSI padded (6 chars), Entry(CMP) 14 chars, all price cols 12 chars
        out += "| # | Stock         | Price      | Conf% |  RSI  | AI       | Entry(CMP)   | SL          | T1          | T2          |\n"
        out += "|---|--------------|------------|-------|------|----------|---------------|--------------|--------------|--------------|\n"
        for i, r in enumerate(sorted_s, 1):
            star = "🏅 " if r.get('_starred') else ""
            ai_data = r.get('ai', {})
            outlook = ai_data.get('outlook', '')
            bear_sym = " 🐻" if outlook == 'BEARISH' else ""
            ai_score_val = ai_data.get('total_score', 0)
            ai_score_str, ai_color = ai_score_emoji(ai_score_val)
            rsi = r.get('rsi', 0)
            prob = r.get('prob', 0)
            price = r['price']
            id_l = get_levels(r, sig)
            entry = price
            sl = id_l['sl']
            t1 = id_l['t1']
            t2 = id_l['t2']
            row = (f"| {i} | {star}{r['symbol']:<12}{bear_sym} | ₹{price:>10.0f} | "
                   f"{prob:>5}% | {rsi:>6.1f} | {ai_color}{ai_score_str:<6} | "
                   f"₹{entry:>13.0f} | ₹{sl:>12.0f} | ₹{t1:>12.0f} | ₹{t2:>12.0f} |")
            out += row + "\n"
        return out

    def fmt_cat_warn_table(cat_list, label, emoji, sig, max_rows=10):
        if not cat_list:
            return f"\n{emoji} {label} [0]\n"
        sorted_s = sorted(cat_list, key=lambda x: (-x.get('prob', 0), -x.get('_confluence', 0)))[:max_rows]
        out = f"\n{emoji} {label} [{len(cat_list)}]\n"
        out += "| Stock          | Price      | Conf% |  RSI  | AI       | Entry(CMP)   | SL          | T1          | T2          | Warning   |\n"
        out += "|----------------|------------|-------|------|----------|---------------|--------------|--------------|--------------|-----------|\n"
        for r in sorted_s:
            star = "🏅 " if r.get('_starred') else ""
            ai_data = r.get('ai', {})
            outlook = ai_data.get('outlook', '')
            bear_sym = " 🐻" if outlook == 'BEARISH' else ""
            ai_score_val = ai_data.get('total_score', 0)
            ai_score_str, ai_color = ai_score_emoji(ai_score_val)
            rsi = r.get('rsi', 0)
            prob = r.get('prob', 0)
            price = r['price']
            id_l = get_levels(r, sig)
            tags = r.get('tags', [])
            tag_str = " ".join(tags) if tags else "—"
            entry = price
            sl = id_l['sl']
            t1 = id_l['t1']
            t2 = id_l['t2']
            row = (f"| {star}{r['symbol']:<12}{bear_sym} | ₹{price:>10.0f} | "
                   f"{prob:>5}% | {rsi:>6.1f} | {ai_color}{ai_score_str:<6} | "
                   f"₹{entry:>13.0f} | ₹{sl:>12.0f} | ₹{t1:>12.0f} | ₹{t2:>12.0f} | {tag_str:<9} |")
            out += row + "\n"
        return out

    # BUILD OUTPUT
    buy_count = len(buy)
    sell_count = len(sell)
    out = f"📊 NIFTY SCANNER v21 (+AI/ML) | {today}\n\n"
    out += f"📉 REGIME: {regime} {regime_icon} (BUY={buy_count} vs SELL={sell_count})\n\n"

    # Cat A — Top Picks (both BUY and SELL)
    cat_a_buy = [r for r in cat_a if r['signal'] == 'BUY']
    cat_a_sell = [r for r in cat_a if r['signal'] == 'SELL']
    if cat_a_buy:
        out += fmt_cat_table(cat_a_buy, "CATEGORY A — TOP PICKS (90%+ Conf)", "🟢", "BUY")
    if cat_a_sell:
        out += fmt_cat_table(cat_a_sell, "CATEGORY A — TOP SHORTS (90%+ Conf)", "🔴", "SELL")

    # Cat A- hedge
    cat_a_minus_buy = [r for r in cat_a_minus if r['signal'] == 'BUY']
    cat_a_minus_sell = [r for r in cat_a_minus if r['signal'] == 'SELL']
    if cat_a_minus_buy:
        out += fmt_cat_warn_table(cat_a_minus_buy, "CATEGORY A- — 2-of-3 CONFIRMED + HEDGE (BUY)", "🟢", "BUY")
    if cat_a_minus_sell:
        out += fmt_cat_warn_table(cat_a_minus_sell, "CATEGORY A- — 2-of-3 CONFIRMED + HEDGE (SHORT)", "🔴", "SELL")

    # Cat B — AI Confirmed
    cat_b_buy = [r for r in cat_b if r['signal'] == 'BUY']
    cat_b_sell = [r for r in cat_b if r['signal'] == 'SELL']
    if cat_b_buy:
        out += fmt_cat_table(cat_b_buy, "CATEGORY B — AI CONFIRMED (BUY)", "🟢", "BUY")
    if cat_b_sell:
        out += fmt_cat_table(cat_b_sell, "CATEGORY B — AI CONFIRMED (SHORT)", "🔴", "SELL")

    # Cat C1 — Signal + AI agree
    cat_c1_buy = [r for r in cat_c1 if r['signal'] == 'BUY']
    cat_c1_sell = [r for r in cat_c1 if r['signal'] == 'SELL']
    if cat_c1_buy:
        out += fmt_cat_warn_table(cat_c1_buy, "CATEGORY C — SIGNAL + AI AGREE (BUY)", "🟡", "BUY")
    if cat_c1_sell:
        out += fmt_cat_warn_table(cat_c1_sell, "CATEGORY C — SIGNAL + AI AGREE (SHORT)", "🟡", "SELL")

    # Cat C2 — Signal only
    cat_c2_buy = [r for r in cat_c2 if r['signal'] == 'BUY']
    cat_c2_sell = [r for r in cat_c2 if r['signal'] == 'SELL']
    if cat_c2_buy:
        out += fmt_cat_warn_table(cat_c2_buy, "CATEGORY C — SIGNAL ONLY (BUY)", "🟡", "BUY")
    if cat_c2_sell:
        out += fmt_cat_warn_table(cat_c2_sell, "CATEGORY C — SIGNAL ONLY (SHORT)", "🟡", "SELL")

    # Cat D — ML Signal Only
    cat_d_buy = [r for r in cat_d if r['signal'] == 'BUY']
    cat_d_sell = [r for r in cat_d if r['signal'] == 'SELL']
    if cat_d_buy:
        out += fmt_cat_warn_table(cat_d_buy, "CATEGORY D — ML SIGNAL ONLY (BUY)", "🟣", "BUY")
    if cat_d_sell:
        out += fmt_cat_warn_table(cat_d_sell, "CATEGORY D — ML SIGNAL ONLY (SHORT)", "🟣", "SELL")

    # Watchlist
    if watchlist:
        out += fmt_cat_warn_table(watchlist, "WATCHLIST — RANGE BOUND", "📋", "BUY")

    out += f"\n⚠️ Not SEBI registered. Validate before trading."
    return out