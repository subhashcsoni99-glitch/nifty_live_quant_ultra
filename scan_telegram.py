#!/usr/bin/env python3
"""
Custom Telegram-format scan for nifty_live_quant_ultra
Shows: Regime, Categories A/B/C/D, TOP BUYS, TOP SHORTS, SUMMARY
With: T1/T2/T3 targets, SL, entry, qty @ Rs10000 risk, hourly ATR, ML confidence
"""
import sys, os, json, glob
from datetime import datetime
sys.path.insert(0, os.path.dirname(__file__))

import yfinance as yf
from nifty_core import (
    NIFTY50_STOCKS, get_ohlc, add_features, build_ml_features,
    get_market_regime, get_adx
)
from scan import get_ml_prediction

CAPITAL = 10000

def load_latest_backtest():
    files = sorted(glob.glob(os.path.join(os.path.dirname(__file__),
                     'models/backtest_v11_*.json')))
    if not files:
        return {}
    data = json.load(open(files[-1]))
    return {r['symbol']: r for r in data['results']}

def get_nifty_price():
    try:
        ticker = yf.Ticker("^NSEI")
        df = ticker.history(period="2d")
        if df is not None and not df.empty:
            close = float(df['Close'].iloc[-1])
            prev = float(df['Close'].iloc[-2]) if len(df) > 1 else close
            chg = (close - prev) / prev * 100
            return close, round(chg, 2)
    except Exception:
        pass
    return None, None

def compute_levels(price, day_atr, atr5, direction='BUY'):
    """T1=ATR×0.5 (tight ATR5), T2=ATR×1.0, T3=ATR×1.5, SL=ATR×1.0 (ratio-sl)"""
    t1_dist = atr5 * 0.5
    t2_dist = day_atr * 1.0
    t3_dist = day_atr * 1.5
    sl_dist = day_atr * 1.0
    risk = sl_dist
    qty = max(1, int(CAPITAL / risk)) if risk > 0 else 0
    hr_atr = round(day_atr / 6.5, 1)
    if direction == 'BUY':
        return {
            'entry': round(price, 1),
            'sl': round(price - sl_dist, 1),
            't1': round(price + t1_dist, 1),
            't2': round(price + t2_dist, 1),
            't3': round(price + t3_dist, 1),
            'risk': round(risk, 1), 'qty': qty,
            'atr14': round(day_atr, 1), 'atr5': round(atr5, 1),
            'hr_atr': hr_atr,
            'hr_t1': round(t1_dist / 6.5, 1),
            'hr_t2': round(t2_dist / 6.5, 1),
            'hr_t3': round(t3_dist / 6.5, 1),
        }
    else:
        return {
            'entry': round(price, 1),
            'sl': round(price + sl_dist, 1),
            't1': round(price - t1_dist, 1),
            't2': round(price - t2_dist, 1),
            't3': round(price - t3_dist, 1),
            'risk': round(risk, 1), 'qty': qty,
            'atr14': round(day_atr, 1), 'atr5': round(atr5, 1),
            'hr_atr': hr_atr,
            'hr_t1': round(t1_dist / 6.5, 1),
            'hr_t2': round(t2_dist / 6.5, 1),
            'hr_t3': round(t3_dist / 6.5, 1),
        }

def scan_stock(name, bt_data):
    try:
        df_day = get_ohlc(name, days=5)
        if df_day is None or len(df_day) < 2:
            return None
        price = float(df_day['Close'].iloc[-1])
        prev_close = float(df_day['Close'].iloc[-2])

        df = get_ohlc(name, days=60)
        if df is None or len(df) < 20:
            return None
        df = add_features(df)

        rsi = float(df['rsi'].iloc[-1])
        day_atr = float(df['atr'].iloc[-1])
        atr5 = float(df['atr5'].iloc[-1])
        adx_val, di_plus, di_minus = get_adx(df)
        adx_trending = bool(adx_val > 20)
        change_pct = round((price - prev_close) / prev_close * 100, 2)

        # Regime
        if adx_trending and adx_val > 20:
            regime = 'BULL' if rsi < 40 else 'BEAR' if rsi > 65 else 'CHOPPY'
        else:
            regime = 'CHOPPY'

        # Backtest data
        bt = bt_data.get(name, {})
        wr = bt.get('win_rate', 0)
        es = bt.get('exit_stats', {})
        cf = es.get('t1_pct', 0)
        trades = bt.get('trades', 0)
        sharpe = bt.get('sharpe', 0)

        # Confluence
        if regime == 'BULL':
            conf = min(100, max(0, 50 + (60 - rsi) * 1.5 + adx_val * 0.8))
        elif regime == 'BEAR':
            conf = min(100, max(0, 50 + (rsi - 40) * 1.5 + adx_val * 0.8))
        else:
            conf = min(100, max(0, 50 - abs(rsi - 50) * 2 + adx_val * 0.5))
        conf = round(conf, 1)

        # ML confidence
        ml = get_ml_prediction(name, df, auto_retrain=False)
        ml_conf = ml.get('confidence', 0) if ml else 0
        ml_dir = ml.get('direction', None) if ml else None

        # Direction emoji
        if regime == 'BULL':
            direction = '📈BULL'
        elif regime == 'BEAR':
            direction = '📉BEAR'
        else:
            direction = '⚠️CHOPPY'

        buy_lvl = compute_levels(price, day_atr, atr5, 'BUY')
        short_lvl = compute_levels(price, day_atr, atr5, 'SHORT')

        score = wr * 0.4 + cf * 0.3 + conf * 0.3

        return {
            'name': name, 'price': price,
            'prev_close': prev_close, 'change_pct': change_pct,
            'rsi': round(rsi, 1), 'regime': regime, 'direction': direction,
            'conf': conf, 'wr': wr, 'cf': cf,
            'bt_ret': bt.get('realized_return', 0),
            'trades': trades, 'sharpe': sharpe,
            'adx': round(adx_val, 1), 'score': round(score, 1),
            'ml_conf': round(ml_conf, 1), 'ml_dir': ml_dir,
            # BUY levels
            'entry': buy_lvl['entry'], 'sl': buy_lvl['sl'],
            't1': buy_lvl['t1'], 't2': buy_lvl['t2'], 't3': buy_lvl['t3'],
            'risk': buy_lvl['risk'], 'qty': buy_lvl['qty'],
            'atr14': buy_lvl['atr14'], 'atr5': buy_lvl['atr5'],
            'hr_atr': buy_lvl['hr_atr'],
            'hr_t1': buy_lvl['hr_t1'], 'hr_t2': buy_lvl['hr_t2'], 'hr_t3': buy_lvl['hr_t3'],
            # SHORT levels
            's_entry': short_lvl['entry'], 's_sl': short_lvl['sl'],
            's_t1': short_lvl['t1'], 's_t2': short_lvl['t2'], 's_t3': short_lvl['t3'],
            's_risk': short_lvl['risk'], 's_qty': short_lvl['qty'],
        }
    except Exception as e:
        return None

def fmt_ml_inline(r):
    """Compact ML for inline display in category lines."""
    ml_conf = r.get('ml_conf', 0)
    ml_d = r.get('ml_dir', '')
    if not ml_conf or not ml_d:
        return ""
    arrow = '↑' if ml_d == 'UP' else '↓'
    return f"ML:{ml_d}{arrow}{ml_conf:.0f}%"

def fmt_ml(r):
    ml_conf = r.get('ml_conf', 0)
    ml_d = r.get('ml_dir', '')
    if not ml_conf or not ml_d:
        return ""
    arrow = '⬆' if ml_d == 'UP' else '⬇'
    return f" *ML:{ml_d}{arrow}:{ml_conf:.1f}%*"

def buy_line(r):
    ml = fmt_ml_inline(r)
    ml_flag = f" [{ml}]" if ml else ""
    return (f"   📈{r['name']} ₹{r['price']}({r['change_pct']:+.1f}%) "
            f"| Entry:₹{r['entry']} SL:₹{r['sl']}(r:₹{r['risk']}) "
            f"T1:₹{r['t1']} T2:₹{r['t2']} T3:₹{r['t3']} "
            f"| CF:{r['cf']}% ATR14:₹{r['atr14']} ATR5:₹{r['atr5']} Qty:{r['qty']}"
            f" | WR:{r['wr']}%{ml_flag}")

def short_line(r):
    ml = fmt_ml_inline(r)
    ml_flag = f" [{ml}]" if ml else ""
    return (f"   📉{r['name']} ₹{r['price']}({r['change_pct']:+.1f}%) "
            f"| Entry:₹{r['s_entry']} SL:₹{r['s_sl']}(r:₹{r['s_risk']}) "
            f"T1:₹{r['s_t1']} T2:₹{r['s_t2']} T3:₹{r['s_t3']} "
            f"| CF:{r['cf']}% ATR14:₹{r['atr14']} ATR5:₹{r['atr5']} Qty:{r['s_qty']}"
            f" | WR:{r['wr']}%{ml_flag}")

def top_card(r, direction='BUY'):
    d_emoji = '📈' if direction == 'BUY' else '📉'
    tag = 'TOP BUY' if direction == 'BUY' else 'TOP SHORT'
    ml = fmt_ml(r)
    if direction == 'BUY':
        entry, sl, t1, t2, t3 = r['entry'], r['sl'], r['t1'], r['t2'], r['t3']
        risk, qty = r['risk'], r['qty']
    else:
        entry, sl, t1, t2, t3 = r['s_entry'], r['s_sl'], r['s_t1'], r['s_t2'], r['s_t3']
        risk, qty = r['s_risk'], r['s_qty']
    return (
        f"🏆⭐ {tag}: {r['name']}\n"
        f"   {d_emoji} Regime:{r['regime']} | RSI:{r['rsi']} | *Conf:{r['conf']}%*{ml} | "
        f"*CF:{r['cf']}%* | WR:{r['wr']}% | Sharpe:{r['sharpe']}\n"
        f"   💰 *Entry:₹{entry}* | SL:₹{sl} (risk₹{risk}) | "
        f"T1:₹{t1} | T2:₹{t2} | T3:₹{t3}\n"
        f"   ⏱️  HR_ATR:₹{r['hr_atr']} | HR_T1:₹{r['hr_t1']} | "
        f"HR_T2:₹{r['hr_t2']} | HR_T3:₹{r['hr_t3']}\n"
        f"   📦 Qty:{qty} @ ₹{r['price']} | *ATR14:₹{r['atr14']}* | ATR5:₹{r['atr5']} | Score:{r['score']}"
    )

def main():
    print("Scanning NIFTY50...", flush=True)
    bt_data = load_latest_backtest()
    nifty_price, nifty_change = get_nifty_price()
    regime_info = get_market_regime()
    regime = regime_info.get('regime', 'CHOPPY')
    regime_icon = '🟢' if regime == 'BULL' else '🔴' if regime == 'BEAR' else '⚠️'
    now = datetime.now().strftime('%d %b %Y %I:%M %p IST')

    results = []
    for i, sym in enumerate(NIFTY50_STOCKS):
        print(f"  {i+1}/{len(NIFTY50_STOCKS)}: {sym}", end='\r', flush=True)
        r = scan_stock(sym, bt_data)
        if r:
            results.append(r)

    print(f"\n✅ Scanned {len(results)}/{len(NIFTY50_STOCKS)} stocks")

    # Filters
    buys = [r for r in results
            if r['conf'] >= 50 and r['cf'] >= 7.0 and r['wr'] >= 0 and r['regime'] != 'BEAR']
    buys.sort(key=lambda x: -x['score'])

    shorts = [r for r in results
              if r['rsi'] > 65 and r['conf'] >= 50 and r['cf'] >= 7.0 and r['wr'] >= 0]
    shorts.sort(key=lambda x: -x['score'])

    # Cat A1 — SUPER: WR≥65%+Conf≥80%+CF≥70%+ML aligned with signal direction
    cat_a1_buy = sorted([r for r in results
                         if r['wr'] >= 65 and r['conf'] >= 80 and r['cf'] >= 70
                         and r.get('ml_dir') == 'UP' and r.get('ml_conf', 0) >= 50],
                        key=lambda x: -x['score'])
    cat_a1_short = sorted([r for r in results
                           if r['wr'] >= 65 and r['conf'] >= 80 and r['cf'] >= 70
                           and r.get('ml_dir') == 'DOWN' and r.get('ml_conf', 0) >= 50
                           and r['regime'] == 'BEAR'],
                          key=lambda x: -x['score'])
    cat_a1 = cat_a1_buy + cat_a1_short
    cat_b = sorted([r for r in results if r['wr'] >= 50 and r['conf'] >= 70 and r['cf'] >= 70],
                   key=lambda x: -x['score'])
    cat_a = sorted([r for r in results
                    if r['wr'] >= 65 and r['conf'] >= 80 and r['cf'] >= 70
                    and not (r.get('ml_dir') == 'UP' and r.get('ml_conf', 0) >= 50)
                    and not (r.get('ml_dir') == 'DOWN' and r.get('ml_conf', 0) >= 50 and r['regime'] == 'BEAR')],
                   key=lambda x: -x['score'])
    cat_b = sorted([r for r in results if r['wr'] >= 50 and r['conf'] >= 70 and r['cf'] >= 70],
                   key=lambda x: -x['score'])
    cat_c = sorted([r for r in results if r['wr'] >= 40 and r['conf'] >= 50 and r['cf'] >= 70],
                   key=lambda x: -x['score'])
    cat_d = sorted([r for r in results if r['rsi'] > 65 and r['conf'] >= 50 and r.get('ml_dir') == 'DOWN'],
                   key=lambda x: -x['score'])

    out = []
    nifty_str = f"NIFTY {nifty_price:,.0f} ({nifty_change:+.1f}%)" if nifty_price else "NIFTY —"
    out.append(f"🗓️ {now}")
    out.append(f"📊 Regime: {regime_icon}{regime} | {nifty_str}")
    out.append("")

    # TOP BUY
    if buys:
        out.append(top_card(buys[0], 'BUY'))
    else:
        out.append("🏆⭐ TOP BUY: No BUY signals meet criteria (Conf≥50% + CF≥70% + WR≥0%)")
    out.append("")

    # TOP SHORT
    if shorts:
        out.append(top_card(shorts[0], 'SHORT'))
    else:
        out.append("🏆⭐ TOP SHORT: No SHORT signals meet criteria (RSI>65 + Conf≥50% + CF≥70% + WR≥0%)")
    out.append("")

    # Helper to categorize direction per stock
    def stock_direction(r):
        if r['regime'] == 'BEAR' or (r.get('ml_dir') == 'DOWN' and r.get('ml_conf', 0) >= 50):
            return 'SHORT'
        return 'BUY'

    # Cat A1 — SUPER (both BUY and SHORT)
    out.append(f"⭐ Cat A1 — SUPER (WR≥65%+Conf≥80%+CF≥70%+ML aligned ≥50%): {len(cat_a1)}")
    if cat_a1_buy:
        out.append("  📈 BUY:")
        for r in cat_a1_buy[:5]:
            out.append(buy_line(r))
    if cat_a1_short:
        out.append("  📉 SHORT:")
        for r in cat_a1_short[:5]:
            out.append(short_line(r))
    if not cat_a1:
        out.append("  — No stocks meet Cat A1 criteria —")
    out.append("")

    # Cat A — HIGH CONF (both BUY and SHORT)
    out.append(f"📈 Cat A — HIGH CONF (WR≥65%+Conf≥80%+CF≥70%): {len(cat_a)}")
    if cat_a:
        out.append("  📈 BUY:")
        for r in cat_a[:5]:
            out.append(buy_line(r))
        out.append("  📉 SHORT:")
        for r in cat_a[:5]:
            out.append(short_line(r))
    else:
        out.append("  — No stocks meet Cat A criteria —")
    out.append("")

    # Cat B — QUALITY (both BUY and SHORT)
    out.append(f"📊 Cat B — QUALITY (WR≥50%+Conf≥70%+CF≥70%): {len(cat_b)}")
    if cat_b:
        out.append("  📈 BUY:")
        for r in cat_b[:5]:
            out.append(buy_line(r))
        out.append("  📉 SHORT:")
        for r in cat_b[:5]:
            out.append(short_line(r))
    out.append("")

    # Cat C — WATCHLIST (both BUY and SHORT)
    out.append(f"📋 Cat C — WATCHLIST (WR≥40%+Conf≥50%+CF≥70%): {len(cat_c)}")
    if cat_c:
        out.append("  📈 BUY:")
        for r in cat_c[:10]:
            out.append(buy_line(r))
        out.append("  📉 SHORT:")
        for r in cat_c[:10]:
            out.append(short_line(r))
    out.append("")

    # Cat D — SHORT BIAS (both BUY and SHORT)
    out.append(f"⚠️ Cat D — SHORT BIAS (RSI>65+Conf≥50%+ML_DOWN): {len(cat_d)}")
    if cat_d:
        out.append("  📉 SHORT:")
        for r in cat_d[:5]:
            out.append(short_line(r))
        out.append("  📈 BUY:")
        for r in cat_d[:5]:
            out.append(buy_line(r))
    out.append("")

    # SUMMARY — BUY
    out.append("📋 SUMMARY — TOP 5 BUY PICKS:")
    for i, r in enumerate(buys[:5], 1):
        ml = fmt_ml(r)
        l1 = f"  {i}. {r['name']} | Entry:₹{r['entry']} | SL:₹{r['sl']} | T1:₹{r['t1']} | T2:₹{r['t2']} | T3:₹{r['t3']}"
        l2 = f"     ATR14:₹{r['atr14']} ATR5:₹{r['atr5']} | Qty:{r['qty']} | Conf:{r['conf']}% WR:{r['wr']}% CF:{r['cf']}% Sharpe:{r['sharpe']}{ml}"
        out.append(l1)
        out.append(l2)

    out.append("")
    # SUMMARY — SHORT
    out.append("📋 SUMMARY — TOP 5 SHORT PICKS:")
    for i, r in enumerate(shorts[:5], 1):
        ml = fmt_ml(r)
        l1 = f"  {i}. {r['name']} | Entry:₹{r['s_entry']} | SL:₹{r['s_sl']} | T1:₹{r['s_t1']} | T2:₹{r['s_t2']} | T3:₹{r['s_t3']}"
        l2 = f"     ATR14:₹{r['atr14']} ATR5:₹{r['atr5']} | Qty:{r['s_qty']} | Conf:{r['conf']}% WR:{r['wr']}% CF:{r['cf']}% Sharpe:{r['sharpe']}{ml}"
        out.append(l1)
        out.append(l2)

    out.append("")
    out.append("⚠️ Not SEBI registered. Validate before trading.")

    print('\n'.join(out))

if __name__ == '__main__':
    main()
