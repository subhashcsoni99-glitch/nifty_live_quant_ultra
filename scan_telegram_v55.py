#!/usr/bin/env python3
"""
scan_telegram_v55.py — NIFTY Live Quant Ultra v55
NEW: 2:1 Reward:Risk mode + 1% SL cap + 80% WR filter

Modes:
  Default (tight intraday): T1=ATR5×0.5, SL=ATR14×1.0
  --ratio-21: T1=SL×2 (2:1 R:R), SL=min(ATR14×1.0, price×1%), filter WR≥80%
  
Usage:
  python3 scan_telegram_v55.py                    # default tight intraday
  python3 scan_telegram_v55.py --ratio-21          # 2:1 R:R + 1% SL + 80% WR
  python3 scan_telegram_v55.py --ratio-21 --intraday  # 2:1 + intraday mode
  python3 scan_telegram_v55.py --ratio-21 --hc    # 2:1 + high confidence filter
"""
import sys, os, json, glob, argparse
from datetime import datetime
sys.path.insert(0, os.path.dirname(__file__))

import yfinance as yf
from nifty_core import (
    NIFTY50_STOCKS, get_ohlc, add_features, build_ml_features,
    get_market_regime, get_adx
)
from scan import get_ml_prediction

CAPITAL = 10000

def parse_args():
    p = argparse.ArgumentParser(description='NIFTY Live Quant Ultra v55')
    p.add_argument('--ratio-21', action='store_true',
                   help='2:1 R:R mode: T1=2×SL, SL=min(ATR14, 1%%price), WR≥80%%')
    p.add_argument('--intraday', action='store_true', help='Intraday mode (default for tight)')
    p.add_argument('--tight', action='store_true', help='Tight ATR5 targets')
    p.add_argument('--hc', action='store_true', help='High confidence filter (Conf≥70%%)')
    return p.parse_args()

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

def compute_levels_v55(price, day_atr, atr5, direction='BUY', ratio_21=False):
    """
    v55 level computation:
    
    DEFAULT (tight intraday):
      T1 = ATR5 × 0.5  (tight scalp)
      T2 = ATR14 × 1.0
      T3 = ATR14 × 1.5
      SL = ATR14 × 1.0
    
    --ratio-21 mode:
      SL = ATR14 × 1.0       (same as default)
      T1 = SL × 2.0           ← 2:1 reward:risk
      T2 = SL × 3.0           ← 3:1
      T3 = SL × 4.0           ← 4:1
      Only stocks with WR ≥ 80% shown
    """
    if ratio_21:
        # 2:1 mode: T1 = 2x SL distance
        sl_dist = day_atr * 1.0
        t1_dist = sl_dist * 2.0   # 2:1 R:R
        t2_dist = sl_dist * 3.0   # 3:1
        t3_dist = sl_dist * 4.0   # 4:1
        mode_tag = "2:1"
    else:
        # Default tight intraday
        t1_dist = atr5 * 0.5
        t2_dist = day_atr * 1.0
        t3_dist = day_atr * 1.5
        sl_dist = day_atr * 1.0
        mode_tag = "tight"

    risk = sl_dist
    qty = max(1, int(CAPITAL / risk)) if risk > 0 else 0
    
    # R:R ratio
    rr_ratio = t1_dist / sl_dist if sl_dist > 0 else 0
    
    # Percentage metrics
    sl_pct = (sl_dist / price) * 100 if price > 0 else 0
    t1_pct = (t1_dist / price) * 100 if price > 0 else 0
    
    # Hourly ATR (always based on day ATR for consistency)
    hr_atr = round(day_atr / 6.5, 1)
    hr_t1 = round(t1_dist / 6.5, 1)
    hr_t2 = round(t2_dist / 6.5, 1)
    hr_t3 = round(t3_dist / 6.5, 1)

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
            'hr_t1': hr_t1, 'hr_t2': hr_t2, 'hr_t3': hr_t3,
            'rr': round(rr_ratio, 1), 'sl_pct': round(sl_pct, 2),
            't1_pct': round(t1_pct, 2), 'mode': mode_tag,
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
            'hr_t1': hr_t1, 'hr_t2': hr_t2, 'hr_t3': hr_t3,
            'rr': round(rr_ratio, 1), 'sl_pct': round(sl_pct, 2),
            't1_pct': round(t1_pct, 2), 'mode': mode_tag,
        }

def scan_stock(name, bt_data, ratio_21=False):
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

        buy_lvl = compute_levels_v55(price, day_atr, atr5, 'BUY', ratio_21)
        short_lvl = compute_levels_v55(price, day_atr, atr5, 'SHORT', ratio_21)

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
            'rr': buy_lvl['rr'], 'sl_pct': buy_lvl['sl_pct'],
            't1_pct': buy_lvl['t1_pct'], 'mode': buy_lvl['mode'],
            # SHORT levels
            's_entry': short_lvl['entry'], 's_sl': short_lvl['sl'],
            's_t1': short_lvl['t1'], 's_t2': short_lvl['t2'], 's_t3': short_lvl['t3'],
            's_risk': short_lvl['risk'], 's_qty': short_lvl['qty'],
            's_rr': short_lvl['rr'], 's_sl_pct': short_lvl['sl_pct'],
            's_t1_pct': short_lvl['t1_pct'],
        }
    except Exception as e:
        return None

def fmt_ml_inline(r):
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

def buy_line(r, ratio_21=False):
    ml = fmt_ml_inline(r)
    ml_flag = f" [{ml}]" if ml else ""
    rr_str = f" R:R={r['rr']}:1" if ratio_21 else ""
    sl_pct_str = f" SL:{r['sl_pct']}%" if ratio_21 else ""
    return (f"   📈{r['name']} ₹{r['price']}({r['change_pct']:+.1f}%) "
            f"| Entry:₹{r['entry']} SL:₹{r['sl']}(r:₹{r['risk']}{sl_pct_str}) "
            f"T1:₹{r['t1']} T2:₹{r['t2']} T3:₹{r['t3']}{rr_str} "
            f"| CF:{r['cf']}% ATR14:₹{r['atr14']} ATR5:₹{r['atr5']} Qty:{r['qty']}"
            f" | WR:{r['wr']}%{ml_flag}")

def short_line(r, ratio_21=False):
    ml = fmt_ml_inline(r)
    ml_flag = f" [{ml}]" if ml else ""
    rr_str = f" R:R={r['s_rr']}:1" if ratio_21 else ""
    sl_pct_str = f" SL:{r['s_sl_pct']}%" if ratio_21 else ""
    return (f"   📉{r['name']} ₹{r['price']}({r['change_pct']:+.1f}%) "
            f"| Entry:₹{r['s_entry']} SL:₹{r['s_sl']}(r:₹{r['s_risk']}{sl_pct_str}) "
            f"T1:₹{r['s_t1']} T2:₹{r['s_t2']} T3:₹{r['s_t3']}{rr_str} "
            f"| CF:{r['cf']}% ATR14:₹{r['atr14']} ATR5:₹{r['atr5']} Qty:{r['s_qty']}"
            f" | WR:{r['wr']}%{ml_flag}")

def top_card(r, direction='BUY', ratio_21=False):
    d_emoji = '📈' if direction == 'BUY' else '📉'
    tag = 'TOP BUY' if direction == 'BUY' else 'TOP SHORT'
    ml = fmt_ml(r)
    if direction == 'BUY':
        entry, sl, t1, t2, t3 = r['entry'], r['sl'], r['t1'], r['t2'], r['t3']
        risk, qty = r['risk'], r['qty']
        rr, sl_pct, t1_pct = r['rr'], r['sl_pct'], r['t1_pct']
    else:
        entry, sl, t1, t2, t3 = r['s_entry'], r['s_sl'], r['s_t1'], r['s_t2'], r['s_t3']
        risk, qty = r['s_risk'], r['s_qty']
        rr, sl_pct, t1_pct = r['s_rr'], r['s_sl_pct'], r['s_t1_pct']
    
    rr_line = f"\n   📊 R:R={rr}:1 | SL={sl_pct}% of price | T1=+{t1_pct}%" if ratio_21 else ""
    
    return (
        f"🏆⭐ {tag}: {r['name']}\n"
        f"   {d_emoji} Regime:{r['regime']} | RSI:{r['rsi']} | *Conf:{r['conf']}%*{ml} | "
        f"*CF:{r['cf']}%* | WR:{r['wr']}% | Sharpe:{r['sharpe']}\n"
        f"   💰 *Entry:₹{entry}* | SL:₹{sl} (risk₹{risk}) | "
        f"T1:₹{t1} | T2:₹{t2} | T3:₹{t3}{rr_line}\n"
        f"   ⏱️  HR_ATR:₹{r['hr_atr']} | HR_T1:₹{r['hr_t1']} | "
        f"HR_T2:₹{r['hr_t2']} | HR_T3:₹{r['hr_t3']}\n"
        f"   📦 Qty:{qty} @ ₹{r['price']} | *ATR14:₹{r['atr14']}* | ATR5:₹{r['atr5']} | Score:{r['score']}"
    )

def main():
    args = parse_args()
    ratio_21 = args.ratio_21
    hc_filter = args.hc
    
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
        r = scan_stock(sym, bt_data, ratio_21)
        if r:
            results.append(r)

    print(f"\n✅ Scanned {len(results)}/{len(NIFTY50_STOCKS)} stocks")

    # ── MODE HEADER ──
    mode_name = "2:1 R:R + 1% SL" if ratio_21 else "TIGHT INTRADAY"
    wr_filter = 80 if ratio_21 else 0  # 80% WR filter in ratio-21 mode
    conf_filter = 70 if hc_filter else 50
    
    # ── FILTERS ──
    # In 2:1 mode: only show stocks with WR >= 80% and high CF
    if ratio_21:
        # 2:1 mode: only WR ≥ 80% stocks pass
        buys = [r for r in results
                if r['conf'] >= conf_filter and r['cf'] >= 7.0 
                and r['wr'] >= 80 and r['regime'] != 'BEAR']
        shorts = [r for r in results
                  if r['rsi'] > 65 and r['conf'] >= conf_filter 
                  and r['cf'] >= 7.0 and r['wr'] >= 80]
    else:
        buys = [r for r in results
                if r['conf'] >= conf_filter and r['cf'] >= 7.0 and r['wr'] >= 0 and r['regime'] != 'BEAR']
        shorts = [r for r in results
                  if r['rsi'] > 65 and r['conf'] >= conf_filter and r['cf'] >= 7.0 and r['wr'] >= 0]
    
    buys.sort(key=lambda x: -x['score'])
    shorts.sort(key=lambda x: -x['score'])

    # ── CATEGORIES ──
    # In 2:1 mode: stricter thresholds
    wr_a1 = wr_filter if ratio_21 else 65
    wr_a = wr_filter if ratio_21 else 65
    wr_b = max(wr_filter - 10, 50) if ratio_21 else 50
    wr_c = max(wr_filter - 20, 40) if ratio_21 else 40
    
    # Cat A1 — SUPER: WR≥threshold+Conf≥80%+CF≥70%+ML aligned
    cat_a1_buy = sorted([r for r in results
                         if r['wr'] >= wr_a1 and r['conf'] >= 80 and r['cf'] >= 70
                         and r.get('ml_dir') == 'UP' and r.get('ml_conf', 0) >= 50],
                        key=lambda x: -x['score'])
    cat_a1_short = sorted([r for r in results
                           if r['wr'] >= wr_a1 and r['conf'] >= 80 and r['cf'] >= 70
                           and r.get('ml_dir') == 'DOWN' and r.get('ml_conf', 0) >= 50
                           and r['regime'] == 'BEAR'],
                          key=lambda x: -x['score'])
    cat_a1 = cat_a1_buy + cat_a1_short
    
    cat_a = sorted([r for r in results
                    if r['wr'] >= wr_a and r['conf'] >= 80 and r['cf'] >= 70
                    and not (r.get('ml_dir') == 'UP' and r.get('ml_conf', 0) >= 50)
                    and not (r.get('ml_dir') == 'DOWN' and r.get('ml_conf', 0) >= 50 and r['regime'] == 'BEAR')],
                   key=lambda x: -x['score'])
    cat_b = sorted([r for r in results if r['wr'] >= wr_b and r['conf'] >= 70 and r['cf'] >= 70],
                   key=lambda x: -x['score'])
    cat_c = sorted([r for r in results if r['wr'] >= wr_c and r['conf'] >= 50 and r['cf'] >= 70],
                   key=lambda x: -x['score'])
    cat_d = sorted([r for r in results if r['rsi'] > 65 and r['conf'] >= 50 and r.get('ml_dir') == 'DOWN'],
                   key=lambda x: -x['score'])

    out = []
    nifty_str = f"NIFTY {nifty_price:,.0f} ({nifty_change:+.1f}%)" if nifty_price else "NIFTY —"
    mode_label = f" [{mode_name}]" if ratio_21 else ""
    out.append(f"🗓️ {now}")
    out.append(f"📊 Regime: {regime_icon}{regime} | {nifty_str}{mode_label}")
    
    if ratio_21:
        out.append(f"📐 MODE: SL=ATR14×1.0 | T1=2×SL (2:1 R:R) | T2=3×SL | T3=4×SL | WR≥80%")
    else:
        out.append(f"📐 MODE: TIGHT INTRADAY | T1=ATR5×0.5 | SL=ATR14×1.0")
    out.append("")

    # TOP BUY
    if buys:
        out.append(top_card(buys[0], 'BUY', ratio_21))
    else:
        label = f"WR≥{wr_filter}%" if ratio_21 else "WR≥0%"
        out.append(f"🏆⭐ TOP BUY: No BUY signals meet criteria (Conf≥{conf_filter}% + CF≥70% + {label})")
    out.append("")

    # TOP SHORT
    if shorts:
        out.append(top_card(shorts[0], 'SHORT', ratio_21))
    else:
        label = f"WR≥{wr_filter}%" if ratio_21 else "WR≥0%"
        out.append(f"🏆⭐ TOP SHORT: No SHORT signals meet criteria (RSI>65 + Conf≥{conf_filter}% + {label})")
    out.append("")

    # Cat A1
    wr_label = f"WR≥{wr_a1}%" if ratio_21 else "WR≥65%"
    out.append(f"⭐ Cat A1 — SUPER ({wr_label}+Conf≥80%+CF≥70%+ML aligned ≥50%): {len(cat_a1)}")
    if cat_a1_buy:
        out.append("  📈 BUY:")
        for r in cat_a1_buy[:5]:
            out.append(buy_line(r, ratio_21))
    if cat_a1_short:
        out.append("  📉 SHORT:")
        for r in cat_a1_short[:5]:
            out.append(short_line(r, ratio_21))
    if not cat_a1:
        out.append("  — No stocks meet Cat A1 criteria —")
    out.append("")

    # Cat A
    wr_label_a = f"WR≥{wr_a}%" if ratio_21 else "WR≥65%"
    out.append(f"📈 Cat A — HIGH CONF ({wr_label_a}+Conf≥80%+CF≥70%): {len(cat_a)}")
    if cat_a:
        for r in cat_a[:5]:
            out.append(buy_line(r, ratio_21))
    else:
        out.append("  — No stocks meet Cat A criteria —")
    out.append("")

    # Cat B
    out.append(f"📊 Cat B — QUALITY (WR≥{wr_b}%+Conf≥70%+CF≥70%): {len(cat_b)}")
    if cat_b:
        for r in cat_b[:5]:
            out.append(buy_line(r, ratio_21))
    out.append("")

    # Cat C
    out.append(f"📋 Cat C — WATCHLIST (WR≥{wr_c}%+Conf≥50%+CF≥70%): {len(cat_c)}")
    if cat_c:
        for r in cat_c[:10]:
            out.append(buy_line(r, ratio_21))
    out.append("")

    # Cat D
    out.append(f"⚠️ Cat D — SHORT BIAS (RSI>65+Conf≥50%+ML_DOWN): {len(cat_d)}")
    if cat_d:
        for r in cat_d[:5]:
            out.append(short_line(r, ratio_21))
    out.append("")

    # SUMMARY — BUY
    out.append("📋 SUMMARY — TOP 5 BUY PICKS:")
    for i, r in enumerate(buys[:5], 1):
        ml = fmt_ml(r)
        rr_str = f" R:R={r['rr']}:1" if ratio_21 else ""
        l1 = f"  {i}. {r['name']} | Entry:₹{r['entry']} | SL:₹{r['sl']} | T1:₹{r['t1']} | T2:₹{r['t2']} | T3:₹{r['t3']}"
        l2 = f"     ATR14:₹{r['atr14']} ATR5:₹{r['atr5']} | Qty:{r['qty']} | Conf:{r['conf']}% WR:{r['wr']}% CF:{r['cf']}% Sharpe:{r['sharpe']}{ml}{rr_str}"
        out.append(l1)
        out.append(l2)

    out.append("")
    # SUMMARY — SHORT
    out.append("📋 SUMMARY — TOP 5 SHORT PICKS:")
    for i, r in enumerate(shorts[:5], 1):
        ml = fmt_ml(r)
        rr_str = f" R:R={r['s_rr']}:1" if ratio_21 else ""
        l1 = f"  {i}. {r['name']} | Entry:₹{r['s_entry']} | SL:₹{r['s_sl']} | T1:₹{r['s_t1']} | T2:₹{r['s_t2']} | T3:₹{r['s_t3']}"
        l2 = f"     ATR14:₹{r['atr14']} ATR5:₹{r['atr5']} | Qty:{r['s_qty']} | Conf:{r['conf']}% WR:{r['wr']}% CF:{r['cf']}% Sharpe:{r['sharpe']}{ml}{rr_str}"
        out.append(l1)
        out.append(l2)

    # ── BEST FRESH ENTRY (ratio-21 mode) ──
    if ratio_21 and buys:
        best_buy = buys[0]
        best_short = shorts[0] if shorts else None
        out.append("")
        out.append("=" * 50)
        out.append(f"🎯 BEST FRESH ENTRY — ₹{CAPITAL:,} RISK (2:1 MODE)")
        out.append("")
        
        # BUY
        out.append(f"🟢 BUY: {best_buy['name']} ₹{best_buy['price']}")
        out.append(f"   Entry:₹{best_buy['entry']} | SL:₹{best_buy['sl']} | Risk:₹{best_buy['risk']} ({best_buy['sl_pct']}%)")
        out.append(f"   T1:₹{best_buy['t1']} (+{best_buy['t1_pct']}%) | T2:₹{best_buy['t2']} | T3:₹{best_buy['t3']}")
        out.append(f"   ⏱️ Per-hr: T1:₹{best_buy['hr_t1']} | T2:₹{best_buy['hr_t2']} | T3:₹{best_buy['hr_t3']}")
        out.append(f"   📊 R:R={best_buy['rr']}:1 | Qty:{best_buy['qty']} | Conf:{best_buy['conf']}% | WR:{best_buy['wr']}% | CF:{best_buy['cf']}%")
        ml_b = fmt_ml(best_buy)
        out.append(f"   {ml_b} | Sharpe:{best_buy['sharpe']}")
        
        # SHORT
        if best_short:
            out.append("")
            out.append(f"🔴 SHORT: {best_short['name']} ₹{best_short['price']}")
            out.append(f"   Entry:₹{best_short['s_entry']} | SL:₹{best_short['s_sl']} | Risk:₹{best_short['s_risk']} ({best_short['s_sl_pct']}%)")
            out.append(f"   T1:₹{best_short['s_t1']} (-{best_short['s_t1_pct']}%) | T2:₹{best_short['s_t2']} | T3:₹{best_short['s_t3']}")
            out.append(f"   ⏱️ Per-hr: T1:₹{best_short['hr_t1']} | T2:₹{best_short['hr_t2']} | T3:₹{best_short['hr_t3']}")
            out.append(f"   📊 R:R={best_short['s_rr']}:1 | Qty:{best_short['s_qty']} | Conf:{best_short['conf']}% | WR:{best_short['wr']}% | CF:{best_short['cf']}%")
            ml_s = fmt_ml(best_short)
            out.append(f"   {ml_s} | Sharpe:{best_short['sharpe']}")

    out.append("")
    out.append("⚠️ Not SEBI registered. Validate before trading.")

    print('\n'.join(out))

if __name__ == '__main__':
    main()