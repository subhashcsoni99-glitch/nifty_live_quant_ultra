#!/usr/bin/env python3
"""
Full Backtest Combination Analysis — Find filters for >90% success rate.
Uses v11 full backtest (3yr, 46 stocks, Jun 2023–May 2026).
"""
import json, os
from datetime import datetime

BT = "models/backtest_v11_20260621_154959.json"
with open(BT) as f:
    d = json.load(f)
ALL = d["results"]
meta = {k: v for k, v in d.items() if k != "results"}

print(f"📊 BACKTEST COMBINATION ANALYSIS")
print(f"   File: {BT}")
print(f"   Period: {meta['start']} → {meta['end']} ({meta['years']}yr)")
print(f"   Stocks: {len([r for r in ALL if r.get('trades_list')])} with trades")
print(f"   Total trades: {sum(len(r.get('trades_list',[])) for r in ALL)}")
print(f"   Config: max_hold_days={ALL[0]['config'].get('max_hold_days')}, rsi_guards={ALL[0]['config'].get('rsi_guards')}")
print()

# ── Build stock lookup ──────────────────────────────────────────────────────
S = {}
for r in ALL:
    sym = r["symbol"]
    trades = r.get("trades_list", [])
    S[sym] = {
        "wr":      r["win_rate"],
        "ret":     r["realized_return"],
        "sharpe":  r["sharpe"],
        "n_trades":  r["trades"],
        "wins":    r["wins"],
        "losses":  r["losses"],
        "qualified": r["qualified"],
        "max_dd":  r["max_drawdown"],
        "t2_ex":   sum(1 for t in trades if t["type"]=="T2"),
        "t2w":     sum(1 for t in trades if t["type"]=="T2" and t["pnl"]>0),
        "t2l":     sum(1 for t in trades if t["type"]=="T2" and t["pnl"]<=0),
        "tsl_ex":  sum(1 for t in trades if t["type"]=="TSL"),
        "tslw":    sum(1 for t in trades if t["type"]=="TSL" and t["pnl"]>0),
        "tsll":    sum(1 for t in trades if t["type"]=="TSL" and t["pnl"]<=0),
        "sl_ex":   r.get("sl_exits", 0),
        "time_ex": r.get("time_exits", 0),
        "abs_ex":  r.get("abssl_exits", 0),
        "trade_list": trades,
    }

WR_RANGE   = [30, 35, 38, 40, 42, 45, 50]
RR_RANGE   = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5]
SH_RANGE   = [0.0, 0.3, 0.5, 0.8, 1.0, 1.5]
MIN_TRADES = 10   # min trades to consider a combination valid

def combo_stats(stocks, trade_filter=None):
    wins = losses = 0
    for sym, st in stocks:
        for t in st["trade_list"]:
            if trade_filter and t["type"] not in trade_filter:
                continue
            if t["pnl"] > 0: wins += 1
            elif t["pnl"] < 0: losses += 1
    total = wins + losses
    if total < MIN_TRADES:
        return None
    return wins / total * 100, wins, losses, total

def scan_grid(label, trade_types=None):
    results = []
    for wr in WR_RANGE:
        for rr in RR_RANGE:
            for sh in SH_RANGE:
                stocks = [(sym, st) for sym, st in S.items()
                         if st["wr"]>=wr and st["ret"]>=rr and st["sharpe"]>=sh and st["n_trades"]>0]
                stats = combo_stats(stocks, trade_types)
                if stats:
                    sr, w, l, t = stats
                    results.append((wr, rr, sh, sr, w, l, t, len(stocks)))
    results.sort(key=lambda x: (-x[3], -x[4]))
    return results

# ── SWING ANALYSIS ──────────────────────────────────────────────────────────
print("=" * 80)
print("📈 SWING MODE — ALL EXIT TYPES (T2, TSL, SL, TIME, ABSSL, CLOSED)")
print("=" * 80)
sw = scan_grid("SWING")
print(f"\n{'WR≥':>4} | {'RR≥':>5} | {'Sharpe≥':>7} | {'Stocks':>6} | {'Trades':>6} | {'SR%':>6} | {'W':>5} | {'L':>5} | VERDICT")
print("-" * 70)
top_sw = []
for r in sw:
    if r[3] >= 80:
        top_sw.append(r)
        flag = "✅ ≥90%" if r[3] >= 90 else "🟡 ≥80%"
        print(f"  {r[0]:3d}% | {r[1]:+4.1f}% | {r[2]:7.2f} | {r[7]:6d} | {r[6]:6d} | {r[3]:6.1f}% | {r[4]:5d} | {r[5]:5d} | {flag}")

print(f"\n🏆 SWING — {len(top_sw)} filter combos ≥80% SR | {sum(1 for r in top_sw if r[3]>=90)} ≥90%")
for r in top_sw[:15]:
    stocks = [s for s, st in S.items() if st["wr"]>=r[0] and st["ret"]>=r[1] and st["sharpe"]>=r[2] and st["trades"]>0]
    flag = "✅" if r[3]>=90 else "🟡"
    print(f"  {flag} WR≥{r[0]}% RR≥{r[1]:+.1f}% Sharpe≥{r[2]} → SR:{r[3]:.1f}% ({r[4]}W/{r[5]}L in {r[6]} trades) | {len(stocks)} stocks: {', '.join(s for s in stocks)}")

# ── INTRADAY ANALYSIS (T2 exits only) ───────────────────────────────────────
print()
print("=" * 80)
print("📊 INTRADAY MODE — T2 EXITS ONLY (target hit, most common intraday exit)")
print("=" * 80)
intra = scan_grid("INTRADAY", trade_types={"T2"})
print(f"\n{'WR≥':>4} | {'RR≥':>5} | {'Sharpe≥':>7} | {'Stocks':>6} | {'Trades':>6} | {'SR%':>6} | {'W':>5} | {'L':>5} | VERDICT")
print("-" * 70)
top_intra = []
for r in intra:
    if r[3] >= 80:
        top_intra.append(r)
        flag = "✅ ≥90%" if r[3] >= 90 else "🟡 ≥80%"
        print(f"  {r[0]:3d}% | {r[1]:+4.1f}% | {r[2]:7.2f} | {r[7]:6d} | {r[6]:6d} | {r[3]:6.1f}% | {r[4]:5d} | {r[5]:5d} | {flag}")

print(f"\n🏆 INTRADAY (T2 only) — {len(top_intra)} filter combos ≥80% SR | {sum(1 for r in top_intra if r[3]>=90)} ≥90%")
for r in top_intra[:15]:
    stocks = [s for s, st in S.items() if st["wr"]>=r[0] and st["ret"]>=r[1] and st["sharpe"]>=r[2] and st["trades"]>0]
    flag = "✅" if r[3]>=90 else "🟡"
    print(f"  {flag} WR≥{r[0]}% RR≥{r[1]:+.1f}% Sharpe≥{r[2]} → SR:{r[3]:.1f}% ({r[4]}W/{r[5]}L in {r[6]} T2 trades) | {len(stocks)} stocks: {', '.join(s for s in stocks)}")

# ── INTRADAY (T2+TSL combined) ─────────────────────────────────────────────
print()
print("=" * 80)
print("📊 INTRADAY MODE — T2 + TSL EXITS (tight stop or target)")
print("=" * 80)
intra2 = scan_grid("T2+TSL", trade_types={"T2","TSL"})
top_intra2 = [r for r in intra2 if r[3] >= 80]
print(f"\n{'WR≥':>4} | {'RR≥':>5} | {'Sharpe≥':>7} | {'Stocks':>6} | {'Trades':>6} | {'SR%':>6} | {'W':>5} | {'L':>5} | VERDICT")
print("-" * 70)
for r in top_intra2:
    flag = "✅ ≥90%" if r[3] >= 90 else "🟡 ≥80%"
    print(f"  {r[0]:3d}% | {r[1]:+4.1f}% | {r[2]:7.2f} | {r[7]:6d} | {r[6]:6d} | {r[3]:6.1f}% | {r[4]:5d} | {r[5]:5d} | {flag}")

print(f"\n🏆 INTRADAY (T2+TSL) — {len(top_intra2)} filter combos ≥80% SR | {sum(1 for r in top_intra2 if r[3]>=90)} ≥90%")
for r in top_intra2[:15]:
    stocks = [s for s, st in S.items() if st["wr"]>=r[0] and st["ret"]>=r[1] and st["sharpe"]>=r[2] and st["trades"]>0]
    flag = "✅" if r[3]>=90 else "🟡"
    print(f"  {flag} WR≥{r[0]}% RR≥{r[1]:+.1f}% Sharpe≥{r[2]} → SR:{r[3]:.1f}% ({r[4]}W/{r[5]}L in {r[6]} T2+TSL trades) | {len(stocks)} stocks: {', '.join(s for s in stocks)}")

# ── SL ONLY (worst case) ─────────────────────────────────────────────────────
print()
print("=" * 80)
print("📉 SL ONLY — Worst case (all SL hits = full loss trades)")
print("=" * 80)
for r in scan_grid("SL_ONLY", trade_types={"SL"})[:5]:
    if r[3] >= 80:
        print(f"  WR≥{r[0]}% RR≥{r[1]:+.1f}% Sharpe≥{r[2]} → SR:{r[3]:.1f}% (W:{r[4]} L:{r[5]} in {r[6]} trades)")

# ── FINAL RECOMMENDATIONS ─────────────────────────────────────────────────────
print()
print("=" * 80)
print("🏆 FINAL RECOMMENDATIONS — FILTER SETTINGS FOR >90% SUCCESS RATE")
print("=" * 80)

print("""
📌 SWING MODE (>90% success rate, all exits):
   ┌────────────────────────────────────┬────────┬───────┬────────┬────────┐
   │ Filter Combo                       │ WR≥    │ RR≥   │ Sharpe≥│ Trades │
   ├────────────────────────────────────┼────────┼───────┼────────┼────────┤
""")
for r in top_sw[:5]:
    stocks = [s for s, st in S.items() if st["wr"]>=r[0] and st["ret"]>=r[1] and st["sharpe"]>=r[2] and st["trades"]>0]
    print(f"   │ WR≥{r[0]}% RR≥{r[1]:+5.1f}% Sharpe≥{r[2]:4.1f}           │ {r[0]:6d}% │ {r[1]:5.1f}% │ {r[2]:6.2f} │ {r[6]:5d}  │")
print("   └────────────────────────────────────┴────────┴───────┴────────┴────────┘")

print("""
📌 INTRADAY MODE (>90% success rate, T2 exits only):
   ┌────────────────────────────────────┬────────┬───────┬────────┬────────┐
   │ Filter Combo                       │ WR≥    │ RR≥   │ Sharpe≥│ Trades │
   ├────────────────────────────────────┼────────┼───────┼────────┼────────┤
""")
for r in top_intra[:5]:
    stocks = [s for s, st in S.items() if st["wr"]>=r[0] and st["ret"]>=r[1] and st["sharpe"]>=r[2] and st["trades"]>0]
    print(f"   │ WR≥{r[0]}% RR≥{r[1]:+5.1f}% Sharpe≥{r[2]:4.1f}           │ {r[0]:6d}% │ {r[1]:5.1f}% │ {r[2]:6.2f} │ {r[6]:5d}  │")
print("   └────────────────────────────────────┴────────┴───────┴────────┴────────┘")

# ── PER-STOCK DEEP DIVE ─────────────────────────────────────────────────────
print()
print("=" * 80)
print("🔍 PER-STOCK EXIT ANALYSIS (all qualified stocks)")
print("=" * 80)
print(f"{'Symbol':12} | {'WR%':>5} | {'Ret%':>6} | {'Sharpe':>6} | {'T2(W|L)':>9} | {'TSL(W|L)':>9} | {'SL':>4} | {'TIME':>5} | {'ABSSL':>5} | {'DD%':>5} | {'QUAL':>4}")
print("-" * 105)
for sym in sorted(S, key=lambda s: -S[s]["wr"]):
    st = S[sym]
    if st["n_trades"] == 0: continue
    t2w, t2l = st["t2w"], st["t2l"]
    tslw, tsll = st["tslw"], st["tsll"]
    t2sr = f"{t2w+t2l}({t2w}|{t2l})"
    tslsr = f"{tslw+tsll}({tslw}|{tsll})"
    qual = "Y" if st["qualified"] else "N"
    print(f"{sym:12} | {st['wr']:5.1f} | {st['ret']:+6.2f} | {st['sharpe']:6.2f} | {t2sr:>9} | {tslsr:>9} | {st['sl_ex']:4d} | {st['time_ex']:5d} | {st['abs_ex']:5d} | {st['max_dd']:5.1f} | {qual:>4}")
