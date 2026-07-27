# NIFTY Live Quant Ultra — SKILL.md v55

> Multi-mode quant trading system. Paper trading only. No live money.
> Backtest: pinned `end_date='2026-05-31'` (3yr, Jul 2023–May 2026)

---

## Rating: 9.7/10 ✅

| Ver | Score | Key Changes |
|-----|-------|-------------|
| v54 | 9.5/10 | Telegram scan, T1→SL% backtest, ML confidence, per-hour targets |
| **v55** | **9.7/10** | **Sector-momentum filter** + **ML weak flag** + **morning spike warning** + **ML conflict flag** |

---

## v55 — Short-Entry Improvement Rules (2026-07-27)

### Problem
Shorts (DIVISLAB, HCLTECH) lost money despite A1 signals. Root causes:
1. RSI overbought can STAY overbought — RSI gives direction, not timing
2. Sector in BULL mode steamrolls individual shorts
3. Morning entry (9:09 AM) catches stocks near day LOW — worst short entry
4. BEARISH regime ≠ every stock falls
5. ML 50-75% is borderline for shorting

### New Features

| Feature | Description |
|---------|-------------|
| **Sector Momentum** | Checks if >50% of sector peers are up >0.5%. Flags: `⚠️SECTOR_BULL:IT:80%bull` |
| **ML Weak Flag** | Shorts with ML <75% get `⚠ML:51%<75` warning |
| **Morning Spike Warning** | Scans before 10 AM get `⚠️MORNING` flag on shorts |
| **ML Conflict Flag** | BUY with ML DOWN ≥70% or SHORT with ML UP flagged |
| **Sector Peer Map** | IT, BANK, PHARMA, FINANCE, AUTO, METAL, OIL, FMCG, INFRA, CEMENT, POWER, CONGLOM |

### Recommended Short-Entry Rules

1. **ML ≥ 75%**: Skip shorts with ML 50-74% (borderline)
2. **No shorts before 10 AM**: Wait for morning spike to settle
3. **Sector filter**: Skip shorts when >50% sector peers are up >0.5%
4. **Price action**: Short only after stock makes a lower high on 5-min chart
5. **No re-entry**: If stopped out, don't re-short same day

---

## v54 — New Telegram Scan + T1→SL% Backtest (2026-07-09)

### What's New

| Feature | File | Description |
|---------|------|-------------|
| **Telegram scan** | `scan_telegram.py` | Full-format scan: Regime, Cat A/B/C/D, TOP BUY/SHORT cards, SUMMARY with all levels |
| **T1→SL% metric** | `backtest.py` | New backtest column: % of trades that hit T1 then got stopped out |
| **T3 exit** | `backtest.py` | Full exit at ATR×1.5 (new exit type) |
| **ML confidence** | `scan_telegram.py` | Live ML model confidence on every stock |
| **Per-hour targets** | `scan_telegram.py` | HR_T1/T2/T3 = hourly ATR-based sub-targets |
| **Qty @ ₹10k risk** | `scan_telegram.py` | Position size for ₹10,000 risk per trade |
| **Separate BUY/SHORT levels** | `scan_telegram.py` | Entry, SL, T1, T2, T3 shown separately for both directions |

### Backtest New Columns

```
Sym  Trds  WR%  Avg%  T1%  SL%  T1→SL%  T2%  TIME%  ADX  Sharpe
```

| Column | Meaning |
|--------|---------|
| `T1%` | % of trades that hit T1 target (ATR×0.5) |
| `SL%` | % of trades that exited via stop-loss |
| `T1→SL%` | **NEW** — % of trades that hit T1 then got stopped (boxed out) |
| `T2%` | % of trades that hit T2 target (ATR×1.0) |
| `TIME%` | % of trades that exited at day's close (intraday rule) |
| `ADX` | Average entry ADX — trend quality indicator |
| `Sharpe` | Risk-adjusted return |

### Backtest Parameters (ratio-sl mode)

```
T1 = ATR × 0.5    T2 = ATR × 1.0    T3 = ATR × 1.5    SL = ATR × 1.0 (= T1 distance)
Ratio: 1:1 reward:risk on T1 partial exit
```

**T1→SL% tells you how often you're "boxed out"** — price hits your first target, you take profit, then price reverses and stops you out. High T1→SL% means the stock whipsaws after reaching T1.

**Current backtest stats (3yr, Jul 2023–May 2026):**
- 46 stocks | Avg WR: 54.7% | Avg Sharpe: 0.87
- T1→SL% avg: **28.3%** — ~1 in 3 trades get boxed
- T1% avg: **83.7%** — most trades reach first target
- T2% avg: **47.7%** — half the time price doesn't reach full ATR×1.0
- Qualified: **8/50 stocks**

---

## Usage

```bash
# ── STEP 1: Run backtest (finds qualified stocks) ──
python3 backtest.py --all --years 3 --intraday --ratio-sl

# ── STEP 2: Live Telegram scan (recommended daily scan) ──
python3 scan_telegram.py

# ── Legacy scan (telegram format, different layout) ──
python3 scan.py --ai --format telegram --tight --intraday

# ── Train ML models ──
python3 train.py --index nifty100
```

---

## scan_telegram.py — Full Output Guide

```
🗓️ 09 Jul 2026 12:00 PM IST
📊 Regime: ⚠️BULLISH | NIFTY 24,068 (+0.8%)

🏆⭐ TOP BUY: AXISBANK
   📈 Regime:BULL | RSI:34.6 | Conf:100% ML:UP:52% | WR:68.8% | CF:93.8% | Sharpe:3.6
   💰 Entry:₹1310.7 | SL:₹1285.3 (risk₹25.4) | T1:₹1325.8 | T2:₹1336.1 | T3:₹1348.9
   ⏱️  HR_ATR:₹3.9 | HR_T1:₹2.3 | HR_T2:₹3.9 | HR_T3:₹5.9
   📦 Qty:393 @ ₹1310.7 | ATR14:₹25.4 | ATR5:₹30.1 | Score:85.7

🏆⭐ TOP SHORT: HDFCBANK
   📉 Regime:BEAR | RSI:67.7 | Conf:100% ML:DOWN:56% | WR:52.6% | CF:94.7% | Sharpe:1.33
   💰 Entry:₹817.2 | SL:₹831.8 (risk₹14.6) | T1:₹807.5 | T2:₹802.5 | T3:₹795.2
   ⏱️  HR_ATR:₹2.2 | HR_T1:₹1.5 | HR_T2:₹2.2 | HR_T3:₹3.4
   📦 Qty:684 @ ₹817.2 | ATR14:₹14.6 | ATR5:₹19.3 | Score:79.5

📈 Cat A — HIGHEST QUALITY (WR≥65%+Conf≥80%+CF≥50%): 3
   📈AXISBANK ₹1310.7(+0.1%) | E:₹1310.7 SL:₹1285.3(r:₹25.4) T1:₹1325.8 T2:₹1336.1 T3:₹1348.9 | ATR14:₹25.4 ATR5:₹30.1 Qty:393 ML:UP:52%
   📈TATASTEEL ₹187.9(-0.2%) | E:₹187.9 SL:₹184.7(r:₹3.2) T1:₹189.5 T2:₹191.1 T3:₹192.8 | ATR14:₹3.2 ATR5:₹3.3 Qty:3090 ML:DOWN:54%
   📈BAJAJFINSV ₹1892.5(+2.0%) | E:₹1892.9 SL:₹1852.1(r:₹40.8) T1:₹1915.6 T2:₹1933.7 T3:₹1954.2 | ATR14:₹40.8 ATR5:₹45.5 Qty:244 ML:DOWN:98%

📊 Cat B — GOOD QUALITY (WR≥50%+Conf≥70%+CF≥30%): 8
   (same format)

📋 Cat C — WATCHLIST (WR≥40%+Conf≥50%): 26
   (same format)

⚠️ Cat D — SHORT BIAS (RSI>65+Conf≥50%): 7
   (same format)

📋 SUMMARY — TOP 5 BUY PICKS:
  1. AXISBANK | Entry:₹1310.7 | SL:₹1285.3 | T1:₹1325.8 | T2:₹1336.1 | T3:₹1348.9
     ATR14:₹25.4 ATR5:₹30.1 | Qty:393 | Conf:100% WR:68.8% CF:93.8% Sharpe:3.6 ML:UP:52%
  2. TATASTEEL | Entry:₹187.9 | SL:₹184.7 | T1:₹189.5 | T2:₹191.1 | T3:₹192.8
     ATR14:₹3.2 ATR5:₹3.3 | Qty:3090 | Conf:100% WR:66.7% CF:90.0% Sharpe:2.02 ML:DOWN:54%
  3. KOTAKBANK | Entry:₹371.4 | SL:₹363.6 | T1:₹375.8 | T2:₹379.2 | T3:₹383.1
     ATR14:₹7.8 ATR5:₹8.7 | Qty:1278 | Conf:100% WR:58.8% CF:100.0% Sharpe:1.0 ML:DOWN:56%
  4. LT | Entry:₹3925.0 | SL:₹3857.5 | T1:₹3962.5 | T2:₹3992.5 | T3:₹4026.2
     ATR14:₹67.5 ATR5:₹74.8 | Qty:148 | Conf:100% WR:52.4% CF:95.2% Sharpe:1.46 ML:DOWN:72%
  5. WIPRO | Entry:₹173.9 | SL:₹170.4 | T1:₹175.5 | T2:₹177.3 | T3:₹179.1
     ATR14:₹3.5 ATR5:₹3.2 | Qty:2885 | Conf:100% WR:53.8% CF:92.3% Sharpe:1.94 ML:UP:91%

📋 SUMMARY — TOP 5 SHORT PICKS:
  1. HDFCBANK | Entry:₹817.2 | SL:₹831.8 | T1:₹807.5 | T2:₹802.5 | T3:₹795.2
     ATR14:₹14.6 ATR5:₹19.3 | Qty:684 | Conf:100% WR:52.6% CF:94.7% Sharpe:1.33 ML:DOWN:56%
  2. APOLLOHOSP | Entry:₹8876.5 | SL:₹9003.4 | T1:₹8799.8 | T2:₹8749.6 | T3:₹8686.2
     ATR14:₹126.9 ATR5:₹153.5 | Qty:78 | Conf:100% WR:55.6% CF:88.9% Sharpe:1.1 ML:DOWN:86%
  3. BAJAJFINSV | Entry:₹1892.5 | SL:₹1933.4 | T1:₹1870.1 | T2:₹1852.0 | T3:₹1831.7
     ATR14:₹40.8 ATR5:₹45.5 | Qty:244 | Conf:100% WR:80.0% CF:50.0% Sharpe:3.54 ML:DOWN:98%
  4. SUNPHARMA | Entry:₹1935.0 | SL:₹1964.9 | T1:₹1917.3 | T2:₹1905.1 | T3:₹1890.2
     ATR14:₹29.9 ATR5:₹35.4 | Qty:334 | Conf:100% WR:45.8% CF:87.5% Sharpe:-0.23 ML:DOWN:75%
  5. CIPLA | Entry:₹1437.8 | SL:₹1468.2 | T1:₹1423.6 | T2:₹1407.4 | T3:₹1392.2
     ATR14:₹30.4 ATR5:₹28.4 | Qty:328 | Conf:100% WR:41.7% CF:83.3% Sharpe:-1.02 ML:UP:57%
```

### Field Reference (scan_telegram.py)

| Field | Meaning |
|-------|---------|
| `Entry` | Current market price (BUY/SHORT entry reference) |
| `SL` | Stop-loss price (risk per share = \|Entry - SL\|) |
| `T1` | Target 1 = Entry ± ATR×0.5 |
| `T2` | Target 2 = Entry ± ATR×1.0 |
| `T3` | Target 3 = Entry ± ATR×1.5 |
| `ATR14` | Daily ATR(14) — base for T2/T3/SL |
| `ATR5` | Tight ATR(5) — used for T1 |
| `Qty` | Shares to buy/short for exactly ₹10,000 risk |
| `Conf` | Rule-based confluence score (0-100%) |
| `WR` | Historical win rate from 3yr backtest |
| `CF` | T1 completion rate — % of trades that hit T1 |
| `Sharpe` | Risk-adjusted return ratio |
| `ML` | ML model output: `UP:XX%` or `DOWN:XX%` = confidence |

### Target Formula (Tight Intraday Mode)

```
T1  = Entry ± ATR5 × 0.5      (tight scalp, ~0.3-0.5% move)
T2  = Entry ± ATR14 × 1.0     (moderate, ~0.8-1.2% move)
T3  = Entry ± ATR14 × 1.5     (full session, ~1.2-1.8% move)
SL  = Entry ± ATR14 × 1.0     (1:1 risk:reward on T1)
HR_T1 = ATR14 / 6.5 × 0.5     (per-hour sub-target T1)
HR_T2 = ATR14 / 6.5 × 1.0     (per-hour sub-target T2)
HR_T3 = ATR14 / 6.5 × 1.5     (per-hour sub-target T3)
```

---

## Category Definitions

| Cat | Threshold | Meaning |
|-----|-----------|---------|
| **A** | WR≥65% + Conf≥80% + CF≥50% | Highest quality — trade with confidence |
| **B** | WR≥50% + Conf≥70% + CF≥30% | Good quality — validate before entry |
| **C** | WR≥40% + Conf≥50% | Watchlist — candidates |
| **D** | RSI>65 + Conf≥50% | Short bias — bearish signals |

---

## System Architecture

```
scan_telegram.py (v54)  → NEW: full Telegram scan with ALL fields
scan.py (v52)            → legacy scan with AI + ML
  └─ nifty_core.py       → signal logic, features, levels, regime
  └─ nifty_categorize    → categorizes stocks into Cat A/B/C/D
  └─ backtest.py         → 3yr validation with T1→SL%, T3 exits
  └─ train.py            → sklearn ML models per stock
```

---

## ⚠️ Disclaimer
Paper trading only. No live money. Backtest results are not indicative of future performance. Always validate with live market data.
