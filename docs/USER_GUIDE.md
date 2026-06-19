# NIFTY Live Quant Ultra — User Guide

> Plain-English guide for traders. No coding required.

---

## What Is This?

**NIFTY Live Quant Ultra** is a quantitative trading research system for NIFTY50 and NIFTY100 stocks. It analyzes stocks using technical indicators, AI scoring, and machine learning — then ranks them by quality so you know which ones to consider.

> ⚠️ **Paper trading only.** This is a research tool. Not SEBI registered. Not financial advice.

---

## Quick Start

### Run a Full Scan (SWING mode)
```bash
cd ~/.openclaw/plugin-skills/nifty_live_quant_ultra
python3 scan.py --index nifty50 --ai --format telegram
```

### Run Intraday Scan
```bash
python3 intraday_core.py
```

### Train ML Models (do once a week)
```bash
python3 train.py --index nifty100
```

### Run Backtest
```bash
python3 backtest.py --all --years 3
```

---

## Understanding the Output

### Market Regime

The scan starts with the **market regime**:

```
📊 Regime: 🟢 BULLISH (📈36 | 📉2)
```

| Regime | Meaning | What it means for you |
|--------|---------|----------------------|
| 🟢 **BULLISH** | Most stocks are AI-bullish | BUY signals are in-trend, SHORTs are hedges |
| 🔴 **BEARISH** | Most stocks are AI-bearish | SELL signals are in-trend, BUY signals are contrarian |
| 🟡 **NEUTRAL** | Mixed signals | Wait, or trade very selectively |

---

## Stock Categories

Every stock in the scan gets sorted into one of these buckets:

### Cat A — Highest Quality ⭐
**Triple confirmed:** Rule-based signal + AI says BULLISH + ML says UP

These are the strongest setups. Requires:
- Signal fires AND AI agrees AND ML model agrees
- Historical win rate ≥ 40%
- Realized return > −2% (not a historical money-loser)

### Cat B — AI Confirmed 🤖
**Double confirmed:** Rule-based signal + AI says BULLISH (ML missing or neutral)

Good setups when ML models aren't available for a stock.

### Cat C1 / Cat C2 — Signal Only
Only the rule-based signal is present. AI and/or ML don't confirm. Higher risk, lower confidence.

### Cat D — ML Signal Only 🧠
Only the ML model fires. No rule-based signal. Shown separately.

### WATCHLIST — Range Bound
No signal. Price is between support and resistance with no clear direction.

### Cat A Shorts
Short (SELL) signals in **Cat A** quality. In a BULL market, these are labeled 🛡️ HEDGE — sector-rotation bets, not directional shorts.

---

## Reading a Stock Line

Here's a typical scan line:

```
📈 COFORGE ₹1,466 | RSI:59 | Conf:82% | RR:+4% 🟢WR:46% | CF:8.1/10
   🎯 Entry(CMP):₹1,466 SL:1349 T1:1641 T2:1816
   💠 Entry(CMP):₹1,466 SL:1450 T1:1474 T2:1481 | ~₹11.7/hr [⭐ TOP_PICK]
```

| Field | Meaning |
|-------|---------|
| `COFORGE` | Stock ticker |
| `₹1,466` | Current market price (CMP) |
| `RSI:59` | Relative Strength Index — 59 is neutral-bullish |
| `Conf:82%` | Rule-based signal confidence — 82% of conditions are met |
| `RR:+4%` | Historical realized return from backtest |
| `🟢WR:46%` | Win rate: 🟢 = 46%+ (🟡 = 40-46%, 🔴 = below 40%) |
| `CF:8.1/10` | Confluence score — overall quality (1-10) |
| `⭐ TOP_PICK` | Qualifies as a top pick (WR≥40%, RR>0, CF≥8) |

### Two Level Sets

| Label | Type | Use for |
|-------|------|---------|
| 🎯 **Entry/SL/T1/T2** | SWING levels | Multi-day positional trades |
| 💠 **Entry/SL/T1/T2** | Hourly levels | Intraday / scalp trades |

---

## The Levels Explained

### Entry (CMP)
The current market price. This is where you'd enter.

### Stop Loss (SL)
The price where you cut losses. Calculated as:
```
SL = Entry − (ATR × multiplier)
```

### T1 — First Target
Partial exit point. When price hits T1:
- 10% of your position is locked in as profit
- Stop loss moves to **breakeven** (you can no longer lose on this trade)
- Remaining 90% continues toward T2

### T2 — Second Target
Full exit point. Take all remaining profit here.

### Per-Hour Target
Expected hourly profit if the trade goes your way (based on hourly ATR). Use this for intraday planning.

### ABSSL — Adaptive Breakeven Stop Loss
An automatic safety net:
- **Before T1:** 8% max loss hard cap
- **After T1 partial:** 5% lock-in hard cap
- **After T2:** disabled — let profits run

---

## ⭐ TOP PICKS — What Are They?

TOP PICKS are the **best of the best**. To earn the ⭐ badge, a stock needs ALL of:

| Requirement | Meaning |
|-------------|---------|
| Category A or B | Signal + AI confirmed, not just noise |
| Win Rate ≥ 40% | 🟢 or 🟡 badge |
| Realized Return > 0% | Historically profitable |
| Confluence Score ≥ 8.0/10 | High overall quality |

TOP PICKS are listed at the **top of the report** (before categories), sorted by WR and CF.

---

## Intraday vs SWING

### SWING (Default — `scan.py`)
- Uses **daily candles**
- Hold for **days to weeks**
- Targets are wider (2×-6× ATR)
- Best for: positional trades with overnight holds

### INTRADAY (`intraday_core.py`)
- Uses **5-minute candles**
- **Same day exit required** (square-off at 3:15 PM IST)
- Tighter targets (1.5×-5× hourly ATR)
- Two signal types:
  - **EMA Crossover** — EMA(9) × EMA(21) on 5-min (rare, high quality)
  - **RSI Oversold/Overbought** — RSI(5m) < 25 or > 75 + hourly RSI confirm

---

## Position Sizing — How Much to Buy?

The system calculates how many shares to buy for **exactly ₹10,000 risk**:

```
Qty = ₹10,000 / (Entry Price − Stop Loss)
```

**Example:** M&M at ₹3,133 with SL ₹2,964
```
Risk per share = 3133 − 2964 = ₹169
Qty = 10000 / 169 = 417 shares
Position value = 417 × 3133 = ₹13,06,461
```

> ⚠️ This is the NOTIONAL position value. The actual risk is ₹10,000. The position size varies with stock price — cheaper stocks = more shares, expensive stocks = fewer shares.

### ⚠️ OVERSIZE Warning
If a position would be > 10% of your capital, the scan shows:
```
⚠️ OVERSIZE(13%)
```
This means the position is large. Consider reducing size or skipping.

---

## Quality Badges

| Badge | Meaning |
|-------|---------|
| 🟢 WR:46% | Win rate 46%+ — good historical performance |
| 🟡 WR:40% | Win rate 40-46% — acceptable |
| 🔴 WR:33% | Win rate below 40% — poor, avoid if possible |
| ⭐ TOP_PICK | Best quality: WR≥40% + RR>0 + CF≥8 |
| 🛡️ HEDGE | Short in BULL regime — hedge bet, not directional |
| 🔴 CONTRARIAN | Signal goes against the market regime |
| ⚠️ NEG_HIST | Historically losing stock (rr < −2% AND wr < 40%) |
| ⏰ STALE | Signal is 4-7 days old |
| 💀 STALE_CRITICAL | Signal is 8+ days old |
| ⚠️ NO_BACKTEST | No 3-year backtest data yet |
| ⚠️ LOW_WINRATE | Win rate below 40% |

---

## How to Read the Summary

```
📋 SUMMARY: Total:50 | 📈LONG:49 | 📉SHORT:1 | ➡️RANGE:0
💰 Position sizing: Risk ≤2% of capital per trade | Max 3 concurrent positions
📊 Quality gates: Cat A/B require WR ≥40% | TOP_PICK requires WR ≥40% + RR>0 + CF≥8
```

- **49 LONG, 1 SHORT** — market is overwhelmingly bullish today
- **Risk ≤2% per trade** — each trade risks at most 2% of capital
- **Max 3 concurrent** — don't spread yourself across more than 3 positions

---

## Best Trade of the Day

At the bottom, the scan shows the **single best BUY** with full intraday + swing levels:

```
🏆 BEST BUY — M&M (Cat B, ⭐ TOP_PICK)
  Entry (CMP): ₹3,133
  Stop Loss:   ₹2,964  (−₹169, −5.4%)
  T1 Target:   ₹3,387  (+₹254, +8.1%)
  T2 Target:   ₹3,641  (+₹508, +16.2%)
  Per Hour:    ₹18.0
  RSI: 51  Conf: 90%  CF: 8.8/10  WR: 57%
  Qty @ ₹10k risk: 417 shares
```

---

## Command Reference

### scan.py
```bash
python3 scan.py --index nifty50 --ai --format telegram
python3 scan.py --index nifty100 --ai --format telegram  # all 100 stocks
python3 scan.py SBIN RELIANCE HDFCBANK --ai              # specific stocks
python3 scan.py --index nifty50 --ai --top 5            # show only top 5
python3 scan.py --backtest-first                          # refresh backtest then scan
python3 scan.py --debug                                   # show confluence breakdown
```

### intraday_core.py
```bash
python3 intraday_core.py                           # scan top GOOD_STOCKS
python3 intraday_core.py SBIN RELIANCE HDFCBANK    # specific stocks
python3 intraday_core.py --debug                   # verbose output
```

### backtest.py
```bash
python3 backtest.py --all --years 3               # full 3-year backtest
python3 backtest.py --stock SBIN                  # single stock
python3 backtest.py --trailing                     # enable trailing SL
python3 backtest.py --sector-cap                   # enforce sector diversification
```

### train.py
```bash
python3 train.py --index nifty50                   # train NIFTY50 models
python3 train.py --index nifty100                   # train all NIFTY100 models
python3 train.py --cleanup                         # remove orphaned models
python3 train.py --coverage                        # check model coverage
python3 train.py --index nifty100 &                # train in background
```

### analyze.py
```bash
python3 analyze.py SBIN --ai --full  # deep single-stock analysis
```

---

## Workflow: How to Use This Daily

### Every Morning (Before Market Open)
```bash
# 1. Run the scan
python3 scan.py --index nifty50 --ai --format telegram

# 2. Check intraday signals if trading during the day
python3 intraday_core.py
```

### Every Sunday (Weekly Maintenance)
```bash
# 1. Retrain ML models (in background)
python3 train.py --index nifty100 &

# 2. Refresh backtest
python3 backtest.py --all --years 3
```

### Every Month
```bash
# Update fundamental data
python3 fetch_fundamental.py
```

---

## Reading the Intraday Report

```
🕐 INTRADAY SCAN | 17 Jun 11:30 AM IST
📊 3 signals | ⛔ Square-off: 3:15 PM sharp

📈 BUY [2]
  RELIANCE ₹1,335 | RSI5:23 RSI_h:45 | CF:75% [RSI_OVERSOLD]
    Entry:₹1,335 | SL:₹1,324 | T1:₹1,346 | T2:₹1,358 | ~₹8.3/hr
    EMA9:1332.5 | EMA21:1330.1 | Open:₹1,332

📉 SELL [1]
  SBIN ₹1,028 | RSI5:78 RSI_h:56 | CF:80% [RSI_OVERBOUGHT]
    Entry:₹1,028 | SL:₹1,038 | T1:₹1,016 | T2:₹1,004 | ~₹5.1/hr
```

- **RSI5** — RSI on 5-minute candles (for signal)
- **RSI_h** — RSI on hourly candles (for confirmation)
- **EMA9 / EMA21** — Current EMA values
- **Open** — Opening price (first 15 min average)
- **⛔ Square-off: 3:15 PM** — ALL positions closed at this time, no matter where price is

---

## Common Questions

**Q: What timeframe does SWING mode use?**
A: Daily candles. You hold positions for days to weeks.

**Q: Can I trade both SWING and INTRADAY simultaneously?**
A: Yes — they're independent. SWING for positional holds, INTRADAY for scalp/same-day.

**Q: How often should I retrain ML models?**
A: Once a week is fine. The `--coverage` flag tells you which stocks need retraining.

**Q: What if a stock has no ML model?**
A: It still appears in the scan (Cat B/C). The ML model is an extra filter, not a requirement.

**Q: What's the difference between the hourly levels (💠) and swing levels (🎯)?**
A: Hourly levels are tighter (1.5× hourly ATR) — use these for intraday or same-week trades. Swing levels are wider (2-6× daily ATR) — use these for positional holds.

**Q: What does "RANGE" mean?**
A: No signal. The stock is in a consolidation — price is between support and resistance with no clear direction. Wait for a breakout.

**Q: Why do some stocks show 🔴 CONTRARIAN?**
A: Because the signal goes against the current market regime. A SELL in a BULL market is labeled 🔴 CONTRARIAN. These are risky — the broader market is against you.

---

## Disclaimer

⚠️ **Paper trading only.** This tool is for research and education.

- Not SEBI registered
- Backtested results do not guarantee future performance
- Past performance ≠ future results
- Always validate with your own research before making trading decisions
- You are responsible for your own trades
