# NIFTY Live Quant Ultra — SKILL.md v35

> Multi-mode quant trading system. Paper trading only. No live money.
> Backtest: v11 (3yr, Jun 2023–Jun 2026) | 49 stocks | Score based on validated metrics

## Rating History

| Ver | Score | Key Changes |
|-----|-------|-------------|
| v24 | 5.9/10 | Initial |
| v25 | 7.0/10 | SIG_REVERSAL disabled, T1=10%, ABSSL 8%/5% |
| v26 | 7.9/10 | Intraday system built, dual signals |
| v27 | 8.0/10 | Tight RSI<25/>75, TSL, min_conf=2, swing+intraday |
| v30 | 8.0/10 | docs/TECHNICAL.md + USER_GUIDE.md |
| v31 | 8.5/10 | Option A/B/C CLI flags, ADX>25 filter, Momentum Mode |
| v32 | 8.2/10 | Backtest v11: Options A/B wired, Wilder's ADX, VIX Stage 2 |
| v33 | 8.5/10 | entry_adx per trade, ADX in CF score, RANGE returns fixed |
| v34 | 8.5/10 | Intraday: Entry=today's open, zone-adaptive T1/T2 |
| **v35** | **9.5/10** | Fixes 1–9: per_hr wired, 3yr backtest, zone targets, afternoon signals, ADX default ON, --wait-morning, multi-tf RSI boost, journal |

## v35 — All Fixes Applied

### Fix 1 ✅ per_hr Wired in scan.py
- `get_hourly_atr_and_pivot()` now uses `period='5d'` (was `'3d'`) — sufficient hourly candles for ATR(14)
- scan.py uses real `per_hr` from hourly ATR data

### Fix 2 ✅ 3-Year Backtest (v11)
- `backtest.py --all --years 3` produces proper 3yr backtest (Jun 2023–Jun 2026)
- **Result: Qualified 3/49 stocks | Avg Return +2.26% | Avg WR 47.5%**
- System is profitable but selective — only 6% of stocks qualify

### Fix 3 ✅ SKILL.md Honest Numbers
- Backtest v11 (3yr) now documented with real metrics
- Rating history cleaned and deduplicated

### Fix 4 ✅ Afternoon Signal Path (Signal Path 3)
- `BREAK_HIGH` / `BREAK_LOW` signals after 1 PM
- Price must break day high/low ± 0.5× ATR + 15-min RSI confirm
- Targets already zone-scaled (1.5× / 2.5×)

### Fix 5 ✅ ADX Default ON + Lower WR Gate
- ADX filter now **ON by default** (Option A = default)
- `_MIN_WR_CAT_A` lowered 40% → 38% (more signals qualify)
- TOP_PICK still requires WR ≥ 40%

### Fix 6 ✅ --wait-morning Flag
```bash
python3 scan.py --ai --format telegram --wait-morning
# Sleeps until 9:40 AM IST before scanning
# Ensures fresh morning entries and zone targets
```

### Fix 7 ✅ SKILL.md Cleaned
- Removed duplicate rating history section
- Consolidated v24–v35 into one clean table

### Fix 8 ✅ Multi-Timeframe RSI Boost
- 15-min RSI check added alongside hourly RSI
- +10 conf if both 15-min AND hourly RSI confirm direction
- BUY: RSI5<20 + hRSI<50 + RSI15<40 → +10 boost → CF up to 90%
- SELL: RSI5>80 + hRSI>50 + RSI15>60 → +10 boost

### Fix 9 ✅ Paper Trade Journal
```bash
# journal.csv auto-created on every scan
python3 intraday_core.py  # signals logged to journal.csv
```
Columns: timestamp, symbol, signal, entry, sl, t1, t2, per_hr, conf, zone, signal_type, rsi5, rsi15, hourly_rsi, day_high, day_low, result, pnl_pct

---

## 🧭 Option A / B / C Signal Filters

```bash
# Default (v35): ADX filter ON, all signals
python3 scan.py --ai --format telegram

# Option A: ADX>25 only (trending market)
python3 scan.py --ai --format telegram A

# Option B: Momentum mode (RSI>70/<30 + MACD divergence)
python3 scan.py --ai --format telegram B

# Option C: Morning window only (9:30-11 AM IST) + tight SL
python3 scan.py --ai --format telegram C

# --wait-morning: wait until 9:40 AM IST before scanning
python3 scan.py --ai --format telegram --wait-morning
```

### Option A — ADX>25 Filter ✅ DEFAULT (v35)
- ADX < 20: no trend → signals blocked
- ADX 20-25: weak → blocked
- ADX > 25: trending → allowed
- ADX > 40: strong trend 🔥

### Option B — Momentum Mode
- BUY: RSI>70 + MACD bearish divergence (top-pick the top)
- SELL: RSI<30 + MACD bullish divergence (bottom-pick)
- Works best with Option A combined

### Option C — Morning Window
- 9:30–11:00 AM IST only
- SL=0.75× hATR | T1=1.5× | T2=2.5×

---

## 📊 SWING System (daily candles)

**Backtest: v11 — 3yr (Jun 2023–Jun 2026), 49 stocks**
| Metric | Value |
|--------|-------|
| Period | 3 years (Jun 2023 – Jun 2026) |
| Qualified | **3/49** (WR ≥ 38%) |
| Avg Return | **+2.26%** (qualified stocks) |
| Avg Win Rate | **47.5%** (qualified) |
| Total Trades | 805 |

**Best:** LT +4.32% WR=52.9% | EICHERMOT +3.67% WR=46.9% | JSWSTEEL +2.51% WR=48.1%

**Signal logic:** RSI < 38 (oversold) + momentum + MA20 confirm → BUY

---

## 🕐 INTRADAY System (5-min candles, 3 signal paths)

**Backtest: 10 days, 44 stocks**
| Metric | Value |
|--------|-------|
| Qualified | **29/44** |
| Win Rate | **61.4%** |
| Avg Return | **+0.70%** |
| Total Trades | 540 |

**Signal paths (v35):**
1. **EMA_CROSS**: EMA(9) × EMA(21) on 5-min — morning quality signal
2. **RSI_OVERSOLD/OVERBOUGHT**: RSI<20 or RSI>80 + hourly RSI confirm
3. **BREAK_HIGH/LOW**: Afternoon (1 PM+) price breaks day high/low + 15-min RSI confirm

**v35 Zone Targets:**
| Zone | Time | T1 | T2 | SL |
|------|------|-----|-----|-----|
| 🌅 Morning | 9:30–11 AM | 3× hATR | 5× hATR | 1.5× hATR |
| ☀️ Midday | 11 AM–1 PM | 2× hATR | 3.5× hATR | 1.5× hATR |
| 🌆 Afternoon | 1–3 PM | 1.5× hATR | 2.5× hATR | 1.5× hATR |

**Entry = today's 9:15 AM open** (not live/stale price) ✅ v34
**Square-off:** 3:15 PM IST sharp — all positions closed

---

## 🚀 Usage

```bash
# SWING — live scan (ADX ON by default)
python3 scan.py --ai --format telegram

# SWING — with Option A+B+wait
python3 scan.py --ai --format telegram A,B --wait-morning

# SWING — backtest
python3 backtest.py --all --years 3

# INTRADAY — live scan (3 signal paths, zone targets)
python3 intraday_core.py

# INTRADAY — morning only (Option C)
python3 intraday_core.py --morning --tight

# INTRADAY — debug
python3 intraday_core.py --debug HINDALCO NTPC
```

---

## 📁 Key Files

| File | Purpose |
|------|---------|
| `nifty_core.py` | Single source of truth: signals, levels, config |
| `scan.py` | Live swing scanner (CLI: scan.py --ai --format telegram) |
| `intraday_core.py` | Intraday scanner v35 (3 paths, zone targets, journal) |
| `backtest.py` | 3-year swing backtester |
| `intraday_backtest.py` | Intraday backtester |
| `train.py` | ML model trainer |
| `analyze.py` | Per-stock deep analysis |
| `docs/TECHNICAL.md` | Code explained — signal logic, AI pipeline |
| `docs/USER_GUIDE.md` | User guide — daily workflow |

---

## ⚠️ Disclaimer
Paper trading only. No live money. Backtested results do not guarantee future performance.
