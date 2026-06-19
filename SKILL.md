# NIFTY Live Quant Ultra — SKILL.md v33

> Multi-mode quant trading system. Paper trading only. No live money.
> Three optional signal filters: **Option A** (ADX>25), **Option B** (Momentum), **Option C** (Morning Window + Tight SL)

## Rating History
| Version | Score | Key Change |
|---------|-------|-----------|
| v24 | 5.9/10 | Initial |
| v25 | 7.0/10 | SIG_REVERSAL disabled, T1=10%, ABSSL 8%/5% |
| v26 | 7.9/10 | Intraday system built, dual signals |
| **v27** | **9.5/10** | Tight RSI<25/>75, TSL, min_conf=2, swing+intraday both strong |
| v30 | 9.5/10 | docs/TECHNICAL.md (code explained) + docs/USER_GUIDE.md added |
| v31 | 9.5/10 | Option A/B/C CLI flags, ADX>25 filter, Momentum Mode, Morning Window |
| v32 | 9.2/10 | Backtest v11: Options A/B wired, Wilder's ADX, ADX in CF score, VIX Stage 2, tightened WR gate |
| **v33** | **9.5/10** | entry_adx per trade, allow_counter removed, momentum+ADX warning, RANGE returns fixed |
| **v34** | **9.5/10** | Intraday: Entry = today's 9:15 AM open (was stale 5-min close). Zone-adaptive T1/T2: morning=3×/5×, midday=2×/3.5×, afternoon=1.5×/2.5×. per_hr = T1_dist / actual hours remaining. Fixed period='5m'→'1d'. |

## v34 — Entry Point + Achievable Targets (CRITICAL FIX)

**Problem:** Previous versions used the **last 5-min candle close** as entry — which was stale (could be hours old). T1 was always 3× ATR regardless of time of day, making targets unreachable in afternoon sessions.

**Fix:**
- **Entry = today's 9:15 AM open** (the actual opening price, fetched from daily candle)
- **T1/T2 scales with zone:**
  - 🌅 Morning (9:30–11 AM): T1=3×hATR, T2=5×hATR — full targets, 6.5h left
  - ☀️ Midday (11 AM–1 PM): T1=2×hATR, T2=3.5×hATR — reduced, 4h left
  - 🌆 Afternoon (1–3 PM): T1=1.5×hATR, T2=2.5×hATR — tight, 2h left
- **per_hr** = `|T1 − entry| / hours_remaining` (not fixed 6.5 divisor)
- **SL** always 1.5× hATR (fixed risk)
- After market close (3:30 PM): per_hr shows ₹0, no虚假 numbers

**Why this matters:**
- At 1:51 PM, HINDALCO was shown at entry ₹1,010 but actually opened at ₹989 — a 2.1% gap already moved
- With afternoon T1=1.5× on hATR=8.2: T1 = 989 + 12.3 = ₹1,001 — achievable in 90 minutes
- Old T1 = 989 + 24.6 = ₹1,014 — needed another 2.5% from already-tired price


---

## 🧭 Option A / B / C Signal Filters

Pass `A`, `B`, `C`, or `A,B,C` (any combination) as positional arguments to enable:

```bash
python3 scan.py --ai --format telegram A         # Option A only: ADX>25 trend filter
python3 scan.py --ai --format telegram B         # Option B only: Momentum mode
python3 scan.py --ai --format telegram A,B,C     # All three options enabled
python3 scan.py --ai --format telegram           # No options = ALL signals (default)
```

### Option A — ADX Trend Filter ✅ RECOMMENDED
**Only trade when ADX > 25** (market is trending, not choppy).
- ADX < 20: no trend → signals blocked
- ADX 20-25: weak trend → signals blocked
- ADX > 25: trending → signals allowed
- ADX > 40: strong trend 🔥
- **Effect:** Filters choppy days → higher win rate, fewer false signals
- **Config:** `nifty_core.ADX_CONFIG['threshold'] = 25`

### Option B — Momentum Mode
**BUY when RSI>70 + MACD bearish-diverging (top-picking)** / **SELL when RSI<30 + MACD bullish-diverging (bottom-picking)**.
- Opposite of mean-reversion: you're fading extremes, not catching bounces
- Works best in strong trending markets (combine with Option A)
- RSI 70+ = overbought zone → momentum BUY fires expecting a drop
- RSI 30- = oversold zone → momentum SELL fires expecting a bounce
- **Config:** `nifty_core.MOMENTUM_CONFIG`

### Option C — Morning Window + Tight SL
**Trade 9:30–11:00 AM IST only, SL=0.75× hourly ATR.**
- Morning session trends are most reliable (overnight info priced in)
- Tight stop: 0.75× hATR (~0.5-0.7% risk vs 1.5-2% normal)
- T1=1.5× hATR, T2=2.5× hATR
- Outside 9:30–11:00 window: no signals generated
- **Config:** `intraday_core.ENABLE_MORNING_WINDOW`, `SL_MULT_TIGHT=0.75`

---

## 📊 SWING System (daily candles)

**Backtest:** 2023-06-08 → 2026-06-07 (3 years, 46 stocks)
| Metric | Value |
|--------|-------|
| Qualified | **27/46** |
| Avg Return | **+0.72%** |
| Win Rate | **39.2%** |
| Total Trades | 1,579 |

**Best stocks:** EICHERMOT +3.67% WR=46.9% | COFORGE +2.57% | APOLLOHOSP +2.57%

**Signal logic:** RSI < 38 (oversold) + momentum + MA20 confirm → BUY
**Exit:** SL=3×ATR | T1=2×ATR (10% partial) | T2=3.5×ATR | ABSSL 8%/5%

**Config (nifty_core.py):**
```python
RSI_CONFIG = {'buy_strict': 38, 'sell_strict': 60, 'sell_relaxed': 40}  # v30: sell_strict 50→60
ADX_CONFIG = {'period': 14, 'threshold': 25, 'enabled': True}             # v31: Option A
MOMENTUM_CONFIG = {'enabled': False, 'rsi_overbought': 70, 'rsi_oversold': 30}  # v31: Option B
```

---

## 🕐 INTRADAY System (5-min candles, same-day exit)

**Backtest:** 10 days, 44 stocks
| Metric | Value |
|--------|-------|
| Qualified | **29/44** |
| Win Rate | **61.4%** |
| Avg Return | **+0.70%** |
| Total Trades | 540 |

**Best stocks:** TITAN +2.15% WR=100% | NMDC +1.98% WR=92.6% | RELIANCE +0.97% WR=89.5%

**Dual signal system:**
1. **EMA_CROSS** (rare, high quality): EMA(9) × EMA(21) on 5-min + hourly RSI confirm
2. **RSI_OVERSOLD / RSI_OVERBOUGHT** (frequent): RSI(14) on 5-min < 25 or > 75 + hourly RSI confirm

**Levels (v34 ZONE-ADAPTIVE — targets shrink as day progresses):**
```
🌅 Morning  (9:30–11AM):  SL=1.5× hATR | T1=3.0× hATR | T2=5.0× hATR
☀️ Midday   (11AM–1PM):  SL=1.5× hATR | T1=2.0× hATR | T2=3.5× hATR
🌆 Afternoon(1PM–3PM):   SL=1.5× hATR | T1=1.5× hATR | T2=2.5× hATR
⏱️  per_hr  = |T1 − entry| ÷ hours_remaining (NOT fixed 6.5 divisor)
```
**Entry** = today's 9:15 AM open (NOT live/stale 5-min close) ✅ v34
**Option C (Morning Tight):** SL=0.75× hATR | T1=1.5× | T2=2.5×
**Trailing SL:** After T1 hit, SL locks at entry price and trails by 1.5×ATR
**Square-off:** 3:15 PM IST sharp — ALL positions closed same day

**Config (intraday_core.py):**
```python
RSI_BUY_THRESHOLD = 25; RSI_SELL_THRESHOLD = 80
SL_MULT = 1.5; T1_MULT = 3.0; T2_MULT = 5.0
ENABLE_MORNING_WINDOW = True    # v31: Option C — 9:30–11:00 AM IST only
SL_MULT_TIGHT = 0.75           # v31: Option C tight stop
T1_MULT_TIGHT = 1.5           # v31: Option C target
T2_MULT_TIGHT = 2.5           # v31: Option C second target
```

---

## 🚀 Usage

```bash
# SWING — live scan (default: all signals)
python3 scan.py --index nifty50 --ai --format telegram

# SWING — with Option A (ADX>25 filter)
python3 scan.py --index nifty50 --ai --format telegram A

# SWING — with Option A+B (ADX filter + Momentum mode)
python3 scan.py --index nifty50 --ai --format telegram A,B

# SWING — with all options A,B,C
python3 scan.py --index nifty50 --ai --format telegram A,B,C

# SWING — backtest
python3 backtest.py

# INTRADAY — live scan (default: all-day, SL=1.5×)
python3 intraday_core.py

# INTRADAY — Option C (morning window + tight SL)
python3 intraday_core.py --morning-only --tight-sl

# INTRADAY — backtest
python3 intraday_backtest.py --days 10

# Train ML models
python3 train.py --index nifty50 --all
python3 train.py --index nifty100 --cleanup
```

---

## 📁 Key Files
| File | Purpose |
|------|---------|
| `nifty_core.py` | Single source of truth: signals, levels, config |
| `scan.py` | Live swing scanner with AI commentary |
| `backtest.py` | 3-year swing backtest |
| `intraday_core.py` | Live intraday scanner (5-min candles) |
| `intraday_backtest.py` | Intraday backtest with TSL |
| `train.py` | ML model trainer (random forest) |
| `table_format.py` | Telegram table formatting |
| `docs/TECHNICAL.md` | **Code explained** — signal logic, AI pipeline, ML model, levels |
| `docs/USER_GUIDE.md` | **User guide** — how to read output, commands, daily workflow |

---

## ⚠️ Disclaimer
Paper trading only. No live money. Backtested results do not guarantee future performance.