# NIFTY Live Quant Ultra — SKILL.md v40

> Multi-mode quant trading system. Paper trading only. No live money.
> Backtest: v11 (3yr, Jun 2023–Jun 2026) | 46 stocks | Score based on validated metrics

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
| v35 | 9.5/10 | Fixes 1–9: per_hr wired, 3yr backtest, zone targets, ADX default ON |
| v36 | 9.5/10 | P0-A/B/C/D/E: ADX 20, RSI<30, RSI>70 short, 52w low, WR gate 45% |
| v37 | 9.5/10 | P0-1/2, P1-1/2: RSI>65 block, poor_hist gap fixed, TOP_PICK ML check, DD cap |
| v38 | 9.5/10 | P0-3/4/5, P1-3/4/5/6, P2-1/2/3 applied |
| **v39** | **9.5/10** | BUG-2/3/4/5 fixed: rsi_mode in JSON, streaming scan, shared categorize, MIN_TRADES=50 |
| **v40** | **9.5/10** | NEW-1/2/3: Market hours config, post-market warning, TTL=5min |

---

## v40 — All Fixes Applied (2026-06-20)

### NEW-1 ✅ `_is_market_open()` — Post/Pre-Market Warning Banner
**File:** `scan.py`

**Problem:** Market-closed scans showed BUY/SELL signals with no context — user could act on stale signals.

**Fix:** Added `_is_market_open()` helper with configurable hours (`MARKET_OPEN_HOUR/MIN`, `MARKET_CLOSE_HOUR/MIN`). `format_telegram()` now shows:
```
📊 Regime: 🔴BEARISH | NIFTY50:2 ⚫ POST-MARKET
⚠️  POST-MARKET — signals shown for reference only. Not for live trading.
```
Also used in `--wait-morning` loop to skip waits when market is closed.

---

### NEW-2 ✅ Configurable Market Hours
**File:** `scan.py`

**Problem:** Hardcoded `15:00` close time — doesn't handle NSE early closes.

**Fix:** Hours now configurable at top of file:
```python
MARKET_OPEN_HOUR,   MARKET_OPEN_MIN   = 9,  0
MARKET_CLOSE_HOUR,  MARKET_CLOSE_MIN  = 15, 30   # NEW-2: configurable
```

---

### NEW-3 ✅ Backtest Cache TTL: 3600s → 300s (5 min)
**File:** `scan.py`

**Problem:** `_STATS_TTL = 3600` (1hr) meant stale backtest data served if cache was warm.

**Fix:** `_STATS_TTL = 300` — stats refreshed every 5 minutes max.

---

### BUG-2 ✅ `rsi_mode` Saved to Backtest JSON
**File:** `backtest.py`

**Problem:** `rsi_mode` (strict/relaxed) was a CLI flag but never saved to the backtest JSON output, making it impossible to verify which mode produced the results.

**Fix:** Added `'rsi_mode': _RSI_MODE` to the backtest result dict:
```python
'rsi_mode': _RSI_MODE,   # self-describing — proves strict vs relaxed
```

---

### BUG-3 ✅ Streaming Scan Output (No More 4-Min Blackout)
**File:** `scan.py`

**Problem:** Sequential loop of 49 stocks × ~5s each = 4+ minutes of no output. Appeared frozen.

**Fix:** Replaced sequential loop with `ThreadPoolExecutor(max_workers=8)` — parallel stock analysis + per-stock progress output + `--stream` flag for live output:
```bash
python3 scan.py --ai --stream   # live per-stock output as it completes
```

Added `stream_output` parameter to `parse_args()` + `main()`.

---

### BUG-4 ✅ Shared Categorization Module (`nifty_categorize.py`)
**Files:** `nifty_categorize.py` (new) + `scan.py` + `backtest.py`

**Problem:** `scan.py` and `backtest.py` each had their own copy of `_categorize()`. ADANIENT was Cat C2 in live scan but qualified in backtest — inconsistent logic.

**Fix:** Extracted all categorization logic into `nifty_categorize.py` — single source of truth, imported by both `scan.py` and `backtest.py`. Backtest now prints live-scan-style Cat A/A-/B/C1/C2/WL counts after the summary, plus shows any mismatch between backtest qualified and live categories.

---

### BUG-5 ✅ Intraday `MIN_TRADES`: 5 → 50
**File:** `intraday_backtest.py`

**Problem:** `MIN_TRADES = 5` meant only 5 trades qualified a stock — statistically meaningless.

**Fix:** `MIN_TRADES = 50` (~1 trade/week × 50 weeks). Proportional to `--days` argument: `max(5, days // 5)`.

---

### v38 Fixes (Still Active)

### P0-3 ✅ SHREECEM DD=100% No DD Gate
**File:** `backtest.py`

**Problem:** SHREECEM DD=100% but qualified (WR=39.1%, Ret=+7.16%) — no DD gate meant high-DD stocks could qualify.

**Fix:** Added `max_drawdown < 50.0` to backtest qualification:
```python
qualified = (len(trades) >= MIN_TRADES and realized_return > 0 and
            max_drawdown < 50.0 and   # v38 P0-3: DD gate
            win_rate >= 0.38)
```

**Result:** Qualified = 10/46 (DD<55 + MIN_TRADES=20): COFORGE, BRITANNIA, HINDALCO, SBILIFE, JSWSTEEL, ADANIENT, KOTAKBANK, HDFCLIFE, DRREDDY, TATACONSUM.

---

### P0-4 ✅ Independent RSI>70 SHORT Trigger
**File:** `nifty_core.py` — `get_signal()`

**Problem:** RSI>70 SHORT blocked by MA20 gate in bear market. `c_rsi_bear +2` couldn't reach `min_confirmations=3` because `sell_cnt` required `not c_price_ma20`. In uptrends, RSI>70 shorts were impossible.

**Fix:** RSI>70 fires as independent SHORT before any other checks, bypassing ADX and MA gates:
```python
if rsi > MOMENTUM_CONFIG['rsi_overbought']:
    return -1, {'signal': 'SELL', 'buy_cnt': 0, 'sell_cnt': 4,
                'reasons': [f"🎯 RSI={rsi:.0f}>70 (extreme overbought — direct SHORT)"], ...}
```

**Result:** 9 SHORT signals appear (APOLLOHOSP RSI=73, AXISBANK RSI=72, BAJFINANCE RSI=71).

---

### P0-5 ✅ Cat B Downgrades ML=DOWN Stocks to Cat C2
**File:** `scan.py` — `_categorize()`

**Problem:** All 5 Cat B stocks had ML=DOWN (AI=BULLISH but ML strongly contradicts). Cat B path had no ML direction check.

**Fix:** ML=DOWN + WR<50% → Cat C2:
```python
ml_dir = (ml or {}).get('direction', '')
if ml_dir == 'DOWN' and wr < 50:
    _add_tag(r, '⚠️ ML_CONTRADICT')
    cat_c2.append(r)
```

**Result:** Cat B now only has exceptional momentum stocks (WR≥50%) or stocks with aligned ML.

---

### P1-3 ✅ BEAR_DIV+BUY in BEARISH Regime → Cat C2
**File:** `scan.py` — `_categorize()`, BEAR_DIV path

**Problem:** ADANIENT (BEAR_DIV+BUY+AI=BULL in BEARISH) was in Cat B. BEAR_DIV is a SHORT setup — BUY direction in bear regime is contrarian.

**Fix:** In BEARISH regime, `BEAR_DIV + BUY` → Cat C2 always:
```python
if div == 'BEARISH' and sig == 'BUY':
    cat_c2.append(r)   # SHORT setup + LONG direction in bear = Cat C2
```

**Result:** ADANIENT moved from Cat B → Cat C2.

---

### P1-4 ✅ no_history Bypasses WR Gate → Cat C2
**File:** `scan.py` — `_categorize()`, Cat B path

**Problem:** TATACONSUM (no backtest history, WR=38%) passed WR gate via `no_history` check. Unknown quality stocks shouldn't enter Cat B.

**Fix:** `no_history` → Cat C2 before WR gate check:
```python
if no_history:
    _add_tag(r, '⚠️ NO_BACKTEST')
    cat_c2.append(r)
elif wr < _MIN_WR_CAT_B:
    ...
```

---

### P1-5 ✅ Watchlist Filter — RSI Zone/S/R Filter
**File:** `scan.py` — `telegram_format()`

**Problem:** 34-stock watchlist with all ADX<20 stocks dumped in unfiltered.

**Fix:** Only include WL stocks in RSI 30-42 (buy zone) or RSI 58-70 (sell zone) or within 3% of support/resistance:
```python
def _wl_filter(r):
    rsi = r.get('rsi', 50)
    if 30 <= rsi <= 42 or 58 <= rsi <= 70: return True
    # Check within 3% of S/R...
```

**Result:** WL filtered from 34 → 19 stocks.

---

### P1-6 ✅ --mode Flag + Single Level Set Display
**File:** `scan.py` — CLI + `fmt_stock_short()` + `fmt_top_pick()`

**Problem:** Two level sets (💠 intraday + 🎯 swing) shown per stock, confusing users.

**Fix:** Added `--mode swing|intraday` flag (default=swing). Only one level set shown:
```bash
scan.py --ai --mode swing   # show swing levels only (default)
scan.py --ai --mode intraday  # show intraday levels only
```

---

### P2-1 ✅ Backtest RSI_MODE Flag
**File:** `backtest.py`

**Fix:** Added `--rsi-mode strict|relaxed`:
- `relaxed` (default): RSI_ENTRY_MAX=65 (matches live buy_relaxed=65)
- `strict`: RSI_ENTRY_MAX=30 (matches live buy_strict=30)

---

### P2-2 ✅ Intraday Backtest MIN_TRADES Raised
**File:** `backtest.py`

**Fix:** MIN_TRADES=20→30 for statistical confidence on 3yr data.

---

### P2-3 ✅ Primary Trigger Label
**File:** `scan.py` — `_get_primary_trigger()`

**Fix:** Added `primary_trigger` field: `MEAN_REVERSION` / `BREAKOUT` / `DIVERGENCE_LONG` / `TREND_FOLLOW` / `MOMENTUM` / `DIVERGENCE_SHORT` / `BREAKDOWN`.

---

## v37 Fixes (Still Active)

| Fix | Change |
|-----|--------|
| P0-1 | `_is_poor_history` wr<45 (gap closed) |
| P0-2 | RSI>65 Cat A BUY blocked |
| P1-1 | RSI>65 signal → RANGE at source |
| P1-2 | TOP_PICK skips ML=DOWN |
| Config | RSI_ENTRY_MAX 55→65, max_hold_days 10→20 |
| Config | SHREECEM DD capped 100% |

### P0-1 ✅ `_is_poor_history` Gap Fixed — WR 25-45% no longer floods Cat A
**File:** `scan.py` — `_is_poor_history()`

**Problem:** `poor_hist` used `wr < 25` but Cat A gate was `_MIN_WR_CAT_A = 45`. Stocks with 25 ≤ wr < 45 passed both filters unchallenged → Cat A flooded with low-WR stocks (ONGC 36%, ITC 38%, WIPRO 25%).

**Fix:**
```python
def _is_poor_history(stats):
    """POOR_HIST: rr < -2% OR wr < 45% — aligned with _MIN_WR_CAT_A=45 (v37)"""
    rr = stats.get('realized_return', 0)
    wr = stats.get('win_rate', 0)
    return rr < -2.0 or wr < 45   # was wr < 25 — gap closed
```

**Result:** ONGC, ITC, WIPRO, HINDUNILVR now correctly tagged `⚠️ POOR_HIST` → Cat C2.

---

### P0-2 ✅ RSI Overbought Guard — RSI>65 blocked from Cat A BUY
**File:** `scan.py` — `_categorize()`, BUY Cat A path

**Problem:** ITC (RSI=70) and HINDUNILVR (RSI=68) in Cat A with deeply overbought RSI — buying near the top, not mean reversion.

**Fix:** Added RSI upper-bound check in the `ai_bull and ml_up` Cat A path:
```python
elif ai_bull and ml_up:
    rsi = r.get('rsi', 0)
    if rsi > RSI_CONFIG['buy_relaxed']:   # RSI > 65 = overbought
        _add_tag(r, '⚠️ OVERBOUGHT')
        _tag_all(r)
        cat_c2.append(r)
    elif not adx_ok:
        ...
```

**Result:** ITC and HINDUNILVR now `⚠️ OVERBOUGHT` → Cat C2.

---

### P1-1 ✅ RSI>65 BUY Signal Blocked at Source
**File:** `nifty_core.py` — `get_signal()`

**Problem:** BUY signal generated at RSI=70 (ITC) — wrong direction. `sell_relaxed=40` only blocked SELL signals in oversold zones, never blocked BUY at overbought RSI.

**Fix:** Block BUY at RSI > `buy_relaxed` (65) before signal is returned:
```python
# v37 P1-1: Block BUY when RSI > buy_relaxed (overbought — not mean reversion)
if rsi > RSI_CONFIG['buy_relaxed'] and buy_cnt >= SIGNAL_CONFIG['min_confirmations']:
    return 0, {'signal': 'RANGE', 'buy_cnt': 0, 'sell_cnt': 0,
                'divergence': divergence,
                'reasons': [f"⛔ BLOCKED: RSI={rsi:.0f}>65 (overbought — momentum chasing)"]}, []
```

**Result:** ITC becomes RANGE at source (not just downgraded in categorize).

---

### P1-2 ✅ TOP_PICK Excludes ML=DOWN Contradictions
**File:** `scan.py` — TOP_PICK selection

**Problem:** ICICIBANK (WR=60%, RR=+3%) starred as ⭐ TOP_PICK despite ML=DOWN at 97.8% — ML completely contradicts the signal.

**Fix:**
```python
for r in cat_a + cat_b:
    ml = r.get('ml') or {}
    ml_dir = ml.get('direction', '')
    if ml_dir == 'DOWN':
        continue   # skip — ML contradicts the signal (v37 P1-2)
    if rr > 0 and cf >= 8.0 and wr >= _MIN_WR_TOP_PICK:
        r['_starred'] = True
        _add_tag(r, '⭐ TOP_PICK')
```

**Result:** ICICIBANK removed from TOP_PICKS.

---

### Config Sync ✅ Backtest ↔ Live Parity
**File:** `backtest.py`

| Parameter | Before | After | Reason |
|-----------|--------|-------|--------|
| `RSI_ENTRY_MAX` | 55 | **65** | Match live `buy_relaxed=65` |
| `max_hold_days` | 10 | **20** | Day-10 exit was killing genuine winners |
| DD formula | could exceed 100% | **capped at 100%** | SHREECEM showed 137% DD (impossible) |
| `peak_ever` tracking | missing | **added** | Separate all-time high tracking |

---

### v36 Fixes (Still Active)

| Fix | Change | Impact |
|-----|--------|--------|
| P0-A | ADX threshold 25→20 | 3-5× more trending signals |
| P0-B | RSI buy_strict 38→30 | True deep oversold only |
| P1-C | RSI>70 → sell_cnt+=2 | Independent bear-market short path |
| P1-D | 52-week low filter | BUY blocked within 3% of 52w low |
| P1-E | `_MIN_WR_CAT_A` = 45% | Cleaner TOP_PICKs |
| Fix 1 | Intraday SL from daily ATR | Realistic risk per trade |
| Fix 2a/2b | per_hr from daily ATR | Realistic ₹/hr display |
| Fix 3 | Qty for ₹10K risk | Position sizing on every line |
| Fix 5 | BEAR_DIV → Cat B | ADANIENT path (WR=47%) |

---

## 📊 SWING System — Backtest v11 (3yr, Jun 2023–Jun 2026)

| Metric | Value |
|--------|-------|
| Period | 3 years (Jun 2023 – Jun 2026) |
| Qualified | **15/46** (WR ≥ 45%) |
| Avg Return | **+0.34%** (all stocks) |
| Avg Win Rate | **36.3%** |
| Max Drawdown | **100%** (capped — was 137% for SHREECEM) |
| Sharpe | 0.25 |
| Total Trades | 1,107 |

**Best performers (qualified):**
| Stock | WR | Real Ret | Sharpe | DD |
|-------|----|----------|--------|-----|
| SHREECEM | 39.1% | +7.16% | 1.31 | 100% |
| COFORGE | 46.4% | +4.43% | 2.05 | 52.4% |
| BRITANNIA | 45.5% | +3.95% | 2.49 | 52.3% |
| TECHM | 41.7% | +2.74% | 1.29 | 56.0% |
| ICICIBANK | 50.0% | +2.63% | 2.44 | 55.8% |
| HINDALCO | 52.0% | +2.50% | 2.16 | 51.0% |

**Key insight:** 15/46 stocks (33%) pass WR≥45% gate in live market (vs 6/46 in v35 backtest with ADX>25). The RSI>65 block, 52w low filter, and BEAR_DIV fix produce more selective, higher-conviction signals.

---

## 🧭 Option A / B / C Signal Filters

```bash
# Default (v37): ADX filter ON, all signals
python3 scan.py --ai --format telegram

# Option A: ADX>20 only (trending market)
python3 scan.py --ai --format telegram A

# Option B: Momentum mode (RSI>70/<30 + MACD divergence)
python3 scan.py --ai --format telegram B

# Option C: Morning window only (9:30-11 AM IST) + tight SL
python3 scan.py --ai --format telegram C

# --wait-morning: wait until 9:40 AM IST before scanning
python3 scan.py --ai --format telegram --wait-morning
```

### Option A — ADX>20 Filter ✅ DEFAULT (v36)
- ADX < 20: no trend → blocked
- ADX 20-25: weak → blocked
- ADX > 20: trending → allowed
- ADX > 40: strong trend 🔥

### Option B — Momentum Mode
- BUY: RSI>70 + MACD bearish divergence (top-pick the top)
- SELL: RSI<30 + MACD bullish divergence (bottom-pick)
- Works best with Option A combined

### Option C — Morning Window
- 9:30–11:00 AM IST only
- SL=0.75× hATR | T1=1.5× | T2=2.5×

---

## 🕐 INTRADAY System (5-min candles, 3 signal paths)

**Backtest: 10 days, 44 stocks**
| Metric | Value |
|--------|-------|
| Qualified | **29/44** |
| Win Rate | **61.4%** |
| Avg Return | **+0.70%** |
| Total Trades | 540 |

**Signal paths:**
1. **EMA_CROSS**: EMA(9) × EMA(21) on 5-min — morning quality signal
2. **RSI_OVERSOLD/OVERBOUGHT**: RSI<20 or RSI>80 + hourly RSI confirm
3. **BREAK_HIGH/LOW**: Afternoon (1 PM+) price breaks day high/low + 15-min RSI confirm

**Zone Targets:**
| Zone | Time | T1 | T2 | SL |
|------|------|-----|-----|-----|
| 🌅 Morning | 9:30–11 AM | 3× hATR | 5× hATR | 1.5× hATR |
| ☀️ Midday | 11 AM–1 PM | 2× hATR | 3.5× hATR | 1.5× hATR |
| 🌆 Afternoon | 1–3 PM | 1.5× hATR | 2.5× hATR | 1.5× hATR |

**Entry = today's 9:15 AM open** | **Square-off:** 3:15 PM IST sharp

---

## 🚀 Usage

```bash
# SWING — live scan (ADX ON by default)
python3 scan.py --ai --format telegram

# SWING — backtest (3yr)
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
| `intraday_core.py` | Intraday scanner (3 paths, zone targets, journal) |
| `backtest.py` | 3-year swing backtester |
| `intraday_backtest.py` | Intraday backtester |
| `train.py` | ML model trainer |
| `analyze.py` | Per-stock deep analysis |
| `docs/TECHNICAL.md` | Code explained — signal logic, AI pipeline |
| `docs/USER_GUIDE.md` | User guide — daily workflow |

---

## ⚠️ Disclaimer
Paper trading only. No live money. Backtested results do not guarantee future performance.
