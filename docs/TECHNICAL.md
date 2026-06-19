# NIFTY Live Quant Ultra — Technical Documentation

> How the system works under the hood. For developers and quant analysts.

---

## 1. Architecture Overview

```
scan.py          ← Main entry point for live SWING scanning
intraday_core.py ← Entry point for live INTRADAY scanning
backtest.py      ← Historical backtest engine (SWING)
intraday_backtest.py ← Historical backtest engine (INTRADAY)
train.py         ← ML model trainer (GradientBoosting + RandomForest)
analyze.py       ← Single-stock deep analysis
nifty_core.py    ← SINGLE SOURCE OF TRUTH for all shared logic
  ├── Price data (NSE India API + Yahoo Finance fallback)
  ├── Feature engineering (RSI, MACD, ATR, MAs, divergence)
  ├── Signal engine (get_signal)
  ├── 9-stage AI opinion pipeline
  └── All configuration constants
models/          ← Trained ML models + backtest JSON results
```

The system has **two modes** that share `nifty_core.py`:

| Mode | Candle | Hold | Exit | Run with |
|------|--------|------|------|----------|
| **SWING** | Daily | Multi-day | SL / T1 / T2 / TSL | `scan.py --ai` |
| **INTRADAY** | 5-min | Same day | SL / T1 / T2 / 3:15 PM sharp | `intraday_core.py` |

---

## 2. Market Regime Detection

**File:** `nifty_core.py` → `get_market_regime()`

```python
def get_market_regime():
    nifty = yf.Ticker("^NSEI").history(period="30d")
    nifty_ma20 = nifty['Close'].rolling(20).mean().iloc[-1]
    nifty_ma50 = nifty['Close'].rolling(50).mean().iloc[-1]
    nifty_price = nifty['Close'].iloc[-1]

    if nifty_price > nifty_ma20 and nifty_price > nifty_ma50:
        regime = "BULLISH"   # price above both averages
    elif nifty_price < nifty_ma20 and nifty_price < nifty_ma50:
        regime = "BEARISH"   # price below both averages
    else:
        regime = "NEUTRAL"   # mixed signals
```

**Cached globally** — one fetch per session, reused for all 50 stocks (prevents 50 redundant API calls).

**Usage:** Regime affects short-signal labeling (a SELL signal in a BULL regime gets tagged 🔴 CONTRARIAN) and confluence scoring.

---

## 3. Feature Engineering

**File:** `nifty_core.py` → `add_features()`

All features are computed from daily OHLCV data:

```python
def add_features(df):
    # Moving averages
    for w in [5, 10, 20, 50, 100, 200]:
        df[f'ma{w}'] = df['Close'].rolling(w).mean()

    # RSI(14)
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['rsi'] = 100 - (100 / (1 + gain / loss))

    # MACD (12, 26, 9)
    ema12 = df['Close'].ewm(span=12).mean()
    ema26 = df['Close'].ewm(span=26).mean()
    df['macd']  = ema12 - ema26
    df['macd_sig'] = df['macd'].ewm(span=9).mean()

    # ATR(14) — True Range
    tr = pd.concat([
        df['High'] - df['Low'],
        abs(df['High'] - df['Close'].shift()),
        abs(df['Low']  - df['Close'].shift())
    ], axis=1).max(axis=1)
    df['atr'] = tr.rolling(14).mean()

    # Volume ratio
    df['vol_ma'] = df['Volume'].rolling(20).mean()
    df['vol_ratio'] = df['Volume'] / (df['vol_ma'] + 1)

    # 5-day return
    df['ret5'] = df['Close'].pct_change(5)

    return df
```

**RSI Thresholds** (`RSI_CONFIG` in `nifty_core.py`):
```python
RSI_CONFIG = {
    'buy_strict':  38,   # RSI < 38 → oversold (BUY signal fires)
    'buy_relaxed': 65,   # allow BUY up to RSI 65 in strong uptrend
    'sell_strict': 50,   # RSI > 50 → overbought
    'sell_relaxed': 36,  # block SELL below 36 (deeply oversold bounce zone)
}
```

> ⚠️ **Important:** These are **mean-reversion** thresholds. Low RSI = price dropped too much, expect bounce. The ML model independently predicts 5-day momentum — they measure different things.

---

## 4. SWING Signal Engine

**File:** `nifty_core.py` → `get_signal()`

```python
def get_signal(df, i):
    """
    Returns: (signal_val, meta_dict, [])
    signal_val: 1=BUY, -1=SELL, 0=RANGE
    """
```

### How a BUY Signal Fires

Each of 7 conditions earns **+1 point** when true:

| # | Condition | Meaning |
|---|-----------|---------|
| 1 | Price > MA20 | Above short-term trend |
| 2 | Price > MA50 | Above medium-term trend |
| 3 | MA50 > MA200 | Long-term uptrend confirmed |
| 4 | RSI < 38 | Oversold (bounce likely) |
| 5 | MACD > MACD Signal | Positive momentum |
| 6 | Volume > 80% above avg | Unusual interest |
| 7 | 5-day return > 0 | Recent upward move |

**Bonus:** If RSI < 38 AND (MA5 > MA20 OR ret5 > 0) → **+1 extra point** (avoid buying falling knives)

**Divergence bonus:** If RSI shows **BULLISH divergence** → **+2 points**

**Minimum to fire:** `min_confirmations = 2` (need ≥ 2 conditions)

### RSI Guards (prevent bad signals)

```python
# Block SELL when RSI < 36 — deeply oversold, bounce zone, don't short
if rsi < RSI_CONFIG['sell_relaxed']:
    sell_cnt = 0

# RSI 36-38 with price > MA20: oversold bounce zone, don't sell into recovery
elif rsi < RSI_CONFIG['buy_strict'] and c_price_ma20:
    sell_cnt = 0
```

### Divergence Detection

```python
def detect_divergence(df):
    # BULLISH: price made a lower low, but RSI made a higher low
    # → price falling but momentum improving = hidden bullish signal
    if recent_price_lower_low AND rsi_higher_low:
        return "BULLISH"

    # BEARISH: price made a higher high, but RSI made a lower high
    # → price rising but momentum weakening = top signal
    if recent_price_higher_high AND rsi_lower_high:
        return "BEARISH"

    return None
```

---

## 5. ATR-Based Levels

**File:** `nifty_core.py` → `calc_levels()`

Three mode presets (`ATR_CONFIG`):

```python
ATR_CONFIG = {
    # INTRADAY: trading view default
    'intraday':       {'sl': 3.0, 't1': 2.0, 't2': 3.5},
    # INTRADAY_TIGHT: scalp levels (swing scan's hourly levels display)
    'intraday_tight': {'sl': 1.5, 't1': 0.75, 't2': 1.5},
    # SWING: wider for multi-day holds
    'swing':          {'sl': 2.0, 't1': 3.0, 't2': 6.0},
}
```

**Level formula:**
```
SL  = Entry − ATR × SL_MULT
T1  = Entry + ATR × T1_MULT   (first target, partial exit)
T2  = Entry + ATR × T2_MULT   (second target, full exit)
```

**ABSSL (Adaptive Break-Even Stop Loss):**
```python
abs_sl_pct = 0.92  # Before T1 hit:  8% max loss
abs_sl_pct = 0.95  # After T1 partial: 5% lock-in
abs_sl_pct = None  # After T2 hit: disabled, let it run
```

**Hourly scalp levels** (shown as 💠 in scan output):
```python
def calc_levels_hourly(price, hourly_atr, sig='BUY'):
    # Tight intraday targets for same-day/hourly trading
    return {
        'sl': price − hourly_atr × 1.0,
        't1': price + hourly_atr × 0.5,
        't2': price + hourly_atr × 1.0,
    }
```

---

## 6. 9-Stage AI Opinion Pipeline

**File:** `nifty_core.py` → `ai_opinion_pipeline()`

Each stock gets scored across 9 stages. Total score = −100 to +100.

```python
def ai_opinion_pipeline(symbol, price, rsi, macd, macd_sig, atr, vol_ratio, ret5, df):
    # Stage 1: Market Regime
    # Stage 2: News Sentiment  (proxied via ret5)
    # Stage 3: Stock Scanner   (setup quality)
    # Stage 4: Trade Validator (S/R + volume)
    # Stage 5: Options Flow    (proxied via RSI)
    # Stage 6: Risk Manager   (ATR-based levels, position size)
    # Stage 7: Execution Timing
    # Stage 8: Trade Replay
    # Stage 9: Learning       (RSI-trend alignment correction)
```

| Stage | What it measures | Score range |
|-------|-----------------|-------------|
| 1. Market Regime | BULL/NEUTRAL/BEAR from NIFTY index | −15 to +15 |
| 2. News Sentiment | 5-day return proxy (>2% = bullish news) | −30 to +30 |
| 3. Stock Scanner | RSI, MACD, volume, MAs above price | −25 to +60 |
| 4. Trade Validator | Near support with volume = CONFIRMED | −10 to +15 |
| 5. Options Flow | RSI proxy for options market sentiment | −10 to +10 |
| 6. Risk Manager | ATR levels, RR ratio, position size | — |
| 7. Execution | Best time of day to enter | 0 |
| 8. Trade Replay | Recent volatility vs history | −5 to +35 |
| 9. Learning | RSI-trend alignment correction | −7.5 to 0 |

**Output example:**
```python
{
    'outlook': 'BULLISH',      # BULLISH / NEUTRAL / BEARISH
    'confidence': 'HIGH',       # HIGH / MEDIUM / LOW
    'total_score': 62.5,        # −100 to +100
    'stages': { ... }
}
```

---

## 7. ML Model (34 Features, 3 Labels)

**File:** `train.py` → `build_features()` + `build_labels()`

**34 Features** per stock per timestamp:

| Group | Features |
|-------|----------|
| Moving Averages | MA5, MA10, MA20, MA50, MA100, MA200 (6) |
| Technical | RSI, MACD, MACD Signal, ATR, Vol ratio, ret5 (6) |
| Price Ratios | C/MA20, C/MA50, MA20/MA50, MA50/MA200, MACD-Sig, RSI×VolRatio (6) |
| ATR Context | (C−MA20)/ATR, (C−MA50)/ATR, MACD/C, VolRatio−1 (4) |
| Returns | 5-day, 10-day, 20-day, RSI/100, ret5×10, 5d vs 20d MA (6) |
| Volatility | 20d StdDev/Mean, Vol MA ratio, 3d vs 10d, 5d vs 30d, ... (6) |

**Total: 34 features**

**Labels** (3 classes):
```python
5-day_return > +2%  →  1 (UP)
5-day_return < −2%  → −1 (DOWN)
else                →  0 (NEUTRAL)
```

> ⚠️ ML labels are **momentum-based** (will price go up 2%+ in 5 days?). The rule-based signal is **mean-reversion** (RSI oversold = expect bounce). They measure different things and are used together as a filter.

**Model architecture:**
```python
GradientBoostingClassifier(learning_rate=0.05, max_depth=6, n_estimators=150)
RandomForestClassifier(max_depth=8, n_estimators=100)
VotingClassifier(estimators=[('gb', gb), ('rf', rf)], voting='soft')
```

---

## 8. Confluence Scoring (1–10)

**File:** `scan.py` → `calc_confluence_score()`

```python
def calc_confluence_score(r, regime='BULLISH'):
    score = (
        sig_conf       * 0.20 +   # Rule-based signal conviction (0–1)
        ai_final       * 0.25 +   # AI opinion pipeline (0–1)
        wr             * 0.20 +   # Historical win rate (0–1)
        align_score    * 0.10 +   # AI T1 vs Signal T1 within 3% = 1.0
        age_score      * 0.10 +   # Fresh < 4d = 1.0, stale = 0.9, critical = 0.7
        regime_score   * 0.15     # Aligned with regime = 1.0, contrarian = 0.3
    )
    return round(score * 10, 1)  # 1–10
```

**NEG_HIST penalty:** If `rr < −2% AND wr < 40%` → score × 0.7 (−30%)

**Stale penalty:**
```python
if age < 4 days:  age_score = 1.0   # no penalty
if 4 ≤ age < 8:  age_score = 0.9   # 10% penalty, tag ⏰ STALE
if age ≥ 8 days:  age_score = 0.7   # 30% penalty, tag 💀 CRITICAL
```

---

## 9. Stock Categorization Logic

**File:** `scan.py` → `_categorize()`

```
All stocks
  └── Signal = RANGE → WATCHLIST

  └── Signal = BUY
      ├── BEARISH DIVERGENCE → Cat C2 (labeled 🔴 CONTRARIAN)
      ├── AI BULLISH + ML UP → Cat A (triple confirmed)
      │     ├── poor history (rr < −2% OR wr < 25%) → downgraded to Cat C2
      │     └── level mismatch (AI T1 vs Signal T1 > 3%) → Cat A-
      ├── AI BULLISH + no ML UP → Cat B (AI confirmed)
      │     ├── wr < 35% → downgraded to Cat C2
      │     └── NEG_HIST → Cat C2
      └── ML UP only → Cat D (ML signal only)

  └── Signal = SELL
      ├── ML DOWN + AI not BULLISH → Cat A (triple confirmed short)
      ├── AI BEARISH → Cat C1 or A- (A- if in BULL regime = hedge)
      └── no ML/AI confirmation → Cat C2
```

**Cat A / Cat B quality gates:**
- Cat A: triple confirmed + WR ≥ 40%
- Cat B: AI confirmed + WR ≥ 35%
- TOP_PICK: Cat A/B + RR > 0 + CF ≥ 8.0 + WR ≥ 40% (⭐ badge)

**Regime coherence:**
- In BULL regime: BUY + no divergence = coherent (score 1.0)
- In BULL regime: BUY + BEAR_DIV = 🔴 CONTRARIAN (score 0.0, big CF penalty)
- In BULL regime: SELL = 🛡️ HEDGE (score 0.6)

---

## 10. SWING Backtest Engine

**File:** `backtest.py` → `backtest_stock()`

**Window:** 3 years (1095 days of data)

**Entry rules:**
1. Signal must be BUY (sig_val == 1)
2. No existing position
3. Not in BLACKLIST (`{'SBIN', 'BHEL', 'TITAN'}`)
4. RSI < 60 at entry (avoid overbought entries)
5. Sector limit check (optional)

**Position sizing:** Risk 1% of capital per trade
```python
risk = capital × 0.01
shares = risk / (ATR × SL_MULT)
```

**Exit types (priority order):**
1. **ABSSL** — Adaptive Break-Even SL: 8% hard cap before T1, 5% after T1 partial
2. **T1 Partial** — One-time sell 10% of position when price hits T1
3. **TSL** — Trailing Stop Loss (activated after T1 partial)
4. **Fixed SL** — ATR × SL_MULT
5. **T2 Full Exit** — When price reaches T2 after T1 was hit
6. **Time Exit** — Max hold 15 days for losing trades only
7. **Signal Exit** — SELL signal fires (disabled by default: `--no-sig-exit`)
8. **End of data** — Close at last available price

**Key metrics:**
```python
# PRIMARY metric (realistic):
realized_return = realized_pnl_sum / initial_capital × 100

# Secondary (inflated by compounding):
compounded_return = ((capital - initial_capital) / initial_capital) × 100

# Sharpe ratio (annualized from per-trade P&L %):
sharpe = (mean_pnl / std_pnl) × sqrt(252 / avg_hold_days)

# Max drawdown (from peak realized, settled capital only):
max_dd = (peak_realized − capital) / peak_realized × 100
```

**Qualified stocks:** 20+ trades AND positive realized return AND WR ≥ 33%

---

## 11. INTRADAY System

**File:** `intraday_core.py` → `get_intraday_signal()`

**Dual signal system:**

### Signal Type 1: EMA Crossover (rare, high quality)
```
BUY:  EMA(9) crosses ABOVE EMA(21) on 5-min candles
      + Hourly RSI < 50 (confirms uptrend)
      + Price > Open price
```
```
SELL: EMA(9) crosses BELOW EMA(21) on 5-min candles
      + Hourly RSI > 50 (confirms downtrend)
      + Price < Open price
```

### Signal Type 2: RSI Oversold/Overbought (frequent)
```
BUY:  RSI(5-min, 14) < 25  AND  Hourly RSI < 50
SELL: RSI(5-min, 14) > 75  AND  Hourly RSI > 50
```

**Levels:**
```python
SL  = Entry ± 1.5 × hourly_ATR
T1  = Entry ± 3.0 × hourly_ATR
T2  = Entry ± 5.0 × hourly_ATR
Per-hour target = hourly_ATR × 0.75
```

**Square-off:** 3:15 PM IST sharp — ALL positions closed, no exceptions.

---

## 12. Quantity Calculation (₹10k Risk)

**Formula:**
```
Qty = ₹10,000 / (Entry Price − Stop Loss)
```

This ensures exactly ₹10,000 risk per stock regardless of price level.

**Example for M&M at ₹3,133 with SL ₹2,964:**
```
Risk per share = 3133 − 2964 = ₹169
Qty = 10000 / 169 = 417 shares
Position value = 417 × 3133 = ₹13,06,461
```

---

## 13. Configuration Reference

### RSI_CONFIG (nifty_core.py)
```python
RSI_CONFIG = {
    'period': 14,
    'buy_strict':  38,   # RSI below this = oversold (BUY zone)
    'buy_relaxed': 65,   # allow BUY up to RSI 65 in uptrend
    'sell_strict': 50,   # RSI above this = overbought (SELL zone)
    'sell_relaxed': 36,  # block SELL below 36 (bounce trap guard)
}
```

### ATR_CONFIG (nifty_core.py)
```python
ATR_CONFIG = {
    'intraday':       {'sl': 3.0, 't1': 2.0, 't2': 3.5},
    'intraday_tight': {'sl': 1.5, 't1': 0.75, 't2': 1.5},
    'swing':          {'sl': 2.0, 't1': 3.0, 't2': 6.0},
    'period': 14,
}
```

### SIGNAL_CONFIG (nifty_core.py)
```python
SIGNAL_CONFIG = {
    'min_confirmations': 2,   # conditions needed to fire a signal
    'volume_spike': 0.8,       # vol must be 80% above 20d avg
    'vol_spike_strong': 1.3,  # strong volume signal
    'momentum_zero': 0,       # ret5 > 0 = positive momentum
}
```

### INTRADAY_CONFIG (intraday_core.py)
```python
RSI_BUY_THRESHOLD  = 25   # RSI(5m) < 25 to fire RSI BUY
RSI_SELL_THRESHOLD = 75   # RSI(5m) > 75 to fire RSI SELL
RSI_H_BULL_THRESHOLD  = 50  # hourly RSI for BUY confirm
RSI_H_BEAR_THRESHOLD  = 50  # hourly RSI for SELL confirm
SL_MULT = 1.5; T1_MULT = 3.0; T2_MULT = 5.0
SQUARE_OFF_HOUR = 15; SQUARE_OFF_MIN = 15  # 3:15 PM IST
```

---

## 14. Key Files Summary

| File | Purpose | Key Function |
|------|---------|-------------|
| `nifty_core.py` | Single source of truth | All config, data fetching, signal logic |
| `scan.py` | Live SWING scanner | `analyze()`, `_categorize()`, `format_telegram()` |
| `intraday_core.py` | Live INTRADAY scanner | `get_intraday_signal()`, `scan_intraday()` |
| `backtest.py` | 3-year SWING backtest | `backtest_stock()` |
| `intraday_backtest.py` | INTRADAY backtest | Same structure, 5-min candles |
| `train.py` | ML model trainer | `train_stock()`, `build_features()` |
| `analyze.py` | Single-stock analysis | `analyze()` with full debug output |
| `compare_cat.py` | Compare Cat A vs B returns | Filters and sorts by category |
| `compare_modes.py` | Compare INTRADAY vs SWING | Mode-specific performance |
| `fetch_fundamental.py` | Scrape PE, MCap, Div Yield | Saves to `models/nifty100_fundamental.csv` |
| `table_format.py` | Terminal table output | ASCII table for terminal users |
