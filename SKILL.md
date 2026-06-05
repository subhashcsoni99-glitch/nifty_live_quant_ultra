# NIFTY Live Quant Ultra - Skill (v18)

Multi-script trading system for NIFTY stocks: scanner, backtester, trainer, analyzer.

**Architecture:** Single source of truth via `nifty_core.py` v3 — all scripts import shared logic from there.

## v18 Changes (9.5/10 review suggestions applied)
1. **NEG_HIST threshold tightened**: only `rr < -2% AND wr < 40%` triggers ⚠️ NEG_HIST (was `rr < 0 AND wr <= 50%`). Small losses are normal — only truly bad history is flagged.
2. **BEAR_DIV counter-trend labeling**: BEAR_DIV in BULL regime → 🔴 CONTRARIAN tag (stocks remain in C2, honestly labeled, not suppressed)
3. **Tiered stale aging**: <4d=FRESH, 4-7d=⏰ STALE, 8+d=💀 STALE_CRITICAL. Confluence score penalty scales: 0% fresh → 10% stale → 30% critical.
4. **Regime-signal coherence**: Confluence score now includes regime fit (15% weight) — BUY in BULL = +bonus, BUY+BEAR_DIV in BULL = contrarian penalty.
5. **Cat B criteria relaxed**: `wr >= 35%, rr >= -2%` (was `wr >= 40%, rr >= 0`). Reflects that small losses early in position are normal; WR matters more than tiny RR.

## v19 Changes (8.5→9.5/10 review suggestions applied)
1. **⭐ TOP_PICK**: Cat A/B stocks with positive RR + CF>=8 → ⭐ tag. Cat A/B blocks sorted by RR desc — positive RR bubbles to top.
2. **Cat A-SHORT HEDGE**: AI_BEARISH + SELL in BULL regime → Cat A- + 🛡️ HEDGE tag (HDFCBANK moves from Cat C1 to Cat A-)
3. **Cat C1 renamed** "SIGNAL + AI AGREE (SHORT)" — clear that shorts belong here
4. **TOP LONG** sorted by (_starred, CF desc) — ⭐ TOP_PICKs appear first with ⭐ badge
5. **TOP SHORT** shows 🛡️ HEDGE count, each hedge short labeled 🛡️

## Scripts

### scan.py (v18) — Live Scanner
```bash
# Core usage
python3 scan.py --symbols SBIN,TCS,HDFCBANK
python3 scan.py --index nifty50 --ai

# Flags
--ai            Full 9-stage AI analysis + ML predictions (category output)
--trailing      Trail SL on positions
--sector-cap    Max 2 per sector
--fundamentals  Filter by PE/PEG/MCap/Dividend (from models/nifty100_fundamental.csv)
--filter-neg-hist  Hide stocks with negative backtest history (rr < 0)
--backtest-first  Run 3-year backtest on all stocks before scan (refreshes stats)
--auto-retrain    Retrain ML models automatically on sklearn version mismatch
--format telegram   Formatted for Telegram (market regime, Cat A/A-/B/C1/C2/D/WATCHLIST)
--format json      JSON output for programmatic use
--index nifty50    NIFTY 50 stocks (actual constituents)
--index nifty100   NIFTY 100 stocks (NIFTY50 + 30 extras, total ~80 stocks)
--symbols SBIN,TCS,HDFCBANK
```

**Level modes (T1/T2 reachable in intraday):**
- `--tight` — Hourly scalp levels: SL=1.5×ATR, T1=0.75×ATR (~0.5-0.7%), T2=1.5×ATR (~1-1.5%). In BEARISH regime, 💠 TIGHT levels shown first.
- `--intraday` (default) — Normal intraday: SL=3×ATR, T1=2×ATR (~3-6%), T2=3.5×ATR (~5-9%)
- `--swing` — Multi-day: SL=2×ATR, T1=3×ATR, T2=6×ATR

**Output categories (v18 — quality hierarchy):**
- **Cat A** (✅ highest): Signal + AI(BULLISH) + ML(UP) + profitable (rr ≥ -2%, WR ≥ 25%) + level-aligned
- **Cat A-** (⚠️): Same as Cat A but with AI/Signal T1 level mismatch (>3%) → "2-of-3 confirmed" + UNCONFIRMED tag
- **Cat B** (🤖): AI HIGH/MEDIUM conviction, no ML UP, WR ≥ 40%, rr ≥ 0. NEG_HIST disqualified unless WR > 50%. WR < 40% → demoted to Cat C2
- **Cat C1**: Signal + AI agree (outlook matches signal direction), not Cat A/A-
- **Cat C2**: Signal only, AI neutral, or negated by quality gates
- **Cat D**: ML signal only (ml_up for BUY / ml_down for SELL), AI neutral
- **WATCHLIST**: RANGE-bound stocks (not counted in quality metrics)

**Quality gates applied:**
- **Level alignment**: |AI_T1 − Signal_T1| / Signal_T1 > 3% → UNCONFIRMED tag + Cat A- (was: no check)
- **NEG_HIST hard rule**: Cat B disqualification unless WR > 50% (DRREDDY, ONGC, INFY, TECHM now correctly filtered)
- **WR threshold**: Cat B minimum WR raised from 28% → 40% (ITC 21%, DRREDDY 30% now correctly excluded)
- **Stale signal**: >3 days without T1 touch → ⏰ STALE tag + confluence penalty
- **Confluence score (1-10)**: (Signal_Conf×0.25 + AI_Conf×0.30 + WR×0.25 + Level_Align×0.10 + Age×0.10) × 10

**Telegram format:** date / Regime / Cat counts / Cat blocks with RR/WR/Confluence / Top LONG+SHORT picks

**v18 changes (all subhash suggestions applied):**
1. Level alignment check: AI_T1 vs Signal_T1 >3% → UNCONFIRMED + Cat A-
2. NEG_HIST hard disqualifier for Cat B (WR ≤ 50%)
3. Cat B WR minimum raised: 28% → 40%
4. Cat A- tier added: 2-of-3 conditions met + level mismatch → not quite Cat A but better than Cat B
5. Stale signal flag: >3 days → ⏰ STALE
6. WATCHLIST bucket: RANGE-bound stocks moved out of quality metrics
7. Confluence score: 1-10 shown per stock in all output formats
8. JSON enriched: ai_t1, confluence_score, level_align, level_gap_pct, signal_age_days
- **All v17 changes retained**

### backtest.py (v10) — Historical Backtest
```bash
python3 backtest.py --all                      # all stocks, last 3 years (default)
python3 backtest.py --stock SBIN --years 3    # single stock
python3 backtest.py --all --years 3 --no-sig-exit  # SL/TSL/ABSSL only (no SELL-signal exits)
python3 backtest.py --all --years 3 --trailing --sector-cap
```
Results saved to `models/backtest_v9_*.json`.

**v10 changes:**
- **BUG FIX**: Fixed dead-code Fixed SL block (was unreachable — missing `if price <= sl:` condition due to indentation error). SL now fires correctly.
- All v9 changes retained (realized_return as primary metric, pnl_list for Sharpe, etc.)

**Key output fields:**
- `realized_return`: primary metric. P&L earned / capital deployed (realistic %).
- `return`: compounded total return (inflated, shown for reference).
- `sharpe`: annualized Sharpe ratio from per-trade P&L list.
- `max_drawdown`: peak-to-trough from realized capital.
- `qualified`: trades ≥ 20 AND win_rate ≥ 35% AND max_drawdown < 5%.

### forward_test.py (v8) — Paper Trading
```bash
python3 forward_test.py 30 --no-ml        # 30 days, rule-based only
python3 forward_test.py 30                 # 30 days, rule-based + ML validation
python3 forward_test.py 30 --no-sig-exit   # no SELL signal exits
```
Uses trained ML models (`models/*_model.joblib`) to validate signals.

**v8 changes:** Same realized return, pnl_list, and no-sig-exit changes as backtest v9.

### train.py (v8) — ML Trainer
```bash
python3 train.py              # trains all GOOD_STOCKS
python3 train.py SBIN         # train single stock
python3 train.py SBIN,TCS     # train multiple stocks
```
Trains GradientBoosting + RandomForest ensemble (34 features). Models saved as `models/*_model.joblib`.

**v8 changes:**
- Stores `sklearn_version` in model metadata on train — allows scan.py to detect version mismatches at load time
- Single-stock and multi-stock CLI support: `python3 train.py SBIN` or `python3 train.py SBIN,TCS`

**Staleness:** Retrains only if model file is > 5 days old (unless explicitly called for single stock).

**NB:** ML labels = 5-day return >+2% (UP) / <-2% (DOWN). This is a **momentum** predictor, not mean-reversion. Use ML as a momentum filter on BUY signals.

### analyze.py — Deep 9-stage AI Analysis (standalone)
Used for detailed per-stock analysis with full 9-stage pipeline.

### fetch_fundamental.py — Fundamental Data Fetcher
```bash
python3 fetch_fundamental.py
```
Updates `models/nifty100_fundamental.csv`. Regenerates only if CSV > 90 days old.

## Signal Logic

**Rule-based signal:** MEAN-REVERSION setup. Fires when RSI < 38 (oversold bounce).
- **BUY**: RSI < 38 + price above MAs + MACD bullish + volume + momentum
- **SELL**: RSI > 62 + price below MAs + MACD bearish + volume + negative momentum
- **RSI guards**: Block BUY when RSI > 70. Block BUY above RSI 65 unless strong uptrend. Block SELL when RSI < 40 (deeply oversold bounce zone).

**ML model:** Momentum predictor. Predicts whether price will be >+2% higher 5 days later. Use as momentum filter: only take BUY when ML also says UP.

## Key Files

- `nifty_core.py` v3 — Single source of truth: stock lists, ATR/RSI/Signal config, price fetch, features, signal engine, 9-stage AI pipeline, S/R, position sizing, ML features
- `models/` — Trained ML models (*_model.joblib with sklearn_version metadata), fundamental data (nifty100_fundamental.csv)
- `models/backtest_v9_*.json` — Backtest results with realized_return, Sharpe, qualified flag
- `models/forward_test_v8_*.json` — Forward test results

## Constraints

- Paper trading only — no live money.
- Not SEBI registered. Validate before trading.
- NSE India API may rate-limit; yfinance as fallback.