# NIFTY Live Quant Ultra — SKILL.md v27

> Multi-mode quant trading system. Paper trading only. No live money.
> Two modes: **SWING** (daily candles, overnight) + **INTRADAY** (5-min candles, same-day exit)

## Rating History
| Version | Score | Key Change |
|---------|-------|-----------|
| v24 | 5.9/10 | Initial |
| v25 | 7.0/10 | SIG_REVERSAL disabled, T1=10%, ABSSL 8%/5% |
| v26 | 7.9/10 | Intraday system built, dual signals |
| **v27** | **9.5/10** | **Tight RSI<25/>75, TSL, min_conf=2, swing+intraday both strong** |

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
RSI_CONFIG = {'buy_strict': 38, 'sell_strict': 50, 'sell_relaxed': 36}
SIGNAL_CONFIG = {'min_confirmations': 2}
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

**Levels:** SL=1×ATR | T1=2×ATR | T2=4×ATR (2:1 RR)
**Trailing SL:** After T1 hit, SL locks at entry price and trails by 1.5×ATR
**Square-off:** 3:15 PM IST sharp — ALL positions closed same day

**Config (intraday_core.py):**
```python
RSI_BUY_THRESHOLD = 25    # was 30
RSI_SELL_THRESHOLD = 75   # was 70
SL_MULT = 1.0; T1_MULT = 2.0; T2_MULT = 4.0
```

---

## 🚀 Usage

```bash
# SWING — live scan
python3 scan.py --index nifty50 --ai --format telegram

# SWING — backtest
python3 backtest.py

# INTRADAY — live scan
python3 intraday_core.py                    # all stocks
python3 intraday_core.py SBIN,TCS --debug   # specific stocks

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

---

## ⚠️ Disclaimer
Paper trading only. No live money. Backtested results do not guarantee future performance.