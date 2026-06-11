#!/usr/bin/env python3
"""NIFTY Live Quant Ultra - Model Trainer v9
Now uses nifty_core.py for all shared logic.
v9: --cleanup removes orphaned models; --coverage prints model coverage report
v8: sklearn version stored in model metadata; single-stock train via CLI
"""
import sys
import os
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import yfinance as yf
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
import sklearn
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.model_selection import train_test_split

from nifty_core import DEFAULT_STOCKS, GOOD_STOCKS, SCANNABLE_STOCKS, NIFTY100_EXTRA, EXCLUDED_STOCKS, MODEL_DIR, get_ohlc, add_features, build_ml_features

os.makedirs(MODEL_DIR, exist_ok=True)

def build_features(df):
    """Build 34-feature arrays (training only). Uses index-based lookback."""
    closes = df['Close'].values
    rows = []
    for i in range(200, len(closes) - 5):
        last = df.iloc[i]
        c = last['Close']
        ma = [last.get(f'ma{w}', c) for w in [5, 10, 20, 50, 100, 200]]
        tech = [last.get('rsi', 50), last.get('macd', 0), last.get('macd_sig', 0),
                last.get('atr', 1), last.get('vol_ratio', 1), last.get('ret5', 0)]
        ratio = [c/(last.get('ma20', c)+1e-10), c/(last.get('ma50', c)+1e-10),
                 last.get('ma20', c)/(last.get('ma50', c)+1e-10),
                 last.get('ma50', c)/(last.get('ma200', c)+1e-10),
                 last.get('macd', 0)-last.get('macd_sig', 0),
                 last.get('rsi', 50)*last.get('vol_ratio', 1)/100]
        av = last.get('atr', 1) + 1
        atr = [(c-last.get('ma20', c))/av, (c-last.get('ma50', c))/av,
               last.get('macd', 0)/(c+1e-10), last.get('vol_ratio', 1)-1]
        rets = []
        for d in [5, 10, 20]:
            idx = max(0, i-d)
            rets.append((c - closes[idx]) / (closes[idx] + 1e-10))
        rets += [last.get('rsi', 50)/100, last.get('ret5', 0)*10,
                 df['Close'].iloc[i-5:i].mean()/(df['Close'].iloc[max(0,i-20):i].mean()+1e-10)]
        vol = [df['Close'].iloc[max(0,i-20):i].std()/(df['Close'].iloc[max(0,i-20):i].mean()+1e-10),
               df['Volume'].iloc[max(0,i-20):i].mean()/(df['Volume'].iloc[max(0,i-50):i].mean()+1e-10),
               df['Close'].iloc[max(0,i-3):i].mean()/(df['Close'].iloc[max(0,i-10):i].mean()+1e-10),
               df['Close'].iloc[max(0,i-5):i].mean()/(df['Close'].iloc[max(0,i-30):i].mean()+1e-10),
               last.get('vol_ratio', 1)*(c/(last.get('ma20', c)+1e-10)),
               last.get('macd', 0)/(last.get('atr', 1)+1e-10)]
        feat = ma + tech + ratio + atr + rets + vol
        feat = [0 if (np.isnan(x) or np.isinf(x)) else x for x in feat]
        rows.append(feat)
    return np.array(rows)

def verify_feature_consistency():
    """Verify train.py build_features produces same feature count as nifty_core.build_ml_features."""
    df = get_ohlc('SBIN', days=730)
    if df is None: return
    df = add_features(df).dropna()
    if len(df) < 210: print(f'  SBIN only {len(df)} rows after dropna, skipping'); return
    train_feat = build_features(df)
    core_feat = build_ml_features(df, 200)
    print(f'  train.py: {train_feat.shape[1]} features | nifty_core: {core_feat.shape[1]} features')
    assert train_feat.shape[1] == core_feat.shape[1], f'Mismatch: {train_feat.shape[1]} vs {core_feat.shape[1]}'

def build_labels(df):
    closes = df['Close'].values
    labels = []
    for i in range(200, len(closes) - 5):
        ret = (closes[i+5] - closes[i]) / closes[i]
        labels.append(1 if ret > 0.02 else (-1 if ret < -0.02 else 0))
    return np.array(labels)

def train_stock(sym):
    p = MODEL_DIR + '/' + sym + '_model.joblib'
    now = datetime.now()
    is_old = os.path.exists(p) and (now - datetime.fromtimestamp(os.path.getmtime(p))).days > 5

    print(f"{now} [{'OLD' if is_old else 'NEW'}] {sym}...", end=' ', flush=True)

    df = get_ohlc(sym, days=730)
    if df is None or len(df) < 250:
        print('FAIL data'); return False, sym

    df = add_features(df)
    if len(df) < 250:
        print('FAIL add_features'); return False, sym
    df = df.dropna()
    if len(df) < 250:
        print('FAIL dropna'); return False, sym

    X = build_features(df)
    y = build_labels(df)
    if len(X) < 100:
        print('FAIL features'); return False, sym

    y_bin = (y == 1).astype(int)
    if y_bin.sum() < 10 or (y_bin == 0).sum() < 10:
        print('FAIL imbalance'); return False, sym

    try:
        Xtr, Xte, ytr, yte = train_test_split(X, y_bin, test_size=0.2, random_state=42, stratify=y_bin)
        gb = GradientBoostingClassifier(learning_rate=0.05, max_depth=6, n_estimators=150, subsample=0.8, random_state=42)
        rf = RandomForestClassifier(max_depth=8, n_estimators=100, random_state=42)
        model = VotingClassifier(estimators=[('gb', gb), ('rf', rf)], voting='soft')
        model.fit(Xtr, ytr)
        tr_acc = model.score(Xtr, ytr)
        te_acc = model.score(Xte, yte)
        # Store sklearn version in model so inference can verify compatibility
        model.sklearn_version = sklearn.__version__
        joblib.dump(model, p)
        age = (now - datetime.fromtimestamp(os.path.getmtime(p))).days
        print(f"OK train={round(tr_acc*100,1)}% test={round(te_acc*100,1)}% age={age}d sklearn={sklearn.__version__}")
        return True, sym
    except Exception as e:
        print(f"FAIL {e}"); return False, sym

def cleanup_orphaned_models():
    """Remove model files for EXCLUDED_STOCKS and stocks with no data in our universe."""
    from nifty_core import EXCLUDED_STOCKS, GOOD_STOCKS, SCANNABLE_STOCKS
    valid_stocks = set(GOOD_STOCKS) | set(SCANNABLE_STOCKS)
    removed = []
    for f in os.listdir(MODEL_DIR):
        if not f.endswith('_model.joblib'):
            continue
        sym = f.replace('_model.joblib', '')
        if sym in EXCLUDED_STOCKS:
            os.remove(os.path.join(MODEL_DIR, f))
            removed.append(f"EXCLUDED:{sym}")
        elif sym not in valid_stocks:
            os.remove(os.path.join(MODEL_DIR, f))
            removed.append(f"ORPHANED:{sym}")
    if removed:
        print(f"🗑️  Cleaned {len(removed)} orphaned models: {removed}")
    else:
        print("✅ No orphaned models found")
    return removed


def print_coverage():
    """Print ML model coverage report for current universe."""
    from nifty_core import GOOD_STOCKS, SCANNABLE_STOCKS
    models = {f.replace('_model.joblib', '')
              for f in os.listdir(MODEL_DIR) if f.endswith('_model.joblib')}
    n50 = set(GOOD_STOCKS)
    n100 = set(SCANNABLE_STOCKS)
    n50_covered = models & n50
    n100_covered = (models & n100) | n50_covered
    missing_n50 = n50 - models
    missing_n100 = n100 - models
    print(f"📊 ML MODEL COVERAGE REPORT")
    print(f"  Models trained: {len(models)}")
    print(f"  NIFTY50: {len(n50_covered)}/{len(n50)} covered | {len(missing_n50)} missing: {sorted(missing_n50)[:10]}{'...' if len(missing_n50)>10 else ''}")
    print(f"  NIFTY100: {len(n100_covered)}/{len(n100)} covered | {len(missing_n100)} missing")
    print(f"  Run: python3 train.py --index nifty100  (background: python3 train.py --index nifty100 &)")
    if models & EXCLUDED_STOCKS:
        excl = models & EXCLUDED_STOCKS
        print(f"  ⚠️  Excluded stocks with models: {sorted(excl)} (run: python3 train.py --cleanup)")
    return len(n50_covered), len(n50), len(n100_covered), len(n100)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='NIFTY ML Model Trainer')
    parser.add_argument('--cleanup', action='store_true',
                        help='Remove orphaned models (EXCLUDED stocks or invalid symbols)')
    parser.add_argument('--coverage', action='store_true',
                        help='Print ML model coverage report without training')
    parser.add_argument('--index', choices=['nifty50', 'nifty100'], default=None,
                        help='Train all stocks in index (default: nifty50 = GOOD_STOCKS)')
    parser.add_argument('stocks', nargs='*', help='Specific stock symbols (optional)')
    args = parser.parse_args(sys.argv[1:])

    if args.cleanup:
        cleanup_orphaned_models()
        return

    if args.coverage:
        print_coverage()
        return

    # Determine stock universe
    if args.index == 'nifty100':
        stocks_to_train = SCANNABLE_STOCKS
        print(f'NIFTY MODEL TRAINER v9 - NIFTY100 universe: {len(stocks_to_train)} stocks')
    else:
        stocks_to_train = GOOD_STOCKS
        print(f'NIFTY MODEL TRAINER v9 - NIFTY50 universe: {len(stocks_to_train)} stocks')

    # Add specific stocks on top (deduplicated)
    extra = [s.strip().upper() for s in args.stocks]
    stocks_to_train = stocks_to_train + [s for s in extra if s not in stocks_to_train]

    if extra:
        print(f'Plus specific: {extra} → total {len(stocks_to_train)} stocks')

    print('=' * 60)
    print_coverage()
    print()
    fresh, retrained, failed = [], [], []
    now = datetime.now()

    for sym in stocks_to_train:
        ok, s = train_stock(sym)
        if ok:
            p = MODEL_DIR + '/' + s + '_model.joblib'
            is_old = os.path.exists(p) and (now - datetime.fromtimestamp(os.path.getmtime(p))).days > 5
            (retrained if is_old else fresh).append(s)
        else:
            failed.append(s)

    print()
    print('=' * 60)
    print('SUMMARY')
    print('=' * 60)
    print(f"Fresh:     {len(fresh)} -> {fresh}")
    print(f"Retrained: {len(retrained)} -> {retrained}")
    print(f"Failed:    {len(failed)} -> {failed}")
    print(f"Total:     {len(fresh)+len(retrained)}/{len(stocks_to_train)} OK")
    print()
    print_coverage()

if __name__ == '__main__':
    main()