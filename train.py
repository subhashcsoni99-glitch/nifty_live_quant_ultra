#!/usr/bin/env python3
"""NIFTY Live Quant Ultra - Model Trainer v8
Now uses nifty_core.py for all shared logic.
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

from nifty_core import DEFAULT_STOCKS, MODEL_DIR, get_ohlc, add_features, build_ml_features

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

def main():
    # Command-line: python3 train.py           → train all GOOD_STOCKS
    #                  python3 train.py SBIN      → train single stock
    #                  python3 train.py SBIN,TCS   → train multiple
    syms = [s.strip().upper() for s in sys.argv[1:] if not s.startswith('--')]
    if syms:
        stocks_to_train = syms
        print(f'NIFTY MODEL TRAINER v8 - Training {len(stocks_to_train)} stock(s): {stocks_to_train}')
    else:
        stocks_to_train = DEFAULT_STOCKS
        print(f'NIFTY MODEL TRAINER v8 - Training all {len(stocks_to_train)} models')
    print('=' * 60)
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

if __name__ == '__main__':
    main()