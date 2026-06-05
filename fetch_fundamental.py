#!/usr/bin/env python3
"""
NIFTY 100 Fundamental Data Fetcher
Fetches PE, PEG, Market Cap, Dividend, 52wk range for all NIFTY 100 stocks
Saves to CSV. Only regenerates if CSV is older than 3 months.
"""
import urllib.request
import json
import os
import csv
import time
from datetime import datetime
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

MODEL_DIR = os.path.dirname(os.path.abspath(__file__)) + "/models"
os.makedirs(MODEL_DIR, exist_ok=True)
CSV_PATH = f"{MODEL_DIR}/nifty100_fundamental.csv"

def get_nifty100_symbols():
    """Fetch NIFTY 100 symbols from NSE"""
    url = 'https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%20100'
    headers = {'User-Agent': 'Mozilla/5.0', 'Referer': 'https://www.nseindia.com/'}
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=10) as resp:
        data = json.loads(resp.read())
        # Filter out index name, get only symbols
        return [s['symbol'] for s in data['data'] if s['symbol'] != 'NIFTY 100']

def fetch_fundamental(sym):
    """Fetch fundamental data for a single stock via yfinance"""
    try:
        tk = yf.Ticker(f"{sym}.NS")
        info = tk.info
        
        pe = info.get('forwardPE', None)
        peg = info.get('pegRatio', None)
        mcap = info.get('marketCap', None)
        div = info.get('dividendYield', 0)
        hi52 = info.get('fiftyTwoWeekHigh', None)
        lo52 = info.get('fiftyTwoWeekLow', None)
        beta = info.get('beta', None)
        price = info.get('currentPrice', info.get('regularMarketPrice', None))
        
        # Convert mcap to string representation
        if mcap:
            if mcap >= 1e12:
                mcap_str = f"{mcap/1e12:.1f}T"
            elif mcap >= 1e10:
                mcap_str = f"{mcap/1e11:.1f}B"
            else:
                mcap_str = f"{mcap/1e9:.1f}B"
        else:
            mcap_str = "N/A"
        
        # Dividend yield as percentage
        if div and div > 0:
            div_str = f"{div*100:.2f}%"
        else:
            div_str = "N/A"
        
        # Distance from 52w high
        if hi52 and price and hi52 > 0:
            dist_from_high = ((price - hi52) / hi52) * 100
        else:
            dist_from_high = None
        
        return {
            'symbol': sym,
            'price': price,
            'pe': round(pe, 1) if pe else None,
            'peg': round(peg, 2) if peg else None,
            'mcap': mcap_str,
            'mcap_raw': mcap,
            'dividend_yield': div_str,
            'div_raw': div * 100 if div else 0,
            'hi52': round(hi52, 1) if hi52 else None,
            'lo52': round(lo52, 1) if lo52 else None,
            'dist_from_high': round(dist_from_high, 1) if dist_from_high is not None else None,
            'beta': round(beta, 2) if beta else None,
            'fetch_time': datetime.now().strftime("%Y-%m-%d %H:%M"),
            'error': None
        }
    except Exception as e:
        return {
            'symbol': sym,
            'price': None, 'pe': None, 'peg': None, 'mcap': None, 'mcap_raw': None,
            'dividend_yield': None, 'div_raw': 0, 'hi52': None, 'lo52': None,
            'dist_from_high': None, 'beta': None,
            'fetch_time': datetime.now().strftime("%Y-%m-%d %H:%M"),
            'error': str(e)
        }

def compute_fundamental_score(f):
    """Score fundamental quality 0-100"""
    score = 0
    
    # PE scoring
    pe = f['pe']
    if pe:
        if pe < 15:
            score += 25
        elif pe < 20:
            score += 15
        elif pe < 25:
            score += 8
        elif pe < 30:
            score += 3
        # PE > 30 = 0 pts
    
    # PEG scoring
    peg = f['peg']
    if peg:
        if peg < 0.8:
            score += 25
        elif peg < 1.0:
            score += 20
        elif peg < 1.3:
            score += 10
        elif peg < 1.5:
            score += 5
    
    # Market cap scoring
    mcap_raw = f.get('mcap_raw')
    if mcap_raw:
        if mcap_raw >= 1e12:
            score += 20
        elif mcap_raw >= 5e11:
            score += 15
        elif mcap_raw >= 1e11:
            score += 10
        elif mcap_raw >= 5e10:
            score += 5
    
    # Dividend scoring
    div = f.get('div_raw', 0)
    if div > 0:
        if div >= 3:
            score += 15
        elif div >= 1.5:
            score += 8
        elif div > 0:
            score += 3
    
    # 52w high proximity scoring
    dist = f.get('dist_from_high')
    if dist is not None:
        if dist >= -5:  # At or near 52w high
            score += 15
        elif dist >= -15:
            score += 10
        elif dist >= -25:
            score += 5
    
    return score

def main():
    # Check if CSV exists and is fresh
    regenerate = True
    if os.path.exists(CSV_PATH):
        mtime = datetime.fromtimestamp(os.path.getmtime(CSV_PATH))
        age_days = (datetime.now() - mtime).days
        print(f"CSV exists, age: {age_days} days")
        if age_days < 90:
            regenerate = False
            print("CSV is fresh (< 90 days), skipping regeneration")
            return
    
    if not regenerate:
        return
    
    print("Fetching NIFTY 100 symbols...")
    symbols = get_nifty100_symbols()
    print(f"Got {len(symbols)} symbols")
    
    results = []
    for i, sym in enumerate(symbols):
        print(f"[{i+1}/{len(symbols)}] {sym}...", end=" ", flush=True)
        f = fetch_fundamental(sym)
        f['fundamental_score'] = compute_fundamental_score(f)
        results.append(f)
        if f['error']:
            print(f"ERR: {f['error']}")
        else:
            print(f"PE={f['pe']} MCap={f['mcap']} Score={f['fundamental_score']}")
        time.sleep(0.3)  # Rate limit
    
    # Write CSV
    fields = ['symbol', 'price', 'pe', 'peg', 'mcap', 'mcap_raw', 'dividend_yield', 
              'div_raw', 'hi52', 'lo52', 'dist_from_high', 'beta', 
              'fundamental_score', 'fetch_time', 'error']
    
    with open(CSV_PATH, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fields)
        writer.writeheader()
        for r in results:
            row = {k: r.get(k, '') for k in fields}
            writer.writerow(row)
    
    print(f"\n✅ Saved {len(results)} stocks to {CSV_PATH}")
    
    # Summary
    scored = [r for r in results if r['fundamental_score'] > 0]
    scored.sort(key=lambda x: -x['fundamental_score'])
    print("\n🏆 TOP 10 by Fundamental Score:")
    for r in scored[:10]:
        print(f"  {r['symbol']}: Score={r['fundamental_score']} | PE={r['pe']} | MCap={r['mcap']} | Div={r['dividend_yield']}")

if __name__ == "__main__":
    main()