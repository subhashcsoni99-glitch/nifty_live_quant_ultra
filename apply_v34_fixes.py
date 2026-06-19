#!/usr/bin/env python3
"""Apply ALL v34 fixes to scan.py (v25 base). Run: python3 apply_v34_fixes.py"""
import os, shutil

SRC = 'scan_v25_restore.py'
DST = 'scan.py'
BACKUP = 'scan_v34_previous.py'
shutil.copy(SRC, BACKUP)
print(f"Backed up v25 -> {BACKUP}")

with open(SRC) as f:
    c = f.read()

fixes_applied = []

def do(old, new, desc):
    global c
    if old in c:
        c = c.replace(old, new)
        fixes_applied.append(f"OK: {desc}")
    else:
        fixes_applied.append(f"MISS: {desc}")

# 1. _get_stats reads ALL files
do("            for fname in files[:3]:",
   "            for fname in all_files:  # v34: read ALL files",
   "_get_stats all_files")

# 2. no_backtest flag
do("    return _STATS_CACHE.get(symbol, {'win_rate': 0, 'realized_return': 0, 'sharpe': 0, 'max_drawdown': 999})",
   "    return _STATS_CACHE.get(symbol, {'win_rate': 0, 'realized_return': 0, 'sharpe': 0, 'max_drawdown': 999, 'no_backtest': True})",
   "_get_stats no_backtest")

# 3. _OPTION_C_ACTIVE global
do("# ─── CLI Args ──────────────────────────────────────────────────────────────",
   "# ─── Global Option Flags ─────────────────────────────────────────────────\n_OPTION_C_ACTIVE = False  # v34: True when --C passed\n\n# ─── CLI Args ──────────────────────────────────────────────────────────────",
   "_OPTION_C_ACTIVE flag")

# 4. parse_args sets _OPTION_C_ACTIVE
do("            if 'C' in opts:\n                # Option C affects intraday_core",
   "            if 'C' in opts:\n                global _OPTION_C_ACTIVE\n                _OPTION_C_ACTIVE = True\n                # Option C affects intraday_core",
   "parse_args sets _OPTION_C_ACTIVE")

# 5. analyze() add momentum_mode param
do("def analyze(sym, use_ai=False, use_trailing=False, fundamental_filter=False, level_mode='intraday', auto_retrain=False):",
   "def analyze(sym, use_ai=False, use_trailing=False, fundamental_filter=False, level_mode='intraday', auto_retrain=False, momentum_mode=False):",
   "analyze() momentum_mode param")
do("    sig_val, meta, _ = core_get_signal(df, len(df) - 1)",
   "    sig_val, meta, _ = core_get_signal(df, len(df) - 1, momentum_mode=momentum_mode)",
   "core_get_signal momentum_mode")
do("        ai = ai_opinion_pipeline(sym, price, rsi, macd, macd_sig, atr, vol_ratio, ret5, df)",
   "        ai = ai_opinion_pipeline(sym, price, rsi, macd, macd_sig, atr, vol_ratio, ret5, df, momentum_mode=momentum_mode)",
   "ai_opinion_pipeline momentum_mode")
do("    levels = get_level_modes_extended(price, atr)\n    sr = calc_support_resistance(df)",
   "    levels = get_level_modes_extended(price, atr)\n    if _OPTION_C_ACTIVE:\n        levels['swing'] = {'sl': round(price - atr * 0.75, 0), 't1': round(price + atr * 1.5, 0), 't2': round(price + atr * 2.5, 0)}\n    sr = calc_support_resistance(df)",
   "Option C tighter SWING")

# 6. ADX check for Cat A
do("        if sig == 'BUY':\n            ml_up = ml_dir == 'UP'\n            ai_bull = ai_dir == 'BULLISH'\n\n            if div == 'BEARISH':",
   "        if sig == 'BUY':\n            ml_up = ml_dir == 'UP'\n            ai_bull = ai_dir == 'BULLISH'\n            s3 = (ai or {}).get('stages', {}).get('3_stock_scanner', {})\n            adx_val = s3.get('adx', 0)\n            adx_trending = s3.get('adx_trending', None)\n            adx_ok = bool(adx_trending is True or (adx_trending is None and adx_val >= 25))\n\n            if div == 'BEARISH':",
   "Cat A ADX check")
do("            elif ai_bull and ml_up:\n                # Cat A: triple confirmed + history not poor\n                if no_history:",
   "            elif ai_bull and ml_up:\n                if not adx_ok:\n                    _add_tag(r, 'LOW_ADX')\n                    _tag_all(r)\n                    cat_a_minus.append(r)\n                elif no_history:",
   "Cat A ADX gate")

# 7. Cat A- new criteria
do("        elif wr < _MIN_WR_CAT_A:  # v25: Cat A gate\n                    _add_tag(r, 'LOW_WINRATE')\n                    _tag_all(r)\n                    cat_a_minus.append(r)",
   "        elif wr >= 15 and (cf >= 6.0 or ai_bull or ml_up):\n                    _tag_all(r)\n                    if not align_ok: _add_tag(r, 'UNCONFIRMED')\n                    cat_a_minus.append(r)\n                elif wr < 15:\n                    _add_tag(r, 'LOW_WINRATE')\n                    _tag_all(r)\n                    cat_c2.append(r)",
   "Cat A- new criteria")

# 8. is_starred v34
do("        if rr > 0 and cf >= 8.0 and wr >= _MIN_WR_TOP_PICK:\n            r['_starred'] = True\n            _add_tag(r, 'TOP_PICK')",
   "    def is_starred(r):\n        rr = r.get('_stats', {}).get('realized_return', 0)\n        cf = r.get('_confluence', 0)\n        wr = r.get('_stats', {}).get('win_rate', 0)\n        conf = r.get('prob', 0)\n        sig = r.get('signal', 'RANGE')\n        rsi = r.get('rsi', 50)\n        adx_l = r.get('_adx_label', '')\n        if sig not in ('BUY', 'SELL'): return False\n        if r.get('_regime_label') == 'CONTRARIAN': return False\n        if sig == 'BUY' and rsi >= 80: return False\n        if sig == 'SELL' and rsi <= 20: return False\n        if sig == 'SELL' and 'TRENDING' not in adx_l and 'STRONG' not in adx_l: return False\n        if cf >= 7.0 and conf >= 65: return True\n        if cf >= 6.0 and conf >= 75: return True\n        if wr >= 35 and rr > 0 and cf >= 5.5: return True\n        return False\n\n    for r in cat_a + cat_a_minus + cat_b + cat_c1 + cat_c2:\n        if is_starred(r):\n            r['_starred'] = True\n            _add_tag(r, 'TOP_PICK')",
   "is_starred v34")

# 9. --strict flag
do("        elif arg == '--debug':\n            debug_mode = True; i += 1\n        elif arg.startswith('--'):",
   "        elif arg == '--debug':\n            debug_mode = True; i += 1\n        elif arg == '--strict':\n            globals()['_MIN_WR_CAT_A'] = 40\n            globals()['_MIN_WR_CAT_B'] = 35\n            i += 1\n        elif arg.startswith('--'):",
   "--strict flag")

# 10. main() passes momentum_mode
do("        r = analyze(sym, use_ai=use_ai, use_trailing=use_trailing,\n                    fundamental_filter=fundamental_filter,\n                    level_mode=level_mode, auto_retrain=auto_retrain)",
   "        r = analyze(sym, use_ai=use_ai, use_trailing=use_trailing,\n                    fundamental_filter=fundamental_filter,\n                    level_mode=level_mode, auto_retrain=auto_retrain,\n                    momentum_mode=momentum_mode)",
   "main() passes momentum_mode")

# Write
with open(DST, 'w') as f:
    f.write(c)

# Syntax check
import py_compile, tempfile
with tempfile.NamedTemporaryFile(suffix='.py', delete=False) as tmp:
    tmp.write(c.encode())
    tmp.flush()
    try:
        py_compile.compile(tmp.name, doraise=True)
        print("SYNTAX OK")
    except py_compile.PyCompileError as e:
        print(f"SYNTAX ERROR: {e}")
        shutil.copy(BACKUP, DST)
        print("Reverted to backup")
    finally:
        os.unlink(tmp.name)

print("=" * 60)
for f in fixes_applied:
    print(f)
