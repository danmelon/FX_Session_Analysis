"""
Session Range — Statistical Significance Analysis
===================================================
Runs chi-squared tests on conditional break probabilities across all pairs.

Run with:  python session_stats_analysis.py

Requirements:
    pip install yfinance pandas numpy scipy tabulate
"""

import pandas as pd
import numpy as np
import yfinance as yf
from scipy.stats import chi2_contingency, fisher_exact
from tabulate import tabulate
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG — match your Streamlit app
# ─────────────────────────────────────────────────────────────────────────────

PAIRS = {
    "EUR/USD": "EURUSD=X",
    "GBP/USD": "GBPUSD=X",
    "USD/JPY": "USDJPY=X",
    "AUD/USD": "AUDUSD=X",
    "USD/CAD": "USDCAD=X",
    "NZD/USD": "NZDUSD=X",
    "USD/CHF": "USDCHF=X",
    "EUR/GBP": "EURGBP=X",
    "EUR/JPY": "EURJPY=X",
    "GBP/JPY": "GBPJPY=X",
}

SESSION_UTC = {
    "Asia":   (0,  9),
    "London": (8, 17),
    "NY":     (13, 22),
}

LOOKBACK_DAYS = 365  # use maximum data for better statistical power
MIN_CELL_COUNT = 5   # chi-squared assumption: all expected cells >= 5

# ─────────────────────────────────────────────────────────────────────────────
# DATA — reuse same logic as Streamlit app
# ─────────────────────────────────────────────────────────────────────────────

def fetch_data(ticker: str, days: int) -> pd.DataFrame:
    df = yf.download(ticker, period="730d", interval="1h", progress=False, auto_adjust=True)
    if df.empty:
        return pd.DataFrame()
    df.index = pd.to_datetime(df.index, utc=True)
    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=days)
    return df[df.index >= cutoff]


def session_ohlc(hourly: pd.DataFrame, session: str) -> pd.DataFrame:
    start_h, end_h = SESSION_UTC[session]
    mask = (hourly.index.hour >= start_h) & (hourly.index.hour < end_h)
    sess = hourly[mask].copy()
    if sess.empty:
        return pd.DataFrame()
    if isinstance(sess.columns, pd.MultiIndex):
        sess.columns = sess.columns.get_level_values(0)
    sess["date"] = sess.index.date
    daily = sess.groupby("date").agg(
        open=("Open", "first"),
        high=("High", "max"),
        low=("Low", "min"),
        close=("Close", "last"),
    )
    daily["range"]     = daily["high"] - daily["low"]
    daily["direction"] = np.where(daily["close"] >= daily["open"], "Bullish", "Bearish")
    return daily


def build_session_df(hourly: pd.DataFrame) -> pd.DataFrame:
    asia   = session_ohlc(hourly, "Asia").add_prefix("asia_")
    london = session_ohlc(hourly, "London").add_prefix("london_")
    ny     = session_ohlc(hourly, "NY").add_prefix("ny_")
    df = asia.join(london, how="inner").join(ny, how="inner").dropna()
    df.index = pd.to_datetime(df.index)
    return df

# ─────────────────────────────────────────────────────────────────────────────
# STATISTICAL TESTS
# ─────────────────────────────────────────────────────────────────────────────

def run_chi_squared(a: int, b: int, c: int, d: int):
    """
    2x2 contingency table:
        
                  Breaks    Does Not Break
    Small range [  a,           b        ]
    Large range [  c,           d        ]

    Returns p-value and the test used.
    Uses Fisher's exact test if any expected cell < MIN_CELL_COUNT.
    """
    table = [[a, b], [c, d]]
    expected = chi2_contingency(table)[3]

    if np.any(expected < MIN_CELL_COUNT):
        # Fisher's exact is more reliable with small samples
        _, p = fisher_exact(table)
        test_used = "Fisher"
    else:
        _, p, _, _ = chi2_contingency(table)
        test_used = "Chi-sq"

    return p, test_used


def analyse_pair(pair_name: str, ticker: str) -> dict | None:
    """Run all conditional probability tests for one pair."""
    print(f"  Fetching {pair_name} ({ticker})...", end=" ", flush=True)

    hourly = fetch_data(ticker, LOOKBACK_DAYS)
    if hourly.empty:
        print("NO DATA")
        return None

    df = build_session_df(hourly)
    n  = len(df)

    if n < 30:
        print(f"SKIPPED (only {n} days)")
        return None

    print(f"{n} days")

    asia_median   = df["asia_range"].median()
    london_median = df["london_range"].median()

    # ── Test 1: Does a small Asia range increase London's break probability? ──
    small_asia = df["asia_range"] < asia_median
    large_asia = ~small_asia

    london_breaks = (df["london_high"] > df["asia_high"]) | (df["london_low"] < df["asia_low"])

    a1 = int(( small_asia &  london_breaks).sum())   # small asia, london breaks
    b1 = int(( small_asia & ~london_breaks).sum())   # small asia, london holds
    c1 = int(( large_asia &  london_breaks).sum())   # large asia, london breaks
    d1 = int(( large_asia & ~london_breaks).sum())   # large asia, london holds

    p1, test1 = run_chi_squared(a1, b1, c1, d1)

    break_rate_small_asia = a1 / (a1 + b1) * 100 if (a1 + b1) > 0 else 0
    break_rate_large_asia = c1 / (c1 + d1) * 100 if (c1 + d1) > 0 else 0
    overall_london_break  = london_breaks.mean() * 100

    # ── Test 2: Does a small London range increase NY's break probability? ────
    small_london = df["london_range"] < london_median
    large_london = ~small_london

    ny_breaks = (df["ny_high"] > df["london_high"]) | (df["ny_low"] < df["london_low"])

    a2 = int(( small_london &  ny_breaks).sum())
    b2 = int(( small_london & ~ny_breaks).sum())
    c2 = int(( large_london &  ny_breaks).sum())
    d2 = int(( large_london & ~ny_breaks).sum())

    p2, test2 = run_chi_squared(a2, b2, c2, d2)

    break_rate_small_london = a2 / (a2 + b2) * 100 if (a2 + b2) > 0 else 0
    break_rate_large_london = c2 / (c2 + d2) * 100 if (c2 + d2) > 0 else 0
    overall_ny_break        = ny_breaks.mean() * 100

    # ── Test 3: Is NY reversal rate different from 50/50 (binomial baseline)? ─
    london_bull    = df["london_direction"] == "Bullish"
    ny_bull        = df["ny_direction"]     == "Bullish"
    ny_reverses    = (london_bull != ny_bull)
    reversal_rate  = ny_reverses.mean() * 100

    # Frame as chi-squared vs expected 50/50
    rev_count  = int(ny_reverses.sum())
    cont_count = n - rev_count
    p3, test3  = run_chi_squared(rev_count, cont_count, n // 2, n - n // 2)

    return {
        "Pair":                   pair_name,
        "Days":                   n,
        # Test 1
        "London Break (overall)": f"{overall_london_break:.1f}%",
        "London Break (small Asia)": f"{break_rate_small_asia:.1f}%",
        "London Break (large Asia)": f"{break_rate_large_asia:.1f}%",
        "T1 p-value":             p1,
        "T1 test":                test1,
        "T1 significant":         "✓" if p1 < 0.05 else "✗",
        # Test 2
        "NY Break (overall)":     f"{overall_ny_break:.1f}%",
        "NY Break (small London)": f"{break_rate_small_london:.1f}%",
        "NY Break (large London)": f"{break_rate_large_london:.1f}%",
        "T2 p-value":             p2,
        "T2 test":                test2,
        "T2 significant":         "✓" if p2 < 0.05 else "✗",
        # Test 3
        "NY Reversal Rate":       f"{reversal_rate:.1f}%",
        "T3 p-value":             p3,
        "T3 test":                test3,
        "T3 significant":         "✓" if p3 < 0.05 else "✗",
    }

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    print("\n" + "="*65)
    print("  FX Session Range — Statistical Significance Analysis")
    print(f"  Lookback: {LOOKBACK_DAYS} days | Significance threshold: p < 0.05")
    print("="*65 + "\n")

    results = []
    for pair_name, ticker in PAIRS.items():
        result = analyse_pair(pair_name, ticker)
        if result:
            results.append(result)

    if not results:
        print("No data retrieved. Check your internet connection.")
        exit()

    df_results = pd.DataFrame(results)

    # ── Print Test 1: Small Asia → London breaks ──────────────────────────────
    print("\n" + "─"*65)
    print("TEST 1: Does a small Asia range increase London breakout probability?")
    print("─"*65)
    t1_cols = ["Pair", "Days", "London Break (overall)",
               "London Break (small Asia)", "London Break (large Asia)",
               "T1 p-value", "T1 test", "T1 significant"]
    t1 = df_results[t1_cols].copy()
    t1["T1 p-value"] = t1["T1 p-value"].apply(lambda x: f"{x:.4f}")
    print(tabulate(t1, headers="keys", tablefmt="rounded_outline", showindex=False))

    sig_t1 = (df_results["T1 significant"] == "✓").sum()
    print(f"\n  → Significant in {sig_t1}/{len(df_results)} pairs")

    # ── Print Test 2: Small London → NY breaks ────────────────────────────────
    print("\n" + "─"*65)
    print("TEST 2: Does a small London range increase NY breakout probability?")
    print("─"*65)
    t2_cols = ["Pair", "Days", "NY Break (overall)",
               "NY Break (small London)", "NY Break (large London)",
               "T2 p-value", "T2 test", "T2 significant"]
    t2 = df_results[t2_cols].copy()
    t2["T2 p-value"] = t2["T2 p-value"].apply(lambda x: f"{x:.4f}")
    print(tabulate(t2, headers="keys", tablefmt="rounded_outline", showindex=False))

    sig_t2 = (df_results["T2 significant"] == "✓").sum()
    print(f"\n  → Significant in {sig_t2}/{len(df_results)} pairs")

    # ── Print Test 3: NY reversal vs 50/50 baseline ───────────────────────────
    print("\n" + "─"*65)
    print("TEST 3: Is NY's reversal rate of London statistically different from 50/50?")
    print("─"*65)
    t3_cols = ["Pair", "Days", "NY Reversal Rate",
               "T3 p-value", "T3 test", "T3 significant"]
    t3 = df_results[t3_cols].copy()
    t3["T3 p-value"] = t3["T3 p-value"].apply(lambda x: f"{x:.4f}")
    print(tabulate(t3, headers="keys", tablefmt="rounded_outline", showindex=False))

    sig_t3 = (df_results["T3 significant"] == "✓").sum()
    print(f"\n  → Significant in {sig_t3}/{len(df_results)} pairs")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "="*65)
    print("  SUMMARY")
    print("="*65)
    print(f"  Pairs analysed        : {len(df_results)}")
    print(f"  T1 significant (p<.05): {sig_t1}/{len(df_results)}  — small Asia → London breaks more")
    print(f"  T2 significant (p<.05): {sig_t2}/{len(df_results)}  — small London → NY breaks more")
    print(f"  T3 significant (p<.05): {sig_t3}/{len(df_results)}  — NY reversal ≠ 50/50")
    print("="*65 + "\n")

    # ── Save to CSV ───────────────────────────────────────────────────────────
    out_path = "session_significance_results.csv"
    df_results.to_csv(out_path, index=False)
    print(f"  Full results saved to: {out_path}\n")