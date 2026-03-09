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

from statsmodels.stats.multitest import multipletests

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.calibration import calibration_curve
from sklearn.model_selection import TimeSeriesSplit

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
        chi2_stat = None  # Fisher doesn't produce a chi2 statistic

    else:
        chi2_stat, p, _, _ = chi2_contingency(table)
        test_used = "Chi-sq"

    return p, test_used, chi2_stat

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineers predictive features from session OHLC data.
    All features use only information available before NY opens.
    """
    fe = df.copy()

    # ── Range-based features ──────────────────────────────────────────────
    # Percentile of today's range within rolling 20-day window
    fe["asia_range_percentile"] = (
        fe["asia_range"]
        .rolling(20)
        .apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)
    )
    fe["london_range_percentile"] = (
        fe["london_range"]
        .rolling(20)
        .apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)
    )

    # London range relative to Asia range
    fe["london_asia_range_ratio"] = fe["london_range"] / fe["asia_range"].replace(0, np.nan)

    # Is today's Asia range larger than yesterday's
    fe["asia_range_expanding"] = (fe["asia_range"] > fe["asia_range"].shift(1)).astype(int)

    # ── Directional features ──────────────────────────────────────────────
    fe["asia_bullish"]       = (fe["asia_direction"]   == "Bullish").astype(int)
    fe["london_bullish"]     = (fe["london_direction"] == "Bullish").astype(int)
    fe["asia_london_agree"]  = (fe["asia_bullish"] == fe["london_bullish"]).astype(int)

    # ── Positional features ───────────────────────────────────────────────
    # Did London open above the midpoint of Asia's range
    asia_mid = (fe["asia_high"] + fe["asia_low"]) / 2
    fe["london_open_above_asia_mid"] = (fe["london_open"] > asia_mid).astype(int)

    # How far London close is from Asia midpoint, normalised by Asia range
    fe["asia_mid_to_london_close"] = (
        (fe["london_close"] - asia_mid) / fe["asia_range"].replace(0, np.nan)
    )

    # ── Time-based features ───────────────────────────────────────────────
    fe["day_of_week"] = pd.to_datetime(fe.index).dayofweek  # 0=Monday, 4=Friday

    # ── Target variable ───────────────────────────────────────────────────
    fe["ny_continues_london"] = (
        fe["london_direction"] == fe["ny_direction"]
    ).astype(int)

    # Drop rows with NaN from rolling calculations
    fe = fe.dropna()

    return fe

def run_logistic_regression(df_features: pd.DataFrame, pair_name: str) -> dict:
    """
    Trains a logistic regression classifier to predict NY continuation.
    Uses chronological 70/30 split to avoid lookahead bias.
    """

    FEATURES = [
        "asia_range_percentile",
        "london_range_percentile",
        "london_asia_range_ratio",
        "asia_range_expanding",
        "asia_bullish",
        "london_bullish",
        "asia_london_agree",
        "london_open_above_asia_mid",
        "asia_mid_to_london_close",
        "day_of_week",
    ]

    TARGET = "ny_continues_london"

    # ── Chronological split ───────────────────────────────────────────────
    split_idx = int(len(df_features) * 0.7)
    train = df_features.iloc[:split_idx]
    test  = df_features.iloc[split_idx:]

    X_train = train[FEATURES]
    y_train = train[TARGET]
    X_test  = test[FEATURES]
    y_test  = test[TARGET]

    # ── Scale features ────────────────────────────────────────────────────
    # Logistic regression is sensitive to feature scale so we normalise
    # Fit scaler on train only — never fit on test data
    scaler  = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    # ── Train model ───────────────────────────────────────────────────────
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_scaled, y_train)

    # ── Predictions ───────────────────────────────────────────────────────
    y_pred       = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

    # ── Metrics ───────────────────────────────────────────────────────────
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc  = roc_auc_score(y_test, y_pred_proba)
    report   = classification_report(y_test, y_pred, output_dict=True)

    # ── Coefficients ──────────────────────────────────────────────────────
    # Positive coefficient = pushes towards continuation
    # Negative coefficient = pushes towards reversal
    coefficients = pd.DataFrame({
        "Feature":     FEATURES,
        "Coefficient": model.coef_[0],
    }).sort_values("Coefficient", ascending=False)

    # ── Baseline accuracy (majority class) ────────────────────────────────
    # What accuracy would we get just by always predicting the most common outcome
    baseline = max(y_test.mean(), 1 - y_test.mean())

    return {
        "pair":         pair_name,
        "n_train":      len(train),
        "n_test":       len(test),
        "accuracy":     accuracy,
        "baseline":     baseline,
        "edge":         accuracy - baseline,
        "roc_auc":      roc_auc,
        "coefficients": coefficients,
        "report":       report,
        "y_test":       y_test,
        "y_pred_proba": y_pred_proba,
    }

def run_walk_forward_cv(df_features: pd.DataFrame, pair_name: str, n_splits: int = 5) -> dict:
    """
    Walk-forward expanding window cross-validation.
    Returns accuracy, ROC AUC, and edge across all folds.
    """

    FEATURES = [
        "asia_range_percentile",
        "london_range_percentile",
        "london_asia_range_ratio",
        "asia_range_expanding",
        "asia_bullish",
        "london_bullish",
        "asia_london_agree",
        "london_open_above_asia_mid",
        "asia_mid_to_london_close",
        "day_of_week",
    ]

    TARGET = "ny_continues_london"

    X = df_features[FEATURES]
    y = df_features[TARGET]

    tscv = TimeSeriesSplit(n_splits=n_splits)

    fold_results = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Scale — fit on train only
        scaler         = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled  = scaler.transform(X_test)

        # Train
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X_train_scaled, y_train)

        # Evaluate
        y_pred       = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        roc_auc  = roc_auc_score(y_test, y_pred_proba)
        baseline = max(y_test.mean(), 1 - y_test.mean())
        edge     = accuracy - baseline

        fold_results.append({
            "fold":       fold,
            "n_train":    len(train_idx),
            "n_test":     len(test_idx),
            "accuracy":   accuracy,
            "baseline":   baseline,
            "edge":       edge,
            "roc_auc":    roc_auc,
        })

    fold_df = pd.DataFrame(fold_results)

    return {
        "pair":          pair_name,
        "fold_df":       fold_df,
        "mean_accuracy": fold_df["accuracy"].mean(),
        "mean_baseline": fold_df["baseline"].mean(),
        "mean_edge":     fold_df["edge"].mean(),
        "mean_roc_auc":  fold_df["roc_auc"].mean(),
        "std_accuracy":  fold_df["accuracy"].std(),
        "std_roc_auc":   fold_df["roc_auc"].std(),
        "positive_edge_folds": (fold_df["edge"] > 0).sum(),
        "total_folds":   n_splits,
    }

def analyse_pair(pair_name: str, ticker: str) -> dict | None:
    """Run all conditional probability tests for one pair."""
    print(f"  Fetching {pair_name} ({ticker})...", end=" ", flush=True)

    hourly = fetch_data(ticker, LOOKBACK_DAYS)
    if hourly.empty:
        print("NO DATA")
        return None

    df = build_session_df(hourly)
    n  = len(df)

    df_features = engineer_features(df)
    print(f"  {pair_name} — {len(df_features)} rows after feature engineering (dropped {n - len(df_features)} for rolling window)")
    


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

    p1, test1, chi2_1 = run_chi_squared(a1, b1, c1, d1)

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

    p2, test2, chi2_2 = run_chi_squared(a2, b2, c2, d2)

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
    p3, test3, chi2_3  = run_chi_squared(rev_count, cont_count, n // 2, n - n // 2)

    # Cramer's V Calculations

    def cramers_v(chi2_stat, n):
        if chi2_stat is None:
            return None
        return np.sqrt(chi2_stat / n)
    
    v1 = cramers_v(chi2_1, n)
    v2 = cramers_v(chi2_2, n)
    v3 = cramers_v(chi2_3, n)

    def interpret_v(v):
        if v is None:
            return "N/A (Fisher)"
        elif v < 0.1:
            return "negligible"
        elif v < 0.3:
            return "Weak"
        elif v < 0.5:
            return "moderate"
        else:
            return "strong"
        
    oos = run_oos_validation(df)

    lr_results = run_logistic_regression(df_features, pair_name)
    cv_results = run_walk_forward_cv(df_features, pair_name)

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
        "T1 Cramers V":      f"{v1:.3f}" if v1 is not None else "N/A",
        "T1 Effect":         interpret_v(v1),
        "T2 Cramers V":      f"{v2:.3f}" if v2 is not None else "N/A",
        "T2 Effect":         interpret_v(v2),
        "T3 Cramers V":      f"{v3:.3f}" if v3 is not None else "N/A",
        "T3 Effect":         interpret_v(v3),

        # Train set
        "OOS Train n":                    oos["Train"]["n"],
        "OOS Train T1 break (small)":     oos["Train"]["T1 break (small asia)"],
        "OOS Train T1 break (large)":     oos["Train"]["T1 break (large asia)"],
        "OOS Train T1 p":                 f"{oos['Train']['T1 p']:.2e}",
        "OOS Train T1 V":                 f"{oos['Train']['T1 V']:.3f}" if oos['Train']['T1 V'] else "N/A",
        "OOS Train T2 break (small)":     oos["Train"]["T2 break (small london)"],
        "OOS Train T2 break (large)":     oos["Train"]["T2 break (large london)"],
        "OOS Train T2 p":                 f"{oos['Train']['T2 p']:.2e}",
        "OOS Train T2 V":                 f"{oos['Train']['T2 V']:.3f}" if oos['Train']['T2 V'] else "N/A",
        "OOS Train T3 reversal rate":     oos["Train"]["T3 reversal rate"],
        "OOS Train T3 p":                 f"{oos['Train']['T3 p']:.2e}",
        "OOS Train T3 V":                 f"{oos['Train']['T3 V']:.3f}" if oos['Train']['T3 V'] else "N/A",
        # Test set
        "OOS Test n":                     oos["Test"]["n"],
        "OOS Test T1 break (small)":      oos["Test"]["T1 break (small asia)"],
        "OOS Test T1 break (large)":      oos["Test"]["T1 break (large asia)"],
        "OOS Test T1 p":                  f"{oos['Test']['T1 p']:.2e}",
        "OOS Test T1 V":                  f"{oos['Test']['T1 V']:.3f}" if oos['Test']['T1 V'] else "N/A",
        "OOS Test T2 break (small)":      oos["Test"]["T2 break (small london)"],
        "OOS Test T2 break (large)":      oos["Test"]["T2 break (large london)"],
        "OOS Test T2 p":                  f"{oos['Test']['T2 p']:.2e}",
        "OOS Test T2 V":                  f"{oos['Test']['T2 V']:.3f}" if oos['Test']['T2 V'] else "N/A",
        "OOS Test T3 reversal rate":      oos["Test"]["T3 reversal rate"],
        "OOS Test T3 p":                  f"{oos['Test']['T3 p']:.2e}",
        "OOS Test T3 V":                  f"{oos['Test']['T3 V']:.3f}" if oos['Test']['T3 V'] else "N/A",
    }, df_features, lr_results, cv_results

def run_oos_validation(df: pd.DataFrame, train_pct: float = 0.7) -> dict:

    def cramers_v(chi2_stat, n):
        if chi2_stat is None:
            return None
        return np.sqrt(chi2_stat / n)
    
    """
    Chronological 70/30 split. Calculates conditional probabilities
    on train set, applies same thresholds to test set.
    """
    split_idx = int(len(df) * train_pct)
    train = df.iloc[:split_idx]
    test  = df.iloc[split_idx:]

    # ── Thresholds from TRAIN set only ──
    asia_median_train   = train["asia_range"].median()
    london_median_train = train["london_range"].median()

    results = {}

    for split_name, split_df in [("Train", train), ("Test", test)]:
        n = len(split_df)

        # Test 1 — small Asia → London breaks
        small_asia     = split_df["asia_range"]   < asia_median_train
        large_asia     = ~small_asia
        london_breaks  = (split_df["london_high"] > split_df["asia_high"]) | \
                         (split_df["london_low"]  < split_df["asia_low"])

        a1 = int(( small_asia &  london_breaks).sum())
        b1 = int(( small_asia & ~london_breaks).sum())
        c1 = int(( large_asia &  london_breaks).sum())
        d1 = int(( large_asia & ~london_breaks).sum())

        p1, test1, chi2_1 = run_chi_squared(a1, b1, c1, d1)
        v1 = cramers_v(chi2_1, n)

        # Test 2 — small London → NY breaks
        small_london  = split_df["london_range"] < london_median_train
        large_london  = ~small_london
        ny_breaks     = (split_df["ny_high"] > split_df["london_high"]) | \
                        (split_df["ny_low"]  < split_df["london_low"])

        a2 = int(( small_london &  ny_breaks).sum())
        b2 = int(( small_london & ~ny_breaks).sum())
        c2 = int(( large_london &  ny_breaks).sum())
        d2 = int(( large_london & ~ny_breaks).sum())

        p2, test2, chi2_2 = run_chi_squared(a2, b2, c2, d2)
        v2 = cramers_v(chi2_2, n)

        # Test 3 — NY reversal vs 50/50
        london_bull  = split_df["london_direction"] == "Bullish"
        ny_bull      = split_df["ny_direction"]     == "Bullish"
        ny_reverses  = (london_bull != ny_bull)
        rev_count    = int(ny_reverses.sum())
        cont_count   = n - rev_count

        p3, test3, chi2_3 = run_chi_squared(rev_count, cont_count, n // 2, n - n // 2)
        v3 = cramers_v(chi2_3, n)

        results[split_name] = {
            "n":                          n,
            "T1 break (small asia)":      f"{a1/(a1+b1)*100:.1f}%" if (a1+b1) > 0 else "N/A",
            "T1 break (large asia)":      f"{c1/(c1+d1)*100:.1f}%" if (c1+d1) > 0 else "N/A",
            "T1 p":                       p1,
            "T1 V":                       v1,
            "T2 break (small london)":    f"{a2/(a2+b2)*100:.1f}%" if (a2+b2) > 0 else "N/A",
            "T2 break (large london)":    f"{c2/(c2+d2)*100:.1f}%" if (c2+d2) > 0 else "N/A",
            "T2 p":                       p2,
            "T2 V":                       v2,
            "T3 reversal rate":           f"{ny_reverses.mean()*100:.1f}%",
            "T3 p":                       p3,
            "T3 V":                       v3,
        }

    return results

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    print("\n" + "="*65)
    print("  FX Session Range — Statistical Significance Analysis")
    print(f"  Lookback: {LOOKBACK_DAYS} days | Significance threshold: p < 0.05")
    print("="*65 + "\n")

    results = []
    features = {}
    lr_results_all = []
    cv_results_all = []

    for pair_name, ticker in PAIRS.items():
        result = analyse_pair(pair_name, ticker)
        if result is not None:
            stats, df_features, lr_result, cv_result = result
            results.append(stats)
            features[pair_name] = df_features
            lr_results_all.append(lr_result)
            cv_results_all.append(cv_result)


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

    # ── Multiple Testing Correction ───────────────────────────────────────────────
    print("\n" + "="*65)
    print("  MULTIPLE TESTING CORRECTION")
    print("="*65)

    # Collect all p-values in order for our Bonferroni & Bejamini-Hochberg
    p_values = []
    labels = []
    for _, row in df_results.iterrows(): # The _ just replaces the 'i' which we usually use and the ',' unpacks the tuple.
        for test, col in [("T1", "T1 p-value"), ("T2", "T2 p-value"), ("T3", "T3 p-value")]:
            p_values.append(float(row[col]))
            labels.append(f"{row['Pair']} — {test}")

    p_array = np.array(p_values)

    # Bonferroni
    bonf_reject, bonf_corrected, _, _ = multipletests(p_array, alpha=0.05, method="bonferroni")

    # Benjamini-Hochberg
    bh_reject, bh_corrected, _, _ = multipletests(p_array, alpha=0.05, method="fdr_bh")

    # Build results table
    correction_df = pd.DataFrame({
    "Test":               labels,
    "Raw p-value":        [f"{p:.2e}" for p in p_array],
    "Bonferroni p":       [f"{p:.2e}" for p in bonf_corrected],
    "Bonferroni sig":     ["✓" if r else "✗" for r in bonf_reject],
    "BH p":               [f"{p:.2e}" for p in bh_corrected],
    "BH sig":             ["✓" if r else "✗" for r in bh_reject],
    })

    print(tabulate(correction_df, headers="keys", tablefmt="rounded_outline", showindex=False))

    print(f"\n  Bonferroni — surviving tests: {bonf_reject.sum()}/30")
    print(f"  BH          — surviving tests: {bh_reject.sum()}/30")
    print(f"\n  Adjusted significance threshold (Bonferroni): p < {0.05/len(p_array):.4f}")


    # Print Cramer's V results:
    print("\n" + "─"*65)
    print("EFFECT SIZES — Cramér's V")
    print("─"*65)
    effect_cols = ["Pair", "Days",
                "T1 Cramers V", "T1 Effect",
                "T2 Cramers V", "T2 Effect",
                "T3 Cramers V", "T3 Effect"]
    print(tabulate(df_results[effect_cols], headers="keys", tablefmt="rounded_outline", showindex=False))

    # Summary
    print(f"\n  Average V — T1 (small Asia → London breaks): "
        f"{df_results['T1 Cramers V'].replace('N/A', np.nan).astype(float).mean():.3f}")
    print(f"  Average V — T2 (small London → NY breaks):   "
        f"{df_results['T2 Cramers V'].replace('N/A', np.nan).astype(float).mean():.3f}")
    print(f"  Average V — T3 (NY reversal vs 50/50):       "
        f"{df_results['T3 Cramers V'].replace('N/A', np.nan).astype(float).mean():.3f}")
    
    
print("\n" + "="*65)
print("  OUT-OF-SAMPLE VALIDATION (70/30 CHRONOLOGICAL SPLIT)")
print("="*65)

for _, row in df_results.iterrows():
    print(f"\n  {row['Pair']}")
    print(f"  {'─'*40}")

    oos_table = {
        "Metric": [
            "N (days)",
            "T1 break small Asia", "T1 break large Asia", "T1 p-value", "T1 Cramér's V",
            "T2 break small London", "T2 break large London", "T2 p-value", "T2 Cramér's V",
            "T3 reversal rate", "T3 p-value", "T3 Cramér's V",
        ],
        "Train (70%)": [
            row["OOS Train n"],
            row["OOS Train T1 break (small)"], row["OOS Train T1 break (large)"],
            row["OOS Train T1 p"], row["OOS Train T1 V"],
            row["OOS Train T2 break (small)"], row["OOS Train T2 break (large)"],
            row["OOS Train T2 p"], row["OOS Train T2 V"],
            row["OOS Train T3 reversal rate"], row["OOS Train T3 p"], row["OOS Train T3 V"],
        ],
        "Test (30%)": [
            row["OOS Test n"],
            row["OOS Test T1 break (small)"], row["OOS Test T1 break (large)"],
            row["OOS Test T1 p"], row["OOS Test T1 V"],
            row["OOS Test T2 break (small)"], row["OOS Test T2 break (large)"],
            row["OOS Test T2 p"], row["OOS Test T2 V"],
            row["OOS Test T3 reversal rate"], row["OOS Test T3 p"], row["OOS Test T3 V"],
        ],
    }
    print(tabulate(oos_table, headers="keys", tablefmt="rounded_outline", showindex=False))

print("\n" + "="*65)
print("  LOGISTIC REGRESSION RESULTS")
print("="*65)

lr_summary = []
for r in lr_results_all:
    lr_summary.append({
        "Pair":       r["pair"],
        "N Train":    r["n_train"],
        "N Test":     r["n_test"],
        "Accuracy":   f"{r['accuracy']*100:.1f}%",
        "Baseline":   f"{r['baseline']*100:.1f}%",
        "Edge":       f"{r['edge']*100:.1f}%",
        "ROC AUC":    f"{r['roc_auc']:.3f}",
    })

print(tabulate(pd.DataFrame(lr_summary), headers="keys", 
               tablefmt="rounded_outline", showindex=False))

print("\n  FEATURE COEFFICIENTS BY PAIR")
print("  (positive = towards continuation, negative = towards reversal)\n")
for r in lr_results_all:
    print(f"  {r['pair']}")
    print(tabulate(r["coefficients"], headers="keys", 
                   tablefmt="rounded_outline", showindex=False))
    print()

print("\n" + "="*65)
print("  WALK-FORWARD CROSS-VALIDATION RESULTS")
print("="*65)

cv_summary = []
for r in cv_results_all:
    cv_summary.append({
        "Pair":             r["pair"],
        "Mean Accuracy":    f"{r['mean_accuracy']*100:.1f}%",
        "Mean Baseline":    f"{r['mean_baseline']*100:.1f}%",
        "Mean Edge":        f"{r['mean_edge']*100:.1f}%",
        "Mean ROC AUC":     f"{r['mean_roc_auc']:.3f}",
        "Std Accuracy":     f"{r['std_accuracy']*100:.1f}%",
        "Positive Folds":   f"{r['positive_edge_folds']}/{r['total_folds']}",
    })

print(tabulate(pd.DataFrame(cv_summary), headers="keys",
               tablefmt="rounded_outline", showindex=False))

print("\n  PER FOLD DETAIL")
for r in cv_results_all:
    print(f"\n  {r['pair']}")
    fold_print = r["fold_df"].copy()
    fold_print["accuracy"] = fold_print["accuracy"].apply(lambda x: f"{x*100:.1f}%")
    fold_print["baseline"] = fold_print["baseline"].apply(lambda x: f"{x*100:.1f}%")
    fold_print["edge"]     = fold_print["edge"].apply(lambda x: f"{x*100:.1f}%")
    fold_print["roc_auc"]  = fold_print["roc_auc"].apply(lambda x: f"{x:.3f}")
    print(tabulate(fold_print, headers="keys",
                   tablefmt="rounded_outline", showindex=False))

    # ── Save to CSV ───────────────────────────────────────────────────────────
    out_path = "session_significance_results.csv"
    df_results.to_csv(out_path, index=False)
    print(f"  Full results saved to: {out_path}\n")