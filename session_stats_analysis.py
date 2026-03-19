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

from xgboost import XGBClassifier

import shap
import matplotlib.pyplot as plt

from statsmodels.nonparametric.smoothers_lowess import lowess

from sklearn.calibration import CalibratedClassifierCV, calibration_curve

from arch import arch_model

from sklearn.pipeline import Pipeline


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

def run_xgboost_cv(df_features: pd.DataFrame, pair_name: str, n_splits: int = 5) -> dict:
    """
    Walk-forward cross-validation using XGBoost.
    Handles class imbalance via scale_pos_weight.
    No feature scaling needed for tree-based models.
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

    tscv        = TimeSeriesSplit(n_splits=n_splits)
    fold_results = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # ── Class imbalance correction ────────────────────────────────────
        neg  = (y_train == 0).sum()
        pos  = (y_train == 1).sum()
        spw  = neg / pos if pos > 0 else 1

        # ── Model ─────────────────────────────────────────────────────────
        model = XGBClassifier(
            n_estimators      = 200,
            max_depth         = 3,
            learning_rate     = 0.05,
            scale_pos_weight  = spw,
            eval_metric       = "logloss",
            early_stopping_rounds = 20,
            random_state      = 42,
            verbosity         = 0,
        )

        # Use last 20% of training data as validation for early stopping
        val_split   = int(len(X_train) * 0.8)
        X_tr        = X_train.iloc[:val_split]
        y_tr        = y_train.iloc[:val_split]
        X_val       = X_train.iloc[val_split:]
        y_val       = y_train.iloc[val_split:]

        model.fit(
            X_tr, y_tr,
            eval_set          = [(X_val, y_val)],
            verbose           = False,
        )

        # ── Evaluate ──────────────────────────────────────────────────────
        y_pred       = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        roc_auc  = roc_auc_score(y_test, y_pred_proba)
        baseline = max(y_test.mean(), 1 - y_test.mean())
        edge     = accuracy - baseline

        # Feature importance for this fold
        importance = pd.DataFrame({
            "Feature":    FEATURES,
            "Importance": model.feature_importances_,
        }).sort_values("Importance", ascending=False)

        fold_results.append({
            "fold":       fold,
            "n_train":    len(train_idx),
            "n_test":     len(test_idx),
            "accuracy":   accuracy,
            "baseline":   baseline,
            "edge":       edge,
            "roc_auc":    roc_auc,
            "importance": importance,
            "n_trees":    model.best_iteration,
        })

    fold_df = pd.DataFrame([{k: v for k, v in f.items() if k != "importance"}
                             for f in fold_results])

    # Average feature importance across all folds
    avg_importance = (
        pd.concat([f["importance"] for f in fold_results])
        .groupby("Feature")["Importance"]
        .mean()
        .reset_index()
        .sort_values("Importance", ascending=False)
    )

    return {
        "pair":             pair_name,
        "fold_df":          fold_df,
        "mean_accuracy":    fold_df["accuracy"].mean(),
        "mean_baseline":    fold_df["baseline"].mean(),
        "mean_edge":        fold_df["edge"].mean(),
        "mean_roc_auc":     fold_df["roc_auc"].mean(),
        "std_accuracy":     fold_df["accuracy"].std(),
        "std_roc_auc":      fold_df["roc_auc"].std(),
        "positive_edge_folds": (fold_df["edge"] > 0).sum(),
        "total_folds":      n_splits,
        "avg_importance":   avg_importance,
    }

def run_shap_analysis(df_features: pd.DataFrame, pair_name: str) -> dict:
    """
    Runs SHAP analysis on XGBoost model trained on full dataset.
    Uses full dataset for SHAP rather than CV folds to maximise
    the number of explanations generated.
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
    train     = df_features.iloc[:split_idx]
    test      = df_features.iloc[split_idx:]

    X_train = train[FEATURES]
    y_train = train[TARGET]
    X_test  = test[FEATURES]
    y_test  = test[TARGET]

    # ── Train XGBoost on train set ────────────────────────────────────────
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    spw = neg / pos if pos > 0 else 1

    model = XGBClassifier(
        n_estimators     = 200,
        max_depth        = 3,
        learning_rate    = 0.05,
        scale_pos_weight = spw,
        eval_metric      = "logloss",
        random_state     = 42,
        verbosity        = 0,
    )
    model.fit(X_train, y_train)

    # ── SHAP values on test set ───────────────────────────────────────────
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # ── Mean absolute SHAP — feature ranking ─────────────────────────────
    mean_shap = pd.DataFrame({
        "Feature":          FEATURES,
        "Mean |SHAP|":      np.abs(shap_values).mean(axis=0),
        "Mean SHAP":        shap_values.mean(axis=0),
    }).sort_values("Mean |SHAP|", ascending=False)

    # Positive mean SHAP = pushes towards continuation
    # Negative mean SHAP = pushes towards reversal
    mean_shap["Direction"] = mean_shap["Mean SHAP"].apply(
        lambda x: "→ Continuation" if x > 0 else "→ Reversal"
    )

    return {
        "pair":        pair_name,
        "shap_values": shap_values,
        "X_test":      X_test,
        "mean_shap":   mean_shap,
        "explainer":   explainer,
        "model":       model,
    }

def run_calibration_analysis(df_features: pd.DataFrame, pair_name: str) -> dict:
    """
    Assesses and corrects probability calibration of XGBoost model.
    Compares uncalibrated, Platt scaling, and isotonic regression.
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
    train     = df_features.iloc[:split_idx]
    test      = df_features.iloc[split_idx:]

    X_train = train[FEATURES]
    y_train = train[TARGET]
    X_test  = test[FEATURES]
    y_test  = test[TARGET]

    # ── Class imbalance ───────────────────────────────────────────────────
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    spw = neg / pos if pos > 0 else 1

   # ── Base XGBoost model — fit separately for uncalibrated probs ────────
    base_model = XGBClassifier(
        n_estimators     = 200,
        max_depth        = 3,
        learning_rate    = 0.05,
        scale_pos_weight = spw,
        eval_metric      = "logloss",
        random_state     = 42,
        verbosity        = 0,
    )
    base_model.fit(X_train, y_train)
    proba_base = base_model.predict_proba(X_test)[:, 1]

    # ── Calibrated models — fit fresh instances using CV ──────────────────
    # cv=3 uses 3-fold CV internally to learn the calibration mapping
    # must pass unfitted estimator instances
    platt_model = CalibratedClassifierCV(
        XGBClassifier(
            n_estimators     = 200,
            max_depth        = 3,
            learning_rate    = 0.05,
            scale_pos_weight = spw,
            eval_metric      = "logloss",
            random_state     = 42,
            verbosity        = 0,
        ),
        cv=3, method="sigmoid"
    )
    platt_model.fit(X_train, y_train)
    proba_platt = platt_model.predict_proba(X_test)[:, 1]

    iso_model = CalibratedClassifierCV(
        XGBClassifier(
            n_estimators     = 200,
            max_depth        = 3,
            learning_rate    = 0.05,
            scale_pos_weight = spw,
            eval_metric      = "logloss",
            random_state     = 42,
            verbosity        = 0,
        ),
        cv=3, method="isotonic"
    )
    iso_model.fit(X_train, y_train)
    proba_iso = iso_model.predict_proba(X_test)[:, 1]

    # ── Calibration curves ────────────────────────────────────────────────
    # n_bins controls granularity — keep low given small test set sizes
    n_bins = 5

    frac_base,  mean_base  = calibration_curve(y_test, proba_base,  n_bins=n_bins)
    frac_platt, mean_platt = calibration_curve(y_test, proba_platt, n_bins=n_bins)
    frac_iso,   mean_iso   = calibration_curve(y_test, proba_iso,   n_bins=n_bins)

    # ── Brier score — lower is better, 0.25 = random, 0 = perfect ────────
    from sklearn.metrics import brier_score_loss
    brier_base  = brier_score_loss(y_test, proba_base)
    brier_platt = brier_score_loss(y_test, proba_platt)
    brier_iso   = brier_score_loss(y_test, proba_iso)

    return {
        "pair":        pair_name,
        "y_test":      y_test,
        "proba_base":  proba_base,
        "proba_platt": proba_platt,
        "proba_iso":   proba_iso,
        "frac_base":   frac_base,
        "mean_base":   mean_base,
        "frac_platt":  frac_platt,
        "mean_platt":  mean_platt,
        "frac_iso":    frac_iso,
        "mean_iso":    mean_iso,
        "brier_base":  brier_base,
        "brier_platt": brier_platt,
        "brier_iso":   brier_iso,
    }


def run_garch_analysis(df_features: pd.DataFrame, hourly: pd.DataFrame, pair_name: str) -> dict:
    """
    Fits a GARCH(1,1) model to daily returns to estimate volatility regimes.
    Re-runs session analysis within each regime to assess regime dependency.
    """

    # ── Compute daily returns from hourly data ────────────────────────────
    if isinstance(hourly.columns, pd.MultiIndex):
        hourly.columns = hourly.columns.get_level_values(0)

    # Use close price, resample to daily
    daily_close = hourly["Close"].resample("1D").last().dropna()
    daily_returns = daily_close.pct_change().dropna() * 100  # percentage returns

    # ── Fit GARCH(1,1) ────────────────────────────────────────────────────
    model = arch_model(
        daily_returns,
        vol   = "Garch",
        p     = 1,
        q     = 1,
        mean  = "constant",
        dist  = "normal",
    )

    try:
        result = model.fit(disp="off")
    except Exception as e:
        print(f"  GARCH fit failed for {pair_name}: {e}")
        return None

    # ── Extract conditional volatility ────────────────────────────────────
    cond_vol = result.conditional_volatility

    # Align with session dataframe on date index
    cond_vol.index = pd.to_datetime(cond_vol.index).normalize().tz_localize(None)
    df_aligned = df_features.copy()
    df_aligned.index = pd.to_datetime(df_aligned.index).normalize().tz_localize(None)

    df_aligned = df_aligned.join(
        cond_vol.rename("cond_vol"), how="inner"
    )

    if len(df_aligned) < 30:
        print(f"  Insufficient aligned data for {pair_name} after GARCH join")
        return None

    # ── Assign volatility regimes by tertile ──────────────────────────────
    df_aligned["vol_regime"] = pd.qcut(
        df_aligned["cond_vol"],
        q      = 3,
        labels = ["Low Vol", "Medium Vol", "High Vol"]
    )

    # ── Re-run session analysis within each regime ────────────────────────
    regime_stats = {}

    for regime in ["Low Vol", "Medium Vol", "High Vol"]:
        subset = df_aligned[df_aligned["vol_regime"] == regime]
        n      = len(subset)

        if n < 10:
            continue

        # Break probabilities
        london_breaks = (
            (subset["london_high"] > subset["asia_high"]) |
            (subset["london_low"]  < subset["asia_low"])
        )
        ny_breaks = (
            (subset["ny_high"]  > subset["london_high"]) |
            (subset["ny_low"]   < subset["london_low"])
        )

        # Directional continuation
        london_bull  = subset["london_direction"] == "Bullish"
        ny_bull      = subset["ny_direction"]     == "Bullish"
        ny_continues = (london_bull == ny_bull)

        # Range averages
        asia_range_avg   = subset["asia_range"].mean()
        london_range_avg = subset["london_range"].mean()
        ny_range_avg     = subset["ny_range"].mean()

        # XGBoost accuracy within regime
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

        if len(subset) >= 20 and TARGET in subset.columns:
            split_idx = int(len(subset) * 0.7)
            X_train   = subset[FEATURES].iloc[:split_idx]
            y_train   = subset[TARGET].iloc[:split_idx]
            X_test    = subset[FEATURES].iloc[split_idx:]
            y_test    = subset[TARGET].iloc[split_idx:]

            if len(X_test) > 5 and len(y_train.unique()) > 1:
                neg = (y_train == 0).sum()
                pos = (y_train == 1).sum()
                spw = neg / pos if pos > 0 else 1

                xgb = XGBClassifier(
                    n_estimators     = 100,
                    max_depth        = 3,
                    learning_rate    = 0.05,
                    scale_pos_weight = spw,
                    eval_metric      = "logloss",
                    random_state     = 42,
                    verbosity        = 0,
                )
                xgb.fit(X_train, y_train)
                y_pred   = xgb.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                baseline = max(y_test.mean(), 1 - y_test.mean())
                edge     = accuracy - baseline
            else:
                accuracy, baseline, edge = None, None, None
        else:
            accuracy, baseline, edge = None, None, None

        regime_stats[regime] = {
            "n":                    n,
            "avg_cond_vol":         subset["cond_vol"].mean(),
            "asia_range_avg":       asia_range_avg,
            "london_range_avg":     london_range_avg,
            "ny_range_avg":         ny_range_avg,
            "london_breaks_asia":   london_breaks.mean() * 100,
            "ny_breaks_london":     ny_breaks.mean() * 100,
            "ny_continues_london":  ny_continues.mean() * 100,
            "xgb_accuracy":         accuracy,
            "xgb_baseline":         baseline,
            "xgb_edge":             edge,
        }

    return {
        "pair":          pair_name,
        "garch_result":  result,
        "cond_vol":      cond_vol,
        "df_aligned":    df_aligned,
        "regime_stats":  regime_stats,
    }

def run_backtest(df_features: pd.DataFrame, pair_name: str) -> dict:
    """
    Backtests a session-based strategy using calibrated XGBoost probabilities.
    Entry: NY open (13:00 UTC)
    Exit:  NY close (22:00 UTC)
    Signal: Platt-calibrated XGBoost probability of NY continuation
    Sizing: Fixed and Kelly
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
    train     = df_features.iloc[:split_idx]
    test      = df_features.iloc[split_idx:]

    X_train = train[FEATURES]
    y_train = train[TARGET]
    X_test  = test[FEATURES]
    y_test  = test[TARGET]

    if len(X_test) < 10:
        return None

    # ── Train calibrated model on train set ───────────────────────────────
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    spw = neg / pos if pos > 0 else 1

    base_model = XGBClassifier(
        n_estimators     = 200,
        max_depth        = 3,
        learning_rate    = 0.05,
        scale_pos_weight = spw,
        eval_metric      = "logloss",
        random_state     = 42,
        verbosity        = 0,
    )
    base_model.fit(X_train, y_train)

    platt_model = CalibratedClassifierCV(
        XGBClassifier(
            n_estimators     = 200,
            max_depth        = 3,
            learning_rate    = 0.05,
            scale_pos_weight = spw,
            eval_metric      = "logloss",
            random_state     = 42,
            verbosity        = 0,
        ),
        cv=3, method="sigmoid"
    )
    platt_model.fit(X_train, y_train)

    # ── Generate signals on test set ──────────────────────────────────────
    proba = platt_model.predict_proba(X_test)[:, 1]

    # ── Estimate win/loss ratio from train set for Kelly ──────────────────
    train_proba  = platt_model.predict_proba(X_train)[:, 1]
    train_pred   = (train_proba > 0.5).astype(int)
    train_actual = y_train.values

    # NY return = close - open, signed by direction
    # Positive if continuation, negative if reversal
    train_ny_return = np.where(
        train["london_bullish"] == 1,
        train["ny_close"] - train["ny_open"],
        train["ny_open"] - train["ny_close"],
    )

    correct_mask   = train_pred == train_actual
    winners        = np.abs(train_ny_return[correct_mask])
    losers         = np.abs(train_ny_return[~correct_mask])

    avg_win  = winners.mean() if len(winners) > 0 else 1
    avg_loss = losers.mean()  if len(losers)  > 0 else 1
    b        = avg_win / avg_loss if avg_loss > 0 else 1

    # ── Build trade log ───────────────────────────────────────────────────
    trades = []

    for i, (idx, row) in enumerate(test.iterrows()):
        p = proba[i]

        # Only trade when model has conviction above threshold
        threshold = 0.55
        if p < threshold:
            continue

        # Direction — follow London if continuation predicted
        london_bull = row["london_bullish"] == 1
        go_long     = london_bull  # continuation = follow London

        # Actual NY return (points)
        ny_return_raw = (row["ny_close"] - row["ny_open"]) / row["ny_open"]
        if abs(ny_return_raw) > 0.05:  # filter returns > 5% as likely data errors
            continue
        # Signed return from our position direction
        trade_return = ny_return_raw if go_long else -ny_return_raw

        # ── Fixed sizing — 1 unit per trade ──────────────────────────────
        fixed_return = trade_return

        # ── Kelly sizing ──────────────────────────────────────────────────
        kelly_f = max(0, p - (1 - p) / b)
        kelly_f = min(kelly_f, 0.1)  # cap at 25% to avoid overbetting
        kelly_return = trade_return * kelly_f

        trades.append({
            "date":          idx,
            "probability":   p,
            "direction":     "Long" if go_long else "Short",
            "london_bull":   london_bull,
            "ny_return":     ny_return_raw,
            "trade_return":  trade_return,
            "fixed_return":  fixed_return,
            "kelly_f":       kelly_f,
            "kelly_return":  kelly_return,
            "correct":       trade_return > 0,
        })

    if len(trades) < 5:
        return None

    trade_df = pd.DataFrame(trades).set_index("date")

    # ── Performance metrics ───────────────────────────────────────────────
    def compute_metrics(returns: pd.Series, label: str) -> dict:
        cumulative    = (1 + returns).cumprod()
        total_return  = cumulative.iloc[-1] - 1
        n_trades      = len(returns)
        win_rate      = (returns > 0).mean()
        avg_win       = returns[returns > 0].mean() if (returns > 0).any() else 0
        avg_loss      = returns[returns < 0].mean() if (returns < 0).any() else 0
        rr_ratio      = abs(avg_win / avg_loss) if avg_loss != 0 else 0

        # Annualised Sharpe (assume 252 trading days)
        daily_mean    = returns.mean()
        daily_std     = returns.std()
        sharpe        = (daily_mean / daily_std * np.sqrt(252)
                         if daily_std > 0 else 0)

        # Max drawdown
        roll_max      = cumulative.cummax()
        drawdown      = (cumulative - roll_max) / roll_max
        max_drawdown  = drawdown.min()

        # Calmar ratio
        ann_return   = (1 + total_return) ** (252 / n_trades) - 1
        calmar        = (ann_return / abs(max_drawdown)
                         if max_drawdown != 0 else 0)

        return {
            "label":        label,
            "n_trades":     n_trades,
            "win_rate":     win_rate,
            "avg_win":      avg_win,
            "avg_loss":     avg_loss,
            "rr_ratio":     rr_ratio,
            "total_return": total_return,
            "sharpe":       sharpe,
            "max_drawdown": max_drawdown,
            "calmar":       calmar,
            "cumulative":   cumulative,
            "ann_return":   ann_return,
        }

    fixed_metrics  = compute_metrics(trade_df["fixed_return"],  "Fixed Sizing")
    kelly_metrics  = compute_metrics(trade_df["kelly_return"],  "Kelly Sizing")

    return {
        "pair":          pair_name,
        "trade_df":      trade_df,
        "fixed_metrics": fixed_metrics,
        "kelly_metrics": kelly_metrics,
        "b":             b,
        "threshold":     threshold,
    }

def analyse_pair(pair_name: str, ticker: str) -> dict:
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
    xgb_results = run_xgboost_cv(df_features, pair_name)
    shap_results = run_shap_analysis(df_features, pair_name)
    cal_results = run_calibration_analysis(df_features, pair_name)
    garch_results = run_garch_analysis(df_features, hourly, pair_name)
    backtest_results = run_backtest(df_features, pair_name)

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
    }, df_features, lr_results, cv_results, xgb_results, shap_results, cal_results, garch_results, backtest_results

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

    results           = []
    features          = {}
    lr_results_all    = []
    cv_results_all    = []
    xgb_results_all   = []
    shap_results_all  = []
    cal_results_all   = []
    garch_results_all = []
    backtest_results_all = []

    for pair_name, ticker in PAIRS.items():
        result = analyse_pair(pair_name, ticker)
        if result is not None:
            stats, df_features, lr_result, cv_result, xgb_result, \
            shap_result, cal_result, garch_result, backtest_result = result
            results.append(stats)
            features[pair_name]       = df_features
            lr_results_all.append(lr_result)
            cv_results_all.append(cv_result)
            xgb_results_all.append(xgb_result)
            shap_results_all.append(shap_result)
            cal_results_all.append(cal_result)
            if garch_result is not None:
                garch_results_all.append(garch_result)
            if backtest_result is not None:
                backtest_results_all.append(backtest_result)

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

print("\n" + "="*65)
print("  XGBOOST WALK-FORWARD CROSS-VALIDATION RESULTS")
print("="*65)

xgb_summary = []
for r in xgb_results_all:
    xgb_summary.append({
        "Pair":           r["pair"],
        "Mean Accuracy":  f"{r['mean_accuracy']*100:.1f}%",
        "Mean Baseline":  f"{r['mean_baseline']*100:.1f}%",
        "Mean Edge":      f"{r['mean_edge']*100:.1f}%",
        "Mean ROC AUC":   f"{r['mean_roc_auc']:.3f}",
        "Std Accuracy":   f"{r['std_accuracy']*100:.1f}%",
        "Positive Folds": f"{r['positive_edge_folds']}/{r['total_folds']}",
    })

print(tabulate(pd.DataFrame(xgb_summary), headers="keys",
               tablefmt="rounded_outline", showindex=False))

# ── Feature importance ────────────────────────────────────────────────────────
print("\n  AVERAGE FEATURE IMPORTANCE BY PAIR")
print("  (averaged across all folds)\n")
for r in xgb_results_all:
    print(f"  {r['pair']}")
    print(tabulate(r["avg_importance"], headers="keys",
                   tablefmt="rounded_outline", showindex=False,
                   floatfmt=".4f"))
    print()

# ── LR vs XGB comparison ──────────────────────────────────────────────────────
print("\n" + "="*65)
print("  LOGISTIC REGRESSION vs XGBOOST COMPARISON")
print("="*65)

comparison = []
for lr, xgb in zip(cv_results_all, xgb_results_all):
    comparison.append({
        "Pair":            xgb["pair"],
        "LR ROC AUC":      f"{lr['mean_roc_auc']:.3f}",
        "XGB ROC AUC":     f"{xgb['mean_roc_auc']:.3f}",
        "AUC Improvement": f"{(xgb['mean_roc_auc'] - lr['mean_roc_auc']):.3f}",
        "LR Edge":         f"{lr['mean_edge']*100:.1f}%",
        "XGB Edge":        f"{xgb['mean_edge']*100:.1f}%",
        "LR +ve Folds":    f"{lr['positive_edge_folds']}/{lr['total_folds']}",
        "XGB +ve Folds":   f"{xgb['positive_edge_folds']}/{xgb['total_folds']}",
    })

print(tabulate(pd.DataFrame(comparison), headers="keys",
               tablefmt="rounded_outline", showindex=False))


for r in cv_results_all:
    print(f"\n  {r['pair']}")
    fold_print = r["fold_df"].copy()
    fold_print["accuracy"] = fold_print["accuracy"].apply(lambda x: f"{x*100:.1f}%")
    fold_print["baseline"] = fold_print["baseline"].apply(lambda x: f"{x*100:.1f}%")
    fold_print["edge"]     = fold_print["edge"].apply(lambda x: f"{x*100:.1f}%")
    fold_print["roc_auc"]  = fold_print["roc_auc"].apply(lambda x: f"{x:.3f}")
    print(tabulate(fold_print, headers="keys",
                   tablefmt="rounded_outline", showindex=False))

print("\n" + "="*65)
print("  SHAP ANALYSIS")
print("="*65)

# ── Mean SHAP table ───────────────────────────────────────────────────────────
print("\n  MEAN ABSOLUTE SHAP VALUES BY PAIR")
print("  (Mean |SHAP| = average impact magnitude on model output)")
print("  (Direction = average push towards continuation or reversal)\n")

for r in shap_results_all:
    print(f"  {r['pair']}")
    print(tabulate(r["mean_shap"], headers="keys",
                   tablefmt="rounded_outline", showindex=False,
                   floatfmt=".4f"))
    print()

# ── Cross-pair feature ranking ────────────────────────────────────────────────
print("\n  AVERAGE MEAN |SHAP| ACROSS ALL PAIRS")
all_shap = pd.concat([r["mean_shap"][["Feature", "Mean |SHAP|"]]
                       for r in shap_results_all])
cross_pair_shap = (
    all_shap.groupby("Feature")["Mean |SHAP|"]
    .mean()
    .reset_index()
    .sort_values("Mean |SHAP|", ascending=False)
)
print(tabulate(cross_pair_shap, headers="keys",
               tablefmt="rounded_outline", showindex=False,
               floatfmt=".4f"))

# ── Plots ─────────────────────────────────────────────────────────────────────
print("\n  Generating SHAP plots — one per pair...")

for r in shap_results_all:
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f"SHAP Analysis — {r['pair']}", fontsize=14, fontweight="bold")

    # Summary plot
    plt.sca(axes[0])
    shap.summary_plot(
        r["shap_values"],
        r["X_test"],
        show=False,
        plot_size=None,
    )
    axes[0].set_title("SHAP Summary — Feature Impact Distribution")

    # Bar plot — mean absolute SHAP
    plt.sca(axes[1])
    shap.summary_plot(
        r["shap_values"],
        r["X_test"],
        plot_type="bar",
        show=False,
        plot_size=None,
    )
    axes[1].set_title("Mean |SHAP| — Average Feature Importance")

    plt.tight_layout()
    filename = f"shap_{r['pair'].replace('/', '_')}.png"
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()

print("\n  SHAP dependence plot for top feature across all pairs...")

for r in shap_results_all:
    top_feature = r["mean_shap"]["Feature"].iloc[0]
    fig, ax     = plt.subplots(figsize=(8, 5))

    shap.dependence_plot(
        top_feature,
        r["shap_values"],
        r["X_test"],
        ax=ax,
        show=False,
        
    )
    x_vals = r["X_test"][top_feature].values
    y_vals = r["shap_values"][:, r["X_test"].columns.get_loc(top_feature)]

    smoothed = lowess(y_vals, x_vals, frac=0.3)
    ax.plot(smoothed[:, 0], smoothed[:, 1], 
            color="red", linewidth=2, label="LOWESS trend")
    ax.legend()
    
    ax.set_title(f"SHAP Dependence — {top_feature} — {r['pair']}")
    plt.tight_layout()
    filename = f"shap_dependence_{r['pair'].replace('/', '_')}.png"
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()
    

print("\n" + "="*65)
print("  PROBABILITY CALIBRATION ANALYSIS")
print("="*65)

# ── Brier score summary table ─────────────────────────────────────────────────
print("\n  BRIER SCORES BY PAIR")
print("  (lower = better | random baseline = 0.25 | perfect = 0.00)\n")

brier_summary = []
for r in cal_results_all:
    best = min(r["brier_base"], r["brier_platt"], r["brier_iso"])
    brier_summary.append({
        "Pair":          r["pair"],
        "Uncalibrated":  f"{r['brier_base']:.4f}",
        "Platt":         f"{r['brier_platt']:.4f}",
        "Isotonic":      f"{r['brier_iso']:.4f}",
        "Best Method":   (
            "Uncalibrated" if best == r["brier_base"] else
            "Platt"        if best == r["brier_platt"] else
            "Isotonic"
        ),
    })

print(tabulate(pd.DataFrame(brier_summary), headers="keys",
               tablefmt="rounded_outline", showindex=False))

# ── Reliability diagrams ──────────────────────────────────────────────────────
print("\n  Generating reliability diagrams...")

for r in cal_results_all:
    fig, ax = plt.subplots(figsize=(7, 6))

    # Perfect calibration reference line
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect calibration")

    ax.plot(r["mean_base"],  r["frac_base"],
            "o-", color="#fb923c", linewidth=2, label="Uncalibrated")
    ax.plot(r["mean_platt"], r["frac_platt"],
            "s-", color="#a78bfa", linewidth=2, label="Platt scaling")
    ax.plot(r["mean_iso"],   r["frac_iso"],
            "^-", color="#38bdf8", linewidth=2, label="Isotonic")

    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives (Actual)")
    ax.set_title(f"Reliability Diagram — {r['pair']}")
    ax.legend(loc="upper left")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)

    # Annotate Brier scores
    ax.text(0.05, 0.88,
            f"Brier — Uncal: {r['brier_base']:.4f} | "
            f"Platt: {r['brier_platt']:.4f} | "
            f"Iso: {r['brier_iso']:.4f}",
            transform=ax.transAxes, fontsize=8, color="#64748b")

    plt.tight_layout()
    filename = f"calibration_{r['pair'].replace('/', '_')}.png"
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()


print("\n" + "="*65)
print("  GARCH VOLATILITY REGIME ANALYSIS")
print("="*65)

for r in garch_results_all:
    print(f"\n  {r['pair']}")
    print(f"  {'─'*55}")

    regime_rows = []
    for regime, s in r["regime_stats"].items():
        regime_rows.append({
            "Regime":             regime,
            "N Days":             s["n"],
            "Avg Vol":            f"{s['avg_cond_vol']:.3f}",
            "Asia Range":         f"{s['asia_range_avg']:.5f}",
            "London Range":       f"{s['london_range_avg']:.5f}",
            "NY Range":           f"{s['ny_range_avg']:.5f}",
            "London Breaks Asia": f"{s['london_breaks_asia']:.1f}%",
            "NY Breaks London":   f"{s['ny_breaks_london']:.1f}%",
            "NY Continues":       f"{s['ny_continues_london']:.1f}%",
            "XGB Edge":           f"{s['xgb_edge']*100:.1f}%" if s['xgb_edge'] is not None else "N/A",
        })

    print(tabulate(pd.DataFrame(regime_rows), headers="keys",
                   tablefmt="rounded_outline", showindex=False))

# ── Conditional volatility plots ──────────────────────────────────────────────
print("\n  Generating GARCH conditional volatility plots...")

for r in garch_results_all:
    df_plot = r["df_aligned"].copy()

    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    fig.suptitle(f"GARCH Volatility Regime Analysis — {r['pair']}",
                 fontsize=13, fontweight="bold")

    # ── Plot 1: Conditional volatility over time ──────────────────────────
    axes[0].plot(df_plot.index, df_plot["cond_vol"],
                 color="#38bdf8", linewidth=1.5)
    axes[0].set_title("Conditional Volatility (GARCH)")
    axes[0].set_ylabel("Volatility")
    axes[0].grid(alpha=0.2)

    # Shade regimes
    colors = {"Low Vol": "#22c55e", "Medium Vol": "#f59e0b", "High Vol": "#ef4444"}
    for regime, color in colors.items():
        mask = df_plot["vol_regime"] == regime
        axes[0].fill_between(df_plot.index, 0, df_plot["cond_vol"],
                             where=mask, alpha=0.15, color=color, label=regime)
    axes[0].legend(loc="upper right", fontsize=8)

    # ── Plot 2: Session ranges by regime ─────────────────────────────────
    regime_labels = list(r["regime_stats"].keys())
    asia_avgs     = [r["regime_stats"][reg]["asia_range_avg"]   for reg in regime_labels]
    london_avgs   = [r["regime_stats"][reg]["london_range_avg"] for reg in regime_labels]
    ny_avgs       = [r["regime_stats"][reg]["ny_range_avg"]     for reg in regime_labels]

    x     = np.arange(len(regime_labels))
    width = 0.25

    axes[1].bar(x - width, asia_avgs,   width, label="Asia",   color="#38bdf8")
    axes[1].bar(x,         london_avgs, width, label="London", color="#a78bfa")
    axes[1].bar(x + width, ny_avgs,     width, label="NY",     color="#fb923c")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(regime_labels)
    axes[1].set_title("Average Session Range by Volatility Regime")
    axes[1].set_ylabel("Range")
    axes[1].legend()
    axes[1].grid(alpha=0.2, axis="y")

    # ── Plot 3: Break and continuation probabilities by regime ────────────
    london_breaks = [r["regime_stats"][reg]["london_breaks_asia"]  for reg in regime_labels]
    ny_breaks     = [r["regime_stats"][reg]["ny_breaks_london"]    for reg in regime_labels]
    ny_continues  = [r["regime_stats"][reg]["ny_continues_london"] for reg in regime_labels]

    axes[2].plot(regime_labels, london_breaks, "o-",
                 color="#a78bfa", linewidth=2, label="London breaks Asia")
    axes[2].plot(regime_labels, ny_breaks,     "s-",
                 color="#fb923c", linewidth=2, label="NY breaks London")
    axes[2].plot(regime_labels, ny_continues,  "^-",
                 color="#f472b6", linewidth=2, label="NY continues London")
    axes[2].axhline(y=50, color="white", linestyle="--",
                    alpha=0.3, linewidth=1, label="50% baseline")
    axes[2].set_title("Session Probabilities by Volatility Regime")
    axes[2].set_ylabel("Probability %")
    axes[2].set_ylim(0, 100)
    axes[2].legend(fontsize=8)
    axes[2].grid(alpha=0.2)

    plt.tight_layout()
    filename = f"garch_{r['pair'].replace('/', '_')}.png"
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()



print("\n" + "="*65)
print("  BACKTEST RESULTS")
print("="*65)

# ── Summary table ─────────────────────────────────────────────────────────────
print("\n  FIXED SIZING")
fixed_summary = []
for r in backtest_results_all:
    m = r["fixed_metrics"]
    fixed_summary.append({
        "Pair":         r["pair"],
        "N Trades":     m["n_trades"],
        "Win Rate":     f"{m['win_rate']*100:.1f}%",
        "Ann Return":   f"{m['ann_return']*100:.1f}",
        "Avg Win":      f"{m['avg_win']:.5f}",
        "Avg Loss":     f"{m['avg_loss']:.5f}",
        "R:R":          f"{m['rr_ratio']:.2f}",
        "Sharpe":       f"{m['sharpe']:.2f}",
        "Max DD":       f"{m['max_drawdown']*100:.1f}%",
        "Calmar":       f"{m['calmar']:.2f}",
    })
print(tabulate(pd.DataFrame(fixed_summary), headers="keys",
               tablefmt="rounded_outline", showindex=False))

print("\n  KELLY SIZING")
kelly_summary = []
for r in backtest_results_all:
    m = r["kelly_metrics"]
    kelly_summary.append({
        "Pair":         r["pair"],
        "N Trades":     m["n_trades"],
        "Win Rate":     f"{m['win_rate']*100:.1f}%",
        "Ann Return":   f"{m['ann_return']*100:.1f}",
        "Sharpe":       f"{m['sharpe']:.2f}",
        "Max DD":       f"{m['max_drawdown']*100:.1f}%",
        "Calmar":       f"{m['calmar']:.2f}",
        "Avg Kelly F":  f"{r['trade_df']['kelly_f'].mean():.3f}",
    })
print(tabulate(pd.DataFrame(kelly_summary), headers="keys",
               tablefmt="rounded_outline", showindex=False))

# ── Equity curves ─────────────────────────────────────────────────────────────
print("\n  Generating equity curve plots...")

for r in backtest_results_all:
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle(f"Backtest — {r['pair']} | "
                 f"Threshold: {r['threshold']} | "
                 f"Win/Loss Ratio (b): {r['b']:.2f}",
                 fontsize=12, fontweight="bold")

    # ── Equity curves ─────────────────────────────────────────────────────
    axes[0, 0].plot(r["fixed_metrics"]["cumulative"].values,
                    color="#38bdf8", linewidth=2, label="Fixed")
    axes[0, 0].plot(r["kelly_metrics"]["cumulative"].values,
                    color="#fb923c", linewidth=2, label="Kelly")
    axes[0, 0].axhline(y=1, color="white", linestyle="--", alpha=0.3)
    axes[0, 0].set_title("Equity Curve")
    axes[0, 0].set_ylabel("Cumulative Return")
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.2)

    # ── Drawdown ──────────────────────────────────────────────────────────
    for metrics, color, label in [
        (r["fixed_metrics"], "#38bdf8", "Fixed"),
        (r["kelly_metrics"], "#fb923c", "Kelly"),
    ]:
        cum   = metrics["cumulative"]
        roll  = cum.cummax()
        dd    = (cum - roll) / roll * 100
        axes[0, 1].fill_between(range(len(dd)), dd, 0,
                                alpha=0.4, color=color, label=label)
    axes[0, 1].set_title("Drawdown %")
    axes[0, 1].set_ylabel("Drawdown %")
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.2)

    # ── Trade returns distribution ─────────────────────────────────────────
    axes[1, 0].hist(r["trade_df"]["trade_return"],
                    bins=20, color="#a78bfa", alpha=0.7, edgecolor="white")
    axes[1, 0].axvline(x=0, color="white", linestyle="--", alpha=0.5)
    axes[1, 0].set_title("Trade Return Distribution")
    axes[1, 0].set_xlabel("Return (points)")
    axes[1, 0].set_ylabel("Frequency")
    axes[1, 0].grid(alpha=0.2)

    # ── Probability distribution of taken trades ───────────────────────────
    axes[1, 1].hist(r["trade_df"]["probability"],
                    bins=15, color="#f472b6", alpha=0.7, edgecolor="white")
    axes[1, 1].axvline(x=r["threshold"], color="white",
                       linestyle="--", alpha=0.5, label=f"Threshold {r['threshold']}")
    axes[1, 1].set_title("Model Probability Distribution (Taken Trades)")
    axes[1, 1].set_xlabel("Calibrated Probability")
    axes[1, 1].set_ylabel("Frequency")
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.2)

    plt.tight_layout()
    filename = f"backtest_{r['pair'].replace('/', '_')}.png"
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()

    # ── Save to CSV ───────────────────────────────────────────────────────────
    out_path = "session_significance_results.csv"
    df_results.to_csv(out_path, index=False)