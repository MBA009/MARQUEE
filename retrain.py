"""
retrain.py — Retrain box office model on TMDB data (v2)
=======================================================
Improvements:
  • XGBoost instead of GradientBoosting (faster, better regularisation)
  • Handles new 16-feature set (adds release_year, renames budget_m → budget_m_adj)
  • Evaluates with more useful metrics
  • Prints calibration examples on real known films
"""

import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

FEATURES_CSV = Path("tmdb_features.csv")
MODEL_PATH   = Path("model_bundle.pkl")

FEATURE_NAMES = [
    "budget_m_adj","marketing_budget_m","release_month","genre_encoded",
    "is_franchise","sequel_number","director_score","cast_score",
    "screen_count","mpaa_rating_encoded","runtime","trailer_views_m",
    "social_sentiment","rt_score","competition_level","release_year",
]


def main():
    if not FEATURES_CSV.exists():
        print(f"❌  {FEATURES_CSV} not found. Run load_tmdb.py first.")
        return

    print(f"📂  Loading {FEATURES_CSV} …")
    df = pd.read_csv(FEATURES_CSV)

    # Handle both old (budget_m) and new (budget_m_adj) column names
    if "budget_m" in df.columns and "budget_m_adj" not in df.columns:
        df = df.rename(columns={"budget_m": "budget_m_adj"})
        print("    (renamed budget_m → budget_m_adj)")

    # Add release_year if missing (old CSV)
    if "release_year" not in df.columns:
        df["release_year"] = 2010.0
        print("    (release_year missing — defaulting to 2010)")

    # Use whichever features are actually present
    available = [f for f in FEATURE_NAMES if f in df.columns]
    missing   = [f for f in FEATURE_NAMES if f not in df.columns]
    if missing:
        print(f"    ⚠  Missing features (will be ignored): {missing}")

    df = df.dropna(subset=available + ["worldwide_gross_m"])
    print(f"    {len(df):,} films loaded, {len(available)} features")

    X = df[available]
    y = np.log1p(df["worldwide_gross_m"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42)

    scaler    = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    # ── Try XGBoost first, fall back to GradientBoosting ─────────────────────
    try:
        import xgboost as xgb
        print("🏋  Training XGBoost …")
        model = xgb.XGBRegressor(
            n_estimators=800,
            max_depth=5,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=5,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        )
        model.fit(
            X_train_s, y_train,
            eval_set=[(X_test_s, y_test)],
            verbose=False,
        )
        model_type = "XGBoost"

    except ImportError:
        from sklearn.ensemble import GradientBoostingRegressor
        print("🏋  XGBoost not found — using GradientBoosting …")
        model = GradientBoostingRegressor(
            n_estimators=500, max_depth=4, learning_rate=0.08,
            min_samples_leaf=15, subsample=0.75, random_state=42,
        )
        model.fit(X_train_s, y_train)
        model_type = "GradientBoosting"

    # ── Evaluate ──────────────────────────────────────────────────────────────
    log_preds      = model.predict(X_test_s)
    r2             = r2_score(y_test, log_preds)
    preds_dollar   = np.expm1(log_preds)
    actuals_dollar = np.expm1(y_test)

    # Median absolute percentage error (more robust than mean)
    mape = float(np.median(
        np.abs(preds_dollar - actuals_dollar) / (actuals_dollar + 1e-6)))

    # Within-2x accuracy (what fraction of predictions are within 2× of actual?)
    within_2x = float(np.mean(
        np.abs(np.log(preds_dollar + 1) - np.log(actuals_dollar + 1)) < np.log(2)))

    # Feature importances
    try:
        importances = dict(zip(available, model.feature_importances_))
    except AttributeError:
        importances = {f: 0.0 for f in available}

    # Medians for imputation
    medians = {f: float(X[f].median()) for f in available}

    bundle = {
        "model":        model,
        "scaler":       scaler,
        "medians":      medians,
        "feature_names": available,
        "metrics": {
            "r2":        round(r2, 4),
            "mape":      round(mape, 4),
            "within_2x": round(within_2x, 4),
            "n_train":   len(X_train),
            "source":    "TMDB real data",
            "model_type": model_type,
        },
        "importances": importances,
    }

    joblib.dump(bundle, MODEL_PATH)

    print(f"\n✅  Model saved → {MODEL_PATH}")
    print(f"    Model type      : {model_type}")
    print(f"    R² (log scale)  : {r2:.4f}")
    print(f"    Median APE      : {mape:.1%}")
    print(f"    Within 2× actual: {within_2x:.1%}  (fraction of test predictions)")
    print(f"    Training films  : {len(X_train):,}")

    print("\n📊  Feature importances:")
    for feat, imp in sorted(importances.items(), key=lambda x: -x[1]):
        bar = "█" * int(imp * 50)
        print(f"    {feat:<28} {bar} {imp:.3f}")

    print("\n🎬  Test-set calibration (random sample):")
    print(f"    {'Actual 2024$M':<16} {'Predicted $M':<16} {'Ratio':<8}")
    print(f"    {'-'*42}")
    idx = np.random.default_rng(7).integers(0, len(preds_dollar), 12)
    for i in idx:
        a, p = actuals_dollar.iloc[i], preds_dollar[i]
        ratio = p/a if a>0 else 0
        flag  = " ⚠" if ratio < 0.4 or ratio > 2.5 else ""
        print(f"    ${a:<15.0f} ${p:<15.0f} {ratio:.2f}x{flag}")

    print(f"\n🚀  Start the server:  python app.py")


if __name__ == "__main__":
    main()