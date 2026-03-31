"""
retrain.py — Retrain the box office model on real TMDB data
===========================================================
Run this AFTER load_tmdb.py has produced tmdb_features.csv.

Usage:
    python retrain.py

What it does:
  1. Loads tmdb_features.csv
  2. Runs the same training pipeline as app.py
  3. Saves a new model_bundle.pkl (overwrites the old one)
  4. Prints evaluation metrics and feature importances
"""

import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

FEATURES_CSV = Path("tmdb_features.csv")
MODEL_PATH   = Path("model_bundle.pkl")

FEATURE_NAMES = [
    "budget_m", "marketing_budget_m", "release_month", "genre_encoded",
    "is_franchise", "sequel_number", "director_score", "cast_score",
    "screen_count", "mpaa_rating_encoded", "runtime", "trailer_views_m",
    "social_sentiment", "rt_score", "competition_level",
]


def main():
    if not FEATURES_CSV.exists():
        print(f"❌  {FEATURES_CSV} not found.")
        print("    Run load_tmdb.py first.")
        return

    print(f"📂  Loading {FEATURES_CSV} …")
    df = pd.read_csv(FEATURES_CSV)
    print(f"    {len(df):,} films loaded")

    X = df[FEATURE_NAMES]
    y = np.log1p(df["worldwide_gross_m"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42)

    print(f"    Train: {len(X_train):,}  Test: {len(X_test):,}")

    scaler      = StandardScaler()
    X_train_s   = scaler.fit_transform(X_train)
    X_test_s    = scaler.transform(X_test)

    print("🏋  Training GradientBoostingRegressor on REAL data …")
    model = GradientBoostingRegressor(
        n_estimators=500, max_depth=4, learning_rate=0.08,
        min_samples_leaf=15, subsample=0.75, random_state=42,
    )
    model.fit(X_train_s, y_train)

    # Evaluate
    log_preds      = model.predict(X_test_s)
    r2             = r2_score(y_test, log_preds)
    preds_dollar   = np.expm1(log_preds)
    actuals_dollar = np.expm1(y_test)
    mape = float(np.median(
        np.abs(preds_dollar - actuals_dollar) / (actuals_dollar + 1e-6)))

    medians = {f: float(X[f].median()) for f in FEATURE_NAMES}

    bundle = {
        "model":       model,
        "scaler":      scaler,
        "medians":     medians,
        "metrics":     {
            "r2":      round(r2, 4),
            "mape":    round(mape, 4),
            "n_train": len(X_train),
            "source":  "TMDB real data",
        },
        "importances": dict(zip(FEATURE_NAMES, model.feature_importances_)),
    }

    joblib.dump(bundle, MODEL_PATH)

    print(f"\n✅  Model retrained and saved to {MODEL_PATH}")
    print(f"    R²             = {r2:.4f}  (higher = better, max 1.0)")
    print(f"    Median Abs %E  = {mape:.1%}  (lower = better)")
    print(f"    Training films = {len(X_train):,}")
    print(f"    Source         = TMDB real data")

    print("\n📊  Feature importances:")
    importances = dict(zip(FEATURE_NAMES, model.feature_importances_))
    for feat, imp in sorted(importances.items(), key=lambda x: -x[1]):
        bar = "█" * int(imp * 60)
        print(f"    {feat:<28}  {bar}  {imp:.3f}")

    # Show some example predictions
    print("\n🎬  Sample predictions on test set (first 8):")
    print(f"    {'Actual ($M)':<14} {'Predicted ($M)':<16} {'Ratio'}")
    print(f"    {'-'*45}")
    sample_actual = np.expm1(y_test.values[:8])
    sample_pred   = preds_dollar[:8]
    for a, p in zip(sample_actual, sample_pred):
        ratio = p / a if a > 0 else 0
        print(f"    ${a:<13.1f} ${p:<15.1f} {ratio:.2f}x")

    print("\n🚀  Start the server to use the retrained model:")
    print("    python app.py")


if __name__ == "__main__":
    main()