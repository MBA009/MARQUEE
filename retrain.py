"""
retrain.py — Retrain box office model on TMDB data (v3)
=======================================================
Single source of truth for the training procedure; app.py imports from here.
The feature list lives in features.py, shared with the scraper.

v3 changes:
  • Only pre-release features that come from real data. Dropped:
      - trailer_views_m, social_sentiment, rt_score — in TMDB they are derived
        from popularity / vote_average, which are measured *after* release
        and leak the target.
      - marketing_budget_m (always 0.5 × budget), screen_count and
        competition_level (formula + random noise) — no real information.
      - mpaa_rating_encoded — TMDB 5000 has no ratings, so it was constant.
  • Time-based evaluation: test on the latest TEST_YEARS years of films,
    train on earlier ones — the way the model is actually used.
  • Per-film 80% prediction intervals from quantile models, calibrated on
    held-out years (conformalized quantile regression) instead of a fixed
    band — a fixed band was overconfident for small films and too wide for
    big ones.
  • The final models are refit on all films after evaluation.
"""

import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from features import FEATURE_NAMES, title_key

warnings.filterwarnings("ignore")
# Progress messages use emoji; don't crash on Windows consoles that can't show them
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")

FEATURES_CSV = Path("tmdb_features.csv")
MODEL_PATH   = Path("model_bundle.pkl")
FILMS_CSV    = Path("training_films.csv")   # every training film and its role

TEST_YEARS = 4                   # latest years held out to measure accuracy
CALIBRATION_YEARS = 3            # years before those, used to calibrate intervals
# Pandemic years (theatres closed) stay in training but would distort the
# accuracy figures and interval widths, so they're never test/calibration years
ANOMALOUS_YEARS = {2020, 2021}
INTERVAL_QUANTILES = (0.10, 0.90)  # 80% prediction interval


def make_model():
    try:
        import xgboost as xgb
        return xgb.XGBRegressor(
            n_estimators=800, max_depth=5, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
            reg_alpha=0.1, reg_lambda=1.0, random_state=42,
            n_jobs=-1, verbosity=0,
        ), "XGBoost"
    except ImportError:
        from sklearn.ensemble import GradientBoostingRegressor
        return GradientBoostingRegressor(
            n_estimators=500, max_depth=4, learning_rate=0.08,
            min_samples_leaf=15, subsample=0.75, random_state=42,
        ), "GradientBoosting"


class QuantilePair:
    """sklearn fallback for the interval model: one model per quantile,
    predicting an (n, 2) array like XGBoost's multi-quantile regressor."""
    def __init__(self, models):
        self.models = models

    def fit(self, X, y):
        for m in self.models:
            m.fit(X, y)
        return self

    def predict(self, X):
        return np.column_stack([m.predict(X) for m in self.models])


def make_interval_model():
    """Predicts the INTERVAL_QUANTILES of log gross (shallower than the point
    model — deeper trees overfit the tails)."""
    try:
        import xgboost as xgb
        return xgb.XGBRegressor(
            objective="reg:quantileerror", quantile_alpha=np.array(INTERVAL_QUANTILES),
            n_estimators=400, max_depth=3, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
            random_state=42, n_jobs=-1, verbosity=0,
        )
    except ImportError:
        from sklearn.ensemble import GradientBoostingRegressor
        return QuantilePair([GradientBoostingRegressor(
            loss="quantile", alpha=a, n_estimators=300, max_depth=3, learning_rate=0.05,
            min_samples_leaf=15, subsample=0.75, random_state=42,
        ) for a in INTERVAL_QUANTILES])


def predict_interval(interval_model, calibration: float, X: pd.DataFrame,
                     point_log: np.ndarray) -> np.ndarray:
    """Calibrated (n, 2) log-scale interval, widened if needed to contain the point prediction."""
    q = interval_model.predict(X)
    low  = np.minimum(q[:, 0] - calibration, point_log)
    high = np.maximum(q[:, 1] + calibration, point_log)
    return np.column_stack([low, high])


def split_years(years: pd.Series):
    """(is_test, is_cal) masks: the latest TEST_YEARS normal years, and the
    CALIBRATION_YEARS normal years before them."""
    normal = sorted(y for y in years.unique() if y not in ANOMALOUS_YEARS)
    test_years = normal[-TEST_YEARS:]
    cal_years  = normal[-(TEST_YEARS + CALIBRATION_YEARS):-TEST_YEARS]
    return years.isin(test_years), years.isin(cal_years)


def _year_span(years) -> str:
    """'2013–2016', or '2017–2019, 2022' when years are skipped."""
    years = sorted(set(int(y) for y in years))
    runs, start = [], None
    for i, y in enumerate(years):
        start = y if start is None else start
        if i == len(years) - 1 or years[i + 1] != y + 1:
            runs.append(f"{start}–{y}" if y != start else str(y))
            start = None
    return ", ".join(runs) or "none"


def evaluate(y_log: pd.Series, pred_log: np.ndarray) -> Dict[str, float]:
    actual = np.expm1(y_log)
    pred   = np.expm1(pred_log)
    return {
        "r2":        round(float(r2_score(y_log, pred_log)), 4),
        # Median absolute percentage error (more robust than mean)
        "mape":      round(float(np.median(np.abs(pred - actual) / (actual + 1e-6))), 4),
        # Fraction of predictions within 2× of the actual gross
        "within_2x": round(float(np.mean(np.abs(pred_log - y_log) < np.log(2))), 4),
    }


def train(df: pd.DataFrame, source: str, verbose: bool = True) -> Dict[str, Any]:
    """Evaluate on a time-based holdout, then refit on all rows. Returns a model bundle."""
    df = df.dropna(subset=FEATURE_NAMES + ["worldwide_gross_m"])
    X  = df[FEATURE_NAMES]
    y  = np.log1p(df["worldwide_gross_m"])

    is_test, is_cal = split_years(df["release_year"])
    test_span = _year_span(df.loc[is_test, "release_year"])
    X_train, y_train = X[~is_test], y[~is_test]
    X_test,  y_test  = X[is_test],  y[is_test]

    model, model_type = make_model()
    if verbose:
        print(f"🏋  Training {model_type} on {len(X_train):,} films "
              f"(other years), testing on {len(X_test):,} released {test_span} …")
    model.fit(X_train, y_train)
    log_preds = model.predict(X_test)
    metrics   = evaluate(y_test, log_preds)

    # Prediction interval (conformalized quantile regression): fit the
    # quantile model on the earlier training years, then widen it by the
    # amount that makes it cover `coverage` of the CALIBRATION_YEARS before the test years
    coverage = INTERVAL_QUANTILES[1] - INTERVAL_QUANTILES[0]
    is_fit = ~is_test & ~is_cal
    interval_model = make_interval_model().fit(X[is_fit], y[is_fit])
    q_cal = interval_model.predict(X[is_cal])
    scores = np.maximum(q_cal[:, 0] - y[is_cal], y[is_cal] - q_cal[:, 1])
    calibration = float(np.quantile(scores, min(1.0, coverage * (1 + 1 / len(scores)))))

    bounds = predict_interval(interval_model, calibration, X_test, log_preds)
    widths = np.exp(bounds[:, 1] - bounds[:, 0])
    interval = {
        "coverage":      coverage,
        "calibration":   round(calibration, 4),
        "test_coverage": round(float(np.mean((y_test >= bounds[:, 0]) & (y_test <= bounds[:, 1]))), 4),
        "median_width":  round(float(np.median(widths)), 2),
    }

    films = pd.DataFrame({
        "tmdb_id":      df["tmdb_id"] if "tmdb_id" in df else None,
        "title":        df["title"] if "title" in df else None,
        "release_year": df["release_year"].astype(int),
        "data_source":  df["data_source"] if "data_source" in df else source,
        "role":         np.where(is_test, "test", np.where(is_cal, "calibration", "train")),
    })

    # Refit on everything for the deployed models
    final_model, _ = make_model()
    final_model.fit(X, y)
    final_interval_model = make_interval_model().fit(X, y)

    try:
        importances = {f: float(v) for f, v in zip(FEATURE_NAMES, final_model.feature_importances_)}
    except AttributeError:
        importances = {f: 0.0 for f in FEATURE_NAMES}

    return {
        "model":         final_model,
        "interval_model": final_interval_model,
        "medians":       {f: float(X[f].median()) for f in FEATURE_NAMES},
        "feature_names": FEATURE_NAMES,
        "interval":      interval,
        "year_range":    (int(df["release_year"].min()), int(df["release_year"].max())),
        "metrics": {
            **metrics,
            "n_train":    len(X),
            "n_test":     len(X_test),
            "test_split": f"films released {test_span}",
            "test_years": test_span,
            "calibration_years": _year_span(df.loc[is_cal, "release_year"]),
            "source":     source,
            "model_type": model_type,
        },
        "importances": importances,
        # Which films the model learned from (identity columns come from
        # load_tmdb.py; synthetic data has none)
        "training_data": {
            "films":      len(df),
            "by_source":  {str(k): int(v) for k, v in films["data_source"].value_counts().items()},
            "by_role":    {str(k): int(v) for k, v in films["role"].value_counts().items()},
            "years":      _year_span(df["release_year"]),
            "trained_at": datetime.now().isoformat(timespec="seconds"),
        },
        "film_index":  {f"{title_key(t)}|{y}": r for t, y, r in
                        zip(films["title"], films["release_year"], films["role"]) if isinstance(t, str)},
        "_films":      films,
        # Held-out predictions, kept for the calibration printout below
        "_holdout": (np.expm1(y_test).to_numpy(), np.expm1(log_preds)),
    }


def save_bundle(bundle: Dict[str, Any]) -> None:
    """Save the model bundle and the list of training films beside it."""
    bundle.pop("_holdout", None)
    films = bundle.pop("_films", None)
    joblib.dump(bundle, MODEL_PATH)
    if films is not None:
        films.to_csv(FILMS_CSV, index=False)


def training_role(bundle: Dict[str, Any], title: str, year: Optional[int]) -> Optional[str]:
    """'train' / 'calibration' / 'test' if the film is in the training data,
    else None. Release years may differ by one between TMDB and Wikipedia."""
    index = bundle.get("film_index", {})
    key = title_key(title)
    years = [year, year - 1, year + 1] if year else []
    return next((index[f"{key}|{y}"] for y in years if f"{key}|{y}" in index), None)


def load_features_csv(path: Path = FEATURES_CSV) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Handle old (budget_m) column name
    if "budget_m" in df.columns and "budget_m_adj" not in df.columns:
        df = df.rename(columns={"budget_m": "budget_m_adj"})
    missing = [f for f in FEATURE_NAMES + ["worldwide_gross_m"] if f not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing columns {missing}. Re-run load_tmdb.py.")
    return df


def main():
    if not FEATURES_CSV.exists():
        print(f"❌  {FEATURES_CSV} not found. Run load_tmdb.py first.")
        return

    print(f"📂  Loading {FEATURES_CSV} …")
    bundle = train(load_features_csv(), source="TMDB real data")
    actuals, preds = bundle.pop("_holdout")
    save_bundle(bundle)

    m, iv = bundle["metrics"], bundle["interval"]
    print(f"\n✅  Model saved → {MODEL_PATH}")
    print(f"    Model type      : {m['model_type']}")
    print(f"    Held-out test   : {m['n_test']:,} {m['test_split']}")
    print(f"    R² (log scale)  : {m['r2']:.4f}")
    print(f"    Median APE      : {m['mape']:.1%}")
    print(f"    Within 2× actual: {m['within_2x']:.1%}")
    print(f"    80% intervals   : cover {iv['test_coverage']:.1%} of held-out films, "
          f"median width ×{iv['median_width']:.1f} (high ÷ low)")
    print(f"    Final model fit : {m['n_train']:,} films (all years)")
    td = bundle["training_data"]
    print(f"    Training data   : {td['films']:,} films, {td['years']} — "
          + ", ".join(f"{n:,} from {s}" for s, n in td["by_source"].items()))
    print(f"    Film list       : {FILMS_CSV} (title, year, source, train/calibration/test)")

    print("\n📊  Feature importances:")
    for feat, imp in sorted(bundle["importances"].items(), key=lambda x: -x[1]):
        bar = "█" * int(imp * 50)
        print(f"    {feat:<28} {bar} {imp:.3f}")

    print("\n🎬  Held-out calibration (random sample):")
    print(f"    {'Actual 2024$M':<16} {'Predicted $M':<16} {'Ratio':<8}")
    print(f"    {'-'*42}")
    idx = np.random.default_rng(7).integers(0, len(preds), 12)
    for i in idx:
        a, p = actuals[i], preds[i]
        ratio = p/a if a>0 else 0
        flag  = " ⚠" if ratio < 0.4 or ratio > 2.5 else ""
        print(f"    ${a:<15.0f} ${p:<15.0f} {ratio:.2f}x{flag}")

    print(f"\n🚀  Start the server:  python app.py")


if __name__ == "__main__":
    main()
