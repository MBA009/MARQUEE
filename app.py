"""
Movie Box Office Predictor — FastAPI Backend
=============================================
Pipeline:
  1. On startup → model_bundle.pkl is loaded, or a model is trained via
     retrain.py on tmdb_features.csv (synthetic data only if that is missing).
  2. POST /api/candidates → films matching a title (to pick between remakes);
     POST /api/search → scrapes Wikipedia / Wikidata / The Numbers
     (scraper.py) to auto-fill movie features.
  3. POST /api/predict → accepts a (possibly partial) feature dict, imputes
     any missing values with dataset medians, and returns the predicted
     worldwide gross (2024 USD) with a per-film 80% prediction interval
     (calibrated on held-out years), plus how each feature moved it.
  4. GET  /            → serves the single-page HTML UI.
"""

import re
import sys
import warnings
from datetime import date
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from features import GENRES, cpi_adjust, encode_genre
from retrain import (FEATURES_CSV, load_features_csv, predict_interval, save_bundle, train,
                     training_role)

warnings.filterwarnings("ignore")
# Startup messages use emoji; don't crash on Windows consoles that can't show them
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
MODEL_PATH = Path("model_bundle.pkl")
RATINGS = ["G", "PG", "PG-13", "R", "NC-17"]
FEATURE_LABELS = {
    "budget_m_adj":        "Production Budget, 2024-adj ($M)",
    "release_month":       "Release Month (1–12)",
    "genre":               "Primary Genre",
    "is_franchise":        "Part of Franchise?",
    "sequel_number":       "Entry # in Series (1 = original)",
    "director_score":      "Director Avg Gross – last 5 films ($M)",
    "cast_score":          "Lead Actor Avg Gross – last 5 films ($M)",
    "runtime":             "Runtime (minutes)",
    "release_year":        "Release Year",
}

# ─────────────────────────────────────────────────────────────────────────────
# Synthetic dataset generation (fallback when tmdb_features.csv is missing).
# Generates extra columns; train() only uses FEATURE_NAMES.
# ─────────────────────────────────────────────────────────────────────────────
def generate_dataset(n: int = 6000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # Release years 2000-2024, weighted toward recent
    release_year       = rng.choice(range(2000, 2025), n,
                                    p=[0.02]*10 + [0.04]*5 + [0.06]*5 + [0.05]*4 + [0.10])
    budget_m_adj       = np.exp(rng.normal(4.2, 1.1, n)).clip(2, 500)
    marketing_budget_m = (budget_m_adj * rng.uniform(0.3, 0.6, n)).clip(1, 250)
    release_month      = rng.choice(range(1, 13), n,
                                    p=[.05,.04,.06,.06,.08,.10,.12,.10,.08,.08,.10,.13])
    genre_encoded      = rng.integers(0, len(GENRES), n)
    is_franchise       = (rng.random(n) < 0.35).astype(float)
    sequel_number      = np.where(is_franchise, rng.integers(2, 8, n), 1).astype(float)
    director_score     = np.exp(rng.normal(4.2, 1.0, n)).clip(1, 900)
    cast_score         = np.exp(rng.normal(4.5, 1.0, n)).clip(1, 900)
    screen_count       = (1500 + budget_m_adj * 9 + rng.normal(0, 400, n)).clip(100, 4500)
    mpaa_rating_encoded= rng.choice(range(len(RATINGS)), n,
                                    p=[.05,.14,.45,.35,.01])
    runtime            = rng.normal(110, 22, n).clip(70, 210)
    trailer_views_m    = (budget_m_adj * rng.uniform(0.04, 0.35, n)).clip(0.1, 150)
    social_sentiment   = rng.beta(5, 3, n).clip(0.1, 1.0)
    rt_score           = rng.normal(62, 22, n).clip(0, 100)
    competition_level  = rng.uniform(1, 10, n)

    summer_boost  = np.where(np.isin(release_month, [5,6,7,8]), 0.20, 0.0)
    holiday_boost = np.where(np.isin(release_month, [11,12]), 0.15, 0.0)
    genre_mult    = np.array([
        0.35, 0.05, -0.05, -0.10, 0.30,
        0.45, -0.05, 0.05, 0.25, -0.20,
    ])[genre_encoded]
    rating_adj    = np.array([0.10, 0.15, 0.0, -0.10, -0.30])[mpaa_rating_encoded]
    year_trend    = (release_year - 2000) * 0.018  # box office grew ~1.8%/yr nominally

    # Calibrated so median notable film ≈ $150M (matches TMDB top-5000 bias)
    log_gross = (
        0.55 * np.log(budget_m_adj + 1)       # budget is strongest signal
        + 0.18 * np.log(marketing_budget_m + 1)
        + 0.20 * np.log(screen_count + 1)
        + 0.55 * is_franchise                  # franchise premium is large
        + 0.06 * np.log1p(sequel_number - 1)  # log-diminishing returns
        + 0.12 * np.log(director_score + 1)
        + 0.14 * np.log(cast_score + 1)
        + 0.004 * rt_score
        + 0.28 * social_sentiment
        + 0.11 * np.log(trailer_views_m + 1)
        + summer_boost + holiday_boost
        + 0.35 * genre_mult + 0.45 * rating_adj
        - 0.025 * competition_level
        + year_trend
        + rng.normal(0, 0.50, n)
        - 1.85                                 # intercept: median ~$130M (matches TMDB top-5000)
    )
    worldwide_gross_m = np.exp(log_gross).clip(0.5, 4000)

    return pd.DataFrame({
        "budget_m_adj":        budget_m_adj,
        "marketing_budget_m":  marketing_budget_m,
        "release_month":       release_month,
        "genre_encoded":       genre_encoded,
        "is_franchise":        is_franchise,
        "sequel_number":       sequel_number,
        "director_score":      director_score,
        "cast_score":          cast_score,
        "screen_count":        screen_count,
        "mpaa_rating_encoded": mpaa_rating_encoded,
        "runtime":             runtime,
        "trailer_views_m":     trailer_views_m,
        "social_sentiment":    social_sentiment,
        "rt_score":            rt_score,
        "competition_level":   competition_level,
        "release_year":        release_year.astype(float),
        "worldwide_gross_m":   worldwide_gross_m,
    })


# ─────────────────────────────────────────────────────────────────────────────
# Model training
# ─────────────────────────────────────────────────────────────────────────────
def train_and_save():
    if FEATURES_CSV.exists():
        print(f"⚙  Training on {FEATURES_CSV} …")
        bundle = train(load_features_csv(), source="TMDB real data")
    else:
        print(f"⚙  {FEATURES_CSV} not found — training on synthetic data …")
        bundle = train(generate_dataset(6000), source="synthetic")
    save_bundle(bundle)
    m = bundle["metrics"]
    print(f"✅  Model saved. Held-out R²={m['r2']:.3f}  MedAPE={m['mape']:.1%}  ({m['model_type']})")
    return bundle

def load_or_train():
    if MODEL_PATH.exists():
        print("📦  Loading existing model …")
        bundle = joblib.load(MODEL_PATH)
        if "interval_model" in bundle and "training_data" in bundle:
            return bundle
        print("⚠  Model bundle is from an older version — retraining …")
    return train_and_save()


# ─────────────────────────────────────────────────────────────────────────────
# FastAPI app
# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI(title="Box Office Predictor")
BUNDLE: Dict[str, Any] = {}


@app.on_event("startup")
def startup():
    global BUNDLE
    BUNDLE = load_or_train()


# ── Pydantic models ───────────────────────────────────────────────────────────
class CandidatesRequest(BaseModel):
    title: str = Field(min_length=1, max_length=200)


class SearchRequest(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    page:  Optional[str] = Field(None, max_length=300)  # exact Wikipedia title, from /api/candidates


class PredictRequest(BaseModel):
    budget_m:            Optional[float] = Field(None, gt=0, le=2000)
    release_month:       Optional[int]   = Field(None, ge=1, le=12)
    genre:               Optional[str]   = None
    is_franchise:        Optional[bool]  = None
    sequel_number:       Optional[int]   = Field(None, ge=1, le=50)
    director_score:      Optional[float] = Field(None, ge=0, le=10000)
    cast_score:          Optional[float] = Field(None, ge=0, le=10000)
    runtime:             Optional[int]   = Field(None, ge=30, le=400)
    release_year:        Optional[int]   = Field(None, ge=1900, le=2100)


# ── Routes ───────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return FileResponse("index.html")


@app.get("/api/model-info")
def model_info():
    return {
        "metrics":     BUNDLE["metrics"],
        "interval":    BUNDLE["interval"],
        "training_data": BUNDLE["training_data"],
        "importances": BUNDLE["importances"],
        "feature_labels": FEATURE_LABELS,
    }


@app.post("/api/candidates")
def film_candidates(req: CandidatesRequest):
    """Films matching a title, so the user can pick between same-named films.
    `candidates` is null if the lookup failed (the UI then searches directly)."""
    from scraper import search_film_candidates
    return {"candidates": search_film_candidates(req.title)}


# Plain `def` routes run in FastAPI's threadpool, so a slow scrape doesn't
# block other requests
@app.post("/api/search")
def search_features(req: SearchRequest):
    """
    Scrape Wikipedia, Wikidata and The Numbers to auto-extract the model's
    features for a film. No API keys required — all public web sources.
    """
    try:
        from scraper import search_movie_features
        data = search_movie_features(req.title, req.page)
        # Was this film in the training data? (exact title + year match)
        film_title = re.sub(r"\s*\([^)]*\)$", "", data.get("wiki_page") or req.title)
        data["training_role"] = training_role(BUNDLE, film_title, data.get("release_year"))
        return JSONResponse(content={"success": True, "data": data})
    except Exception as e:
        import traceback
        return JSONResponse(content={
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()[-800:],
        })


@app.post("/api/predict")
def predict(req: PredictRequest):
    """
    Predict worldwide box-office gross.
    Missing features are imputed from training-set medians.
    The budget is given in release-year dollars and inflation-adjusted here,
    as in training; director/cast scores are already in 2024 USD.
    Returns point estimate (2024 USD), 80% prediction interval, per-feature
    effects, a list of which fields were imputed, and warnings.
    """
    medians  = BUNDLE["medians"]
    model    = BUNDLE["model"]
    feat_names = BUNDLE["feature_names"]

    budget_adj = (cpi_adjust(req.budget_m, req.release_year or date.today().year)
                  if req.budget_m is not None else None)

    warnings_out = []
    year_lo, year_hi = BUNDLE.get("year_range", (None, None))
    if req.release_year and year_hi and req.release_year > year_hi:
        warnings_out.append(
            f"The training data ends in {year_hi}; the model treats a "
            f"{req.release_year} release like a {year_hi} one.")
    elif req.release_year and year_lo and req.release_year < year_lo:
        warnings_out.append(
            f"The training data starts in {year_lo}; the model treats a "
            f"{req.release_year} release like a {year_lo} one.")

    raw_input = {
        "budget_m_adj":        budget_adj,
        "release_month":       req.release_month,
        "genre_encoded":       encode_genre(req.genre),
        "is_franchise":        (1.0 if req.is_franchise else 0.0)
                               if req.is_franchise is not None else None,
        "sequel_number":       req.sequel_number,
        "director_score":      req.director_score,
        "cast_score":          req.cast_score,
        "runtime":             req.runtime,
        "release_year":        req.release_year,
    }

    imputed_fields = []
    final_values   = {}
    for feat in feat_names:
        val = raw_input.get(feat)
        if val is None:
            final_values[feat] = medians.get(feat, 0.0)
            imputed_fields.append(feat)
        else:
            final_values[feat] = float(val)

    X = pd.DataFrame([final_values], columns=feat_names)

    log_pred = float(model.predict(X)[0])
    gross_m  = float(np.expm1(log_pred))

    # Per-film 80% prediction interval (calibrated quantile models)
    low_log, high_log = predict_interval(
        BUNDLE["interval_model"], BUNDLE["interval"]["calibration"], X, np.array([log_pred]))[0]
    ci_low, ci_high = float(np.expm1(low_log)), float(np.expm1(high_log))

    # How each feature moved this prediction: XGBoost's built-in SHAP values.
    # The model works in log dollars, so each value is a multiplier on the
    # baseline (the model's typical film): baseline × all factors ≈ prediction.
    effects = None
    try:
        import xgboost as xgb
        shap = model.get_booster().predict(xgb.DMatrix(X), pred_contribs=True)[0]
        effects = {
            "baseline_m": round(float(np.expm1(shap[-1])), 1),
            "factors":    {f: round(float(np.exp(v)), 3) for f, v in zip(feat_names, shap[:-1])},
        }
    except (ImportError, AttributeError):
        pass  # non-XGBoost fallback model: no per-prediction explanation

    tier = ("Blockbuster 🎬" if gross_m > 500
            else "Major Hit 🌟" if gross_m > 200
            else "Solid Performer 🎥" if gross_m > 80
            else "Modest Release 🎞" if gross_m > 25
            else "Limited Run 📽")

    return {
        "predicted_gross_m": round(gross_m, 1),
        "ci_low_m":          round(ci_low, 1),
        "ci_high_m":         round(ci_high, 1),
        "tier":              tier,
        "imputed_fields":    imputed_fields,
        "warnings":          warnings_out,
        "effects":           effects,
        "interval_coverage": BUNDLE["interval"]["coverage"],
        "interval_test_coverage": BUNDLE["interval"]["test_coverage"],
        "training_years":    BUNDLE.get("year_range"),
        "model_metrics":     BUNDLE["metrics"],
    }


@app.post("/api/retrain")
def retrain():
    global BUNDLE
    BUNDLE = train_and_save()
    return {"success": True, "metrics": BUNDLE["metrics"]}


if __name__ == "__main__":
    import uvicorn
    # Local only; pass host="0.0.0.0" to expose it on your network
    uvicorn.run("app:app", host="127.0.0.1", port=8000)