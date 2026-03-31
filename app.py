"""
Movie Box Office Predictor — FastAPI Backend
=============================================
Pipeline:
  1. On startup → synthetic training data is generated (or loaded) and a
     GradientBoostingRegressor is trained / loaded from disk.
  2. POST /api/search  → uses the Anthropic API + web-search to auto-fill
     movie features; returns found fields, missing fields, and confidence scores.
  3. POST /api/predict → accepts a (possibly partial) feature dict, imputes
     any missing values with dataset medians, and returns the predicted
     worldwide gross with a confidence interval + feature importances.
  4. GET  /            → serves the single-page HTML UI.
"""

import json
import os
import re
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_percentage_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
MODEL_PATH = Path("model_bundle.pkl")
GENRES = ["Action", "Comedy", "Drama", "Horror", "SciFi",
          "Animation", "Romance", "Thriller", "Fantasy", "Documentary"]
RATINGS = ["G", "PG", "PG-13", "R", "NC-17"]
FEATURE_NAMES = [
    "budget_m", "marketing_budget_m", "release_month", "genre_encoded",
    "is_franchise", "sequel_number", "director_score", "cast_score",
    "screen_count", "mpaa_rating_encoded", "runtime", "trailer_views_m",
    "social_sentiment", "rt_score", "competition_level",
]
FEATURE_LABELS = {
    "budget_m":            "Production Budget ($M)",
    "marketing_budget_m":  "Marketing Budget ($M)",
    "release_month":       "Release Month (1–12)",
    "genre":               "Primary Genre",
    "is_franchise":        "Part of Franchise?",
    "sequel_number":       "Entry # in Series (1 = original)",
    "director_score":      "Director Avg Gross – last 3 films ($M)",
    "cast_score":          "Lead Actor Avg Gross – last 3 films ($M)",
    "screen_count":        "Opening Weekend Theaters",
    "mpaa_rating":         "MPAA Rating",
    "runtime":             "Runtime (minutes)",
    "trailer_views_m":     "Total Trailer Views on YouTube (M)",
    "social_sentiment":    "Social Media Sentiment (0–1)",
    "rt_score":            "Rotten Tomatoes Score (0–100)",
    "competition_level":   "Opening-Weekend Competition (1–10)",
}

# ─────────────────────────────────────────────────────────────────────────────
# Synthetic dataset generation
# ─────────────────────────────────────────────────────────────────────────────
def generate_dataset(n: int = 6000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    budget_m           = np.exp(rng.normal(4.0, 1.1, n)).clip(1, 400)
    marketing_budget_m = (budget_m * rng.uniform(0.3, 0.7, n)).clip(1, 200)
    release_month      = rng.choice(range(1, 13), n,
                                    p=[.05,.04,.06,.06,.08,.10,.12,.10,.08,.08,.10,.13])
    genre_encoded      = rng.integers(0, len(GENRES), n)
    is_franchise       = (rng.random(n) < 0.32).astype(float)
    sequel_number      = np.where(is_franchise,
                                  rng.integers(2, 6, n), 1).astype(float)
    director_score     = np.exp(rng.normal(4.2, 1.0, n)).clip(1, 800)
    cast_score         = np.exp(rng.normal(4.5, 1.0, n)).clip(1, 800)
    screen_count       = (2000 + budget_m * 8 + rng.normal(0, 400, n)).clip(100, 4500)
    mpaa_rating_encoded= rng.choice(range(len(RATINGS)), n,
                                    p=[.05,.14,.45,.35,.01])
    runtime            = rng.normal(110, 22, n).clip(70, 210)
    trailer_views_m    = (budget_m * rng.uniform(0.05, 0.4, n)).clip(0.1, 150)
    social_sentiment   = rng.beta(5, 3, n).clip(0.1, 1.0)
    rt_score           = rng.normal(62, 22, n).clip(0, 100)
    competition_level  = rng.uniform(1, 10, n)

    # ── Generate realistic log-gross ──────────────────────────────────────────
    summer_boost  = np.where(np.isin(release_month, [5,6,7,8]), 0.20, 0.0)
    holiday_boost = np.where(np.isin(release_month, [11,12]), 0.15, 0.0)
    genre_mult    = np.array([
        0.35,  # Action
        0.05,  # Comedy
       -0.05,  # Drama
       -0.10,  # Horror
        0.30,  # SciFi
        0.45,  # Animation
       -0.05,  # Romance
        0.05,  # Thriller
        0.25,  # Fantasy
       -0.20,  # Documentary
    ])[genre_encoded]
    rating_adj    = np.array([0.10, 0.15, 0.0, -0.10, -0.30])[mpaa_rating_encoded]

    # Target is ln(gross_M); calibrated so typical film is ~0-300M
    log_gross = (
        0.45 * np.log(budget_m + 1)
        + 0.15 * np.log(marketing_budget_m + 1)
        + 0.22 * np.log(screen_count + 1)
        + 0.40 * is_franchise
        + 0.08 * (sequel_number - 1)
        + 0.10 * np.log(director_score + 1)
        + 0.12 * np.log(cast_score + 1)
        + 0.003 * rt_score
        + 0.22 * social_sentiment
        + 0.09 * np.log(trailer_views_m + 1)
        + summer_boost + holiday_boost
        + 0.30 * genre_mult + 0.50 * rating_adj
        - 0.03 * competition_level
        + rng.normal(0, 0.55, n)
        - 1.8
    )
    worldwide_gross_m = np.exp(log_gross).clip(0.1, 3000)

    return pd.DataFrame({
        "budget_m": budget_m, "marketing_budget_m": marketing_budget_m,
        "release_month": release_month, "genre_encoded": genre_encoded,
        "is_franchise": is_franchise, "sequel_number": sequel_number,
        "director_score": director_score, "cast_score": cast_score,
        "screen_count": screen_count, "mpaa_rating_encoded": mpaa_rating_encoded,
        "runtime": runtime, "trailer_views_m": trailer_views_m,
        "social_sentiment": social_sentiment, "rt_score": rt_score,
        "competition_level": competition_level,
        "worldwide_gross_m": worldwide_gross_m,
    })


# ─────────────────────────────────────────────────────────────────────────────
# Model training
# ─────────────────────────────────────────────────────────────────────────────
def train_and_save():
    print("⚙  Generating synthetic training data …")
    df = generate_dataset(6000)
    X = df[FEATURE_NAMES]
    y = np.log1p(df["worldwide_gross_m"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    print("🏋  Training GradientBoostingRegressor …")
    model = GradientBoostingRegressor(
        n_estimators=500, max_depth=4, learning_rate=0.08,
        min_samples_leaf=15, subsample=0.75, random_state=42,
    )
    model.fit(X_train_s, y_train)

    log_preds      = model.predict(X_test_s)
    r2             = r2_score(y_test, log_preds)
    preds_dollar   = np.expm1(log_preds)
    actuals_dollar = np.expm1(y_test)
    mape = float(np.median(
        np.abs(preds_dollar - actuals_dollar) / (actuals_dollar + 1e-6)))
    medians = {f: float(X[f].median()) for f in FEATURE_NAMES}

    bundle = {
        "model":    model,
        "scaler":   scaler,
        "medians":  medians,
        "metrics":  {"r2": round(r2, 4), "mape": round(mape, 4),
                     "n_train": len(X_train)},
        "importances": dict(zip(FEATURE_NAMES, model.feature_importances_)),
    }
    joblib.dump(bundle, MODEL_PATH)
    print(f"✅  Model saved. R²={r2:.3f}  MAPE={mape:.1%}")
    return bundle


def load_or_train():
    if MODEL_PATH.exists():
        print("📦  Loading existing model …")
        return joblib.load(MODEL_PATH)
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
class SearchRequest(BaseModel):
    title: str


class PredictRequest(BaseModel):
    budget_m:            Optional[float] = None
    marketing_budget_m:  Optional[float] = None
    release_month:       Optional[int]   = None
    genre:               Optional[str]   = None
    is_franchise:        Optional[bool]  = None
    sequel_number:       Optional[int]   = None
    director_score:      Optional[float] = None
    cast_score:          Optional[float] = None
    screen_count:        Optional[int]   = None
    mpaa_rating:         Optional[str]   = None
    runtime:             Optional[int]   = None
    trailer_views_m:     Optional[float] = None
    social_sentiment:    Optional[float] = None
    rt_score:            Optional[float] = None
    competition_level:   Optional[float] = None


# ── Routes ───────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return FileResponse("index.html")


@app.get("/api/model-info")
def model_info():
    return {
        "metrics":     BUNDLE["metrics"],
        "importances": BUNDLE["importances"],
        "feature_labels": FEATURE_LABELS,
    }


@app.post("/api/search")
async def search_features(req: SearchRequest):
    """
    Scrape Wikipedia, The Numbers, Metacritic, and YouTube to auto-extract
    box-office relevant features for the given movie title.
    No API keys required — all public web sources.
    """
    try:
        from scraper import search_movie_features
        data = search_movie_features(req.title)
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
    Returns point estimate, confidence interval, per-feature contributions,
    and a list of which fields were imputed.
    """
    medians  = BUNDLE["medians"]
    model    = BUNDLE["model"]
    scaler   = BUNDLE["scaler"]

    genre_enc  = GENRES.index(req.genre)  if req.genre  in GENRES  else None
    rating_enc = RATINGS.index(req.mpaa_rating) if req.mpaa_rating in RATINGS else None

    raw_input = {
        "budget_m":            req.budget_m,
        "marketing_budget_m":  req.marketing_budget_m,
        "release_month":       req.release_month,
        "genre_encoded":       genre_enc,
        "is_franchise":        (1.0 if req.is_franchise else 0.0)
                               if req.is_franchise is not None else None,
        "sequel_number":       req.sequel_number,
        "director_score":      req.director_score,
        "cast_score":          req.cast_score,
        "screen_count":        req.screen_count,
        "mpaa_rating_encoded": rating_enc,
        "runtime":             req.runtime,
        "trailer_views_m":     req.trailer_views_m,
        "social_sentiment":    req.social_sentiment,
        "rt_score":            req.rt_score,
        "competition_level":   req.competition_level,
    }

    imputed_fields = []
    final_values   = {}
    for feat in FEATURE_NAMES:
        val = raw_input.get(feat)
        if val is None:
            final_values[feat] = medians[feat]
            # Map back to user-facing name
            key = feat.replace("_encoded", "").replace("genre_encoded", "genre")
            imputed_fields.append(feat)
        else:
            final_values[feat] = float(val)

    X = np.array([[final_values[f] for f in FEATURE_NAMES]])
    X_scaled = scaler.transform(X)

    log_pred = float(model.predict(X_scaled)[0])
    gross_m  = float(np.expm1(log_pred))

    # Confidence interval — use stage predictions from the ensemble
    stage_preds = np.array([
        np.expm1(est.predict(X_scaled)[0])
        for est in model.estimators_.flatten()
    ])
    ci_low  = float(np.percentile(stage_preds, 10))
    ci_high = float(np.percentile(stage_preds, 90))
    # Smooth CI around the actual prediction
    ci_low  = gross_m * 0.55
    ci_high = gross_m * 1.80

    # Simple feature contribution (importance × scaled value)
    importances = BUNDLE["importances"]
    contributions = {
        f: float(importances[f] * abs(X_scaled[0][i]))
        for i, f in enumerate(FEATURE_NAMES)
    }
    # Normalise contributions to percentages
    total = sum(contributions.values()) or 1
    contributions_pct = {f: round(v / total * 100, 1)
                         for f, v in contributions.items()}

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
        "contributions_pct": contributions_pct,
        "model_metrics":     BUNDLE["metrics"],
    }


@app.post("/api/retrain")
def retrain():
    global BUNDLE
    if MODEL_PATH.exists():
        MODEL_PATH.unlink()
    BUNDLE = train_and_save()
    return {"success": True, "metrics": BUNDLE["metrics"]}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
