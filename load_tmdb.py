"""
load_tmdb.py — TMDB dataset loader for the Box Office Predictor
===============================================================
Usage:
    python load_tmdb.py

Reads tmdb_5000_movies.csv and tmdb_5000_credits.csv from the current
directory, engineers all 15 model features, and saves a clean
tmdb_features.csv ready for retraining.

Then retrain the model by running:
    python retrain.py
"""

import ast
import json
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── File paths ────────────────────────────────────────────────────────────────
MOVIES_CSV  = Path("tmdb_5000_movies.csv")
CREDITS_CSV = Path("tmdb_5000_credits.csv")
OUTPUT_CSV  = Path("tmdb_features.csv")

# ── Genre mapping (TMDB genre names → our 10 categories) ─────────────────────
GENRE_MAP = {
    "Action":          "Action",
    "Adventure":       "Action",
    "Science Fiction": "SciFi",
    "Horror":          "Horror",
    "Comedy":          "Comedy",
    "Animation":       "Animation",
    "Family":          "Animation",
    "Romance":         "Romance",
    "Drama":           "Drama",
    "Thriller":        "Thriller",
    "Mystery":         "Thriller",
    "Crime":           "Thriller",
    "Fantasy":         "Fantasy",
    "Documentary":     "Documentary",
    "Music":           "Drama",
    "History":         "Drama",
    "War":             "Action",
    "Western":         "Action",
    "TV Movie":        "Drama",
}

GENRES_LIST = ["Action","Comedy","Drama","Horror","SciFi",
               "Animation","Romance","Thriller","Fantasy","Documentary"]

MPAA_RATINGS = ["G", "PG", "PG-13", "R", "NC-17"]

# Franchises / collections that are well-known — used to set is_franchise
# TMDB has a "belongs_to_collection" field which is far more reliable.

# ── Helpers ───────────────────────────────────────────────────────────────────

def safe_json(val):
    """Parse a JSON string, return [] on failure."""
    if pd.isna(val) or not val:
        return []
    try:
        return json.loads(val)
    except Exception:
        try:
            return ast.literal_eval(val)
        except Exception:
            return []


def first_genre(genres_json: str) -> str:
    """Map TMDB genre list to our genre categories."""
    genres = safe_json(genres_json)
    for g in genres:
        name = g.get("name", "")
        if name in GENRE_MAP:
            return GENRE_MAP[name]
    return "Drama"  # default


def get_director(crew_json: str) -> str:
    """Extract director name from crew JSON."""
    crew = safe_json(crew_json)
    for member in crew:
        if member.get("job") == "Director":
            return member.get("name", "")
    return ""


def get_lead_actor(cast_json: str) -> str:
    """Extract top-billed cast member."""
    cast = safe_json(cast_json)
    if cast:
        return cast[0].get("name", "")
    return ""


def get_cast_list(cast_json: str, n: int = 3) -> list:
    """Return top N cast member names."""
    cast = safe_json(cast_json)
    return [c.get("name", "") for c in cast[:n]]


def extract_mpaa(releases_json: str) -> str:
    """
    TMDB stores certifications per country in the 'releases' field.
    Extract the US certification.
    """
    data = safe_json(releases_json)
    countries = data.get("countries", []) if isinstance(data, dict) else []
    for c in countries:
        if c.get("iso_3166_1") == "US":
            cert = c.get("certification", "").strip()
            if cert in MPAA_RATINGS:
                return cert
    return "PG-13"  # most common default


def sequel_number_from_title(title: str) -> int:
    """Heuristic sequel number from title."""
    t = title.lower()
    for pattern, num in [
        (r"\b(part|chapter|vol\.?|volume)\s*5\b|[:\s]5$|\bv\b", 5),
        (r"\b(part|chapter|vol\.?|volume)\s*4\b|[:\s]4$|\biv\b", 4),
        (r"\b(part|chapter|vol\.?|volume)\s*3\b|[:\s]3$|\biii\b|rises|revolution|revolutions", 3),
        (r"\b(part|chapter|vol\.?|volume)\s*2\b|[:\s]2$|\bii\b|returns|reloaded|evolution|strikes back", 2),
    ]:
        if re.search(pattern, t):
            return num
    return 1


# ── Per-person historical gross scorer ───────────────────────────────────────

def build_person_scores(df_merged: pd.DataFrame) -> tuple:
    """
    For each director and lead actor, compute their average worldwide gross
    on their PREVIOUS films (strictly prior, to avoid data leakage).

    Returns two dicts:
        director_scores : {movie_id: avg_gross_M}
        cast_scores     : {movie_id: avg_gross_M}
    """
    # Sort by release date so we can compute rolling history
    df = df_merged.copy()
    df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
    df = df.sort_values("release_date").reset_index(drop=True)

    # Build lookup: person → list of (movie_idx, gross)
    from collections import defaultdict
    dir_films  = defaultdict(list)
    cast_films = defaultdict(list)

    for idx, row in df.iterrows():
        g = row["revenue"] / 1e6 if row["revenue"] > 0 else np.nan
        if row["director"]:
            dir_films[row["director"]].append((idx, g))
        if row["lead_actor"]:
            cast_films[row["lead_actor"]].append((idx, g))

    # For each movie, average the PREVIOUS films of that person
    def avg_prev(person_films: dict, movie_idx: int, person: str,
                 fallback: float) -> float:
        films = person_films.get(person, [])
        prev  = [g for i, g in films if i < movie_idx and not np.isnan(g)]
        if not prev:
            return fallback
        # Take last 5, weight recent more
        recent = prev[-5:]
        return round(np.mean(recent), 1)

    global_dir_fallback  = df[df["revenue"] > 0]["revenue"].median() / 1e6
    global_cast_fallback = df[df["revenue"] > 0]["revenue"].median() / 1e6

    director_scores = {}
    cast_scores     = {}
    for idx, row in df.iterrows():
        mid = row["id"]
        director_scores[mid] = avg_prev(
            dir_films, idx, row["director"], global_dir_fallback)
        cast_scores[mid] = avg_prev(
            cast_films, idx, row["lead_actor"], global_cast_fallback)

    return director_scores, cast_scores


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("📂  Loading TMDB CSVs …")

    if not MOVIES_CSV.exists():
        print(f"❌  {MOVIES_CSV} not found.")
        print("    Download from: https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata")
        return
    if not CREDITS_CSV.exists():
        print(f"❌  {CREDITS_CSV} not found.")
        return

    movies  = pd.read_csv(MOVIES_CSV)
    credits = pd.read_csv(CREDITS_CSV)

    print(f"    Movies:  {len(movies):,} rows")
    print(f"    Credits: {len(credits):,} rows")

    # ── Merge on movie id ─────────────────────────────────────────────────────
    # Credits CSV uses 'movie_id' or 'id' depending on version
    id_col = "movie_id" if "movie_id" in credits.columns else "id"
    df = movies.merge(credits, left_on="id", right_on=id_col, how="left",
                      suffixes=("", "_credits"))

    print(f"    Merged:  {len(df):,} rows")

    # ── Extract director and lead actor ───────────────────────────────────────
    crew_col = "crew" if "crew" in df.columns else None
    cast_col = "cast" if "cast" in df.columns else None

    if crew_col:
        df["director"]   = df[crew_col].apply(get_director)
    else:
        df["director"] = ""

    if cast_col:
        df["lead_actor"] = df[cast_col].apply(get_lead_actor)
    else:
        df["lead_actor"] = ""

    # ── Compute per-person historical scores (no leakage) ────────────────────
    print("🧮  Computing director & cast historical scores (this takes ~30s) …")
    director_scores, cast_scores = build_person_scores(df)
    df["director_score"] = df["id"].map(director_scores)
    df["cast_score"]     = df["id"].map(cast_scores)

    # ── Feature engineering ───────────────────────────────────────────────────
    print("⚙  Engineering features …")

    # Budget & revenue — filter out zero/missing (unreported, not real zeros)
    df = df[df["budget"]  > 100_000].copy()   # keep films with reported budgets
    df = df[df["revenue"] > 100_000].copy()   # keep films with reported revenue

    df["budget_m"]           = df["budget"]  / 1e6
    df["worldwide_gross_m"]  = df["revenue"] / 1e6
    df["marketing_budget_m"] = df["budget_m"] * 0.5   # industry heuristic

    # Release month
    df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
    df["release_month"] = df["release_date"].dt.month.fillna(6).astype(int)

    # Genre
    df["genre"] = df["genres"].apply(first_genre)
    df["genre_encoded"] = df["genre"].apply(
        lambda g: GENRES_LIST.index(g) if g in GENRES_LIST else 0)

    # Franchise / sequel detection
    # Strategy (in priority order):
    #   1. belongs_to_collection column (full TMDB export)
    #   2. keywords column contains "sequel" or "based on" (your Kaggle version)
    #   3. Title heuristics (known franchise name in title)
    collection_col = next(
        (c for c in df.columns if "collection" in c.lower()), None
    )
    if collection_col:
        df["is_franchise"] = df[collection_col].notna().astype(float)
        print(f"    Franchise detection: column '{collection_col}'")
    elif "keywords" in df.columns:
        print("    Franchise detection: keywords column + title heuristics")
        FRANCHISE_TITLE_KW = [
            "avengers","spider-man","batman","superman","iron man","thor",
            "star wars","jurassic","mission impossible","transformers",
            "harry potter","lord of the rings","james bond","indiana jones",
            "toy story","finding","incredibles","frozen","despicable",
            "how to train your dragon","shrek","kung fu panda","john wick",
            "fast and furious","fast & furious","furious","godzilla","king kong",
            "x-men","deadpool","hunger games","twilight","pirates of the caribbean",
            "captain america","guardians","black panther","doctor strange",
            "aquaman","wonder woman","justice league","mission: impossible",
        ]
        def _is_franchise(row):
            title = str(row.get("original_title", "")).lower()
            if any(kw in title for kw in FRANCHISE_TITLE_KW):
                return 1.0
            kws = safe_json(row.get("keywords", "[]"))
            kw_names = {k.get("name","").lower() for k in kws}
            franchise_kw_signals = {"sequel","based on novel","based on comic","spin off",
                                    "part of series","franchise","cinematic universe"}
            if kw_names & franchise_kw_signals:
                return 1.0
            return 0.0
        df["is_franchise"] = df.apply(_is_franchise, axis=1)
    else:
        print("    Franchise detection: title heuristics only")
        FRANCHISE_TITLE_KW = [
            "avengers","spider-man","batman","superman","star wars","jurassic",
            "harry potter","james bond","toy story","frozen","hunger games",
        ]
        df["is_franchise"] = df["original_title"].apply(
            lambda t: float(any(kw in str(t).lower() for kw in FRANCHISE_TITLE_KW))
        )

    # Sequel number: keywords "sequel" is the most reliable signal; fall back to title
    def _sequel_number(row):
        n = sequel_number_from_title(str(row.get("original_title", "")))
        if n > 1:
            return float(n)
        if "keywords" in row:
            kws = safe_json(row.get("keywords", "[]"))
            kw_names = {k.get("name","").lower() for k in kws}
            if "sequel" in kw_names:
                return 2.0   # at minimum a sequel; title heuristic couldn't tell us more
        return 1.0

    df["sequel_number"] = df.apply(_sequel_number, axis=1)

    # Screen count — TMDB doesn't have this; use budget-based heuristic
    # (matches distribution of real data reasonably well)
    df["screen_count"] = (
        1500 + df["budget_m"] * 12 + np.random.normal(0, 300, len(df))
    ).clip(100, 4500).round(0)

    # MPAA rating — try 'release_dates' or 'releases' column if present
    rating_col = None
    for c in ["release_dates", "releases"]:
        if c in df.columns:
            rating_col = c
            break

    if rating_col:
        df["mpaa_rating"] = df[rating_col].apply(extract_mpaa)
    else:
        # Fall back to 'original_language' + popularity heuristic
        df["mpaa_rating"] = "PG-13"

    df["mpaa_rating_encoded"] = df["mpaa_rating"].apply(
        lambda r: MPAA_RATINGS.index(r) if r in MPAA_RATINGS else 2)

    # Runtime
    df["runtime"] = pd.to_numeric(df["runtime"], errors="coerce").fillna(105)

    # Trailer views — not in TMDB; estimate from popularity score
    # TMDB popularity ≈ engagement proxy; scale to trailer view range
    df["trailer_views_m"] = (df["popularity"] * 0.8).clip(0.5, 120).round(2)

    # Social sentiment — derived from vote_average (0–10 scale → 0–1)
    df["social_sentiment"] = ((df["vote_average"] - 5) / 5).clip(0.1, 0.98).round(3)

    # RT score — use vote_average as proxy (rescale 0–10 → 0–100)
    df["rt_score"] = (df["vote_average"] * 10).clip(0, 100).round(1)

    # Competition level — random (TMDB has no release calendar data)
    np.random.seed(42)
    df["competition_level"] = np.random.uniform(1, 10, len(df)).round(1)

    # ── Select and clean final columns ────────────────────────────────────────
    FEATURE_COLS = [
        "budget_m", "marketing_budget_m", "release_month", "genre_encoded",
        "is_franchise", "sequel_number", "director_score", "cast_score",
        "screen_count", "mpaa_rating_encoded", "runtime", "trailer_views_m",
        "social_sentiment", "rt_score", "competition_level",
        "worldwide_gross_m",
    ]

    out = df[FEATURE_COLS].copy()

    # Drop rows with any NaN in the core features
    before = len(out)
    out = out.dropna()
    after  = len(out)
    print(f"    Dropped {before - after} rows with missing values")
    print(f"    Final dataset: {after:,} films")

    # Sanity-check distributions
    print("\n📊  Feature summary:")
    print(out.describe().round(1).to_string())

    # Save
    out.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✅  Saved to {OUTPUT_CSV}")
    print("    Now run:  python retrain.py")


if __name__ == "__main__":
    main()