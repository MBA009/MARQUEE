"""
fetch_tmdb.py — Download newer films from the TMDB API for training
===================================================================
The Kaggle TMDB 5000 dataset stops in 2016. This script fetches the
highest-grossing films of later years from the TMDB API and writes them in
the same format as the Kaggle CSVs, so load_tmdb.py builds features for both
with identical code:

    tmdb_api_movies.csv   (columns as tmdb_5000_movies.csv)
    tmdb_api_credits.csv  (columns as tmdb_5000_credits.csv)

Only training needs a key; the app itself stays keyless. Get a free key at
https://www.themoviedb.org/settings/api and provide either
    TMDB_API_KEY      — the "API Key" (v3), or
    TMDB_READ_TOKEN   — the "API Read Access Token" (v4)
as an environment variable or a line `TMDB_API_KEY=...` in a .env file.

Usage:
    python fetch_tmdb.py                 # 2017 → last year, 300 films/year
    python fetch_tmdb.py --from-year 2015 --per-year 400
    python load_tmdb.py && python retrain.py

Responses are cached in .tmdb_cache/, so an interrupted run resumes.
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import requests

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")

API         = "https://api.themoviedb.org/3"
CACHE_DIR   = Path(".tmdb_cache")
MOVIES_OUT  = Path("tmdb_api_movies.csv")
CREDITS_OUT = Path("tmdb_api_credits.csv")
ENV_FILE    = Path(".env")
PAGE_SIZE   = 20      # TMDB discover results per page
MIN_AGE_DAYS = 180    # skip films still in release: their gross is incomplete
WORKERS     = 8       # TMDB allows ~40 requests/second


def _credentials() -> Dict[str, Any]:
    """Request auth from TMDB_API_KEY / TMDB_READ_TOKEN (environment or .env)."""
    values = dict(os.environ)
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
            key, sep, val = line.partition("=")
            if sep and key.strip() not in values:
                values[key.strip()] = val.strip().strip('"').strip("'")
    if values.get("TMDB_READ_TOKEN"):
        return {"headers": {"Authorization": f"Bearer {values['TMDB_READ_TOKEN']}"}, "params": {}}
    if values.get("TMDB_API_KEY"):
        return {"headers": {}, "params": {"api_key": values["TMDB_API_KEY"]}}
    sys.exit("❌  No TMDB credentials. Set TMDB_API_KEY (or TMDB_READ_TOKEN) in the "
             "environment or in a .env file — see the top of fetch_tmdb.py.")


SESSION = requests.Session()


def _get(path: str, auth: Dict[str, Any], **params) -> Optional[dict]:
    for attempt in range(5):
        try:
            resp = SESSION.get(f"{API}{path}", params={**auth["params"], **params},
                               headers=auth["headers"], timeout=20)
        except requests.RequestException:
            time.sleep(2 * (attempt + 1))
            continue
        if resp.status_code == 200:
            return resp.json()
        if resp.status_code == 401:
            sys.exit("❌  TMDB rejected the credentials (401). Check the key.")
        if resp.status_code == 404:
            return None
        wait = float(resp.headers.get("Retry-After", 2 * (attempt + 1)))
        time.sleep(min(wait, 30))
    return None


def discover_ids(year: int, count: int, until: date, auth) -> List[int]:
    """Film ids released in `year` (and before `until`), highest revenue first."""
    ids: List[int] = []
    last = min(date(year, 12, 31), until)
    for page in range(1, (count + PAGE_SIZE - 1) // PAGE_SIZE + 1):
        data = _get("/discover/movie", auth, page=page, sort_by="revenue.desc",
                    include_adult="false",
                    **{"primary_release_date.gte": f"{year}-01-01",
                       "primary_release_date.lte": last.isoformat()})
        if not data or not data.get("results"):
            break
        ids += [m["id"] for m in data["results"]]
        if page >= data.get("total_pages", page):
            break
    return ids[:count]


def movie_details(movie_id: int, auth) -> Optional[dict]:
    """Film details + credits + keywords, cached on disk."""
    path = CACHE_DIR / f"{movie_id}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    data = _get(f"/movie/{movie_id}", auth, append_to_response="credits,keywords")
    if data:
        path.write_text(json.dumps(data), encoding="utf-8")
    return data


def to_kaggle_rows(d: dict) -> Dict[str, dict]:
    """One API response → rows shaped like tmdb_5000_movies / _credits."""
    pick = lambda items, *keys: json.dumps([{k: i.get(k) for k in keys} for i in items])
    movie = {
        "id":             d["id"],
        "title":          d.get("title"),
        "original_title": d.get("original_title"),
        "budget":         d.get("budget") or 0,
        "revenue":        d.get("revenue") or 0,
        "release_date":   d.get("release_date"),
        "runtime":        d.get("runtime"),
        "genres":         pick(d.get("genres", []), "id", "name"),
        "keywords":       pick(d.get("keywords", {}).get("keywords", []), "id", "name"),
        "popularity":     d.get("popularity"),
        "vote_average":   d.get("vote_average"),
        "vote_count":     d.get("vote_count"),
        "original_language": d.get("original_language"),
    }
    credits = d.get("credits", {})
    credit = {
        "movie_id": d["id"],
        "title":    d.get("title"),
        "cast":     pick(sorted(credits.get("cast", []), key=lambda c: c.get("order", 999)),
                         "name", "order", "character"),
        "crew":     pick(credits.get("crew", []), "name", "job", "department"),
    }
    return {"movie": movie, "credit": credit}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--from-year", type=int, default=2017)
    parser.add_argument("--to-year",   type=int, default=date.today().year)
    parser.add_argument("--per-year",  type=int, default=300,
                        help="highest-grossing films fetched per year")
    args = parser.parse_args()

    auth  = _credentials()
    until = date.today() - timedelta(days=MIN_AGE_DAYS)
    CACHE_DIR.mkdir(exist_ok=True)

    ids: List[int] = []
    for year in range(args.from_year, min(args.to_year, until.year) + 1):
        year_ids = discover_ids(year, args.per_year, until, auth)
        print(f"📅  {year}: {len(year_ids)} films")
        ids += year_ids
    ids = list(dict.fromkeys(ids))

    print(f"⬇  Fetching details for {len(ids):,} films (cached in {CACHE_DIR}/) …")
    rows = []
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        for i, d in enumerate(pool.map(lambda m: movie_details(m, auth), ids), 1):
            if d:
                rows.append(to_kaggle_rows(d))
            if i % 250 == 0:
                print(f"    {i:,}/{len(ids):,}")

    movies  = pd.DataFrame([r["movie"] for r in rows])
    credits = pd.DataFrame([r["credit"] for r in rows])
    movies.to_csv(MOVIES_OUT, index=False)
    credits.to_csv(CREDITS_OUT, index=False)
    usable = int(((movies["budget"] > 500_000) & (movies["revenue"] > 500_000)).sum()) if len(movies) else 0
    print(f"\n✅  {len(movies):,} films → {MOVIES_OUT}, {CREDITS_OUT}")
    print(f"    {usable:,} have both budget and revenue (usable for training)")
    print("    Next: python load_tmdb.py && python retrain.py")


if __name__ == "__main__":
    main()
