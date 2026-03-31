# 🎬 MARQUEE — Movie Box Office Predictor
### Zero API keys required

An end-to-end ML system that predicts a film's worldwide box-office gross.
Feature extraction uses free public web scraping.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  Browser UI (index.html)                                        │
│    ① Enter movie title                                         │
│    ② Review auto-scraped features  (edit / fill gaps)          │
│    ③ Click "Run Prediction"  →  gross + feature chart          │
└────────────────────┬────────────────────────────────────────────┘
                     │ HTTP / JSON
┌────────────────────▼────────────────────────────────────────────┐
│  FastAPI  (app.py)                                              │
│                                                                 │
│  POST /api/search   →  scraper.py (see below)                   │
│  POST /api/predict  →  GBR model  →  gross + CI + contributions │
│  GET  /api/model-info                                           │
│  POST /api/retrain                                              │
└────────────────────┬────────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────────┐
│  scraper.py  — 5 free data sources, zero keys                   │
│                                                                 │
│  1. Wikipedia MediaWiki API                                     │
│       budget, gross, runtime, genre, director,                  │
│       cast, MPAA rating, release date, franchise/sequel         │
│                                                                 │
│  2. The Numbers (the-numbers.com)                               │
│       budget cross-check, opening screen count                  │
│                                                                 │
│  3. Metacritic (scrape)                                         │
│       Metascore as proxy for RT score                           │
│                                                                 │
│  4. YouTube (scrape)                                            │
│       Official trailer view count                               │
│                                                                 │
│  5. Wikipedia filmography pages (recursive)                     │
│       Director & lead actor historical avg gross                │
└────────────────────┬────────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────────┐
│  ML Model  (model_bundle.pkl)                                   │
│    • GradientBoostingRegressor  (500 trees, depth 4)            │
│    • 15 features                                                │
│    • Trained on 6,000 synthetic films                           │
│    • Log-transformed target  →  R² ≈ 0.66, Median APE ≈ 39%   │
└─────────────────────────────────────────────────────────────────┘
```

## Quick start

```bash
# 1. Install (no API keys needed!)
pip install -r requirements.txt

# 2. Start the server
python app.py

# 3. Open the browser
open http://localhost:8000
```

First launch generates synthetic training data and trains the model (~15s).
Subsequent starts load the cached `model_bundle.pkl` instantly.

---

## Data sources used by the scraper

| Source | Fields | Notes |
|--------|--------|-------|
| Wikipedia MediaWiki API | budget, gross, runtime, genre, director, cast, rating, release date, franchise | Free, no key, very reliable |
| The Numbers | budget cross-check, screen count | Public site, politely scraped |
| Metacritic | Critic score (0–100) | Used as RT score proxy |
| YouTube | Trailer view count | Page scrape, view count in source |
| Wikipedia filmography | Director/actor avg gross | Recursive page lookups |

**Marketing budget** is estimated at 50% of production budget (industry average).  
**Social sentiment** is derived from trailer view count using a log-scaled formula.  
**Competition level** is not scrapeable and will always be median-imputed.

---

## Scraper robustness

- Every field is wrapped in `try/except` — partial failure never crashes the search
- Each HTTP request has 2 retries with polite back-off delays
- Fields that can't be found return `None` and are flagged as "ESTIMATED" in the UI
- The model imputes medians for all missing features so prediction always works

---

## Feature list

| Feature | Source | Auto-found? |
|---------|--------|-------------|
| Production budget | Wikipedia / The Numbers | ✅ Usually |
| Marketing budget | Estimated (50% of budget) | ✅ Always (if budget known) |
| Release month | Wikipedia | ✅ Usually |
| Genre | Wikipedia | ✅ Usually |
| Franchise / Sequel | Wikipedia + title heuristics | ✅ Usually |
| Director avg gross | Wikipedia filmography | ⚠️ Sometimes |
| Cast avg gross | Wikipedia filmography | ⚠️ Sometimes |
| Screen count | The Numbers | ⚠️ Sometimes |
| MPAA rating | Wikipedia | ✅ Usually |
| Runtime | Wikipedia | ✅ Usually |
| Trailer views | YouTube scrape | ⚠️ Sometimes |
| Social sentiment | Derived from trailer views | ⚠️ If trailer views found |
| Critic score | Metacritic | ⚠️ Sometimes |
| Competition level | — | ❌ Always imputed |

---

## Using real training data

The model ships with synthetic data so it works out of the box.
To train on real historical films:

1. Download the [TMDB 5000 Movie Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata) (free Kaggle account required)
2. Map TMDB columns to the 15 features in `app.py`
3. Replace `generate_dataset()` with a CSV loader
4. Hit `POST /api/retrain` to retrain

---

## Tips

- The scraper takes 10–30 seconds per movie (multiple web requests, filmography lookups)
- For well-documented films (big studio releases), expect 10–13 fields auto-filled
- For obscure/indie films, expect 4–6 fields; the rest will be median-imputed
- You can always manually edit any field in the UI before running prediction
