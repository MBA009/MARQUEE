# 🎬 MARQUEE — Movie Box Office Predictor
### Zero API keys required to run

An end-to-end ML system that predicts a film's worldwide box-office gross.
Feature extraction uses free public web sources. (Only the optional
training-data download, `fetch_tmdb.py`, needs a free TMDB key.)

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  Browser UI (index.html)                                        │
│    ① Enter movie title → pick the film if several match        │
│    ② Review auto-scraped features  (edit / fill gaps)          │
│    ③ Run Prediction → gross + range + what drove it            │
│       (+ actual gross for released films)                       │
└────────────────────┬────────────────────────────────────────────┘
                     │ HTTP / JSON
┌────────────────────▼────────────────────────────────────────────┐
│  FastAPI  (app.py)                                              │
│                                                                 │
│  POST /api/candidates → films matching a title (Wikidata)       │
│  POST /api/search   →  scraper.py (see below)                   │
│  POST /api/predict  →  XGBoost  →  gross + 80% PI + SHAP       │
│  GET  /api/model-info                                           │
│  POST /api/retrain                                              │
└────────────────────┬────────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────────┐
│  scraper.py  — 3 free data sources, zero keys                   │
│                                                                 │
│  1. Wikipedia MediaWiki API                                     │
│       budget, gross, runtime, genre, director,                  │
│       cast, release date, franchise/sequel                      │
│                                                                 │
│  2. Wikidata SPARQL (+ Wikipedia billing check)                 │
│       film search; director & lead actor avg gross              │
│                                                                 │
│  3. The Numbers (the-numbers.com)                               │
│       budget, when Wikipedia has none                           │
└────────────────────┬────────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────────┐
│  ML Model  (model_bundle.pkl)                                   │
│    • XGBoost (800 trees, depth 5), log-transformed target       │
│    • 9 pre-release features, 4,393 films (1990–2026):           │
│      2,687 Kaggle TMDB 5000 + 1,706 TMDB API                    │
│    • Held-out (2023–2026 films): R² ≈ 0.51, Median APE ≈ 59%,  │
│      53% of predictions within 2× of actual                     │
│    • Second model: per-film 80% range (quantile + calibration)  │
└─────────────────────────────────────────────────────────────────┘
```

## What you need to run it

| Needed | What | Notes |
|--------|------|-------|
| Always | Python 3.13 and `pip install -r requirements.txt` | Versions are pinned to the ones the saved model was made with |
| Always | `app.py`, `features.py`, `retrain.py`, `scraper.py`, `franchise_lookup.py`, `index.html` | The app imports all of these |
| Always | `model_bundle.pkl` (or `tmdb_features.csv` to train one at startup) | Without either, the app trains on synthetic data, which is only a demo |
| For auto-fill | Internet access | Wikipedia, Wikidata, The Numbers. Without it you can still type the features in by hand |
| Only to retrain | `load_tmdb.py`, `tmdb_5000_*.csv`, and optionally `fetch_tmdb.py` + a TMDB key | See [Training on real data](#training-on-real-data) |

No API key is needed to run the predictor. Generated files that can be
deleted and rebuilt: `.wiki_cache.sqlite`, `.tmdb_cache/`,
`training_films.csv`, and `tmdb_api_*.csv` (re-download with
`fetch_tmdb.py`).

## Quick start

```bash
# 1. Install (no API keys needed!)
pip install -r requirements.txt

# 2. Start the server
python app.py

# 3. Open the browser
open http://localhost:8000
```

The server listens on localhost only. To reach it from other devices, change
`host` in the `uvicorn.run` call at the bottom of `app.py` to `"0.0.0.0"`.

If `model_bundle.pkl` is missing (or from an older version), the server
trains on `tmdb_features.csv` (~15s), falling back to synthetic data only if
that file is missing. Subsequent starts load the cached bundle instantly.

---

## What changed from the original version

The original reported R² ≈ 0.69, but most of that came from information only
available *after* a film's release. The app also computed its inputs
differently from the training data, and the scraper filled in few fields.
The changes below make the accuracy figures honest, then make the live
predictions match how the model was trained.

**1. Honest model**
- Removed seven features: three leaked post-release data, three were
  synthetic noise and one was constant (details under
  [Features that were removed](#features-that-were-removed)). 16 → 9
  features. Held-out R² went from ~0.65 to ~0.51; the lower number is the
  honest one.
- Evaluation now uses a time-based split instead of a random one (see
  [Training and test split](#training-and-test-split)).
- `POST /api/retrain` used to replace the real model with one trained on
  synthetic data; it now retrains on the real data.
- The "Model Accuracy" figure (which was just 1 − median error) was replaced
  with "Within 2× of actual".

**2. Same features in training and in the app** (`features.py`)
- The budget you enter is converted to 2024 dollars, as in training
  (official BLS CPI figures). It used to be fed in unadjusted.
- Director and lead-actor scores now use the training definition: average
  gross of the person's last 5 films *before* this one, in 2024 dollars,
  counting only films they directed or were top-billed in. The app used to
  average their 5 *highest-grossing* films, in unadjusted dollars, sometimes
  including the film itself.
- Release year and month come from the infobox release date, and genre
  from the article's first sentence. The year used to default to the
  current year, and genre was almost never found.
- Infobox parsing was rewritten: nested templates, cast lists, budget
  ranges, `&nbsp;`, and non-USD amounts. Many films previously came back
  with no fields at all.
- The app warns when a film's release year is outside the training data.

**3. Per-film ranges and explanations**
- The fixed ×0.26–×3.1 band was replaced by a model that predicts each
  film's own 80% range. The fixed band caught only 65% of small films and
  was needlessly wide for big ones.
- The chart now shows each feature's effect as a multiplier on a typical
  film (SHAP values), e.g. "budget ×5.1", with increases and decreases in
  different colours.

**4. Usability**
- A film picker appears when several films share a title (*Dune* 1984 vs
  2021).
- Released films show their actual gross next to the prediction.
- Searches no longer block the server and are faster: lookups run in
  parallel, results are cached, and the unused Metacritic and YouTube
  scrapes were dropped.
- When a site rate-limits the scraper, it stops waiting and the status line
  says which fields may be missing as a result.
- Form inputs and the API both validate ranges and show readable errors.
- The server listens on localhost only and no longer auto-reloads.
- The scripts no longer crash on Windows consoles that can't show emoji.

**5. Newer training data**
- `fetch_tmdb.py` downloads films from 2017 onward from the TMDB API, in
  the same format as the Kaggle data. See
  [Training on real data](#training-on-real-data).
- The test and calibration years now move with the data, and franchise
  detection uses English titles to match the live scraper.
- Every retrain records exactly which films it used: `training_films.csv`,
  a summary in the model, the app header, and a per-film note in results.
  See [Which data was the model trained on?](#which-data-was-the-model-trained-on).
- `requirements.txt` now pins the versions the model was actually built
  with. The old pins (e.g. numpy 1.26) didn't install on Python 3.13 and
  couldn't load the saved model.

---

## Data sources used by the scraper

| Source | Fields | Notes |
|--------|--------|-------|
| Wikipedia MediaWiki API | budget, actual gross, runtime, genre, director, cast, release date, franchise | Free, no key; rate-limited |
| Wikidata SPARQL | Film search (picker); director/actor film history with box office | One query each |
| The Numbers | Budget, only when Wikipedia has none | Public site, politely scraped |

Metacritic and YouTube were dropped: the model no longer uses critic scores
or trailer views (see "Features that were removed" below).

For released films, the result shows the actual worldwide gross from
Wikipedia next to the prediction, and says whether the film was in the
training data. Only films that weren't are a true test of the model.

---

## Scraper robustness

- Every field is wrapped in `try/except` — partial failure never crashes the search
- Each HTTP request has up to 3 tries with back-off; a long `Retry-After`
  (rate limiting) skips that site for the cooldown instead of waiting, and
  the search notes say so
- Wikipedia pages and Wikidata results are cached for 7 days in
  `.wiki_cache.sqlite` (safe to delete). Wikimedia rate-limits
  unauthenticated clients, so a new search uses about 2 Wikipedia requests
  and 2 Wikidata queries, and a repeated one uses none
- The director/actor lookup and The Numbers run in parallel, and searches
  don't block the server (other requests are answered meanwhile)
- Fields that can't be found return `None` and are flagged as "ESTIMATED" in the UI
- The model imputes medians for all missing features so prediction always works

---

## Feature list

| Feature | Source | Auto-found? |
|---------|--------|-------------|
| Production budget (release-year $; inflation-adjusted by the server) | Wikipedia / The Numbers | ✅ Usually |
| Release month | Wikipedia infobox (US release preferred) | ✅ Usually |
| Release year | Wikipedia infobox | ✅ Usually |
| Genre | Wikipedia lead sentence ("is a 2010 science fiction action film") | ✅ Usually |
| Franchise / Sequel | `franchise_lookup.py` + title heuristics | ✅ Usually |
| Director avg gross (last 5 prior films, 2024 $) | Wikidata | ✅ Usually |
| Lead actor avg gross (last 5 prior top-billed films, 2024 $) | Wikidata + Wikipedia | ⚠️ Sometimes |
| Runtime | Wikipedia | ✅ Usually |

Missing features are median-imputed. Predictions are in 2024 USD.

Training (`load_tmdb.py`) and the scraper compute features the same way,
using the shared definitions in `features.py`:

- **Inflation**: BLS CPI-U annual averages, converted to 2024 dollars.
- **Director / lead-actor score**: mean worldwide gross of the person's 5
  most recent films released *before* the film being predicted. For actors,
  only films where they were top-billed count, since TMDB's lead actor is
  the first cast member. If Wikipedia is too rate-limited to check billing,
  the actor score is left blank rather than counting supporting roles.

The model treats release years after its newest training data like the
last year it has, and the UI says so. With only the Kaggle data that is
2016; run `fetch_tmdb.py` (below) to extend it.

### Features that were removed

Earlier versions used 16 features. Seven were dropped because they made the
reported accuracy look better than it was, without helping real predictions:

- **Trailer views, social sentiment, critic score**: in the TMDB training
  data these were derived from `popularity` and `vote_average`, which are
  measured *after* release, so they leaked the answer. Removing them drops
  held-out R² from ~0.65 to ~0.51. The lower number is the honest one.
- **Marketing budget** (always 0.5 × budget), **screen count** and
  **competition level** (formula + random noise): no real information.
- **MPAA rating**: the TMDB 5000 CSV has no ratings, so it was PG-13 for
  every film.

---

## Training on real data

```bash
python fetch_tmdb.py  # optional: films from 2017 on, via the TMDB API (needs a key)
python load_tmdb.py   # tmdb_5000_*.csv (+ tmdb_api_*.csv) → tmdb_features.csv
python retrain.py     # → model_bundle.pkl, prints held-out metrics
```

The bundled Kaggle TMDB 5000 data stops in 2016. `fetch_tmdb.py` adds the
highest-grossing films of each later year (300 per year by default; films
released in the last 6 months are skipped because their gross is still
growing). It writes them in the Kaggle format, so `load_tmdb.py` builds
features for old and new films with the same code, and API rows replace
Kaggle rows for the same film.

It needs a free TMDB key (https://www.themoviedb.org/settings/api). Only
this script uses it; the app stays keyless. Set `TMDB_API_KEY` (or the v4
`TMDB_READ_TOKEN`) in the environment, or put it in a `.env` file:

```
TMDB_API_KEY=your_key_here
```

`.env` is in `.gitignore`. Responses are cached in `.tmdb_cache/`, so an
interrupted run resumes where it stopped.

*This product uses the TMDB API but is not endorsed or certified by TMDB.*

`retrain.py` holds the training procedure used by both the CLI and the
server (`POST /api/retrain`); the feature definitions live in `features.py`.

### Training and test split

The split is by **release year**, not random, because the model is used to
predict films that come out after the ones it learned from. A random split
would let it learn from 2025 films and be graded on 2023 ones.

How the years are chosen: the **test set is the latest 4 years** in the
data, and **calibration is the 3 years before them**. 2020–2021 stay in
training but are never test or calibration years, because pandemic-era box
office would distort both. The split is recomputed on every retrain.

Current model (4,393 films, 1990–2026):

| Set | Years | Films | Used for |
|-----|-------|-------|----------|
| Training | 1990–2022 | 3,783 | Fitting the gross model |
| Test | 2023–2026 | 610 | All reported accuracy figures (R², median error, within 2×, range coverage) |

The range model splits the training years once more:

| Set | Years | Films | Used for |
|-----|-------|-------|----------|
| Fit | 1990–2017, 2020–2021 | 3,197 | Fitting the 10th/90th-percentile model |
| Calibration | 2018–2019, 2022 | 586 | Widening the ranges until they cover 80% of these films |
| Test | 2023–2026 | 610 | Checking coverage: the ranges catch 84.9% |

With only the Kaggle data (1990–2016), the same rules give test 2013–2016
(467 films) and calibration 2010–2012.

Things to know:

- **The deployed models are refit on all films**, test years included,
  once the figures are measured. The reported accuracy comes from the
  models that never saw the test films. The deployed ones have seen slightly
  more data, so the figures are, if anything, a little pessimistic.
- **There is no separate validation set for tuning.** The gross model's
  settings were kept from the original project. The range model's tree
  depth was chosen by comparing a few options on the 2013–2016 films, which
  makes its coverage figures slightly optimistic.
- **The ranges are currently on the cautious side**: they catch 85% of test
  films rather than 80%, with a median width of ×20 from low to high. The
  calibration years include 2022, when cinemas were still recovering, so
  the model learned to allow for more surprise than 2023–2026 needed.
  Narrow ranges still go to big franchise films and wide ones to small films.

### Which data was the model trained on?

Every retrain records this in three places:

- **`training_films.csv`**: one row per film with its TMDB id, title,
  release year, source (`kaggle` or `tmdb_api`) and role (`train`,
  `calibration` or `test`). Open it in a spreadsheet to check a film.
- **The app header** shows the total, year span and source counts, e.g.
  "Trained: 4,393 films, 1990–2026 (2,687 Kaggle + 1,706 TMDB API)".
  `GET /api/model-info` returns the same summary plus when it was trained.
- **Each result** for a released film says whether that film was in the
  training data, using title and release year. Only films that weren't
  are a true test.

The sources: `tmdb_5000_*.csv` (Kaggle, films up to 2016) and
`tmdb_api_*.csv` (written by `fetch_tmdb.py`, 2017 onward). A film released
after the last retrain is new to the model. To include newer films, run
`fetch_tmdb.py`, then `load_tmdb.py` and `retrain.py`.

Each prediction is explained with XGBoost's SHAP values. The model works in
log dollars, so each feature's effect is a multiplier on a typical film
(about $93M): "budget ×5.1, franchise ×1.6, …" multiply out to the estimate.

---

## Tips

- A search takes about 2–10 seconds; repeat searches are near-instant thanks to the cache
- For well-documented films (big studio releases), expect all 9 model fields auto-filled
- For obscure/indie films, expect fewer; the rest will be median-imputed
- If the status line says a site rate-limited the search, wait a minute and search again
- You can always manually edit any field in the UI before running prediction
