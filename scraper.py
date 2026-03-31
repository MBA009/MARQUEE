"""
scraper.py — Free, no-API-key movie data scraper
=================================================
Sources (all free, no key required):
  1. Wikipedia MediaWiki API  — genre, director, cast, budget, gross, runtime,
                                release date, MPAA rating, franchise info
  2. The Numbers              — budget cross-check, opening-weekend gross,
                                screen count
  3. Director / actor scoring — recursively looks up filmography pages on
                                Wikipedia + The Numbers to compute avg gross

Falls back gracefully: every field is wrapped in try/except and returns None
rather than crashing. The caller (app.py) imputes medians for anything None.
"""

import re
import time
import urllib.parse
from typing import Any, Dict, List, Optional, Tuple

import requests
from bs4 import BeautifulSoup

# ── Session with realistic browser headers ────────────────────────────────────
SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
})
TIMEOUT = 12  # seconds per request

# Genre mapping from Wikipedia category keywords → our 10 genres
GENRE_MAP = {
    "action":       "Action",
    "adventure":    "Action",
    "superhero":    "Action",
    "comedy":       "Comedy",
    "romantic comedy": "Romance",
    "drama":        "Drama",
    "horror":       "Horror",
    "psychological horror": "Horror",
    "science fiction": "SciFi",
    "sci-fi":       "SciFi",
    "animated":     "Animation",
    "animation":    "Animation",
    "romance":      "Romance",
    "romantic":     "Romance",
    "thriller":     "Thriller",
    "mystery":      "Thriller",
    "crime":        "Thriller",
    "fantasy":      "Fantasy",
    "documentary":  "Documentary",
}

MPAA_RATINGS = {"G", "PG", "PG-13", "R", "NC-17"}

# Franchises with strong signal — checked against title/series keywords
KNOWN_FRANCHISES = [
    "avengers", "spider-man", "batman", "superman", "iron man", "thor",
    "captain america", "guardians of the galaxy", "fast and furious", "fast &",
    "star wars", "jurassic", "mission: impossible", "transformers",
    "harry potter", "fantastic beasts", "lord of the rings", "the hobbit",
    "james bond", "indiana jones", "matrix", "terminator", "alien", "predator",
    "toy story", "finding", "incredibles", "frozen", "despicable me",
    "how to train your dragon", "shrek", "kung fu panda", "minions",
    "john wick", "bourne", "x-men", "deadpool", "ant-man", "black panther",
    "doctor strange", "aquaman", "wonder woman", "justice league",
    "pirates of the caribbean", "hunger games", "twilight", "divergent",
    "maze runner", "percy jackson", "godzilla", "king kong",
]


# ─────────────────────────────────────────────────────────────────────────────
# Low-level helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get(url: str, params: dict = None, retries: int = 2) -> Optional[requests.Response]:
    """GET with retry and polite delay."""
    for attempt in range(retries):
        try:
            resp = SESSION.get(url, params=params, timeout=TIMEOUT)
            if resp.status_code == 200:
                return resp
        except Exception:
            pass
        time.sleep(1.5 * (attempt + 1))
    return None


def _parse_money(text: str) -> Optional[float]:
    """
    Extract a dollar amount in millions from messy text like:
      '$150 million'  '€120,000,000'  '$1.2 billion'  'US$85.4 million'
    Returns float (millions) or None.
    """
    if not text:
        return None
    text = text.lower().replace(",", "").replace("\xa0", " ")
    # Remove citation markers like [1]
    text = re.sub(r"\[\d+\]", "", text)

    # Billion
    m = re.search(r"[\$€£]?\s*([\d.]+)\s*billion", text)
    if m:
        return float(m.group(1)) * 1000

    # Million
    m = re.search(r"[\$€£]?\s*([\d.]+)\s*million", text)
    if m:
        return float(m.group(1))

    # Raw large number (≥ 1,000,000)
    m = re.search(r"[\$€£]?\s*(\d{7,})", text)
    if m:
        return float(m.group(1)) / 1_000_000

    return None


def _clean_wikitext(text: str) -> str:
    """Strip wiki markup, templates, and HTML entities from a string."""
    # Remove [[Link|display]] → display
    text = re.sub(r"\[\[(?:[^\]|]*\|)?([^\]]*)\]\]", r"\1", text)
    # Remove {{template|...}} recursively (simple single-level)
    text = re.sub(r"\{\{[^{}]*\}\}", "", text)
    # Remove remaining markup
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"'{2,}", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _extract_runtime(text: str) -> Optional[int]:
    """Extract runtime in minutes from strings like '148 minutes', '2h 28min'."""
    text = text.lower()
    # Try hours+minutes FIRST (e.g. 2h 28min, 2 hours 28 minutes)
    m = re.search(r"(\d+)\s*h(?:r|ours?)?\s*(\d+)\s*m", text)
    if m:
        return int(m.group(1)) * 60 + int(m.group(2))
    # Hours only (e.g. 2 hours)
    m = re.search(r"(\d+)\s*h(?:r|ours?)", text)
    if m:
        return int(m.group(1)) * 60
    # Minutes only (e.g. 148 minutes, 148 mins)
    m = re.search(r"(\d+)\s*min", text)
    if m:
        return int(m.group(1))
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Wikipedia API
# ─────────────────────────────────────────────────────────────────────────────

WIKI_API = "https://en.wikipedia.org/w/api.php"

def wiki_search(title: str) -> Optional[str]:
    """
    Search Wikipedia for a movie title and return the best-matching page title.
    Appends 'film' to the query to prefer film articles.
    """
    resp = _get(WIKI_API, params={
        "action": "query", "list": "search",
        "srsearch": f"{title} film", "format": "json",
        "srlimit": 5, "srnamespace": 0,
    })
    if not resp:
        return None
    hits = resp.json().get("query", {}).get("search", [])
    for hit in hits:
        t = hit["title"].lower()
        if "film" in t or "movie" in t or title.lower().split()[0] in t:
            return hit["title"]
    return hits[0]["title"] if hits else None


def wiki_get_wikitext(page_title: str) -> Optional[str]:
    """Fetch raw wikitext for a Wikipedia page."""
    resp = _get(WIKI_API, params={
        "action": "query", "prop": "revisions",
        "rvprop": "content", "rvslots": "main",
        "format": "json", "titles": page_title,
    })
    if not resp:
        return None
    pages = resp.json().get("query", {}).get("pages", {})
    for page in pages.values():
        try:
            return page["revisions"][0]["slots"]["main"]["*"]
        except (KeyError, IndexError):
            pass
    return None


def wiki_get_html(page_title: str) -> Optional[BeautifulSoup]:
    """Fetch parsed HTML for a Wikipedia page."""
    resp = _get(WIKI_API, params={
        "action": "parse", "page": page_title,
        "prop": "text", "format": "json",
    })
    if not resp:
        return None
    html = resp.json().get("parse", {}).get("text", {}).get("*", "")
    return BeautifulSoup(html, "lxml") if html else None


def _parse_infobox_wikitext(wikitext: str) -> Dict[str, str]:
    """
    Extract key-value pairs from a Wikipedia film infobox in wikitext format.
    Returns a flat dict of {field_name: raw_value}.
    """
    fields: Dict[str, str] = {}
    # Find the infobox block
    m = re.search(r"\{\{Infobox film(.*?)(?=\n\[\[|\n==|\Z)", wikitext,
                  re.DOTALL | re.IGNORECASE)
    if not m:
        return fields

    block = m.group(1)
    # Each field: | key = value (possibly multi-line until next |)
    for match in re.finditer(r"\|\s*([\w\s]+?)\s*=\s*(.*?)(?=\n\s*\||\Z)",
                              block, re.DOTALL):
        key = match.group(1).strip().lower().replace(" ", "_")
        val = match.group(2).strip()
        fields[key] = val

    return fields


def _parse_infobox_html(soup: BeautifulSoup) -> Dict[str, str]:
    """
    Fallback: extract infobox data from Wikipedia's rendered HTML table.
    """
    fields: Dict[str, str] = {}
    table = soup.find("table", class_=re.compile("infobox"))
    if not table:
        return fields
    for row in table.find_all("tr"):
        th = row.find("th")
        td = row.find("td")
        if th and td:
            key = th.get_text(" ", strip=True).lower().replace(" ", "_")
            val = td.get_text(" ", strip=True)
            fields[key] = val
    return fields


def scrape_wikipedia(title: str) -> Dict[str, Any]:
    """
    Main Wikipedia scraper. Returns a dict with extracted fields:
      budget_m, gross_m, runtime, genre, mpaa_rating, release_month,
      director, lead_actor, is_franchise, sequel_number, series_name
    """
    result: Dict[str, Any] = {}

    page_title = wiki_search(title)
    if not page_title:
        return result

    result["wiki_page"] = page_title

    # Try wikitext first (more structured), fall back to HTML
    wikitext = wiki_get_wikitext(page_title)
    if wikitext:
        fields = _parse_infobox_wikitext(wikitext)
    else:
        soup = wiki_get_html(page_title)
        fields = _parse_infobox_html(soup) if soup else {}

    if not fields:
        return result

    # ── Budget ─────────────────────────────────────────────────────────────
    for key in ("budget", "budget_usd"):
        if key in fields:
            v = _parse_money(_clean_wikitext(fields[key]))
            if v:
                result["budget_m"] = round(v, 1)
                break

    # ── Worldwide gross ────────────────────────────────────────────────────
    for key in ("gross", "gross_usd", "box_office"):
        if key in fields:
            v = _parse_money(_clean_wikitext(fields[key]))
            if v:
                result["gross_m"] = round(v, 1)
                break

    # ── Runtime ───────────────────────────────────────────────────────────
    for key in ("runtime", "running_time"):
        if key in fields:
            v = _extract_runtime(_clean_wikitext(fields[key]))
            if v:
                result["runtime"] = v
                break

    # ── Release date → month ──────────────────────────────────────────────
    for key in ("release_date", "released"):
        if key in fields:
            raw = fields[key]
            # Look for 4-digit year and month
            m = re.search(r"\b(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|"
                          r"May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|"
                          r"Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\b", raw, re.I)
            if m:
                months = {"jan":1,"feb":2,"mar":3,"apr":4,"may":5,"jun":6,
                          "jul":7,"aug":8,"sep":9,"oct":10,"nov":11,"dec":12}
                result["release_month"] = months.get(m.group(1)[:3].lower())
            else:
                # Try numeric date
                m2 = re.search(r"\b(\d{4})[|-](\d{1,2})", raw)
                if m2:
                    result["release_month"] = int(m2.group(2))
            if "release_month" in result:
                break

    # ── Genre ─────────────────────────────────────────────────────────────
    for key in ("genre", "genres"):
        if key in fields:
            raw = _clean_wikitext(fields[key]).lower()
            for keyword, mapped in GENRE_MAP.items():
                if keyword in raw:
                    result["genre"] = mapped
                    break
            break

    # ── MPAA rating ───────────────────────────────────────────────────────
    for key in ("rating", "rated", "film_rating", "certification"):
        if key in fields:
            raw = _clean_wikitext(fields[key]).upper()
            for r in MPAA_RATINGS:
                if r in raw.split() or raw.strip() == r:
                    result["mpaa_rating"] = r
                    break
            if "mpaa_rating" in result:
                break

    # ── Director ──────────────────────────────────────────────────────────
    for key in ("director", "directors"):
        if key in fields:
            raw = _clean_wikitext(fields[key])
            # Take first named person
            name = raw.split(",")[0].split("\n")[0].strip()
            if name:
                result["director"] = name
            break

    # ── Lead cast ─────────────────────────────────────────────────────────
    for key in ("starring", "stars", "cast"):
        if key in fields:
            raw = _clean_wikitext(fields[key])
            names = [n.strip() for n in re.split(r"[,\n]", raw) if n.strip()]
            if names:
                result["lead_actor"] = names[0]
                result["cast_list"]  = names[:4]
            break

    # ── Franchise / sequel detection ──────────────────────────────────────
    full_text = (wikitext or "").lower()
    title_lower = title.lower()

    is_franchise = False
    series_name  = None
    for franchise in KNOWN_FRANCHISES:
        if franchise in title_lower or franchise in full_text[:3000]:
            is_franchise = True
            series_name  = franchise.title()
            break

    # Check infobox for series field
    for key in ("series", "franchise", "based_on"):
        if key in fields and fields[key].strip():
            is_franchise = True
            series_name  = _clean_wikitext(fields[key]).split("\n")[0].strip()
            break

    result["is_franchise"] = is_franchise
    if series_name:
        result["series_name"] = series_name

    # Estimate sequel number from title patterns
    sequel_patterns = [
        (r"\b2\b|ii\b|part\s*2|chapter\s*2|returns\b|reloaded\b", 2),
        (r"\b3\b|iii\b|part\s*3|chapter\s*3|revolution\b|rises\b",  3),
        (r"\b4\b|iv\b|part\s*4|chapter\s*4",                        4),
        (r"\b5\b|v\b|part\s*5|chapter\s*5",                         5),
    ]
    sequel_number = 1
    for pattern, num in sequel_patterns:
        if re.search(pattern, title_lower):
            sequel_number = num
            break
    result["sequel_number"] = sequel_number

    return result


# ─────────────────────────────────────────────────────────────────────────────
# The Numbers — budget + screen count cross-check
# ─────────────────────────────────────────────────────────────────────────────

def _the_numbers_slug(title: str) -> str:
    """Convert title to The Numbers URL slug."""
    slug = title.lower()
    slug = re.sub(r"[^a-z0-9\s-]", "", slug)
    slug = re.sub(r"\s+", "-", slug.strip())
    return slug


def scrape_the_numbers(title: str) -> Dict[str, Any]:
    """
    Scrape The Numbers (the-numbers.com) for budget and screen count.
    Returns dict with budget_m (float) and/or screen_count (int).
    """
    result: Dict[str, Any] = {}
    slug = _the_numbers_slug(title)
    url  = f"https://www.the-numbers.com/movie/{slug}#tab=summary"

    resp = _get(url)
    if not resp:
        # Try search
        resp = _get("https://www.the-numbers.com/search", params={"searchterm": title})
        if not resp:
            return result
        soup = BeautifulSoup(resp.text, "lxml")
        link = soup.select_one("div#page_filling_chart a")
        if link:
            resp2 = _get("https://www.the-numbers.com" + link["href"])
            if resp2:
                resp = resp2
            else:
                return result
        else:
            return result

    soup = BeautifulSoup(resp.text, "lxml")

    # Budget row in the summary table
    for row in soup.select("table tr"):
        cells = row.find_all("td")
        if len(cells) >= 2:
            label = cells[0].get_text(strip=True).lower()
            value = cells[1].get_text(strip=True)
            if "production budget" in label:
                v = _parse_money(value)
                if v:
                    result["budget_m"] = round(v, 1)
            elif "widest release" in label or "opening weekend" in label:
                m = re.search(r"([\d,]+)\s*theater", value, re.I)
                if m:
                    result["screen_count"] = int(m.group(1).replace(",", ""))

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Director / actor historical scoring
# ─────────────────────────────────────────────────────────────────────────────

def _get_filmography_grosses(person_name: str, max_films: int = 5) -> List[float]:
    """
    Find a director's or actor's last N films and return their worldwide
    grosses (in $M) by looking up each film on Wikipedia.
    Returns a list of floats (may be empty if nothing found).
    """
    grosses: List[float] = []

    # Search for person's Wikipedia page
    page_title = wiki_search(person_name)
    if not page_title:
        return grosses

    wikitext = wiki_get_wikitext(page_title)
    if not wikitext:
        return grosses

    # Find film titles linked in filmography section
    # Look for lines in sections like "Filmography", "Director", "Actor"
    section_match = re.search(
        r"==\s*(?:Film(?:ography)?|Director|Director(?:ial)?|Acting)\s*==.*?(?===)",
        wikitext, re.DOTALL | re.IGNORECASE
    )
    section_text = section_match.group(0) if section_match else wikitext

    # Extract [[Film Title (year film)|...]] style links
    film_links = re.findall(
        r"\[\[([^\]|]+?\(\d{4}\s*film\))[|\]]",
        section_text, re.IGNORECASE
    )
    # Also plain year-parenthesized links
    film_links += re.findall(
        r"\[\[([A-Z][^\]|]+?\(\d{4}\))[|\]]",
        section_text
    )

    # De-dupe and take last N
    seen = set()
    unique_films = []
    for f in reversed(film_links):
        if f not in seen:
            seen.add(f)
            unique_films.append(f)
        if len(unique_films) >= max_films:
            break

    for film_page in unique_films:
        try:
            wt = wiki_get_wikitext(film_page)
            if not wt:
                continue
            fields = _parse_infobox_wikitext(wt)
            for key in ("gross", "gross_usd", "box_office"):
                if key in fields:
                    v = _parse_money(_clean_wikitext(fields[key]))
                    if v and v > 0.5:
                        grosses.append(v)
                        break
        except Exception:
            continue

    return grosses


def score_person(person_name: str) -> Optional[float]:
    """
    Compute a person's 'score' = average worldwide gross of their
    last 3-5 films. Returns $M float or None.
    """
    if not person_name:
        return None
    try:
        grosses = _get_filmography_grosses(person_name, max_films=5)
        if not grosses:
            return None
        # Weight recent films more heavily
        grosses = sorted(grosses)[-5:]
        return round(sum(grosses) / len(grosses), 1)
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# RT score via Metacritic (more scraper-friendly than RT)
# ─────────────────────────────────────────────────────────────────────────────

def scrape_metacritic_score(title: str) -> Optional[float]:
    """
    Scrape Metacritic for a Metascore (0–100). We use this as a proxy
    for RT critic score since RT blocks scrapers more aggressively.
    Metascore is well-correlated with RT.
    """
    slug = title.lower()
    slug = re.sub(r"[^a-z0-9\s]", "", slug)
    slug = re.sub(r"\s+", "-", slug.strip())

    url  = f"https://www.metacritic.com/movie/{slug}/"
    resp = _get(url)
    if not resp:
        return None

    soup = BeautifulSoup(resp.text, "lxml")

    # Metascore is in a div with class "metascore_w" or similar
    score_el = (
        soup.select_one("span.metascore_w") or
        soup.select_one('[class*="metascore"]') or
        soup.select_one("div.score_number span")
    )
    if score_el:
        try:
            return float(score_el.get_text(strip=True))
        except ValueError:
            pass

    # Try JSON-LD structured data
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            import json
            data = json.loads(script.string)
            rating = (data.get("aggregateRating") or {}).get("ratingValue")
            if rating:
                return float(rating)
        except Exception:
            pass

    return None


# ─────────────────────────────────────────────────────────────────────────────
# Social sentiment proxy via Google Trends / YouTube trailer views
# ─────────────────────────────────────────────────────────────────────────────

def scrape_trailer_views(title: str) -> Optional[float]:
    """
    Estimate YouTube trailer views (in millions) by querying the YouTube
    search page (no API key needed) for the official trailer.
    This scrapes view counts from the search result snippets.
    """
    query = urllib.parse.quote_plus(f"{title} official trailer")
    url   = f"https://www.youtube.com/results?search_query={query}"
    resp  = _get(url)
    if not resp:
        return None

    # YouTube embeds view counts as "X views" in the page source
    matches = re.findall(
        r'"viewCountText":\{"simpleText":"([\d,]+)\s*views"\}',
        resp.text
    )
    if not matches:
        # Fallback pattern
        matches = re.findall(r'"videoViewCountRenderer".*?"([\d,]+)\s*views"',
                             resp.text)
    if matches:
        # Take the first (most relevant) result
        views = int(matches[0].replace(",", ""))
        return round(views / 1_000_000, 2)

    return None


# ─────────────────────────────────────────────────────────────────────────────
# Master search function (called by app.py)
# ─────────────────────────────────────────────────────────────────────────────

def search_movie_features(title: str) -> Dict[str, Any]:
    """
    Orchestrate all scrapers and return a unified feature dict compatible
    with the /api/search response format used by the frontend.

    Returns:
        {
            "title":         str,
            "budget_m":      float | None,
            "marketing_budget_m": float | None,
            "release_month": int | None,
            "genre":         str | None,
            "is_franchise":  bool,
            "sequel_number": int,
            "director_score": float | None,
            "cast_score":    float | None,
            "screen_count":  int | None,
            "mpaa_rating":   str | None,
            "runtime":       int | None,
            "trailer_views_m": float | None,
            "social_sentiment": float | None,
            "rt_score":      float | None,
            "competition_level": None,   # too dynamic to scrape
            "found":         [list of field names found],
            "missing":       [list of field names not found],
            "notes":         str,
        }
    """
    print(f"[scraper] Searching Wikipedia for: {title}")
    wiki_data = scrape_wikipedia(title)

    print(f"[scraper] Wikipedia page: {wiki_data.get('wiki_page', 'not found')}")

    # Cross-check The Numbers for budget / screen count
    tn_data: Dict[str, Any] = {}
    if wiki_data.get("wiki_page"):
        print("[scraper] Checking The Numbers …")
        tn_data = scrape_the_numbers(title)

    # Critic score
    rt_score = None
    print("[scraper] Checking Metacritic …")
    mc_score = scrape_metacritic_score(title)
    if mc_score:
        rt_score = mc_score  # Metascore ≈ RT; both ~0–100

    # Trailer views
    trailer_views = None
    print("[scraper] Checking YouTube trailer views …")
    trailer_views = scrape_trailer_views(title)

    # Director score
    director       = wiki_data.get("director")
    director_score = None
    if director:
        print(f"[scraper] Scoring director: {director}")
        director_score = score_person(director)

    # Cast score
    lead_actor  = wiki_data.get("lead_actor")
    cast_score  = None
    if lead_actor:
        print(f"[scraper] Scoring lead actor: {lead_actor}")
        cast_score = score_person(lead_actor)

    # Marketing budget heuristic (industry avg ≈ 50% of production budget)
    budget_m           = wiki_data.get("budget_m") or tn_data.get("budget_m")
    marketing_budget_m = round(budget_m * 0.5, 1) if budget_m else None

    # Social sentiment proxy: trailer-views-based (high view count → positive buzz)
    social_sentiment = None
    if trailer_views:
        # Normalise: 100M+ views → ~0.90, 10M → ~0.70, 1M → ~0.55
        import math
        score = 0.45 + 0.12 * math.log10(max(trailer_views, 0.1) + 1)
        social_sentiment = round(min(score, 0.98), 2)

    # Assemble output
    output = {
        "title":              title,
        "budget_m":           budget_m,
        "marketing_budget_m": marketing_budget_m,
        "release_month":      wiki_data.get("release_month"),
        "genre":              wiki_data.get("genre"),
        "is_franchise":       wiki_data.get("is_franchise", False),
        "sequel_number":      wiki_data.get("sequel_number", 1),
        "director_score":     director_score,
        "cast_score":         cast_score,
        "screen_count":       wiki_data.get("screen_count") or tn_data.get("screen_count"),
        "mpaa_rating":        wiki_data.get("mpaa_rating"),
        "runtime":            wiki_data.get("runtime"),
        "trailer_views_m":    trailer_views,
        "social_sentiment":   social_sentiment,
        "rt_score":           rt_score,
        "competition_level":  None,   # requires release-calendar knowledge
    }

    # Determine found / missing
    skip_keys = {"title"}
    found   = [k for k, v in output.items() if k not in skip_keys and v is not None]
    missing = [k for k, v in output.items() if k not in skip_keys and v is None]

    notes_parts = []
    if wiki_data.get("wiki_page"):
        notes_parts.append(f"Wikipedia: {wiki_data['wiki_page']}")
    if director:
        notes_parts.append(f"Director: {director}")
    if lead_actor:
        notes_parts.append(f"Lead: {lead_actor}")
    if wiki_data.get("series_name"):
        notes_parts.append(f"Series: {wiki_data['series_name']}")
    if mc_score:
        notes_parts.append(f"Metascore: {mc_score}")

    output["found"]   = found
    output["missing"] = missing
    output["notes"]   = " · ".join(notes_parts) or "Scraped from public sources"

    print(f"[scraper] Done. Found {len(found)} fields, missing {len(missing)}")
    return output
