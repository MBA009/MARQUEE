"""
scraper.py — Free, no-API-key movie data scraper
=================================================
Sources (all free, no key required):
  1. Wikipedia MediaWiki API  — genre, director, cast, budget, gross, runtime,
                                release date, franchise info
  2. Wikidata                 — film search (to pick between same-named
                                films) and director / actor film histories
  3. The Numbers              — budget, when Wikipedia has none

Model inputs follow the definitions in features.py, the same ones used to
build the training data:
  • budget_m is returned in release-year dollars; app.py inflation-adjusts it.
  • director_score / cast_score = average worldwide gross (2024 USD) of the
    person's last 5 films released *before* this one, counting only films
    they directed / were top-billed in.

Falls back gracefully: every field is wrapped in try/except and returns None
rather than crashing. The caller (app.py) imputes medians for anything None.
"""

import html
import json
import re
import sqlite3
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import requests
from bs4 import BeautifulSoup

from features import cpi_adjust, person_score
from franchise_lookup import (
    franchise_name as fl_franchise_name,
    is_franchise as fl_is_franchise,
    sequel_number as fl_sequel_number,
)

# ── Session with realistic browser headers (non-Wikipedia sites) ──────────────
SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
})
# Wikimedia asks API clients to identify themselves and throttles those that don't
WIKI_HEADERS = {
    "User-Agent": "MarqueeBoxOfficePredictor/1.0 (educational box-office model; python-requests)",
}
TIMEOUT = 12  # seconds per request

# Genre keywords → our 10 genres. A film's genre is the keyword that appears
# first in its Wikipedia description ("is a 2023 epic biographical thriller
# film" → Drama); at the same position the longer keyword wins.
GENRE_KEYWORDS = {
    "romantic comedy": "Romance",
    "science fiction": "SciFi",
    "sci-fi":          "SciFi",
    "space opera":     "SciFi",
    "superhero":       "Action",
    "action":          "Action",
    "adventure":       "Action",
    "martial arts":    "Action",
    "war":             "Action",
    "western":         "Action",
    "comedy":          "Comedy",
    "animated":        "Animation",
    "animation":       "Animation",
    "horror":          "Horror",
    "slasher":         "Horror",
    "thriller":        "Thriller",
    "mystery":         "Thriller",
    "crime":           "Thriller",
    "heist":           "Thriller",
    "fantasy":         "Fantasy",
    "documentary":     "Documentary",
    "romance":         "Romance",
    "romantic":        "Romance",
    "drama":           "Drama",
    "biographical":    "Drama",
    "historical":      "Drama",
    "musical":         "Drama",
    "coming-of-age":   "Drama",
}

MPAA_RATINGS = {"G", "PG", "PG-13", "R", "NC-17"}

MONTHS = {"jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
          "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12}


# ─────────────────────────────────────────────────────────────────────────────
# Low-level helpers
# ─────────────────────────────────────────────────────────────────────────────

MAX_RETRY_WAIT = 10  # seconds; longer Retry-After → skip the host until then
_cooldown_until: Dict[str, float] = {}   # host → time it may be retried
RATE_LIMITED_HOSTS: set = set()         # hosts that throttled us (for UI notes)


def _get(url: str, params: dict = None, headers: dict = None,
         retries: int = 3) -> Optional[requests.Response]:
    """GET with retry, honouring Retry-After on 429/503."""
    host = urllib.parse.urlparse(url).netloc
    if time.time() < _cooldown_until.get(host, 0):
        RATE_LIMITED_HOSTS.add(host)
        return None
    for attempt in range(retries):
        try:
            resp = SESSION.get(url, params=params, headers=headers, timeout=TIMEOUT)
        except requests.RequestException:
            resp = None
        if resp is not None:
            if resp.status_code == 200:
                return resp
            if resp.status_code == 404:
                return None
        if attempt == retries - 1:
            break
        wait = 1.5 * (attempt + 1)
        if resp is not None and resp.status_code in (429, 503):
            RATE_LIMITED_HOSTS.add(host)
            try:
                wait = float(resp.headers.get("Retry-After", wait))
            except ValueError:
                pass
            if wait > MAX_RETRY_WAIT:
                _cooldown_until[host] = time.time() + wait
                return None
        time.sleep(wait)
    return None


_MONEY = re.compile(
    r"\$\s*(\d+(?:\.\d+)?)(?:\s*(?:–|-|to)\s*\$?\s*(\d+(?:\.\d+)?))?\s*(million|billion)?")
_MONEY_NO_SYMBOL = re.compile(
    r"(\d+(?:\.\d+)?)(?:\s*(?:–|-|to)\s*(\d+(?:\.\d+)?))?\s*(million|billion)")
_FOREIGN_CURRENCY = re.compile(
    r"[₹£€¥]|\b(?:crore|lakh|rupees?|yen|yuan|won|euros?|pounds?|rmb|inr)\b")


def _parse_money(text: str) -> Optional[float]:
    """
    Extract a US-dollar amount in millions from messy text like:
      '$150 million'  '$1.2 billion'  'US$85.4 million'  '$356–400 million'
    Ranges return their midpoint. Amounts only in other currencies return
    None rather than being mistaken for dollars.
    """
    if not text:
        return None
    text = text.lower().replace(",", "").replace("\xa0", " ")
    # Remove citation markers like [1]
    text = re.sub(r"\[\d+\]", "", text)

    m = _MONEY.search(text)
    if not m:
        if _FOREIGN_CURRENCY.search(text):
            return None
        m = _MONEY_NO_SYMBOL.search(text)
    if not m:
        # Raw large number (≥ 1,000,000)
        m7 = re.search(r"(\d{7,})", text)
        return float(m7.group(1)) / 1_000_000 if m7 else None

    low  = float(m.group(1))
    high = float(m.group(2)) if m.group(2) else low
    value = (low + high) / 2
    unit = m.group(3)
    if unit == "billion":
        return value * 1000
    if unit == "million":
        return value
    return value / 1_000_000 if value >= 1_000_000 else None


_LINK = re.compile(r"\[\[(?:[^\[\]|{}]*\|)?([^\[\]{}]*)\]\]")
_LIST_NAMES = r"(?:plain ?list|ubl|unbulleted ?list|flat ?list|hlist|bulleted ?list)"
_LIST_TEMPLATE  = re.compile(r"\{\{\s*" + _LIST_NAMES + r"\s*\|([^{}]*)\}\}", re.IGNORECASE)
_OTHER_TEMPLATE = re.compile(r"\{\{(?!\s*" + _LIST_NAMES + r"\s*\|)[^{}]*\}\}", re.IGNORECASE)


def _clean_wikitext(text: str) -> str:
    """Strip wiki markup, templates, refs and HTML from a field value, keeping
    the contents of list templates as a comma-separated list."""
    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)
    text = re.sub(r"<ref[^>]*/>", "", text)
    text = re.sub(r"<ref[^>]*>.*?</ref>", "", text, flags=re.DOTALL)
    text = re.sub(r"<br\s*/?>", ", ", text)
    text = re.sub(r"\{\{\s*(?:nbsp|snd|spaces)\s*\}\}", " ", text, flags=re.IGNORECASE)
    # Templates whose first argument is the visible text
    text = re.sub(r"\{\{\s*(?:US\$|USD)\s*\|\s*([^{}|]*)[^{}]*\}\}", r"$\1", text,
                  flags=re.IGNORECASE)
    text = re.sub(r"\{\{\s*(?:nowrap|nobr|small|abbr)\s*\|\s*([^{}|]*)[^{}]*\}\}", r"\1", text,
                  flags=re.IGNORECASE)
    # [[Link|display]] → display (plain links first, so list-template pipes
    # are unambiguous)
    text = _LINK.sub(r"\1", text)
    # Innermost first: drop other templates (e.g. {{efn}} footnotes), unwrap
    # list templates once they contain no templates, then resolve links that
    # wrapped a list
    prev = None
    while prev != text:
        prev = text
        text = _OTHER_TEMPLATE.sub("", text)
        text = _LIST_TEMPLATE.sub(
            lambda m: ", ".join(p.strip(" *\n") for p in re.split(r"[|\n]", m.group(1))
                                if p.strip(" *\n")),
            text)
        text = _LINK.sub(r"\1", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"'{2,}", "", text)
    text = re.sub(r"\n\s*\*\s*", ", ", text)   # bulleted lines → list
    text = re.sub(r"^\s*\*\s*", "", text)
    text = html.unescape(text)                  # &nbsp; &amp; …
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _split_names(cleaned: str) -> List[str]:
    return [n.strip() for n in cleaned.split(",") if n.strip()]


def _norm_title(title: str) -> str:
    t = title.strip().replace("_", " ")
    return t[:1].upper() + t[1:]


def _link_targets(raw: str) -> List[str]:
    """Page titles linked from a raw wikitext value, in order."""
    return [_norm_title(t) for t in re.findall(r"\[\[([^\[\]|{}#]+)", raw)]


def _strip_disambiguation(page_title: str) -> str:
    """'Oppenheimer (film)' → 'Oppenheimer'"""
    return re.sub(r"\s*\([^)]*\)$", "", page_title)


def _template_span(text: str, name: str) -> Optional[Tuple[int, int, int]]:
    """Locate {{name ...}} with brace matching → (start, body_start, end)."""
    m = re.search(r"\{\{\s*" + name + r"\b", text, re.IGNORECASE)
    if not m:
        return None
    depth, i = 0, m.start()
    while i < len(text) - 1:
        pair = text[i:i + 2]
        if pair == "{{":
            depth += 1
            i += 2
        elif pair == "}}":
            depth -= 1
            i += 2
            if depth == 0:
                return m.start(), m.end(), i
        else:
            i += 1
    return None


def _split_top_level(body: str) -> List[str]:
    """Split template arguments on '|' that are not inside {{ }} or [[ ]]."""
    parts, buf, depth, i = [], [], 0, 0
    while i < len(body):
        pair = body[i:i + 2]
        if pair in ("{{", "[["):
            depth += 1
            buf.append(pair)
            i += 2
        elif pair in ("}}", "]]"):
            depth -= 1
            buf.append(pair)
            i += 2
        elif body[i] == "|" and depth == 0:
            parts.append("".join(buf))
            buf = []
            i += 1
        else:
            buf.append(body[i])
            i += 1
    parts.append("".join(buf))
    return parts


def _parse_release(raw: str) -> Tuple[Optional[int], Optional[int]]:
    """
    (year, month) from an infobox release-date value. For {{Film date}} with
    several entries, prefers the United States release.
    """
    if not raw:
        return None, None

    span = _template_span(raw, "film date")
    if span:
        args = [a.strip() for a in _split_top_level(raw[span[1]:span[2] - 2])[1:]
                if "=" not in a]
        entries, i = [], 0
        while i < len(args):
            if re.fullmatch(r"\d{4}", args[i]):
                year = int(args[i])
                month = int(args[i + 1]) if (i + 1 < len(args) and args[i + 1].isdigit()
                                             and len(args[i + 1]) <= 2) else None
                j = i + 1
                while j < len(args) and args[j].isdigit() and len(args[j]) <= 2:
                    j += 1
                loc = args[j] if j < len(args) and not re.fullmatch(r"\d{4}", args[j]) else ""
                entries.append((year, month, loc))
                i = j
            else:
                i += 1
        if entries:
            year, month, _ = next((e for e in entries if "united states" in e[2].lower()),
                                  entries[0])
            return year, month if month and 1 <= month <= 12 else None

    m = re.search(r"\{\{\s*(?:start date|release date)[^|}]*\|\s*(\d{4})\s*(?:\|\s*(\d{1,2}))?",
                  raw, re.IGNORECASE)
    if m:
        month = int(m.group(2)) if m.group(2) else None
        return int(m.group(1)), month if month and 1 <= month <= 12 else None

    text = _clean_wikitext(raw).lower()
    m = re.search(r"\b((?:19|20)\d{2})-(\d{1,2})-\d{1,2}\b", text)
    if m:
        return int(m.group(1)), int(m.group(2))
    m = re.search(r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+"
                  r"(?:\d{1,2},?\s+)?((?:19|20)\d{2})\b", text)
    if m:
        return int(m.group(2)), MONTHS[m.group(1)]
    m = re.search(r"\b((?:19|20)\d{2})\b", text)
    if m:
        return int(m.group(1)), None
    return None, None


def _match_genre(text: str) -> Optional[str]:
    text = text.lower()
    best = None
    for keyword, genre in GENRE_KEYWORDS.items():
        m = re.search(r"\b" + re.escape(keyword) + r"\b", text)
        if m:
            rank = (m.start(), -len(keyword))
            if best is None or rank < best[0]:
                best = (rank, genre)
    return best[1] if best else None


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
WIKI_BATCH = 50  # max titles per API query
# API errors returned with HTTP 200 that are worth retrying
WIKI_RETRYABLE = {"cirrussearch-too-busy-error", "ratelimited", "maxlag"}


def _wiki_api(**params) -> Optional[dict]:
    for attempt in range(3):
        resp = _get(WIKI_API, params={"format": "json", "formatversion": 2, **params},
                    headers=WIKI_HEADERS)
        if not resp:
            return None
        try:
            data = resp.json()
        except ValueError:
            return None
        code = (data.get("error") or {}).get("code")
        if code in WIKI_RETRYABLE and attempt < 2:
            time.sleep(2 * (attempt + 1))
            continue
        return None if code else data
    return None


def wiki_search(title: str) -> Optional[str]:
    """
    Search Wikipedia for a movie title and return the best-matching page title.
    Appends 'film' to the query to prefer film articles.
    """
    data = _wiki_api(action="query", list="search", srsearch=f"{title} film",
                     srlimit=5, srnamespace=0)
    if not data:
        return None
    hits = data.get("query", {}).get("search", [])
    for hit in hits:
        t = hit["title"].lower()
        if "film" in t or "movie" in t or title.lower().split()[0] in t:
            return hit["title"]
    return hits[0]["title"] if hits else None


# Wikimedia rate-limits aggressively, and directors / actors recur across
# searches, so fetched pages (and confirmed-missing ones) and Wikidata query
# results are cached on disk.
CACHE_PATH = Path(__file__).with_name(".wiki_cache.sqlite")
CACHE_TTL  = 7 * 86400  # seconds


def _cache(sql: str, args: tuple = ()) -> list:
    with sqlite3.connect(CACHE_PATH, timeout=10) as db:
        db.execute("CREATE TABLE IF NOT EXISTS pages "
                   "(key TEXT PRIMARY KEY, fetched REAL, final TEXT, content TEXT)")
        return db.execute(sql, args).fetchall()


def _cached_json(key: str, fetch: Callable[[], Any]) -> Any:
    """Return the cached JSON value for `key`, else fetch() and cache it
    unless it is None (a failed fetch)."""
    try:
        rows = _cache("SELECT content FROM pages WHERE key=? AND fetched>?",
                      (key, time.time() - CACHE_TTL))
        if rows:
            return json.loads(rows[0][0])
    except (sqlite3.Error, ValueError):
        pass
    value = fetch()
    if value is not None:
        try:
            _cache("INSERT OR REPLACE INTO pages VALUES (?, ?, ?, ?)",
                   (key, time.time(), "", json.dumps(value)))
        except sqlite3.Error:
            pass
    return value


def wiki_get_pages(titles: List[str], lead_only: bool = False) -> Dict[str, Tuple[str, str]]:
    """
    Fetch wikitext for many pages, WIKI_BATCH per request, following
    redirects. With lead_only, only section 0 (infobox + lead) is fetched.
    Returns {requested title: (resolved title, wikitext)} in request order;
    missing pages are omitted.
    """
    prefix = "lead:" if lead_only else "full:"
    found: Dict[str, Tuple[str, str]] = {}
    to_fetch = []
    for t in titles:
        try:
            rows = _cache("SELECT final, content FROM pages WHERE key=? AND fetched>?",
                          (prefix + t, time.time() - CACHE_TTL))
        except sqlite3.Error:
            rows = []
        if not rows:
            to_fetch.append(t)
        elif rows[0][0] is not None:
            found[t] = (rows[0][0], rows[0][1])

    for i in range(0, len(to_fetch), WIKI_BATCH):
        chunk = to_fetch[i:i + WIKI_BATCH]
        params = dict(action="query", prop="revisions", rvprop="content",
                      rvslots="main", redirects=1, titles="|".join(chunk))
        if lead_only:
            params["rvsection"] = 0
        data = _wiki_api(**params)
        if not data:
            continue
        q = data.get("query", {})
        normalized = {n["from"]: n["to"] for n in q.get("normalized", [])}
        redirects  = {r["from"]: r["to"] for r in q.get("redirects", [])}
        content = {p["title"]: p["revisions"][0]["slots"]["main"]["content"]
                   for p in q.get("pages", []) if p.get("revisions")}
        rows = []
        for t in chunk:
            final = normalized.get(t, t)
            final = redirects.get(final, final)
            if final in content:
                found[t] = (final, content[final])
                rows.append((prefix + t, time.time(), final, content[final]))
            elif "continue" not in data:   # confirmed missing, not just truncated
                rows.append((prefix + t, time.time(), None, None))
        try:
            with sqlite3.connect(CACHE_PATH, timeout=10) as db:
                db.executemany("INSERT OR REPLACE INTO pages VALUES (?, ?, ?, ?)", rows)
        except sqlite3.Error:
            pass
    return {t: found[t] for t in titles if t in found}


def wiki_get_wikitext(page_title: str) -> Optional[str]:
    """Fetch raw wikitext for a Wikipedia page."""
    page = wiki_get_pages([page_title]).get(page_title)
    return page[1] if page else None


def find_film_page(title: str) -> Tuple[Optional[str], Optional[str]]:
    """
    (page title, wikitext) of the film's article. Tries '<title> (film)' and
    '<title>' directly — one request, and not subject to search throttling —
    then falls back to Wikipedia search.
    """
    t = title.strip()
    for final, text in wiki_get_pages([f"{t} (film)", t]).values():
        if _template_span(text, "Infobox film"):
            return final, text
    page_title = wiki_search(t)
    return (page_title, wiki_get_wikitext(page_title)) if page_title else (None, None)


def wiki_get_html(page_title: str) -> Optional[BeautifulSoup]:
    """Fetch parsed HTML for a Wikipedia page."""
    data = _wiki_api(action="parse", page=page_title, prop="text")
    html = (data or {}).get("parse", {}).get("text", "")
    return BeautifulSoup(html, "lxml") if html else None


def _parse_infobox_wikitext(wikitext: str) -> Dict[str, str]:
    """
    Extract key-value pairs from a Wikipedia film infobox in wikitext format.
    Returns a flat dict of {field_name: raw_value}.
    """
    span = _template_span(wikitext, "Infobox film")
    if not span:
        return {}
    fields: Dict[str, str] = {}
    for part in _split_top_level(wikitext[span[1]:span[2] - 2]):
        if "=" in part:
            key, val = part.split("=", 1)
            key = key.strip().lower().replace(" ", "_")
            if key:
                fields[key] = val.strip()
    return fields


def _lead_description(wikitext: str) -> str:
    """The '<title> is a 2019 American superhero film' phrase from the lead."""
    span = _template_span(wikitext, "Infobox film")
    lead = _clean_wikitext(wikitext[span[2]:span[2] + 4000] if span else wikitext[:4000])
    m = re.search(r"\b(?:is|was) an? (.{0,150}?)\b(?:film|movie)\b", lead)
    return m.group(1) if m else ""


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
            val = td.get_text(", ", strip=True)
            fields[key] = val
    return fields


def scrape_wikipedia(title: str, page: Optional[str] = None) -> Dict[str, Any]:
    """
    Main Wikipedia scraper, for the exact `page` if given, else a lookup of
    `title`. Returns a dict with extracted fields:
      budget_m, gross_m, runtime, genre, mpaa_rating, release_year,
      release_month, director(_page), lead_actor(_page), is_franchise,
      sequel_number, series_name
    """
    result: Dict[str, Any] = {}

    if page:
        page_title, wikitext = wiki_get_pages([page]).get(page, (None, None))
    else:
        page_title, wikitext = find_film_page(title)
    if not page_title:
        return result

    result["wiki_page"] = page_title

    # Try wikitext first (more structured), fall back to HTML
    fields = _parse_infobox_wikitext(wikitext) if wikitext else {}
    if not fields:
        soup = wiki_get_html(page_title)
        fields = _parse_infobox_html(soup) if soup else {}

    if not fields:
        return result

    description = _lead_description(wikitext) if wikitext else ""

    # ── Budget (release-year dollars) ───────────────────────────────────────
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

    # ── Release date → year + month ───────────────────────────────────────
    for key in ("released", "release_date", "release_dates"):
        if key in fields:
            year, month = _parse_release(fields[key])
            if year:
                result["release_year"] = year
            if month:
                result["release_month"] = month
            if year:
                break
    if "release_year" not in result:
        m = re.search(r"\b((?:19|20)\d{2})\b", description)
        if m:
            result["release_year"] = int(m.group(1))

    # ── Genre (rarely in the infobox; usually from the lead sentence) ──────
    for raw in (fields.get("genre"), fields.get("genres"), description):
        genre = _match_genre(_clean_wikitext(raw)) if raw else None
        if genre:
            result["genre"] = genre
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

    # ── Director / lead cast: display name + linked Wikipedia page ─────────
    for out_key, keys in (("director", ("director", "directors")),
                          ("lead_actor", ("starring", "stars", "cast"))):
        for key in keys:
            if key in fields:
                names   = _split_names(_clean_wikitext(fields[key]))
                targets = _link_targets(fields[key])
                if names:
                    result[out_key] = names[0]
                    result[f"{out_key}_page"] = targets[0] if targets else None
                    if out_key == "lead_actor":
                        result["cast_list"] = names[:4]
                break

    # ── Franchise / sequel detection — use authoritative lookup table ──────
    film_title = _strip_disambiguation(page_title)
    # Pass any infobox keywords as hints
    kw_hints = []
    for key in ("series", "franchise", "based_on"):
        if key in fields and fields[key].strip():
            kw_hints.append(_clean_wikitext(fields[key]).split("\n")[0].strip())
    result["is_franchise"]  = fl_is_franchise(film_title, kw_hints)
    result["sequel_number"] = fl_sequel_number(film_title, kw_hints)
    fn = fl_franchise_name(film_title)
    if fn:
        result["series_name"] = fn

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
# Director / actor historical scoring (Wikidata + Wikipedia)
# ─────────────────────────────────────────────────────────────────────────────

WIKIDATA_SPARQL = "https://query.wikidata.org/sparql"
TOP_BILLED_CHECK = 15  # most recent prior films checked for the lead's billing


def _wikidata_histories(film_page: str, lead_actor: Optional[str]
                        ) -> Optional[Dict[str, List[Tuple[int, float, str]]]]:
    """Cached wrapper around _fetch_wikidata_histories."""
    return _cached_json(f"history:{film_page}|{lead_actor or ''}",
                        lambda: _fetch_wikidata_histories(film_page, lead_actor))


def _fetch_wikidata_histories(film_page: str, lead_actor: Optional[str]
                              ) -> Optional[Dict[str, List[Tuple[int, float, str]]]]:
    """
    One Wikidata query: starting from the target film, every film directed by
    its directors, and every film featuring the cast member named
    `lead_actor`, that has a US-dollar box office. Returns
    {"director": [...], "cast": [...]} of (first release year, max USD gross
    $M — the worldwide figure, enwiki title), or None if the query failed.
    """
    cast_branch = ""
    if lead_actor:
        cast_branch = f"""UNION {{
        ?target wdt:P161 ?person. ?person rdfs:label ?label.
        FILTER(LANG(?label) = "en" && LCASE(STR(?label)) = {json.dumps(lead_actor.lower())})
        BIND("cast" AS ?role) ?film wdt:P161 ?person.
      }}"""
    query = f"""
    SELECT ?role ?title (MIN(?date) AS ?released) (MAX(?usd) AS ?gross) WHERE {{
      ?target ^schema:about [ schema:isPartOf <https://en.wikipedia.org/>;
                              schema:name {json.dumps(film_page)}@en ].
      {{ ?target wdt:P57 ?person. BIND("director" AS ?role) ?film wdt:P57 ?person. }}
      {cast_branch}
      ?film wdt:P577 ?date;
            p:P2142/psv:P2142 [ wikibase:quantityAmount ?usd; wikibase:quantityUnit wd:Q4917 ].
      ?wp schema:about ?film; schema:isPartOf <https://en.wikipedia.org/>; schema:name ?title.
    }} GROUP BY ?role ?title"""
    resp = _get(WIKIDATA_SPARQL, params={"query": query, "format": "json"},
                headers=WIKI_HEADERS)
    if not resp:
        return None
    out: Dict[str, List[Tuple[int, float, str]]] = {"director": [], "cast": []}
    try:
        for b in resp.json()["results"]["bindings"]:
            year = int(b["released"]["value"][:4])
            out[b["role"]["value"]].append(
                (year, float(b["gross"]["value"]) / 1e6, b["title"]["value"]))
    except (ValueError, KeyError):
        return None
    return out


def _is_top_billed(fields: Dict[str, str], page: Optional[str], name: str) -> bool:
    raw = fields.get("starring")
    if not raw:
        return False
    targets = _link_targets(raw)
    names = _split_names(_clean_wikitext(raw))
    return ((page is not None and targets[:1] == [_norm_title(page)])
            or (bool(names) and names[0].lower() == name.strip().lower()))


def score_people(film_page: str, lead_actor: Optional[str], lead_actor_page: Optional[str],
                 before_year: int) -> Tuple[Optional[float], Optional[float], List[str]]:
    """
    (director_score, cast_score, notes). A score is the average worldwide
    gross (2024 $M) of the person's last 5 films released before
    `before_year` — the same definition as the training data
    (features.person_score). As in training, the cast score only counts
    films where the lead actor was top-billed; that is checked against the
    Wikipedia infobox of their most recent prior films.
    """
    notes: List[str] = []
    hist = _wikidata_histories(film_page, lead_actor)
    if hist is None:
        return None, None, notes

    def prior(films):
        return sorted(f for f in films if f[0] < before_year and f[2] != film_page)

    director_score = person_score([cpi_adjust(g, y) for y, g, _ in prior(hist["director"])])

    cast = prior(hist["cast"])[-TOP_BILLED_CHECK:]
    if cast and lead_actor:
        pages = wiki_get_pages([t for _, _, t in cast], lead_only=True)
        if not pages:
            # Counting supporting roles would inflate the score relative to
            # training, so leave it to be imputed (or entered by the user)
            notes.append("Lead-actor score skipped: couldn't check billing on Wikipedia")
            return director_score, None, notes
        cast = [f for f in cast if f[2] in pages and _is_top_billed(
            _parse_infobox_wikitext(pages[f[2]][1]), lead_actor_page, lead_actor)]
    cast_score = person_score([cpi_adjust(g, y) for y, g, _ in cast])
    return director_score, cast_score, notes


# ─────────────────────────────────────────────────────────────────────────────
# Film candidates (Wikidata search) — lets the user pick between same-named films
# ─────────────────────────────────────────────────────────────────────────────

def search_film_candidates(title: str, limit: int = 8) -> Optional[List[Dict[str, Any]]]:
    """Cached wrapper around _fetch_film_candidates."""
    return _cached_json(f"candidates:{limit}:{title.strip().lower()}",
                        lambda: _fetch_film_candidates(title, limit))


def _fetch_film_candidates(title: str, limit: int) -> Optional[List[Dict[str, Any]]]:
    """
    Films whose Wikidata label or alias matches `title`, in search-rank order:
    [{"page": enwiki title, "year": int | None, "description": str}].
    One Wikidata query; None if it failed (the caller falls back to a direct
    Wikipedia lookup).
    """
    query = f"""
    SELECT ?item ?title ?desc (MIN(?date) AS ?released) (MIN(?ord) AS ?rank) WHERE {{
      SERVICE wikibase:mwapi {{
        bd:serviceParam wikibase:endpoint "www.wikidata.org"; wikibase:api "EntitySearch";
                        mwapi:search {json.dumps(title)}; mwapi:language "en"; mwapi:limit "25".
        ?item wikibase:apiOutputItem mwapi:item.
        ?ord wikibase:apiOrdinal true.
      }}
      ?item wdt:P31/wdt:P279* wd:Q11424.
      ?wp schema:about ?item; schema:isPartOf <https://en.wikipedia.org/>; schema:name ?title.
      OPTIONAL {{ ?item schema:description ?desc. FILTER(LANG(?desc) = "en") }}
      OPTIONAL {{ ?item wdt:P577 ?date. }}
    }} GROUP BY ?item ?title ?desc ORDER BY ?rank"""
    resp = _get(WIKIDATA_SPARQL, params={"query": query, "format": "json"},
                headers=WIKI_HEADERS)
    if not resp:
        return None
    try:
        bindings = resp.json()["results"]["bindings"]
    except (ValueError, KeyError):
        return None
    candidates = []
    for b in bindings[:limit]:
        released = b.get("released", {}).get("value", "")
        candidates.append({
            "page":        b["title"]["value"],
            "year":        int(released[:4]) if released[:4].isdigit() else None,
            "description": b.get("desc", {}).get("value", ""),
        })
    return candidates


# ─────────────────────────────────────────────────────────────────────────────
# Master search function (called by app.py)
# ─────────────────────────────────────────────────────────────────────────────

MODEL_FIELDS = ["budget_m", "release_month", "release_year", "genre", "is_franchise",
                "sequel_number", "director_score", "cast_score", "runtime"]


def search_movie_features(title: str, page: Optional[str] = None) -> Dict[str, Any]:
    """
    Scrape the model's input features for a film. `page` is its exact English
    Wikipedia title (from search_film_candidates); without it, `title` is
    looked up directly and then via Wikipedia search.

    Returns:
        {
            "title", "wiki_page",
            "budget_m":       float | None,   # release-year dollars
            "release_month", "release_year", "genre", "is_franchise",
            "sequel_number", "runtime",
            "director_score": float | None,   # 2024 USD
            "cast_score":     float | None,   # 2024 USD
            "actual_gross_m":     float | None,  # worldwide gross so far, release-year $
            "actual_gross_adj_m": float | None,  # same, 2024 USD
            "found", "missing": [model field names],
            "notes": str,
        }
    """
    RATE_LIMITED_HOSTS.clear()
    print(f"[scraper] Looking up Wikipedia page for: {page or title}")
    wiki_data = scrape_wikipedia(title, page)
    wiki_page = wiki_data.get("wiki_page")
    print(f"[scraper] Wikipedia page: {wiki_page or 'not found'}")

    release_year = wiki_data.get("release_year")
    director     = wiki_data.get("director")
    lead_actor   = wiki_data.get("lead_actor")
    budget_m     = wiki_data.get("budget_m")

    # Director / lead-actor scores (Wikidata) and, only if Wikipedia had no
    # budget, The Numbers — run in parallel
    director_score = cast_score = None
    people_notes: List[str] = []
    if wiki_page:
        with ThreadPoolExecutor(max_workers=2) as pool:
            people = pool.submit(score_people, wiki_page, lead_actor,
                                 wiki_data.get("lead_actor_page"),
                                 release_year or date.today().year + 1)
            numbers = (pool.submit(scrape_the_numbers, _strip_disambiguation(wiki_page))
                       if budget_m is None else None)
            director_score, cast_score, people_notes = people.result()
            if numbers:
                budget_m = numbers.result().get("budget_m")

    actual = wiki_data.get("gross_m")
    output = {
        "title":              title,
        "wiki_page":          wiki_page,
        "budget_m":           budget_m,
        "release_month":      wiki_data.get("release_month"),
        "release_year":       release_year,
        "genre":              wiki_data.get("genre"),
        "is_franchise":       wiki_data.get("is_franchise"),
        "sequel_number":      wiki_data.get("sequel_number"),
        "director_score":     director_score,
        "cast_score":         cast_score,
        "runtime":            wiki_data.get("runtime"),
        "actual_gross_m":     actual,
        "actual_gross_adj_m": (round(cpi_adjust(actual, release_year), 1)
                               if actual and release_year else None),
    }

    found   = [k for k in MODEL_FIELDS if output[k] is not None]
    missing = [k for k in MODEL_FIELDS if output[k] is None]

    notes_parts = []
    if wiki_page:
        notes_parts.append(f"Wikipedia: {wiki_page}")
    if director:
        notes_parts.append(f"Director: {director}")
    if lead_actor:
        notes_parts.append(f"Lead: {lead_actor}")
    if wiki_data.get("series_name"):
        notes_parts.append(f"Series: {wiki_data['series_name']}")
    notes_parts += people_notes
    if RATE_LIMITED_HOSTS:
        notes_parts.append("Rate-limited by " + ", ".join(sorted(RATE_LIMITED_HOSTS))
                           + " — some fields may be missing; try again in a minute")

    output["found"]   = found
    output["missing"] = missing
    output["notes"]   = " · ".join(notes_parts) or "Scraped from public sources"

    print(f"[scraper] Done. Found {len(found)} fields, missing {len(missing)}")
    return output
