"""
features.py — Feature definitions shared by training and inference
==================================================================
Training (load_tmdb.py, retrain.py) and inference (scraper.py, app.py) both
import from here, so a feature means the same thing on both sides.
"""

import re
from typing import List, Optional

FEATURE_NAMES = [
    "budget_m_adj", "release_month", "genre_encoded", "is_franchise",
    "sequel_number", "director_score", "cast_score", "runtime", "release_year",
]

GENRES = ["Action", "Comedy", "Drama", "Horror", "SciFi",
          "Animation", "Romance", "Thriller", "Fantasy", "Documentary"]


def encode_genre(genre: Optional[str]) -> Optional[int]:
    return GENRES.index(genre) if genre in GENRES else None


def title_key(title: str) -> str:
    """Normalised title for matching a film across TMDB and Wikipedia
    ('Avengers: Endgame' and 'Avengers – Endgame' → 'avengers endgame')."""
    t = re.sub(r"[^a-z0-9 ]", " ", str(title).lower().replace("&", " and "))
    return re.sub(r"\s+", " ", t).strip()


# ── Inflation ────────────────────────────────────────────────────────────────
# BLS CPI-U, U.S. city average, annual averages (1982–84 = 100).
# Add new years as BLS publishes them; later years are treated as BASE_YEAR
# dollars and earlier ones as the first year in the table.
CPI_U = {
    1970: 38.8,  1971: 40.5,  1972: 41.8,  1973: 44.4,  1974: 49.3,
    1975: 53.8,  1976: 56.9,  1977: 60.6,  1978: 65.2,  1979: 72.6,
    1980: 82.4,  1981: 90.9,  1982: 96.5,  1983: 99.6,  1984: 103.9,
    1985: 107.6, 1986: 109.6, 1987: 113.6, 1988: 118.3, 1989: 124.0,
    1990: 130.7, 1991: 136.2, 1992: 140.3, 1993: 144.5, 1994: 148.2,
    1995: 152.4, 1996: 156.9, 1997: 160.5, 1998: 163.0, 1999: 166.6,
    2000: 172.2, 2001: 177.1, 2002: 179.9, 2003: 184.0, 2004: 188.9,
    2005: 195.3, 2006: 201.6, 2007: 207.3, 2008: 215.3, 2009: 214.5,
    2010: 218.1, 2011: 224.9, 2012: 229.6, 2013: 233.0, 2014: 236.7,
    2015: 237.0, 2016: 240.0, 2017: 245.1, 2018: 251.1, 2019: 255.7,
    2020: 258.8, 2021: 271.0, 2022: 292.7, 2023: 304.7, 2024: 313.7,
}
BASE_YEAR = 2024


def cpi_adjust(amount_m: float, year: Optional[float]) -> float:
    """Convert a dollar amount from `year` dollars to BASE_YEAR dollars."""
    if year is None:
        return amount_m
    y = min(max(int(year), min(CPI_U)), max(CPI_U))
    return amount_m * CPI_U[BASE_YEAR] / CPI_U[y]


# ── Director / lead-actor score ──────────────────────────────────────────────
PERSON_HISTORY = 5  # number of most recent prior films averaged


def person_score(prior_grosses_adj: List[float]) -> Optional[float]:
    """
    Average worldwide gross (BASE_YEAR $M) of a person's most recent prior
    films. `prior_grosses_adj` must be in chronological order and contain only
    films released before the one being scored — in training, films where the
    person is the director / top-billed cast member; the scraper mirrors this.
    """
    recent = prior_grosses_adj[-PERSON_HISTORY:]
    return round(sum(recent) / len(recent), 1) if recent else None
