"""
franchise_lookup.py — Authoritative franchise & sequel number registry
======================================================================
Used by both scraper.py (live search) and load_tmdb.py (batch training).

Structure:
    FRANCHISE_ENTRIES : dict[str, int]
        Lowercase movie title  →  entry number in its series
        e.g. "avengers: doomsday" → 6

    FRANCHISE_NAMES : dict[str, str]
        Lowercase movie title  →  canonical franchise name
        e.g. "avengers: doomsday" → "Marvel Cinematic Universe"

    is_franchise(title, keywords=None) -> bool
    sequel_number(title, keywords=None) -> int
    franchise_name(title) -> str | None
"""

from typing import Optional, List

# ---------------------------------------------------------------------------
# Explicit entry-number registry
# Key = lowercase title (strip punctuation inconsistencies where needed)
# Value = entry number (1 = first/original, 2 = first sequel, etc.)
# ---------------------------------------------------------------------------
FRANCHISE_ENTRIES: dict = {
    # ── Marvel Cinematic Universe ───────────────────────────────────────────
    "iron man": 1,
    "the incredible hulk": 2,
    "iron man 2": 3,
    "thor": 4,
    "captain america: the first avenger": 5,
    "the avengers": 6,
    "iron man 3": 7,
    "thor: the dark world": 8,
    "captain america: the winter soldier": 9,
    "guardians of the galaxy": 10,
    "avengers: age of ultron": 11,
    "ant-man": 12,
    "captain america: civil war": 13,
    "doctor strange": 14,
    "guardians of the galaxy vol. 2": 15,
    "spider-man: homecoming": 16,
    "thor: ragnarok": 17,
    "black panther": 18,
    "avengers: infinity war": 19,
    "ant-man and the wasp": 20,
    "captain marvel": 21,
    "avengers: endgame": 22,
    "spider-man: far from home": 23,
    "black widow": 24,
    "shang-chi and the legend of the ten rings": 25,
    "eternals": 26,
    "spider-man: no way home": 27,
    "doctor strange in the multiverse of madness": 28,
    "thor: love and thunder": 29,
    "black panther: wakanda forever": 30,
    "ant-man and the wasp: quantumania": 31,
    "guardians of the galaxy vol. 3": 32,
    "the marvels": 33,
    "deadpool & wolverine": 34,
    "captain america: brave new world": 35,
    "thunderbolts": 36,
    "avengers: doomsday": 37,
    "avengers doomsday": 37,
    "avengers: secret wars": 38,

    # ── DC Extended Universe ────────────────────────────────────────────────
    "man of steel": 1,
    "batman v superman: dawn of justice": 2,
    "suicide squad": 3,
    "wonder woman": 4,
    "justice league": 5,
    "aquaman": 6,
    "shazam!": 7,
    "birds of prey": 8,
    "wonder woman 1984": 9,
    "the suicide squad": 10,
    "black adam": 11,
    "shazam! fury of the gods": 12,
    "the flash": 13,
    "blue beetle": 14,
    "aquaman and the lost kingdom": 15,
    "joker": 1,  # standalone
    "joker: folie à deux": 2,

    # ── Star Wars ───────────────────────────────────────────────────────────
    "star wars": 1,
    "star wars: a new hope": 1,
    "the empire strikes back": 2,
    "return of the jedi": 3,
    "star wars: the phantom menace": 4,
    "star wars: attack of the clones": 5,
    "star wars: revenge of the sith": 6,
    "star wars: the force awakens": 7,
    "rogue one: a star wars story": 7,   # anthology
    "star wars: the last jedi": 8,
    "solo: a star wars story": 8,        # anthology
    "star wars: the rise of skywalker": 9,

    # ── Fast & Furious ──────────────────────────────────────────────────────
    "the fast and the furious": 1,
    "2 fast 2 furious": 2,
    "the fast and the furious: tokyo drift": 3,
    "fast & furious": 4,
    "fast five": 5,
    "fast & furious 6": 6,
    "furious 7": 7,
    "the fate of the furious": 8,
    "hobbs & shaw": 8,   # spinoff
    "f9": 9,
    "fast x": 10,

    # ── Mission: Impossible ─────────────────────────────────────────────────
    "mission: impossible": 1,
    "mission: impossible 2": 2,
    "mission: impossible – ghost protocol": 4,
    "mission: impossible - ghost protocol": 4,
    "mission: impossible – rogue nation": 5,
    "mission: impossible - rogue nation": 5,
    "mission: impossible – fallout": 6,
    "mission: impossible - fallout": 6,
    "mission: impossible – dead reckoning part one": 7,
    "mission: impossible - dead reckoning part one": 7,
    "mission: impossible – the final reckoning": 8,
    "mission: impossible - the final reckoning": 8,

    # ── James Bond ──────────────────────────────────────────────────────────
    "casino royale": 1,   # Craig era start — model will see modern films
    "quantum of solace": 2,
    "skyfall": 3,
    "spectre": 4,
    "no time to die": 5,

    # ── Jurassic Park / World ───────────────────────────────────────────────
    "jurassic park": 1,
    "the lost world: jurassic park": 2,
    "jurassic park iii": 3,
    "jurassic world": 4,
    "jurassic world: fallen kingdom": 5,
    "jurassic world dominion": 6,

    # ── Transformers ────────────────────────────────────────────────────────
    "transformers": 1,
    "transformers: revenge of the fallen": 2,
    "transformers: dark of the moon": 3,
    "transformers: age of extinction": 4,
    "transformers: the last knight": 5,
    "bumblebee": 5,   # prequel/spinoff
    "transformers: rise of the beasts": 6,

    # ── Harry Potter / Fantastic Beasts ────────────────────────────────────
    "harry potter and the philosopher's stone": 1,
    "harry potter and the sorcerer's stone": 1,
    "harry potter and the chamber of secrets": 2,
    "harry potter and the prisoner of azkaban": 3,
    "harry potter and the goblet of fire": 4,
    "harry potter and the order of the phoenix": 5,
    "harry potter and the half-blood prince": 6,
    "harry potter and the deathly hallows – part 1": 7,
    "harry potter and the deathly hallows - part 1": 7,
    "harry potter and the deathly hallows – part 2": 8,
    "harry potter and the deathly hallows - part 2": 8,
    "fantastic beasts and where to find them": 9,
    "fantastic beasts: the crimes of grindelwald": 10,
    "fantastic beasts: the secrets of dumbledore": 11,

    # ── Lord of the Rings / Hobbit ──────────────────────────────────────────
    "the lord of the rings: the fellowship of the ring": 1,
    "the lord of the rings: the two towers": 2,
    "the lord of the rings: the return of the king": 3,
    "the hobbit: an unexpected journey": 4,
    "the hobbit: the desolation of smaug": 5,
    "the hobbit: the battle of the five armies": 6,

    # ── Pirates of the Caribbean ────────────────────────────────────────────
    "pirates of the caribbean: the curse of the black pearl": 1,
    "pirates of the caribbean: dead man's chest": 2,
    "pirates of the caribbean: at world's end": 3,
    "pirates of the caribbean: on stranger tides": 4,
    "pirates of the caribbean: dead men tell no tales": 5,

    # ── Hunger Games ────────────────────────────────────────────────────────
    "the hunger games": 1,
    "the hunger games: catching fire": 2,
    "the hunger games: mockingjay – part 1": 3,
    "the hunger games: mockingjay - part 1": 3,
    "the hunger games: mockingjay – part 2": 4,
    "the hunger games: mockingjay - part 2": 4,
    "the hunger games: the ballad of songbirds & snakes": 5,

    # ── Toy Story ───────────────────────────────────────────────────────────
    "toy story": 1,
    "toy story 2": 2,
    "toy story 3": 3,
    "toy story 4": 4,

    # ── Frozen ──────────────────────────────────────────────────────────────
    "frozen": 1,
    "frozen ii": 2,
    "frozen 2": 2,
    "frozen 3": 3,

    # ── How to Train Your Dragon ────────────────────────────────────────────
    "how to train your dragon": 1,
    "how to train your dragon 2": 2,
    "how to train your dragon: the hidden world": 3,
    "how to train your dragon 3": 3,

    # ── Despicable Me / Minions ─────────────────────────────────────────────
    "despicable me": 1,
    "despicable me 2": 2,
    "minions": 2,    # spinoff
    "despicable me 3": 3,
    "minions: the rise of gru": 3,
    "despicable me 4": 4,

    # ── Shrek ───────────────────────────────────────────────────────────────
    "shrek": 1,
    "shrek 2": 2,
    "shrek the third": 3,
    "shrek forever after": 4,

    # ── Kung Fu Panda ───────────────────────────────────────────────────────
    "kung fu panda": 1,
    "kung fu panda 2": 2,
    "kung fu panda 3": 3,
    "kung fu panda 4": 4,

    # ── Spider-Man (Sony) ───────────────────────────────────────────────────
    "spider-man": 1,
    "spider-man 2": 2,
    "spider-man 3": 3,
    "the amazing spider-man": 1,
    "the amazing spider-man 2": 2,
    "venom": 1,
    "venom: let there be carnage": 2,
    "venom: the last dance": 3,
    "morbius": 1,
    "kraven the hunter": 1,

    # ── X-Men ───────────────────────────────────────────────────────────────
    "x-men": 1,
    "x2": 2,
    "x-men: the last stand": 3,
    "x-men origins: wolverine": 4,
    "x-men: first class": 5,
    "the wolverine": 6,
    "x-men: days of future past": 7,
    "x-men: apocalypse": 8,
    "logan": 9,
    "x-men: dark phoenix": 10,
    "the new mutants": 11,
    "deadpool": 1,
    "deadpool 2": 2,

    # ── Batman (standalone / DCAU) ──────────────────────────────────────────
    "batman begins": 1,
    "the dark knight": 2,
    "the dark knight rises": 3,
    "the batman": 1,
    "the batman – part ii": 2,

    # ── Indiana Jones ───────────────────────────────────────────────────────
    "raiders of the lost ark": 1,
    "indiana jones and the temple of doom": 2,
    "indiana jones and the last crusade": 3,
    "indiana jones and the kingdom of the crystal skull": 4,
    "indiana jones and the dial of destiny": 5,

    # ── John Wick ───────────────────────────────────────────────────────────
    "john wick": 1,
    "john wick: chapter 2": 2,
    "john wick: chapter 3 – parabellum": 3,
    "john wick: chapter 3 - parabellum": 3,
    "john wick: chapter 4": 4,
    "ballerina": 5,

    # ── Avatar ──────────────────────────────────────────────────────────────
    "avatar": 1,
    "avatar: the way of water": 2,
    "avatar: fire and ash": 3,

    # ── Godzilla / MonsterVerse ─────────────────────────────────────────────
    "godzilla": 1,
    "kong: skull island": 2,
    "godzilla: king of the monsters": 3,
    "godzilla vs. kong": 4,
    "godzilla x kong: the new empire": 5,

    # ── Planet of the Apes ──────────────────────────────────────────────────
    "rise of the planet of the apes": 1,
    "dawn of the planet of the apes": 2,
    "war for the planet of the apes": 3,
    "kingdom of the planet of the apes": 4,

    # ── The Matrix ──────────────────────────────────────────────────────────
    "the matrix": 1,
    "the matrix reloaded": 2,
    "the matrix revolutions": 3,
    "the matrix resurrections": 4,

    # ── Alien / Predator ────────────────────────────────────────────────────
    "alien": 1,
    "aliens": 2,
    "alien 3": 3,
    "alien: resurrection": 4,
    "prometheus": 5,
    "alien: covenant": 6,
    "alien: romulus": 7,
    "predator": 1,
    "predator 2": 2,
    "predators": 3,
    "the predator": 4,
    "prey": 5,
    "predator: badlands": 6,

    # ── The Conjuring Universe ──────────────────────────────────────────────
    "the conjuring": 1,
    "annabelle": 2,
    "the conjuring 2": 3,
    "annabelle: creation": 4,
    "the nun": 5,
    "the curse of la llorona": 6,
    "annabelle comes home": 7,
    "the conjuring: the devil made me do it": 8,
    "the nun ii": 9,

    # ── Bourne ──────────────────────────────────────────────────────────────
    "the bourne identity": 1,
    "the bourne supremacy": 2,
    "the bourne ultimatum": 3,
    "the bourne legacy": 4,
    "jason bourne": 5,

    # ── Ice Age ─────────────────────────────────────────────────────────────
    "ice age": 1,
    "ice age: the meltdown": 2,
    "ice age: dawn of the dinosaurs": 3,
    "ice age: continental drift": 4,
    "ice age: collision course": 5,

    # ── Cars ────────────────────────────────────────────────────────────────
    "cars": 1,
    "cars 2": 2,
    "cars 3": 3,

    # ── Incredibles ─────────────────────────────────────────────────────────
    "the incredibles": 1,
    "incredibles 2": 2,

    # ── Finding Nemo ────────────────────────────────────────────────────────
    "finding nemo": 1,
    "finding dory": 2,

    # ── Monsters, Inc. ──────────────────────────────────────────────────────
    "monsters, inc.": 1,
    "monsters university": 2,

    # ── Madagascar ──────────────────────────────────────────────────────────
    "madagascar": 1,
    "madagascar: escape 2 africa": 2,
    "madagascar 3: europe's most wanted": 3,

    # ── Twilight ────────────────────────────────────────────────────────────
    "twilight": 1,
    "the twilight saga: new moon": 2,
    "the twilight saga: eclipse": 3,
    "the twilight saga: breaking dawn – part 1": 4,
    "the twilight saga: breaking dawn – part 2": 5,

    # ── Maze Runner ─────────────────────────────────────────────────────────
    "the maze runner": 1,
    "maze runner: the scorch trials": 2,
    "maze runner: the death cure": 3,

    # ── Divergent ───────────────────────────────────────────────────────────
    "divergent": 1,
    "insurgent": 2,
    "allegiant": 3,

    # ── Terminator ──────────────────────────────────────────────────────────
    "the terminator": 1,
    "terminator 2: judgment day": 2,
    "terminator 3: rise of the machines": 3,
    "terminator salvation": 4,
    "terminator genisys": 5,
    "terminator: dark fate": 6,

    # ── Ocean's ─────────────────────────────────────────────────────────────
    "ocean's eleven": 1,
    "ocean's twelve": 2,
    "ocean's thirteen": 3,
    "ocean's 8": 4,

    # ── Hotel Transylvania ──────────────────────────────────────────────────
    "hotel transylvania": 1,
    "hotel transylvania 2": 2,
    "hotel transylvania 3: summer vacation": 3,

    # ── Paddington ──────────────────────────────────────────────────────────
    "paddington": 1,
    "paddington 2": 2,
    "paddington in peru": 3,

    # ── Mamma Mia ───────────────────────────────────────────────────────────
    "mamma mia!": 1,
    "mamma mia! here we go again": 2,

    # ── Pitch Perfect ───────────────────────────────────────────────────────
    "pitch perfect": 1,
    "pitch perfect 2": 2,
    "pitch perfect 3": 3,

    # ── Insidious ───────────────────────────────────────────────────────────
    "insidious": 1,
    "insidious: chapter 2": 2,
    "insidious: chapter 3": 3,
    "insidious: the last key": 4,
    "insidious: the red door": 5,

    # ── Scream ──────────────────────────────────────────────────────────────
    "scream": 1,
    "scream 2": 2,
    "scream 3": 3,
    "scream 4": 4,
    "scream (2022)": 5,
    "scream vi": 6,
    "scream 7": 7,

    # ── Halloween ───────────────────────────────────────────────────────────
    "halloween (2018)": 1,
    "halloween kills": 2,
    "halloween ends": 3,

    # ── A Quiet Place ───────────────────────────────────────────────────────
    "a quiet place": 1,
    "a quiet place part ii": 2,
    "a quiet place: day one": 3,

    # ── Five Nights at Freddy's ─────────────────────────────────────────────
    "five nights at freddy's": 1,

    # ── Sonic ───────────────────────────────────────────────────────────────
    "sonic the hedgehog": 1,
    "sonic the hedgehog 2": 2,
    "sonic the hedgehog 3": 3,

    # ── The Super Mario Bros. ───────────────────────────────────────────────
    "the super mario bros. movie": 1,

    # ── Inside Out ──────────────────────────────────────────────────────────
    "inside out": 1,
    "inside out 2": 2,

    # ── Elemental ───────────────────────────────────────────────────────────
    "elemental": 1,

    # ── Moana ───────────────────────────────────────────────────────────────
    "moana": 1,
    "moana 2": 2,

    # ── Zootopia ────────────────────────────────────────────────────────────
    "zootopia": 1,
    "zootopia+": 1,

    # ── Encanto / Raya / Turning Red / Lightyear / Elemental ───────────────
    "encanto": 1,
    "raya and the last dragon": 1,
    "lightyear": 1,

    # ── Barbie / Oppenheimer (standalone) ──────────────────────────────────
    "barbie": 1,
    "oppenheimer": 1,
}

# ---------------------------------------------------------------------------
# Known franchise names (for display / is_franchise detection)
# ---------------------------------------------------------------------------
FRANCHISE_NAMES: dict = {
    # MCU
    "iron man": "Marvel Cinematic Universe",
    "avengers": "Marvel Cinematic Universe",
    "thor": "Marvel Cinematic Universe",
    "captain america": "Marvel Cinematic Universe",
    "black panther": "Marvel Cinematic Universe",
    "spider-man: homecoming": "Marvel Cinematic Universe",
    "spider-man: far from home": "Marvel Cinematic Universe",
    "spider-man: no way home": "Marvel Cinematic Universe",
    "doctor strange": "Marvel Cinematic Universe",
    "guardians of the galaxy": "Marvel Cinematic Universe",
    "ant-man": "Marvel Cinematic Universe",
    "black widow": "Marvel Cinematic Universe",
    "eternals": "Marvel Cinematic Universe",
    "shang-chi": "Marvel Cinematic Universe",
    "thor: ragnarok": "Marvel Cinematic Universe",
    "captain marvel": "Marvel Cinematic Universe",
    "deadpool": "Marvel Cinematic Universe",
    "the marvels": "Marvel Cinematic Universe",
    "thunderbolts": "Marvel Cinematic Universe",
    # DC
    "batman v superman": "DC Extended Universe",
    "wonder woman": "DC Extended Universe",
    "aquaman": "DC Extended Universe",
    "the flash": "DC Extended Universe",
    # Franchise buckets
    "star wars": "Star Wars",
    "fast": "Fast & Furious",
    "furious": "Fast & Furious",
    "mission: impossible": "Mission: Impossible",
    "james bond": "James Bond",
    "jurassic": "Jurassic",
    "transformers": "Transformers",
    "harry potter": "Wizarding World",
    "fantastic beasts": "Wizarding World",
    "lord of the rings": "Middle-Earth",
    "hobbit": "Middle-Earth",
    "pirates of the caribbean": "Pirates of the Caribbean",
    "hunger games": "The Hunger Games",
}

# ---------------------------------------------------------------------------
# Franchise-positive title keywords (any match → is_franchise = True)
# ---------------------------------------------------------------------------
_FRANCHISE_KW = [
    "avengers", "spider-man", "batman", "superman", "iron man", "thor",
    "captain america", "guardians of the galaxy", "black panther", "ant-man",
    "doctor strange", "captain marvel", "black widow", "eternals", "shang-chi",
    "the marvels", "thunderbolts", "deadpool", "wolverine",
    "star wars", "rogue one", "the force awakens", "the last jedi",
    "the rise of skywalker", "the phantom menace", "attack of the clones",
    "revenge of the sith", "a new hope", "the empire strikes back",
    "jurassic", "transformers", "mission: impossible", "james bond",
    "harry potter", "fantastic beasts", "lord of the rings", "the hobbit",
    "pirates of the caribbean", "hunger games", "twilight", "maze runner",
    "divergent", "toy story", "finding nemo", "finding dory", "the incredibles",
    "incredibles 2", "frozen", "monsters, inc", "monsters university",
    "how to train your dragon", "shrek", "kung fu panda", "despicable me",
    "minions", "ice age", "madagascar", "cars", "john wick", "fast and furious",
    "fast & furious", "hobbs", "furious 7", "fast five", "fast x",
    "godzilla", "king kong", "kong:", "planet of the apes", "the matrix",
    "terminator", "alien", "predator", "bourne", "ocean's",
    "hotel transylvania", "paddington", "pitch perfect", "insidious",
    "scream", "halloween", "a quiet place", "sonic the hedgehog",
    "the batman", "the dark knight", "batman begins", "avatar",
    "indiana jones", "venom", "x-men", "the wolverine", "logan",
    "inside out", "moana", "zootopia", "the conjuring", "annabelle",
    "aquaman", "wonder woman", "justice league", "suicide squad",
    "the flash", "blue beetle", "shazam", "joker",
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _normalise(title: str) -> str:
    """Lowercase, strip leading 'the '."""
    t = title.lower().strip()
    # Remove common punctuation variants
    t = t.replace("\u2013", "-").replace("\u2014", "-")
    t = t.replace("\u2018", "'").replace("\u2019", "'")
    return t


def sequel_number(title: str, keywords: Optional[List[str]] = None) -> int:
    """
    Return the entry number of `title` in its franchise.
    1 = original / standalone.

    Falls back to keyword signals and then title pattern heuristics.
    """
    t = _normalise(title)

    # 1. Exact lookup
    if t in FRANCHISE_ENTRIES:
        return FRANCHISE_ENTRIES[t]

    # 2. Prefix/substring lookup (handles e.g. slight punctuation differences)
    for key, num in FRANCHISE_ENTRIES.items():
        if key in t or t in key:
            if abs(len(key) - len(t)) < 10:   # avoid spurious short-key matches
                return num

    # 3. Keyword signals (from TMDB keywords field)
    if keywords:
        kw_set = {k.lower() for k in keywords}
        if "sequel" in kw_set:
            return 2

    # 4. Title-pattern heuristics (last resort)
    import re
    t2 = t
    for pattern, num in [
        (r"\bpart\s*(?:v|5)\b|[:\s]5$",          5),
        (r"\bpart\s*(?:iv|4)\b|[:\s]4$|\biv\b",   4),
        (r"\bpart\s*(?:iii|3)\b|[:\s]3$|\biii\b|"
         r"rises\b|revolutions\b|ultimatum\b",      3),
        (r"\bpart\s*(?:ii|2)\b|[:\s]2$|\bii\b|"
         r"reloaded\b|returns\b|strikes back\b|"
         r"evolution\b|apocalypse\b",               2),
    ]:
        if re.search(pattern, t2):
            return num

    return 1


# Titles that are explicitly standalone one-offs (entry 1, no series)
_CONFIRMED_STANDALONE = {
    "oppenheimer", "barbie", "everything everywhere all at once",
    "get out", "us", "nope", "parasite", "midsommar", "hereditary",
    "the revenant", "1917", "dunkirk", "tenet", "interstellar",
    "inception", "the grand budapest hotel", "la la land",
    "whiplash", "birdman", "the shape of water", "nomadland",
    "coda", "triangle of sadness", "tár", "poor things",
    "the holdovers", "anatomy of a fall", "all quiet on the western front",
    "blackkklansman", "jojo rabbit", "marriage story",
}

# Franchise root titles that DO have sequels (entry-1 films that are part of series)
_FRANCHISE_ROOTS = {k for k, v in FRANCHISE_ENTRIES.items() if v == 1} - _CONFIRMED_STANDALONE


def is_franchise(title: str, keywords: Optional[List[str]] = None) -> bool:
    """Return True if the movie is part of a known franchise or series."""
    t = _normalise(title)

    # Explicit standalone override
    if t in _CONFIRMED_STANDALONE:
        return False

    # 1. Exact match with entry > 1 → definitely a sequel/franchise
    if t in FRANCHISE_ENTRIES and FRANCHISE_ENTRIES[t] > 1:
        return True

    # 2. Exact match with entry == 1 → only franchise if it has known sequels
    if t in FRANCHISE_ENTRIES and t in _FRANCHISE_ROOTS:
        return True

    # 3. Substring match: only count as franchise if the matched key is a root
    #    with sequels, or if the matched entry is > 1
    for key, num in FRANCHISE_ENTRIES.items():
        if key != t and (key in t or t in key) and abs(len(key) - len(t)) < 8:
            if num > 1 or key in _FRANCHISE_ROOTS:
                return True

    # 4. Known franchise keyword in title
    if any(kw in t for kw in _FRANCHISE_KW):
        return True

    # 5. TMDB keywords
    if keywords:
        kw_set = {k.lower() for k in keywords}
        if kw_set & {"sequel", "spin off", "based on comic", "cinematic universe",
                     "based on novel or book", "part of series"}:
            return True

    return False


def franchise_name(title: str) -> Optional[str]:
    """Return the canonical franchise name for a title, or None."""
    t = _normalise(title)
    for key, name in FRANCHISE_NAMES.items():
        if key in t:
            return name
    return None