"""
ivan.sports.nba.names

One spelling of a player's name for everything NBA. Every source writes a name its own way: nba_api keeps the generational
suffix ("Jimmy Butler III", "Gary Payton II", "Kelly Oubre Jr."), DraftKings and Rotowire drop it ("Jimmy Butler"), the
books and Rotowire use a nickname where nba_api has the given name ("GG Jackson" / "Gregory Jackson"; "Yang Hansen" /
"Hansen Yang"). The shared normalize_name (ivan.core.utils: accents, punctuation, case, its own overrides) leaves those
apart, so a starter had no history and a headline no player (2026-11-05: starters and news not in the box scores).

    normalize_name(name)   the shared normalize_name, then the generational suffix dropped (jr, sr, ii, iii, iv, v at the
                           end), then the NBA aliases (ALIASES): the key every NBA table and frame uses
    SUFFIXES, ALIASES

Every NBA module imports normalize_name from here, so a name is keyed the same way on input from every source (the DK
pool, Rotowire's starters and news, the props, the box scores, the play-by-play, the standings exports) and in every
read. Rows stored before this rule are re-keyed once by scripts/nba/normalize_names.py.
"""
from __future__ import annotations

from ivan.core.utils import normalize_name as _core_normalize_name

SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}

# the spelling the other sources use -> the one nba_api's box scores use (after the shared normalize and the suffix drop);
# both sides are written normalized
ALIASES = {
    "gregory jackson": "gg jackson",       # GG Jackson II: the books and Rotowire write the given name
    "yang hansen": "hansen yang",          # Rotowire writes the family name first
    "cameron thomas": "cam thomas",
    "kenyon martin": "kj martin",          # Kenyon Martin Jr.
    "ronald holland": "ron holland",       # the shared overrides map "ronald holland ii"; without the suffix it reaches here
    "nicolas claxton": "nic claxton",      # the shared overrides map it too; kept so the rule reads whole here
    "alexandre sarr": "alex sarr",
}


def strip_suffix(name: str) -> str:
    """'jimmy butler iii' -> 'jimmy butler'; a trailing generational suffix only, and never the whole name."""
    toks = name.split()
    while len(toks) > 1 and toks[-1] in SUFFIXES:
        toks.pop()
    return " ".join(toks)


def normalize_name(name) -> str:
    """The NBA key of a name: the shared normalize_name, the generational suffix dropped, the NBA aliases applied."""
    n = strip_suffix(_core_normalize_name(str(name or "")))
    return ALIASES.get(n, n)
