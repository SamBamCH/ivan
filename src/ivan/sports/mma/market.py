# src/ivan/sports/mma/market.py
"""
Fighter moneylines from the-odds-api, matched to the slate's fights.

the-odds-api's MMA feed carries h2h, spreads and total rounds for
DraftKings, nothing on how the fight ends, so this fills the Win column
only. Events are matched to fights as pairs: both of an event's names
against both fighters of a fight, by string similarity after normalising,
which survives transliteration differences ("Darya Zheleznyakova" vs
"daria zhelezniakova") that a token matcher would not.
"""
from __future__ import annotations

import logging
from difflib import SequenceMatcher
from typing import Optional

from ivan.sports.common.odds import get_game_odds
from ivan.sports.mma.ingest import name_tokens

log = logging.getLogger(__name__)
SPORT_KEY = "mma_mixed_martial_arts"


def fetch_moneylines(api_key: str, bookmakers: str = "draftkings", regions: str = "us") -> tuple[list[dict], dict]:
    """[{home, away, commence_time, prices: {name: american}}], quota headers."""
    data, headers = get_game_odds(sport=SPORT_KEY, api_key=api_key, regions=regions, markets="h2h",
                                  oddsFormat="american", dateFormat="iso", bookmakers=bookmakers)
    events = []
    for ev in data or []:
        prices: dict[str, float] = {}
        for bk in ev.get("bookmakers", []):
            for m in bk.get("markets", []):
                if m.get("key") == "h2h":
                    for o in m.get("outcomes", []):
                        prices.setdefault(o["name"], float(o["price"]))
        events.append({"home": ev["home_team"], "away": ev["away_team"], "commence_time": ev.get("commence_time"), "prices": prices})
    quota = {k: headers.get(k) for k in ("x-requests-used", "x-requests-remaining", "x-requests-last")} if headers else {}
    return events, quota


def _sim(a: str, b: str) -> float:
    return SequenceMatcher(None, " ".join(name_tokens(a)), " ".join(name_tokens(b))).ratio()


def _last(a: str) -> str:
    t = name_tokens(a)
    return t[-1] if t else ""


def match_events_to_fights(events: list[dict], fights: list[dict], min_score: float = 0.75) -> dict[int, float]:
    """
    fights: [{"id", "fighters": [{"id", "name"}, {"id", "name"}]}]. Returns
    fighter id -> moneyline for every fight an event matches. A fight is
    matched by the best pairing of the event's two names to its two
    fighters; the pairing's mean similarity must reach min_score, with a
    bonus when last names agree exactly.
    """
    out: dict[int, float] = {}
    for f in fights:
        a, b = f["fighters"]
        best, best_ev, best_pair = 0.0, None, None
        for ev in events:
            if not ev["prices"]:
                continue
            for ea, eb in ((ev["home"], ev["away"]), (ev["away"], ev["home"])):
                s = (_sim(ea, a["name"]) + _sim(eb, b["name"])) / 2
                s += 0.1 * ((_last(ea) == _last(a["name"])) + (_last(eb) == _last(b["name"]))) / 2
                if s > best:
                    best, best_ev, best_pair = s, ev, (ea, eb)
        if best_ev is None or best < min_score:
            log.warning("no moneyline event matched fight %s (best score %.2f)", f["id"], best)
            continue
        ea, eb = best_pair
        if ea in best_ev["prices"] and eb in best_ev["prices"]:
            out[a["id"]] = best_ev["prices"][ea]
            out[b["id"]] = best_ev["prices"][eb]
    return out


def moneylines_for_slate(api_key: Optional[str], fights: list[dict], bookmakers: str = "draftkings") -> tuple[dict[int, float], dict]:
    """Win odds per fighter id for the slate, plus quota; empty when there is no key."""
    if not api_key:
        return {}, {}
    events, quota = fetch_moneylines(api_key, bookmakers=bookmakers)
    return match_events_to_fights(events, fights), quota
