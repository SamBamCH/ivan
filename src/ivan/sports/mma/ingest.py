# src/ivan/sports/mma/ingest.py
"""
Build the slate: the DraftKings fighter pool, the fight-level and fighter-
level markets, and each fighter's win probability and projection.

The pool comes from the DraftKings API (the slate is picked automatically:
the Classic MMA draft group with the most contests, or --draft-group) or
from DK's "Export to CSV" (player_ids.csv). Moneylines come from
the-odds-api when ODDS_API_KEY is set. Everything else is hand-entered in
two CSVs in data/mma/input (paths in config/mma.json):

  fight_odds.csv   fight,u60 sec,1st round,itd        one row per bout
  player_odds.csv  player,1st round win,ITD,Win,sig strikes,takedowns

On a new card ingest writes both files fresh, names in and Win filled from
the API, for you to complete; on a card they already describe it reads
them as overrides (a non-empty cell wins over the API) and leaves them
alone. Odds names are matched to DK names by exact normalised match, then
by last name, then by any unique shared token; config name_aliases settles
the rest.
"""
from __future__ import annotations

import csv
import logging
import re
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

from sqlalchemy.orm import Session

from ivan.sports.mma.fight import DEFAULT_SCORING, Fight, Fighter, project_fighter
from ivan.sports.mma.models import MMAFight, MMAPlayer
from ivan.sports.mma.odds import to_float

log = logging.getLogger(__name__)

GENERATIONAL = {"jr", "sr", "ii", "iii", "iv"}


def name_tokens(name: str) -> list[str]:
    s = unicodedata.normalize("NFKD", name or "").encode("ascii", "ignore").decode()
    return [t for t in re.sub(r"[^a-z ]", "", s.lower()).split() if t]


def fight_key(a: str, b: str) -> str:
    """Deterministic fight id from the two fighters' DK abbreviations."""
    return "-".join(sorted([a.strip(), b.strip()]))


# ------------------------------------------------------------------ the DK pool
def _parse_game_info(game_info: str) -> tuple[Optional[str], Optional[str], Optional[datetime]]:
    """'Jauregui@Demopoulos 09/26/2026 05:00PM ET' -> (away, home, start)."""
    parts = (game_info or "").rsplit(" ", 3)
    if not parts or "@" not in parts[0]:
        return None, None, None
    away, home = [x.strip() for x in parts[0].split("@", 1)]
    start = None
    if len(parts) >= 3:
        try:
            start = datetime.strptime(f"{parts[1]} {parts[2]}", "%m/%d/%Y %I:%M%p")
        except ValueError:
            start = None
    return away, home, start


def load_players_csv(path: str | Path) -> list[dict]:
    """Fighter rows from the DK export. Fighters without a listed fight are dropped with a warning."""
    rows = []
    with open(path, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            name = (r.get("Name") or "").strip()
            if not name:
                continue
            away, home, start = _parse_game_info(r.get("Game Info", ""))
            team = (r.get("TeamAbbrev") or "").strip()
            if not (away and home):
                log.warning("%s has no fight listed (Game Info %r); dropped from the slate", name, r.get("Game Info"))
                continue
            opponent = home if team == away else away if team == home else None
            if opponent is None:
                log.warning("%s: TeamAbbrev %r matches neither side of %r; dropped", name, team, r.get("Game Info"))
                continue
            rows.append({"id": int(r["ID"]), "name": name, "team": team, "opponent": opponent,
                         "fight_id": fight_key(team, opponent), "gametime": start,
                         "salary": int(str(r.get("Salary", "0")).replace(",", "") or 0),
                         "status": (r.get("Status") or "").strip() or None,
                         "avg_fpts": to_float(r.get("AvgPointsPerGame"))})
    return rows


def load_players_dk(draft_group_id: int) -> list[dict]:
    """
    The pool from the DraftKings API, the same call the other sports make.
    Untested against a live MMA slate as of 2026-10-10: DK's competition
    payload for MMA may not name the two fighters the way the shared parser
    expects, in which case the opponent is missing and the fighter is dropped
    with a warning. The CSV export is the known-good path.
    """
    from ivan.sports.common.draftkings import ingest_players
    rows = []
    for p in ingest_players("mma", dedupe_by_name=False, draft_group_id=draft_group_id):
        if not p.team or not p.opponent:
            log.warning("%s: no opponent in the DK payload; dropped", p.name)
            continue
        rows.append({"id": int(p.dk_id), "name": p.name, "team": p.team, "opponent": p.opponent,
                     "fight_id": fight_key(p.team, p.opponent), "gametime": p.gametime,
                     "salary": int(p.salary), "status": p.status, "avg_fpts": None})
    return rows


# ------------------------------------------------------------------ odds files
class NameMatcher:
    """Resolve an odds-file name to a pool row; see the module docstring for the order tried."""

    def __init__(self, rows: Iterable[dict], aliases: Optional[dict] = None):
        self.rows = list(rows)
        self.tokens = {r["id"]: name_tokens(r["name"]) for r in self.rows}
        self.aliases = {k.lower().strip(): v for k, v in (aliases or {}).items()}

    def find(self, name: str) -> Optional[dict]:
        name = self.aliases.get((name or "").lower().strip(), name)
        q = name_tokens(name)
        if not q:
            return None
        for r in self.rows:                                      # 1. exact
            if self.tokens[r["id"]] == q:
                return r
        if len(q) == 1:                                          # 2. a lone token as a last name
            hits = [r for r in self.rows if self.tokens[r["id"]] and self.tokens[r["id"]][-1] == q[0]]
            if len(hits) == 1:
                return hits[0]
        hits = [r for r in self.rows if self.tokens[r["id"]] and self.tokens[r["id"]][-1] == q[-1]]
        if len(hits) == 1:                                       # 3. unique last-name match
            return hits[0]
        def surname(tokens: list[str]) -> Optional[str]:         # the last token that is not a generational suffix
            core = [t for t in tokens if t not in GENERATIONAL]
            return core[-1] if core else None

        hits = [r for r in self.rows if surname(self.tokens[r["id"]]) == surname(q)]
        if len(hits) == 1:                                       # 4. unique surname match ignoring Jr / III
            log.info("matched odds name %r to DK name %r", name, hits[0]["name"])
            return hits[0]
        qs = set(q) - GENERATIONAL                               # 5. a shared token that is a surname on one side
        hits = [r for r in self.rows                             #    (name order swapped); a shared first name alone never matches
                if surname(self.tokens[r["id"]]) in qs or surname(q) in (set(self.tokens[r["id"]]) - GENERATIONAL)]
        if len(hits) == 1:
            log.info("matched odds name %r to DK name %r", name, hits[0]["name"])
            return hits[0]
        return None


def load_fight_odds_csv(path: str | Path) -> list[dict]:
    out = []
    with open(path, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            fight = (r.get("fight") or "").strip()
            if " vs " not in fight:
                continue
            a, b = [x.strip() for x in fight.split(" vs ", 1)]
            out.append({"a": a, "b": b, "u60_odds": to_float(r.get("u60 sec")),
                        "r1_odds": to_float(r.get("1st round")), "itd_odds": to_float(r.get("itd"))})
    return out


def load_player_odds_csv(path: str | Path) -> list[dict]:
    out = []
    with open(path, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            name = (r.get("player") or "").strip()
            if not name:
                continue
            out.append({"name": name, "r1_win_odds": to_float(r.get("1st round win")), "itd_odds": to_float(r.get("ITD")),
                        "win_odds": to_float(r.get("Win")), "sig_strikes": to_float(r.get("sig strikes")),
                        "takedowns": to_float(r.get("takedowns"))})
    return out


# ------------------------------------------------------------------ build the slate
def five_round_fight_ids(fights: dict[str, dict], n: int) -> set[str]:
    """The latest n fights on the card by start time: the main event(s) are five rounds."""
    timed = [(fid, f["gametime"]) for fid, f in fights.items() if f.get("gametime")]
    if n <= 0 or not timed:
        if n > 0:
            log.warning("no fight times parsed; every fight is treated as 3 rounds")
        return set()
    timed.sort(key=lambda x: x[1], reverse=True)
    return {fid for fid, _ in timed[:n]}


def pick_slate(game_type: str = "Classic") -> dict:
    """The MMA draft group to ingest: the `game_type` slate with the most contests."""
    from ivan.sports.common.draftkings import list_slates
    slates = list_slates("mma")
    wanted = [s for s in slates if s["game_type"] == game_type] or slates
    if not wanted:
        raise SystemExit("DraftKings lists no upcoming MMA slates")
    return max(wanted, key=lambda s: s["contests"])


FIGHT_ODDS_COLS = ["fight", "u60 sec", "1st round", "itd"]
PLAYER_ODDS_COLS = ["player", "1st round win", "ITD", "Win", "sig strikes", "takedowns"]


def write_odds_templates(fights: dict[str, dict], fight_path: Path, player_path: Path, win_odds: dict[int, float]) -> None:
    """Fresh fight_odds.csv / player_odds.csv for this card: names in, Win from the API, the rest blank."""
    ordered = sorted(fights.values(), key=lambda f: (f["gametime"] or datetime.min, f["id"]))
    for path in (fight_path, player_path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            prev = path.with_name(f"{path.stem}.prev{path.suffix}")
            path.replace(prev)
            log.info("%s described another card; moved to %s", path.name, prev.name)
    with open(fight_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(FIGHT_ODDS_COLS)
        for fi in ordered:
            a, b = sorted(fi["fighters"], key=lambda x: -x["salary"])
            w.writerow([f"{a['name']} vs {b['name']}", "", "", ""])
    with open(player_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(PLAYER_ODDS_COLS)
        for fi in ordered:
            for x in sorted(fi["fighters"], key=lambda x: -x["salary"]):
                wo = win_odds.get(x["id"])
                w.writerow([x["name"], "", "", "" if wo is None else int(wo), "", ""])


def ingest_slate(session: Session, cfg: dict, *, players_csv: str | Path | None = None,
                 draft_group_id: Optional[int] = None, fight_odds_csv: str | Path | None = None,
                 player_odds_csv: str | Path | None = None, num_5_round_fights: Optional[int] = None,
                 odds_api_key: Optional[str] = None, fetch_contests: bool = True) -> dict:
    """
    Replace the slate tables (mma_player, mma_fight) and return a summary.
    Pool: players_csv if given, else the DK API (draft_group_id, or the
    slate pick_slate chooses). Markets: the-odds-api moneylines when a key
    is given, then the two CSVs as overrides. Contests and payout tables for
    the slate are fetched too unless fetch_contests is False. Nothing is
    written until every input has been read.
    """
    paths = cfg.get("paths", {})
    scoring = {**DEFAULT_SCORING, **cfg.get("scoring", {})}
    if num_5_round_fights is None:
        num_5_round_fights = int(cfg.get("slate", {}).get("num_5_round_fights", 1))
    fight_path = Path(fight_odds_csv or paths["fight_odds_csv"])
    player_path = Path(player_odds_csv or paths["player_odds_csv"])

    slate = None
    if players_csv is not None:
        rows = load_players_csv(players_csv)
    else:
        if draft_group_id is None:
            slate = pick_slate(cfg.get("slate", {}).get("game_type", "Classic"))
            draft_group_id = slate["id"]
        rows = load_players_dk(int(draft_group_id))
    if not rows:
        raise SystemExit("no fighters loaded; nothing changed")

    fights: dict[str, dict] = {}
    for r in rows:
        f = fights.setdefault(r["fight_id"], {"id": r["fight_id"], "fighters": [], "gametime": r["gametime"],
                                              "u60_odds": None, "r1_odds": None, "itd_odds": None})
        f["fighters"].append(r)
        f["gametime"] = f["gametime"] or r["gametime"]
    for fid in [fid for fid, f in fights.items() if len(f["fighters"]) != 2]:
        log.warning("fight %s has %d fighters, skipped", fid, len(fights[fid]["fighters"]))
        for r in fights[fid]["fighters"]:
            rows.remove(r)
        del fights[fid]
    by_id = {r["id"]: r for r in rows}

    # 1. moneylines from the-odds-api
    quota: dict = {}
    api_win: dict[int, float] = {}
    if odds_api_key:
        from ivan.sports.mma.market import moneylines_for_slate
        api_win, quota = moneylines_for_slate(odds_api_key, list(fights.values()), cfg.get("odds_api", {}).get("bookmakers", "draftkings"))
        for pid, price in api_win.items():
            by_id[pid]["win_odds"] = price

    # 2. the hand-entered CSVs: overrides when they describe this card, templates otherwise
    matcher = NameMatcher(rows, cfg.get("name_aliases"))
    unmatched: list[str] = []
    fight_rows = load_fight_odds_csv(fight_path) if fight_path.exists() else []
    player_rows = load_player_odds_csv(player_path) if player_path.exists() else []
    # the files describe this card when most of their names are in the pool; one stray lookalike is not enough
    names = [fo["a"] for fo in fight_rows] + [fo["b"] for fo in fight_rows] + [po["name"] for po in player_rows]
    n_hit = sum(1 for n in names if matcher.find(n))
    matched_any = bool(names) and n_hit * 2 > len(names)
    if names and not matched_any:
        log.warning("%s / %s: %d of %d names are in this slate (an earlier card?): writing fresh templates",
                    fight_path.name, player_path.name, n_hit, len(names))
    if not matched_any:
        write_odds_templates(fights, fight_path, player_path, api_win)
        fight_rows, player_rows = [], []
    fights_with_odds = 0
    for fo in fight_rows:
        a, b = matcher.find(fo["a"]), matcher.find(fo["b"])
        if not a or not b or a["fight_id"] != b["fight_id"]:
            unmatched.append(f"{fo['a']} vs {fo['b']}")
            continue
        vals = {k: fo[k] for k in ("u60_odds", "r1_odds", "itd_odds") if fo[k] is not None}
        fights[a["fight_id"]].update(vals)
        if vals:
            fights_with_odds += 1
    players_in_csv = 0
    for po in player_rows:
        r = matcher.find(po["name"])
        if not r:
            unmatched.append(po["name"])
            continue
        by_id[r["id"]].update({k: v for k, v in po.items() if k != "name" and v is not None})
        players_in_csv += 1

    five = five_round_fight_ids(fights, num_5_round_fights)
    for fid in fights:
        fights[fid]["rounds"] = 5 if fid in five else 3
    for r in rows:
        f = Fighter(id=r["id"], name=r["name"], salary=r["salary"], win_odds=r.get("win_odds"), itd_odds=r.get("itd_odds"),
                    r1_win_odds=r.get("r1_win_odds"), sig_strikes=r.get("sig_strikes") or 0.0, takedowns=r.get("takedowns") or 0.0)
        r["win_prob"], r["projection"] = project_fighter(f, fights[r["fight_id"]]["itd_odds"], scoring)

    # 3. write: clear what we replace, insert, commit once
    session.query(MMAPlayer).delete(synchronize_session=False)
    session.query(MMAFight).delete(synchronize_session=False)
    for fid, f in fights.items():
        f1, f2 = f["fighters"]
        session.add(MMAFight(id=fid, fighter1_id=f1["id"], fighter2_id=f2["id"], rounds=f["rounds"], gametime=f["gametime"],
                             u60_odds=f["u60_odds"], r1_odds=f["r1_odds"], itd_odds=f["itd_odds"]))
    for r in rows:
        session.add(MMAPlayer(id=r["id"], name=r["name"], team=r["team"], opponent=r["opponent"], fight_id=r["fight_id"],
                              gametime=r["gametime"], salary=r["salary"], status=r["status"], avg_fpts=r["avg_fpts"],
                              win_odds=r.get("win_odds"), itd_odds=r.get("itd_odds"), r1_win_odds=r.get("r1_win_odds"),
                              sig_strikes=r.get("sig_strikes"), takedowns=r.get("takedowns"),
                              win_prob=r["win_prob"], projection=r["projection"]))
    session.commit()

    # 4. the slate's contests and their payout tables
    contests: list[dict] = []
    contest_counts: dict = {}
    if fetch_contests and draft_group_id is not None:
        from ivan.sports.mma import contests as dkc
        try:
            contest_counts = dkc.fetch_contests(session)
            contest_counts.update(dkc.fetch_payouts(session, draft_group_id=int(draft_group_id)))
        except Exception as e:                                   # the slate is in; contests are a convenience
            log.warning("contest fetch failed: %s", e)
        contests = dkc.contests_for_slate(session, int(draft_group_id))

    missing_cols = {"fight": {}, "player": {}}
    for fid, f in fights.items():
        m = [k for k in ("u60_odds", "r1_odds", "itd_odds") if f[k] is None]
        if m:
            missing_cols["fight"][fid] = m
    for r in rows:
        m = [k for k in ("win_odds", "itd_odds", "r1_win_odds", "sig_strikes", "takedowns") if r.get(k) is None]
        if m:
            missing_cols["player"][r["name"]] = m

    return {"draft_group_id": draft_group_id, "slate": slate, "players": len(rows), "fights": len(fights),
            "players_with_odds": sum(1 for r in rows if r.get("win_odds") is not None), "api_win_odds": len(api_win),
            "players_in_csv": players_in_csv, "fights_with_odds": fights_with_odds, "five_round": sorted(five),
            "unmatched": unmatched, "templates_written": not matched_any, "fight_odds_csv": str(fight_path),
            "player_odds_csv": str(player_path), "missing": missing_cols, "quota": quota,
            "contests": contests, "contest_counts": contest_counts,
            "no_player_odds": sorted(r["name"] for r in rows if r.get("win_odds") is None),
            "no_fight_odds": sorted(fid for fid, f in fights.items() if f["itd_odds"] is None)}


def format_summary(s: dict) -> str:
    lines = []
    if s.get("slate"):
        sl = s["slate"]
        lines.append(f"slate: DK draft group {sl['id']} ({sl['game_type']}, {sl['contests']} contests, {sl['start']}) {sl['name']!r}")
    lines.append(f"{s['players']} fighters in {s['fights']} fights; win odds for {s['players_with_odds']} "
                 f"({s['api_win_odds']} from the-odds-api, {s['players_in_csv']} rows from {Path(s['player_odds_csv']).name}); "
                 f"finish markets for {s['fights_with_odds']} fights")
    if s.get("quota"):
        q = s["quota"]
        lines.append(f"the-odds-api: {q.get('x-requests-last')} credit(s) used, {q.get('x-requests-remaining')} remaining")
    if s["five_round"]:
        lines.append("5-round fights: " + ", ".join(s["five_round"]))
    if s["templates_written"]:
        lines.append(f"new card: wrote {Path(s['fight_odds_csv']).name} and {Path(s['player_odds_csv']).name} with the names "
                     "(and Win odds) filled in. Add the finish markets and volume lines, then run ingest again.")
    nf, npl = s["missing"]["fight"], s["missing"]["player"]
    if nf:
        lines.append(f"fights still without finish markets (defaults apply): {len(nf)} of {s['fights']}"
                     + ("" if len(nf) > 6 else ": " + ", ".join(nf)))
    if npl:
        cols = sorted({c for m in npl.values() for c in m})
        lines.append(f"fighters with blank columns: {len(npl)} of {s['players']} (missing: {', '.join(cols)})")
    if s["unmatched"]:
        lines.append("odds rows that matched nobody in the pool: " + "; ".join(s["unmatched"]))
    if s.get("contest_counts"):
        c = s["contest_counts"]
        lines.append(f"contests: {c.get('upcoming', 0)} upcoming MMA contests ({c.get('new', 0)} new); payout tables fetched for "
                     f"{c.get('fetched', 0)}, {c.get('failed', 0)} failed")
    if s.get("contests"):
        lines.append("this slate's contests (pass the id to sim.py --contest-id):")
        for c in s["contests"][:12]:
            lines.append(f"  {c['id']}  ${c['fee']:<6.2f} {c['entries']:>7,} entries  {'payouts' if c['payouts'] else 'NO PAYOUTS':<10} {c['name']}")
        if len(s["contests"]) > 12:
            lines.append(f"  ... and {len(s['contests']) - 12} more")
    return "\n".join(lines)


# ------------------------------------------------------------------ read the slate back
def load_fights(session: Session, scoring: Optional[dict] = None) -> list[Fight]:
    """Fight objects for the simulator and the generator, from the slate tables."""
    scoring = {**DEFAULT_SCORING, **(scoring or {})}
    players = {p.id: p for p in session.query(MMAPlayer).all()}

    def fighter(pid: int) -> Fighter:
        p = players[pid]
        return Fighter(id=p.id, name=p.name, salary=p.salary, win_odds=p.win_odds, itd_odds=p.itd_odds,
                       r1_win_odds=p.r1_win_odds, sig_strikes=p.sig_strikes or 0.0, takedowns=p.takedowns or 0.0,
                       win_prob=p.win_prob if p.win_prob is not None else 0.5)

    out = []
    for f in session.query(MMAFight).order_by(MMAFight.gametime, MMAFight.id).all():
        if f.fighter1_id not in players or f.fighter2_id not in players:
            continue
        out.append(Fight(id=f.id, f1=fighter(f.fighter1_id), f2=fighter(f.fighter2_id), rounds=f.rounds,
                         u60_odds=f.u60_odds, r1_odds=f.r1_odds, itd_odds=f.itd_odds, scoring=scoring))
    return out
