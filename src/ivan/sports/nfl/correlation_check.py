# src/ivan/sports/nfl/correlation_check.py
"""
The working slate's correlations, game by game, laid out to be checked before a sim trusts them.

For every game of the pool (correlated_samples.assemble_games, the same build the sim samples from):

- the slot pairs that move a lineup (QB-WR1, QB-PTS, RB1-DST, QB x opp QB, ...): the league's Spearman, the team's own
  history with its count and standard error, the same with tonight's QB only, what the sim is given (own shrunk toward
  the league by LEAGUE_SHRINK_GAMES), and what the last sim drew for the players in those slots; flags where the team's
  own history disagrees with what is used beyond its noise, where a new QB makes the history somebody else's, and where
  the draw missed its target;
- the players in depth order with their target (requested) Pearson matrix and, when a sim report of a contest on the
  game is stored, the sampled one;
- per pair, the history behind it (pair_history: one point per team-game, tonight's QB's games marked).

Terminal: scripts/nfl/correlations.py (the desk's correlation section is the same call).
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sqlalchemy.orm import Session

from ivan.sports.nfl.refactor import team_correlation as tc

# slot pairs worth checking: same team, then (team slot, opponent slot)
WITHIN_PAIRS = [("QB", "WR1"), ("QB", "WR2"), ("QB", "WR3"), ("QB", "TE1"), ("QB", "RB1"), ("QB", "PTS"), ("RB1", "PTS"),
                ("WR1", "WR2"), ("RB1", "RB2"), ("RB1", "DST"), ("QB", "DST"), ("K", "PTS"), ("DST", "PTS")]
CROSS_PAIRS = [("QB", "QB"), ("QB", "WR1"), ("WR1", "WR1"), ("RB1", "RB1"), ("QB", "DST"), ("RB1", "DST"), ("DST", "DST"), ("PTS", "PTS")]
NEW_QB_GAMES = 8            # tonight's QB with fewer games than this in the team's history: the matrix is mostly another QB's
STRUCTURAL_TOL = 0.12      # a DST pair's draw this far from its history (Pearson): the structural build disagrees with it
DIFFERS_Z = 2.0             # own history vs used beyond this many standard errors: worth a look


def _f(x) -> Optional[float]:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def spearman_se(rho: Optional[float], n: int) -> Optional[float]:
    """Approximate standard error of a Spearman correlation from n pairs (Bonett & Wright)."""
    if n is None or n <= 3:
        return None
    r = rho or 0.0
    return math.sqrt((1 + r * r / 2) / (n - 3))


def _rho(x: np.ndarray, y: np.ndarray, min_obs: int = tc.MIN_PAIR_OBS) -> Tuple[Optional[float], int]:
    m = ~np.isnan(x) & ~np.isnan(y)
    n = int(m.sum())
    if n < min_obs:
        return None, n
    r, _ = spearmanr(x[m], y[m])
    return _f(r), n


def _cross_frame(session: Session, team: str, before=None) -> pd.DataFrame:
    """A team's games with its opponent's slots alongside (opp_QB, ...)."""
    w = tc.league_wide_table(session)
    g = tc.team_games(session, team, before)
    return g.merge(tc._opponent_frame(w), left_on=["season", "week", "opponent"], right_on=["season", "week", "team"],
                   how="inner", suffixes=("", "_o"))


def _qb_now(players: list, team: str) -> Optional[str]:
    qbs = [p for p in players if p.team == team and p.slot_label == "QB"]
    qbs.sort(key=lambda p: -(p.fpts_mean or 0))
    return qbs[0].name if qbs else None


def _same_qb(names: pd.Series, qb: Optional[str]) -> np.ndarray:
    if not qb:
        return np.zeros(len(names), bool)
    q = qb.strip().lower()
    return names.fillna("").str.strip().str.lower().eq(q).to_numpy()


def _people(g) -> Tuple[List[int], List[dict]]:
    """Indices of g.players keeping one row per person (the FLEX row of a showdown pair) and their labels."""
    keep: Dict[str, int] = {}
    for j, p in enumerate(g.players):
        k = (p.name, p.team)
        if k not in keep or (g.players[keep[k]].roster_slot_id == 511 and p.roster_slot_id != 511):
            keep[k] = j
    idx = sorted(keep.values())
    labels = [{"name": g.players[j].name, "team": g.players[j].team, "slot": g.players[j].slot_label,
               "position": g.players[j].position, "salary": g.players[j].salary, "mean": _f(g.mu[j]), "sd": _f(g.sigma[j]),
               "own": _f(g.players[j].ownership_proj)} for j in idx]
    return idx, labels


def _sampled_for(reports: list, teams: tuple, labels: List[dict]) -> Optional[dict]:
    """The newest stored sim report's sampled matrix for this game, aligned to labels by (name, team), or None."""
    for rep in reports:
        corr = (rep.report or {}).get("correlation") or {}
        for rg in corr.get("games", []):
            if set(rg.get("teams") or []) != set(teams):
                continue
            pos = {(l["name"], l["team"]): i for i, l in enumerate(rg.get("labels", []))}
            k = len(labels)
            M = np.full((k, k), np.nan)
            S = np.asarray(rg.get("sampled") or [], float)
            ix = [pos.get((l["name"], l["team"])) for l in labels]
            for a in range(k):
                for b in range(k):
                    if ix[a] is not None and ix[b] is not None and S.size:
                        M[a, b] = S[ix[a], ix[b]]
            return {"matrix": M, "contest": rep.contest_name, "contest_id": rep.contest_id, "at": rep.created_at,
                    "n_samples": corr.get("n_samples"), "se": _f(corr.get("standard_error")),
                    "max_abs_error": _f(rg.get("max_abs_error")), "mean_abs_error": _f(rg.get("mean_abs_error"))}
    return None


def _slot_people(labels: List[dict], team: str, slot: str) -> List[int]:
    return [i for i, l in enumerate(labels) if l["team"] == team and l["slot"] == slot]


def _pair_row(kind: str, ta: str, sa: str, tb: str, sb: str, league: float, own: Optional[float], n: int,
              own_qb: Optional[float], n_qb: int, used: float, target_m: np.ndarray, sampled: Optional[dict], labels: List[dict]) -> dict:
    se = spearman_se(own, n)
    ia, ib = _slot_people(labels, ta, sa), _slot_people(labels, tb, sb)
    tgt = smp = None
    who = None
    if ia and ib and ia[0] != ib[0]:
        tgt = _f(target_m[ia[0], ib[0]])
        if sampled is not None:
            smp = _f(sampled["matrix"][ia[0], ib[0]])
        who = (labels[ia[0]]["name"], labels[ib[0]]["name"])
    flags = []
    if own is not None and se and abs(own - used) > DIFFERS_Z * se:
        flags.append("own history differs")
    if own is None:
        flags.append("no own history")
    # a DST is built from the opponent's drawn score plus takeaways, not drawn on the matrix: its pairs have no target of
    # their own, so the draw is held against the history it should reproduce (Spearman to Pearson, a looser bar)
    structural = "DST" in (sa, sb)
    if smp is not None and tgt is not None and sampled.get("se"):
        if structural:
            hist_p = 2 * math.sin(math.pi * used / 6)
            if abs(smp - hist_p) > STRUCTURAL_TOL:
                flags.append("DST draw vs history")
        elif abs(smp - tgt) > 3 * sampled["se"]:
            flags.append("draw missed target")
    return {"kind": kind, "a_team": ta, "a": sa, "b_team": tb, "b": sb, "league": _f(league), "own": own, "n": n, "se": se,
            "own_qb": own_qb, "n_qb": n_qb, "used": _f(used), "shift": _f(used - league) if league is not None else None,
            "target": tgt, "sampled": smp, "players": who, "structural": structural, "flags": flags}


def game_check(session: Session, g, reports: list) -> dict:
    """One game's check (see the module docstring). g is a correlated_samples.GameSpec."""
    from ivan.sports.nfl.actuals import history_cutoff
    from ivan.sports.nfl.refactor.dst import matchup_lines
    a, b = g.teams
    before = history_cutoff(session)
    league_w, league_x, n_league = tc.league_matrices(session)
    idx, labels = _people(g)
    k = len(labels)
    target = np.asarray(g.requested, float)[np.ix_(idx, idx)] if k else np.zeros((0, 0))
    sampled = _sampled_for(reports, g.teams, labels)

    teams = {}
    for t in (a, b):
        used_w, ok = tc.build_team_correlation(session, t, before=before)
        hist = tc.team_games(session, t, before)
        qb = _qb_now(g.players, t)
        same = _same_qb(hist.qb_name, qb) if len(hist) else np.zeros(0, bool)
        summary = tc.team_history_summary(session, t)
        teams[t] = {"used": used_w, "own_source": ok, "hist": hist, "same_qb": same, "qb": qb, "qb_games": int(same.sum()),
                    "games": int(len(hist)), "last": summary.get("last"), "hist_qb": summary.get("qb")}
    used_x, cross_ok = tc.build_cross_team_correlation(session, a, b)

    pairs: List[dict] = []
    for t in (a, b):
        T = teams[t]
        X = T["hist"]
        for sa, sb in WITHIN_PAIRS:
            i, j = tc.SLOT_IDX[sa], tc.SLOT_IDX[sb]
            own, n = _rho(X[sa].to_numpy(float), X[sb].to_numpy(float)) if len(X) else (None, 0)
            sq = T["same_qb"]
            own_qb, n_qb = _rho(X[sa].to_numpy(float)[sq], X[sb].to_numpy(float)[sq]) if sq.any() else (None, 0)
            pairs.append(_pair_row(t, t, sa, t, sb, league_w[i, j], own, n, own_qb, n_qb, T["used"][i, j], target, sampled, labels))
    # cross: A's slot against B's slot, both directions (B's QB vs A's WR1 is another pair)
    fa, fb = _cross_frame(session, a), _cross_frame(session, b)
    for (t1, t2, f1, f2) in ((a, b, fa, fb), (b, a, fb, fa)):
        for sa, sb in CROSS_PAIRS:
            if t1 == b and sa == sb:          # symmetric slot pairs once
                continue
            i, j = tc.SLOT_IDX[sa], tc.SLOT_IDX[sb]
            # pooled like build_cross_team_correlation: t1's slot vs its opponents', and t2's opponents' vs t2's slot
            x = np.concatenate([f1[sa].to_numpy(float), f2[f"opp_{sa}"].to_numpy(float)]) if len(f1) or len(f2) else np.zeros(0)
            y = np.concatenate([f1[f"opp_{sb}"].to_numpy(float), f2[sb].to_numpy(float)]) if len(f1) or len(f2) else np.zeros(0)
            own, n = _rho(x, y) if len(x) else (None, 0)
            used = used_x[i, j] if t1 == a else used_x[j, i]
            pairs.append(_pair_row("cross", t1, sa, t2, sb, league_x[i, j], own, n, None, 0, used, target, sampled, labels))

    alerts = []
    for t in (a, b):
        T = teams[t]
        if not T["own_source"]:
            alerts.append({"level": "warn", "text": f"{t}: {T['games']} games of history, under {tc.MIN_GAMES_WITHIN}: the league's matrix is used"})
        if T["qb"] and T["qb_games"] < NEW_QB_GAMES:
            alerts.append({"level": "warn", "text": f"{t}: tonight's QB {T['qb']} has {T['qb_games']} of the team's {T['games']} games; "
                                                    f"its QB pairs are mostly {T['hist_qb'] if T['hist_qb'] and T['hist_qb'] != T['qb'] else 'other QBs'}'"})
    differs = [p for p in pairs if "own history differs" in p["flags"]]
    missed = [p for p in pairs if "draw missed target" in p["flags"]]
    if differs:
        alerts.append({"level": "info", "text": f"{len(differs)} pair(s) where the team's own history is beyond {DIFFERS_Z:g} se of what is used "
                                                f"(shrunk {tc.LEAGUE_SHRINK_GAMES:g} games toward the league)"})
    dst = [p for p in pairs if "DST draw vs history" in p["flags"]]
    if dst:
        alerts.append({"level": "warn", "text": f"{len(dst)} DST pair(s) whose draw is more than {STRUCTURAL_TOL:g} from the history "
                                                f"(the DST is built from the drawn score, not the matrix)"})
    if missed:
        alerts.append({"level": "bad", "text": f"{len(missed)} pair(s) where the last sim's draw is off its target by more than 3 se"})
    if sampled and sampled.get("max_abs_error") is not None and sampled.get("se") and sampled["max_abs_error"] > 4 * sampled["se"]:
        alerts.append({"level": "bad", "text": f"sampled vs target: largest error {sampled['max_abs_error']:.3f} (noise ~{sampled['se']:.3f})"})

    lines = None
    ml = matchup_lines(session, a, b)
    if ml is not None and g.implied:
        total, home_margin, a_home = ml
        home, away = (a, b) if a_home else (b, a)
        fav = home if home_margin > 0 else away
        lines = {"home": home, "away": away, "total": _f(total), "spread": _f(-abs(home_margin)), "fav": fav if abs(home_margin) > 0 else None,
                 "implied": {t: _f(v) for t, v in g.implied.items()}}
    level = "bad" if any(x["level"] == "bad" for x in alerts) else "warn" if any(x["level"] == "warn" for x in alerts) else "ok"
    gametime = min((p.gametime for p in g.players if p.gametime), default=None)
    return {"key": f"{a}-{b}", "teams": [a, b], "away": lines["away"] if lines else a, "home": lines["home"] if lines else b,
            "gametime": gametime, "lines": lines, "sources": g.sources, "cross_own": cross_ok, "n_league": n_league,
            "history": {t: {"games": teams[t]["games"], "qb": teams[t]["qb"], "qb_games": teams[t]["qb_games"], "last": teams[t]["last"],
                            "own": teams[t]["own_source"]} for t in (a, b)},
            "labels": labels, "target": np.round(target, 3).tolist(),
            "sampled": None if sampled is None else {**{k: v for k, v in sampled.items() if k != "matrix"},
                                                     "matrix": [[_f(round(v, 3)) if v == v else None for v in row] for row in sampled["matrix"]]},
            "pairs": pairs, "alerts": alerts, "level": level}


def slate_check(session: Session) -> dict:
    """Every game of the working slate (see the module docstring), the independent rows, the method."""
    from ivan.sports.nfl.correlated_samples import assemble_games
    from ivan.sports.nfl.models import ContestSimReport
    games, independent, method, _ = assemble_games(session)
    reports = session.query(ContestSimReport).order_by(ContestSimReport.created_at.desc()).all()
    out = [game_check(session, g, reports) for g in games]
    out.sort(key=lambda x: (x["gametime"] or pd.Timestamp.max, x["key"]))
    return {"games": out, "method": method, "shrink": tc.LEAGUE_SHRINK_GAMES,
            "independent": [{"name": p.name, "team": p.team, "slot": p.slot_label, "reason": getattr(p, "_independent_reason", "")}
                            for p in independent if p.roster_slot_id != 511]}


def pair_history(session: Session, team: str, a: str, b: str, opponent: Optional[str] = None) -> dict:
    """
    The team-games behind one pair: slot a of `team` against slot b of `team` (or of its opponent in each game when
    `opponent` is given; the opponent tonight only names the pair). One point per game, tonight's QB's games marked.
    """
    from ivan.sports.nfl.actuals import history_cutoff
    if opponent:
        f = _cross_frame(session, team)
        x, y = f[a].to_numpy(float), f[f"opp_{b}"].to_numpy(float)
    else:
        f = tc.team_games(session, team, history_cutoff(session))
        x, y = f[a].to_numpy(float), f[b].to_numpy(float)
    from ivan.sports.nfl.models import NFLPlayer
    qb = _qb_now(session.query(NFLPlayer).filter(NFLPlayer.team == team).all(), team)
    same = _same_qb(f.qb_name, qb) if len(f) else np.zeros(0, bool)
    pts = []
    for k in range(len(f)):
        if np.isnan(x[k]) or np.isnan(y[k]):
            continue
        pts.append({"x": float(x[k]), "y": float(y[k]), "season": int(f.season.iloc[k]), "week": int(f.week.iloc[k]),
                    "opp": str(f.opponent.iloc[k]), "qb": str(f.qb_name.iloc[k] or ""), "same_qb": bool(same[k])})
    rho, n = _rho(x, y)
    rho_qb, n_qb = _rho(x[same], y[same]) if same.any() else (None, 0)
    recent = pts[-16:]
    rho_recent, n_recent = _rho(np.array([p["x"] for p in recent]), np.array([p["y"] for p in recent])) if recent else (None, 0)
    return {"team": team, "a": a, "b": b, "opponent": opponent, "qb": qb, "points": pts, "rho": rho, "n": n, "se": spearman_se(rho, n),
            "rho_qb": rho_qb, "n_qb": n_qb, "rho_recent": rho_recent, "n_recent": n_recent}
