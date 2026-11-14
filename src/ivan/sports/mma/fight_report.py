# src/ivan/sports/mma/fight_report.py
"""
Each fight's simulation, summarised: who wins and how, how long it lasts,
and each fighter's fantasy points (mean, spread, the points by result, the
distribution). The same fight objects the contest sim draws from
(ingest.load_fights, simulate_detail), on more worlds than the contest sim
uses and a fixed seed, so the figures do not move between two reads of the
same card.

Also what the card is missing: a fight with no finish price runs on the
default finish chance, a fighter with no strikes / takedowns line scores on
the win bonus alone, and input files saved after the last ingest are not in
the pool yet. `card_report` carries all three as `alerts`.

    card_report(session, cfg)     the report (the slate page and scripts/mma/fights.py)
    format_card(report)           the terminal table
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
from sqlalchemy import func
from sqlalchemy.orm import Session

from ivan.core.seed import seed_for
from ivan.core.utils import PROJECT_ROOT
from ivan.sports.mma.fight import DECISION, ROUND_SECONDS
from ivan.sports.mma.ingest import load_fights
from ivan.sports.mma.models import LineupSimulationResult, MMAFight, MMAPlayer
from ivan.sports.mma.odds import american_to_prob

N_SIMS = 20000          # worlds per fight for the report (the contest sim's simulation.n_sims is smaller; this only smooths the read)
SEED = 20261114         # fixed, so a card reads the same twice; IVAN_SEED overrides (ivan.core.seed)
BIN = 10                # fantasy points per bar of the distribution
BIG = 100               # "a big score": the share of worlds at or above it
PCTS = (10, 25, 50, 75, 90)


def segment_keys(rounds: int) -> list[str]:
    """How a fight can end, in time order: r1 (the first 60 seconds included), r2 ... then the decision."""
    return [f"r{r}" for r in range(1, rounds + 1)] + [DECISION]


def segment_label(key: str) -> str:
    return "Dec" if key == DECISION else key.upper()


def _fighter_stats(fpts: np.ndarray, wins: np.ndarray, finished: np.ndarray, sig: np.ndarray, td: np.ndarray, salary: int) -> dict:
    won, lost = fpts[wins], fpts[~wins]
    q = np.percentile(fpts, PCTS)
    mean = float(fpts.mean())
    return {"win": float(wins.mean()), "win_finish": float((wins & finished).mean()), "win_decision": float((wins & ~finished).mean()),
            "mean": mean, "sd": float(fpts.std()), **{f"p{p}": float(v) for p, v in zip(PCTS, q)}, "max": float(fpts.max()),
            "mean_win": float(won.mean()) if won.size else None, "mean_loss": float(lost.mean()) if lost.size else None,
            "big": float((fpts >= BIG).mean()), "sig": float(sig.mean()), "td": float(td.mean()),
            "value": mean / (salary / 1000.0) if salary else None}


def _finish_source(fight_row: Optional[MMAFight], p1: MMAPlayer, p2: MMAPlayer, model: str) -> str:
    """Where the fight's finish chance came from: fight | fighters | default."""
    if fight_row is not None and fight_row.itd_odds not in (None, 0):
        return "fight"
    if model == "hazard" and p1.itd_odds not in (None, 0) and p2.itd_odds not in (None, 0):
        return "fighters"
    return "default"


def simulate_card(session: Session, cfg: dict, n_sims: Optional[int] = None, seed: Optional[int] = None) -> dict:
    """Every fight of the pool on file simulated n_sims times; one dict per fight, latest fight first (the card's billing order)."""
    n = int(n_sims or N_SIMS)
    rng = np.random.default_rng(seed if seed is not None else seed_for("mma.fights", SEED))
    fight_cfg = cfg.get("fight") or {}
    model = str(fight_cfg.get("model", "hazard")).lower()
    players = {p.id: p for p in session.query(MMAPlayer).all()}
    rows = {f.id: f for f in session.query(MMAFight).all()}
    out = []
    for f in load_fights(session, cfg.get("scoring"), fight_cfg):
        d = f.simulate_detail(n, rng)
        t, finished, f1_wins = d["t"], d["finished"], d["f1_wins"]
        rnd = np.minimum(np.floor(t / ROUND_SECONDS).astype(int) + 1, f.rounds)
        ends = []
        for key in segment_keys(f.rounds):
            m = ~finished if key == DECISION else (finished & (rnd == int(key[1:])))
            ends.append({"key": key, "label": segment_label(key), "p1": float((m & f1_wins).mean()), "p2": float((m & ~f1_wins).mean())})
        sides = []
        for fighter, wins, fpts, sig, td in ((f.f1, f1_wins, d["fpts1"], d["sig1"], d["td1"]), (f.f2, ~f1_wins, d["fpts2"], d["sig2"], d["td2"])):
            p = players[fighter.id]
            blank = [label for label, v in (("moneyline", p.win_odds), ("ITD", p.itd_odds), ("strikes line", p.sig_strikes), ("takedowns line", p.takedowns)) if v is None]
            sides.append({"id": p.id, "name": p.name, "salary": int(p.salary), "win_odds": p.win_odds, "itd_odds": p.itd_odds,
                          "sig_line": p.sig_strikes, "td_line": p.takedowns, "market_win": p.win_prob, "projection": p.projection,
                          "blank": blank, "fpts": fpts, **_fighter_stats(fpts, wins, finished, sig, td, int(p.salary))})
        row = rows.get(f.id)
        out.append({"id": f.id, "rounds": int(f.rounds), "gametime": row.gametime if row is not None else None,
                    "itd_odds": row.itd_odds if row is not None else None, "finish_source": _finish_source(row, players[f.f1.id], players[f.f2.id], model),
                    "p_finish": float(finished.mean()), "p_u60": float((finished & (t < 60)).mean()), "minutes": float(t.mean() / 60.0),
                    "ends": ends, "fighters": sides})
    out.sort(key=lambda r: (r["gametime"] or datetime.min, r["id"]), reverse=True)
    return {"n_sims": n, "model": model, "fights": out}


def add_histograms(fights: list[dict]) -> dict:
    """Each fighter's share of worlds per BIN points, on one axis for the whole card (the last bar holds everything above it). Drops the raw draws."""
    draws = [s["fpts"] for f in fights for s in f["fighters"]]
    top = max((float(np.percentile(a, 99.5)) for a in draws), default=BIN)
    xmax = int(max(BIN, np.ceil(top / BIN) * BIN))
    edges = np.arange(0, xmax + BIN, BIN, dtype=float)
    ymax = 0.0
    for f in fights:
        for s in f["fighters"]:
            fpts = np.minimum(s.pop("fpts"), xmax - 1e-9)
            s["hist"] = (np.histogram(fpts, bins=edges)[0] / fpts.shape[0]).tolist()
            ymax = max(ymax, max(s["hist"]))
    return {"bin": BIN, "xmax": xmax, "ymax": ymax}


def input_files(cfg: dict) -> list[Path]:
    paths = cfg.get("paths", {})
    return [PROJECT_ROOT / paths[k] if not Path(paths[k]).is_absolute() else Path(paths[k]) for k in ("player_odds_csv", "fight_odds_csv") if paths.get(k)]


def _utc(dt: Optional[datetime]) -> Optional[datetime]:
    """The tables' created_at columns are UTC without a zone."""
    return dt.replace(tzinfo=timezone.utc) if dt is not None and dt.tzinfo is None else dt


def card_alerts(session: Session, cfg: dict, fights: list[dict]) -> tuple[list[dict], dict]:
    """
    What is wrong with the card's inputs, worst first: [{level: bad|warn, text}], and the timeline
    ({ingested, sims: [{contest, lineups, at, stale}]}, local times).
    """
    alerts: list[dict] = []
    ingested = _utc(session.query(func.max(MMAPlayer.created_at)).scalar())
    n_fights, n_fighters = len(fights), sum(len(f["fighters"]) for f in fights)
    # inputs saved after the pool was built: the pool does not hold them
    if ingested is not None:
        for path in input_files(cfg):
            if path.exists() and datetime.fromtimestamp(path.stat().st_mtime, timezone.utc) > ingested:
                saved = datetime.fromtimestamp(path.stat().st_mtime)
                alerts.append({"level": "bad", "text": f"{path.name} was saved at {saved:%H:%M}, after the last ingest ({ingested.astimezone():%H:%M}): "
                                                       "the pool does not hold it. Run scripts/mma/ingest.py again, then sim.py and choose.py."})
    no_finish = [f for f in fights if f["finish_source"] == "default"]
    if no_finish:
        alerts.append({"level": "bad" if len(no_finish) == n_fights else "warn",
                       "text": f"{len(no_finish)} of {n_fights} fights have no finish price (ITD): each runs on the default finish chance "
                               f"({float((cfg.get('fight') or {}).get('default_itd_prob', 0.5)):.0%}), split between the fighters by the moneyline."})
    no_volume = [s for f in fights for s in f["fighters"] if s["sig_line"] is None and s["td_line"] is None]
    if no_volume:
        alerts.append({"level": "bad" if len(no_volume) == n_fighters else "warn",
                       "text": f"{len(no_volume)} of {n_fighters} fighters have no strikes or takedowns line: they score the win bonus and nothing else."})
    no_win = [s for f in fights for s in f["fighters"] if s["win_odds"] is None]
    if no_win:
        alerts.append({"level": "warn", "text": f"{len(no_win)} fighters have no moneyline (a coin flip): " + ", ".join(s["name"].title() for s in no_win[:6])})
    sims = []
    for name, n, at in (session.query(LineupSimulationResult.contest_name, func.count(LineupSimulationResult.id), func.max(LineupSimulationResult.created_at))
                        .group_by(LineupSimulationResult.contest_name).all()):
        at = _utc(at)
        sims.append({"contest": name, "lineups": int(n), "at": at.astimezone() if at else None,
                     "stale": bool(at and ingested and at < ingested)})
    stale = [s for s in sims if s["stale"]]
    if stale:
        alerts.append({"level": "bad", "text": f"{len(stale)} of {len(sims)} contest sims are older than the pool (ingested {ingested.astimezone():%H:%M}): run scripts/mma/sim.py again."})
    alerts.sort(key=lambda a: a["level"] != "bad")
    return alerts, {"ingested": ingested.astimezone() if ingested else None, "sims": sorted(sims, key=lambda s: s["contest"])}


def card_report(session: Session, cfg: dict, n_sims: Optional[int] = None, seed: Optional[int] = None) -> dict:
    """The whole read: {n_sims, model, fights, axis (the shared distribution axis), alerts, timeline}."""
    rep = simulate_card(session, cfg, n_sims, seed)
    rep["axis"] = add_histograms(rep["fights"])
    rep["alerts"], rep["timeline"] = card_alerts(session, cfg, rep["fights"])
    return rep


def _odds(v) -> str:
    return "—" if v is None else f"{v:+.0f}"


def format_card(rep: dict) -> str:
    lines = [f"{len(rep['fights'])} fights, {rep['n_sims']:,} worlds each, fight model {rep['model']}"]
    for a in rep["alerts"]:
        lines.append(f"  [{a['level']}] {a['text']}")
    for f in rep["fights"]:
        a, b = f["fighters"]
        when = f"{f['gametime']:%H:%M}" if f["gametime"] else ""
        src = {"fight": "fight ITD price", "fighters": "the fighters' ITD prices", "default": "NO finish price, default"}[f["finish_source"]]
        lines.append("")
        lines.append(f"{a['name'].title()} vs {b['name'].title()}  ·  {f['rounds']} rounds  {when}  ·  finish {f['p_finish']:.0%} ({src})  ·  {f['minutes']:.1f} min on average")
        ends = "  ".join(f"{e['label']} {e['p1']:.0%}/{e['p2']:.0%}" for e in f["ends"])
        lines.append(f"  ends (first / second fighter wins): {ends}")
        lines.append(f"  {'fighter':<24} {'salary':>6} {'ML':>6} {'mkt':>5} {'win':>5} {'fin':>5} {'dec':>5} {'mean':>6} {'p25':>5} {'p50':>5} {'p75':>5} {'p90':>5} "
                     f"{'wins':>6} {'loses':>6} {f'{BIG}+':>5} {'/$1k':>5} {'SS':>5} {'TD':>4}  blank")
        for s in f["fighters"]:
            def num(v, w=6, d=1):
                return f"{'—':>{w}}" if v is None else f"{v:>{w}.{d}f}"
            lines.append(f"  {s['name'].title()[:24]:<24} {s['salary']:>6} {_odds(s['win_odds']):>6} "
                         f"{(american_to_prob(s['win_odds']) if s['win_odds'] is not None else 0.5):>5.0%} {s['win']:>5.0%} {s['win_finish']:>5.0%} {s['win_decision']:>5.0%} "
                         f"{s['mean']:>6.1f} {s['p25']:>5.0f} {s['p50']:>5.0f} {s['p75']:>5.0f} {s['p90']:>5.0f} {num(s['mean_win'])} {num(s['mean_loss'])} "
                         f"{s['big']:>5.0%} {num(s['value'], 5, 2)} {s['sig']:>5.1f} {s['td']:>4.1f}  {', '.join(s['blank'])}")
    return "\n".join(lines)
