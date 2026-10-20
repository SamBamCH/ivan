# src/ivan/web/pipeline_nfl.py
"""
The NFL pre-lock pipeline as the web UI sees it: ordered steps, each a thin
wrapper around one of the CLI scripts in scripts/nfl/. Nothing here knows how
the scripts work internally; it only builds their command lines and reports
coarse progress from the database while they run.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Callable, Optional

import pandas as pd
from sqlalchemy import text

from ivan.core.utils import PROJECT_ROOT
from ivan.sports.nfl.models import ContestSimReport
from ivan.sports.nfl.stacks import describe_lineup
from ivan.web.charts import bars_svg, iso_quality_lines, scatter_svg
from ivan.sports.nfl.paths import INPUT_DIR, OUTPUT_DIR

PYTHON = sys.executable
SCRIPTS = PROJECT_ROOT / "scripts" / "nfl"
WEB_INPUT_DIR = INPUT_DIR / "web"
WEB_OUTPUT_DIR = OUTPUT_DIR / "web"
CURRENT_ENTRIES = WEB_INPUT_DIR / "entries.csv"
WEIGHTS_FILE = WEB_INPUT_DIR / "weights.json"        # per-contest choose weights, remembered between runs

WEIGHT_KEYS = ("method", "cash_pct", "avg_roi_win", "diversity_lambda", "bankroll")
WEIGHT_LABELS = {"method": "Method", "cash_pct": "Cash %", "avg_roi_win": "ROI", "diversity_lambda": "Diversity", "bankroll": "Bankroll x fees"}
WEIGHT_METHODS = ("greedy", "portfolio")
DEFAULT_WEIGHTS = {"cash_pct": 0.5, "avg_roi_win": 0.5, "diversity_lambda": 0.1, "method": "greedy", "bankroll": 3.0}


@dataclass
class Field:
    name: str
    label: str
    kind: str = "number"            # number | checkbox
    default: object = None
    help: str = ""
    step: str = "1"
    min: object = None


@dataclass
class Step:
    key: str
    title: str
    blurb: str
    build: Callable[[dict], list[list[str]]]
    fields: list[Field] = field(default_factory=list)
    needs_entries: bool = False
    progress: Optional[Callable[[object], str]] = None   # (session) -> one-line status


def _num(form: dict, name: str, default, cast=int):
    v = form.get(name)
    return default if v in (None, "") else cast(v)


def _flag(form: dict, name: str) -> bool:
    return str(form.get(name, "")).lower() in ("on", "true", "1")


def _script(name: str) -> list[str]:
    return [PYTHON, str(SCRIPTS / name)]


_slate_cache: dict = {"at": 0.0, "items": []}


def slates(max_age: float = 120) -> list[dict]:
    """Upcoming DK slates, cached briefly so page loads don't hammer DraftKings."""
    if time.time() - _slate_cache["at"] > max_age:
        from ivan.sports.common.draftkings import SUPPORTED_GAME_TYPES, list_slates
        items = list_slates("nfl")
        for s in items:
            s["supported"] = s["game_type"] in SUPPORTED_GAME_TYPES
        _slate_cache.update(at=time.time(), items=items)
    return _slate_cache["items"]


def _require_slate(form: dict) -> str:
    dg = str(form.get("draft_group", "")).strip()
    if not dg.isdigit():
        raise ValueError("Pick a slate first. The list under step 1 loads from DraftKings.")
    return dg


def _players_progress(s) -> str:
    n, teams = s.execute(text("SELECT count(*), count(DISTINCT team) FROM nfl_player")).one()
    return f"{n} player rows across {teams} teams"


def _lineups_progress(s) -> str:
    return f"{s.execute(text('SELECT count(*) FROM lineup')).scalar():,} lineups in DB"


def _sims_progress(s) -> str:
    done = s.execute(text("SELECT count(DISTINCT contest_name) FROM lineup_simulation_results")).scalar()
    return f"{done} contest(s) simulated"


def _contests_progress(s) -> str:
    n = s.execute(text("SELECT count(*) FROM payouts")).scalar()
    return f"{n:,} payout rows in DB"


STEPS: list[Step] = [
    Step(
        key="ingest", title="Ingest the slate",
        blurb="Pulls players and salaries from DraftKings and player odds from the-odds-api, fits each player's fantasy-point distribution, then projects ownership by sampling a field of lineups. Run this first, and again close to lock for fresh odds and ownership.",
        fields=[Field("prelock", "Re-fetch players from DraftKings", "checkbox", True,
                      help="Leave on. Turn off only to refresh odds without touching the player pool."),
                Field("update_odds", "Update odds", "checkbox", True)],
        build=lambda f: [_script("ingest.py")
                         + (["-p", "--draft-group", _require_slate(f)] if _flag(f, "prelock") else [])
                         + ["--update-odds", "true" if _flag(f, "update_odds") else "false"]],
        progress=_players_progress,
    ),
    Step(
        key="contests", title="Fetch contests & payouts",
        blurb="Looks up the upcoming DraftKings contests and their payout tables. The simulator needs these for every contest in your entries file.",
        build=lambda f: [_script("get_contest_ids.py"), _script("get_payouts.py")],
        progress=_contests_progress,
    ),
    Step(
        key="generate", title="Generate the field",
        blurb="Builds the pool of candidate lineups that the simulator picks from. More lineups = more choice, longer run.",
        fields=[Field("num_lineups", "Lineups to generate", default=10000, min=100, step="100"),
                Field("num_jobs", "Parallel workers", default=max(1, min(8, os.cpu_count() or 1)), min=1,
                      help="Use the number of CPU cores, or fewer if the machine should stay responsive.")],
        build=lambda f: [_script("generate_lineups.py") + [str(_num(f, "num_lineups", 10000)), str(_num(f, "num_jobs", 4))]],
        progress=_lineups_progress,
    ),
    Step(
        key="sim", title="Simulate your contests",
        blurb="Simulates every contest in your entries file against the generated field and scores each candidate lineup (cash %, ROI, rank). "
              "The chooser then shrinks each lineup's simulated ROI by its own standard error before picking.",
        fields=[Field("n", "Simulations per contest", default=1000, min=100, step="100"),
                Field("c", "Field cap", default=10000, min=100, step="100",
                      help="Largest field size to simulate; bigger contests are scaled down to this."),
                Field("d", "Allow duplicate lineups in classic slates", "checkbox", False)],
        needs_entries=True,
        build=lambda f: [_script("sim.py") + ["--entries", str(CURRENT_ENTRIES), "-n", str(_num(f, "n", 1000)),
                                              "-c", str(_num(f, "c", 10000))] + (["-d"] if _flag(f, "d") else [])],
        progress=_sims_progress,
    ),
    Step(
        key="choose", title="Choose lineups",
        blurb="Picks the best lineups for each contest from the simulation results and writes a DraftKings upload file. "
              "The weights say what 'best' means, per contest: cash % (how often a lineup cashes), ROI (how much it wins "
              "when it wins) and diversity (higher = more different lineups across that contest's entries). "
              "Your last weights for each contest are remembered.",
        needs_entries=True,
        build=lambda f: _build_choose(f),
    ),
]


def load_weights() -> dict:
    """Saved per-contest weights, keyed by contest id (strings), plus 'default'."""
    try:
        data = json.loads(WEIGHTS_FILE.read_text())
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def choose_rows(entries: Optional[dict]) -> list[dict]:
    """One row per contest in the entries file with the weights to prefill: saved, else default."""
    if not entries or entries.get("error"):
        return []
    saved = load_weights()
    rows = []
    for c in entries["contests"]:
        cid = str(c["Contest ID"]).strip()
        w = {**DEFAULT_WEIGHTS, **{k: v for k, v in saved.get(cid, {}).items() if k in WEIGHT_KEYS}}
        rows.append({"id": cid, "name": c["Contest Name"], "entries": c["entries"], "weights": w, "saved": cid in saved})
    return rows


def _build_choose(form: dict) -> list[list[str]]:
    """Read the per-contest weight inputs, remember them, and hand choose.py a weights file."""
    entries = entries_summary()
    if not entries or entries.get("error"):
        raise ValueError("Upload an entries file first.")
    weights = {k: v for k, v in load_weights().items() if k != "default"}
    for row in choose_rows(entries):
        w = dict(row["weights"])
        for key in WEIGHT_KEYS:
            raw = str(form.get(f"w-{row['id']}-{key}", "")).strip()
            if raw == "":
                continue
            if key == "method":
                if raw not in WEIGHT_METHODS:
                    raise ValueError(f"'{raw}' is not a chooser method (greedy or portfolio) for {row['name']}.")
                w[key] = raw
                continue
            try:
                val = float(raw)
            except ValueError:
                raise ValueError(f"'{raw}' is not a number ({WEIGHT_LABELS[key]} for {row['name']}).")
            if val < 0:
                raise ValueError(f"{WEIGHT_LABELS[key]} for {row['name']} must be 0 or more.")
            w[key] = val
        weights[row["id"]] = w
    WEB_INPUT_DIR.mkdir(parents=True, exist_ok=True)
    WEIGHTS_FILE.write_text(json.dumps({"default": DEFAULT_WEIGHTS, **weights}, indent=1))
    return [_script("choose.py") + ["--entries", str(CURRENT_ENTRIES), "--no-prompt",
                                    "--weights-json", str(WEIGHTS_FILE), "-o", str(WEB_OUTPUT_DIR / "chosen.csv")]]
STEP_BY_KEY = {s.key: s for s in STEPS}


def entries_summary() -> Optional[dict]:
    """What's in the currently uploaded entries file, or None if there isn't one."""
    if not CURRENT_ENTRIES.exists():
        return None
    try:
        df = pd.read_csv(CURRENT_ENTRIES, engine="python", usecols=["Entry ID", "Contest ID", "Contest Name"], dtype=str)
        df = df.dropna(subset=["Entry ID"])
        df = df[df["Entry ID"].str.strip() != ""]
        contests = (df.groupby(["Contest ID", "Contest Name"], sort=False).size()
                      .reset_index(name="entries").to_dict("records"))
        error = None
    except Exception as e:
        contests, error = [], f"Could not read this file as a DraftKings entries export: {e}"
    return {"path": CURRENT_ENTRIES, "mtime": CURRENT_ENTRIES.stat().st_mtime,
            "contests": contests, "total": sum(c["entries"] for c in contests), "error": error}


def slate_status(session) -> dict:
    players, teams, first, last = session.execute(
        text("SELECT count(*), count(DISTINCT team), min(gametime), max(gametime) FROM nfl_player")).one()
    teams_list = [r[0] for r in session.execute(text("SELECT DISTINCT team FROM nfl_player ORDER BY team"))]
    lineups = session.execute(text("SELECT count(*) FROM lineup")).scalar()
    sims = session.execute(text(
        "SELECT contest_name, count(*) FROM lineup_simulation_results GROUP BY contest_name ORDER BY contest_name")).all()
    owned = session.execute(text("SELECT count(*) FROM nfl_player WHERE ownership_proj IS NOT NULL AND ownership_proj > 0")).scalar()
    return {"players": players, "teams": teams_list, "first_game": first, "last_game": last,
            "lineups": lineups, "sims": [{"contest": c, "rows": n} for c, n in sims], "players_with_ownership": owned}


RESETS = {
    "sims":    ("Clear simulation results", ["lineup_simulation_results", "contest_sim_report"]),
    "lineups": ("Clear lineups and simulation results", ["lineup_simulation_results", "contest_sim_report", "lineup"]),
}


def reset_working(session, what: str) -> dict:
    """
    Delete this slate's generated state (what = 'sims' | 'lineups'). Same as
    scripts/nfl/clear_sims.py / clear_all.py. VACUUMs afterwards when the freed
    pages are a meaningful share of the file, so the DB does not balloon.
    """
    _, tables = RESETS[what]
    counts = {}
    for t in tables:
        counts[t] = session.execute(text(f"DELETE FROM {t}")).rowcount
    session.commit()
    pages, free = (session.execute(text("PRAGMA page_count")).scalar(),
                   session.execute(text("PRAGMA freelist_count")).scalar())
    if pages and free / pages > 0.25:
        session.close()
        with session.get_bind().connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
            conn.execute(text("VACUUM"))
    return counts


PLAYER_COLUMNS = ["name", "team", "opponent", "position", "roster_slot_id", "salary", "status",
                  "fpts_mean", "fpts_std", "fpts_p10", "fpts_p50", "fpts_p90", "ownership_proj", "gametime", "id"]


def players_frame(session) -> pd.DataFrame:
    """The slate's player pool, one row per DK draftable (showdown has CPT and FLEX rows)."""
    df = pd.read_sql(text("SELECT * FROM nfl_player"), session.get_bind())
    if df.empty:
        return df
    df["slot"] = df["roster_slot_id"].map({511: "CPT", 512: "FLEX", 66: "QB", 67: "RB", 68: "WR", 69: "TE", 71: "DST"}).fillna(df["roster_slot_id"].astype(str))
    order = [c for c in ["slot"] + PLAYER_COLUMNS if c in df.columns] + [c for c in df.columns if c not in PLAYER_COLUMNS + ["slot"]]
    return df[order].sort_values(["roster_slot_id", "salary"], ascending=[True, False]).reset_index(drop=True)


# ------------------------------------------------------------------ contest reports
GROUP_ORDER = ["CPT", "FLEX", "QB", "RB", "WR", "TE", "K", "DST"]


def contest_reports(session) -> list[dict]:
    """Simulated contests with a report, newest first."""
    rows = session.query(ContestSimReport).order_by(ContestSimReport.created_at.desc()).all()
    return [{"id": r.contest_id, "name": r.contest_name, "created_at": r.created_at,
             "counts": (r.report or {}).get("check_counts", {})} for r in rows]


def contest_report(session, contest_id: int) -> Optional[dict]:
    row = session.get(ContestSimReport, contest_id)
    return row.report if row else None


def report_view(report: dict) -> dict:
    """Derived, display-only groupings for contests.html."""
    f = report.get("field", {})
    groups: dict[str, list] = {}
    for r in f.get("exposures", []):
        groups.setdefault(r.get("slot") or r.get("pos") or "?", []).append(r)
    ordered = sorted(groups.items(), key=lambda kv: GROUP_ORDER.index(kv[0]) if kv[0] in GROUP_ORDER else 99)
    max_hist = max((h["lineups"] for h in f.get("dupe_hist", [])), default=1)
    checks = report.get("checks", [])
    rank = {"bad": 0, "warn": 1, "info": 2, "ok": 3}
    return {"exposure_groups": [{"name": k, "rows": v} for k, v in ordered],
            "max_hist": max_hist,
            "checks_sorted": sorted(checks, key=lambda c: rank.get(c["status"], 9)),
            "attention": [c for c in checks if c["status"] in ("bad", "warn")]}


def corr_css(v) -> str:
    """Cell background for a correlation in [-1, 1]: blue positive, red negative."""
    if v is None:
        return ""
    v = max(-1.0, min(1.0, float(v)))
    a = abs(v) ** 0.8 * 0.85
    return f"background:rgba(31,95,191,{a:.2f})" if v >= 0 else f"background:rgba(180,35,24,{a:.2f})"


# ------------------------------------------------------------------ chosen lineups (step 5 output)
OUTPUT_KINDS = {"showdown": "DraftKings upload (showdown)", "classic": "DraftKings upload (classic)",
                "exposures": "Exposures", "lineups": "Chosen lineups (detail)", "choose": "Chooser diagnostics (JSON)"}
CLASSIC_SLOT_ORDER = ["QB", "RB", "WR", "TE", "FLEX", "DST"]


def output_files() -> list[dict]:
    """Every CSV step 5 has written to the web output dir, newest first."""
    files = sorted(list(WEB_OUTPUT_DIR.glob("*.csv")) + list(WEB_OUTPUT_DIR.glob("*.json")),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    out = []
    for p in files:
        parts = p.name.split(".")
        kind = parts[1] if len(parts) >= 4 else ""
        out.append({"name": p.name, "kind": OUTPUT_KINDS.get(kind, kind), "key": kind, "size": p.stat().st_size,
                    "mtime": p.stat().st_mtime, "stamp": parts[2] if len(parts) >= 4 else ""})
    return out


def chosen_runs() -> list[dict]:
    """Step 5 runs (one per timestamp) with the files each produced, newest first."""
    runs: dict[str, dict] = {}
    for f in output_files():
        if not f["stamp"]:
            continue
        r = runs.setdefault(f["stamp"], {"stamp": f["stamp"], "mtime": f["mtime"], "files": {}})
        r["files"][f["key"]] = f["name"]
        r["mtime"] = max(r["mtime"], f["mtime"])
    return sorted(runs.values(), key=lambda r: r["mtime"], reverse=True)


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    """NaN -> None so templates can test `is not none`."""
    return df.astype(object).where(pd.notnull(df), None)


def _players_text(players: list[dict], fmt: str) -> str:
    """'CPT darnold · FLEX smithnjigba, kupp, ...' / 'QB x · RB a, b · WR ... · DST d'."""
    parts, last = [], None
    for p in players:
        if p["slot"] != last:
            parts.append([p["slot"], []])
            last = p["slot"]
        parts[-1][1].append(p["name"])
    return " · ".join(f"{slot} {', '.join(names)}" for slot, names in parts)


def chosen_view(lineups_csv: Path, contest_id: Optional[str]) -> dict:
    """
    Exposures and lineup detail for one step-5 run, over every contest
    (contest_id None) or one contest. Reads the chosen.lineups CSV.
    """
    df = pd.read_csv(lineups_csv, dtype={"entry_id": str, "contest_id": str, "contest_name": str})
    contests = (df.drop_duplicates(["contest_id", "entry_id"])
                  .groupby(["contest_id", "contest_name"], sort=False).size().reset_index(name="entries"))
    contests = [{"id": r["contest_id"], "name": r["contest_name"], "entries": int(r["entries"])} for _, r in contests.iterrows()]
    if contest_id and contest_id not in {c["id"] for c in contests}:
        contest_id = None
    scope = df if not contest_id else df[df["contest_id"] == contest_id]
    n = int(scope[["contest_id", "entry_id"]].drop_duplicates().shape[0])
    fmt = str(scope["format"].iloc[0]) if len(scope) else "showdown"
    scope = scope.assign(group=scope["slot"].where(scope["format"] == "showdown", scope["position"]))

    # exposures: how many of the n lineups each player is in, grouped by CPT/FLEX or position
    exp = (scope.groupby(["group", "player_id"], sort=False)
                .agg(name=("name", "first"), team=("team", "first"), position=("position", "first"),
                     salary=("salary", "first"), projection=("projection", "first"), ownership=("ownership", "first"),
                     count=("entry_id", "nunique")).reset_index())
    exp["exposure"] = exp["count"] / max(n, 1)
    groups = []
    for g in sorted(exp["group"].unique(), key=lambda x: GROUP_ORDER.index(x) if x in GROUP_ORDER else 99):
        sub = exp[exp["group"] == g].sort_values(["exposure", "salary"], ascending=[False, False])
        groups.append({"name": g, "rows": _clean(sub).to_dict("records")})

    # lineups, per contest, in the order step 5 wrote them
    lineups = []
    for (cid, cname), sub in scope.groupby(["contest_id", "contest_name"], sort=False):
        rows = []
        for i, (eid, lu) in enumerate(sub.groupby("entry_id", sort=False), 1):
            lu = lu.sort_values("slot_no")
            players = _clean(lu).to_dict("records")
            d = describe_lineup(players)
            rows.append({"n": i, "entry_id": eid, "lineup_id": int(lu["lineup_id"].iloc[0]),
                         "players_text": _players_text(players, fmt),
                         "salary": int(lu["salary"].fillna(0).sum()), "projection": float(lu["projection"].fillna(0).sum()),
                         "teams": d["teams"], "stacks": d["stacks"],
                         "cash_pct": players[0].get("cash_pct"), "avg_roi": players[0].get("avg_roi")})
        lineups.append({"id": cid, "name": cname, "rows": rows,
                        "mean_cash": _mean([r["cash_pct"] for r in rows]), "mean_roi": _mean([r["avg_roi"] for r in rows])})

    return {"contests": contests, "contest_id": contest_id, "n_lineups": n, "format": fmt,
            "groups": groups, "lineups": lineups, "legacy": False}


def _mean(xs) -> Optional[float]:
    xs = [float(x) for x in xs if x is not None and x == x]
    return sum(xs) / len(xs) if xs else None


def legacy_exposures_view(exposures_csv: Path) -> dict:
    """Runs from before the per-lineup file existed: overall exposures only."""
    df = pd.read_csv(exposures_csv)
    if "roster_slot" in df.columns and "format" in df.columns:
        df["group"] = df["roster_slot"].where(df["format"] == "showdown", df["position"])
    else:
        df["group"] = df["position"]
    groups = []
    for g in sorted(df["group"].dropna().unique(), key=lambda x: GROUP_ORDER.index(x) if x in GROUP_ORDER else 99):
        sub = df[df["group"] == g].sort_values("exposure", ascending=False)
        groups.append({"name": g, "rows": _clean(sub).to_dict("records")})
    return {"contests": [], "contest_id": None, "n_lineups": None, "format": None,
            "groups": groups, "lineups": [], "legacy": True}


# ------------------------------------------------------------------ the chooser, visualised
def _pct(v: float) -> str:
    return f"{v * 100:.0f}%"


def _pct_signed(v: float) -> str:
    return f"{v * 100:+.0f}%"


def choose_view(choose_json: Path, lineups_csv: Optional[Path], contest_id: Optional[str]) -> dict:
    """
    What the chooser did for one contest of one step-5 run: the quality plane
    (cash % vs ROI with the weight direction), the solution-space map (PCA of
    the simulated point traces), the greedy walk, and the picks in order.
    """
    data = json.loads(choose_json.read_text())
    contests = [{"id": cid, "name": d["contest_name"], "entries": d["entries"]} for cid, d in data.items()]
    if not contests:
        return {"contests": []}
    if contest_id not in data:
        contest_id = contests[0]["id"]
    d = data[contest_id]
    pts = d["points"]
    w = d["weights"]
    w1, w2 = float(w.get("cash_pct", 0)), float(w.get("avg_roi_win", 1))
    lam = float(d["lambda"])

    order = sorted(range(len(pts)), key=lambda i: pts[i]["q"])
    shade = {i: k / max(1, len(pts) - 1) for k, i in enumerate(order)}

    def kind(p):
        return "pick" if p["pick"] else ("top" if p["top"] else "cand")

    def title(p):
        t = f"lineup {p['lineup_id']} · cash {p['cash'] * 100:.1f}% · ROI {p['roi'] * 100:+.0f}% · quality {p['q']:.2f}"
        return t + (f" · pick {p['pick']}" if p["pick"] else (" · in the top by quality alone, not picked" if p["top"] else ""))

    def mark(p, i, x, y):
        return {"x": x, "y": y, "kind": kind(p), "shade": shade[i],
                "label": str(p["pick"]) if p["pick"] and p["pick"] <= 30 else None, "title": title(p)}

    quality_svg = scatter_svg([mark(p, i, p["cash"], p["roi"]) for i, p in enumerate(pts)],
                              xlabel="cash % (share of simulations in the money)", ylabel="average ROI per simulation",
                              xfmt=_pct, yfmt=_pct_signed,
                              iso_lines=iso_quality_lines(w1, w2, d["norm"]["cash"], d["norm"]["roi"]))
    evr = d.get("explained_variance", [0, 0])
    map_svg = scatter_svg([mark(p, i, p["x"], p["y"]) for i, p in enumerate(pts)],
                          xlabel=f"map axis 1 ({evr[0]:.0%} of the differences between lineups)",
                          ylabel=f"map axis 2 ({evr[1]:.0%})", xfmt=lambda v: f"{v:.0f}", yfmt=lambda v: f"{v:.0f}")
    tops_q = sorted((p["q"] for p in pts if p["top"]), reverse=True)
    walk_svg = bars_svg(d["trace"], lam, tops_q)

    picks = [p for p in pts if p["pick"]]
    tops = [p for p in pts if p["top"]]
    overlap = len({p["lineup_id"] for p in picks} & {p["lineup_id"] for p in tops})

    players_by_lid: dict[int, str] = {}
    if lineups_csv and lineups_csv.exists():
        df = pd.read_csv(lineups_csv, dtype={"contest_id": str, "entry_id": str})
        sub = df[df["contest_id"] == contest_id]
        for lid, lu in sub.groupby("lineup_id", sort=False):
            lu = lu.sort_values("slot_no").drop_duplicates("slot_no")
            players_by_lid[int(lid)] = _players_text(_clean(lu).to_dict("records"), str(lu["format"].iloc[0]))
    by_lid = {p["lineup_id"]: p for p in pts}
    top_ids = {p["lineup_id"] for p in tops}
    rows = []
    for t in d["trace"]:
        p = by_lid.get(t["lineup_id"], {})
        rows.append({"pick": t["pick"], "lineup_id": t["lineup_id"], "players": players_by_lid.get(int(t["lineup_id"]), ""),
                     "cash": p.get("cash"), "roi": p.get("roi"), "q": t["q"], "div": t["div"], "bonus": lam * t["div"],
                     "score": t["score"], "in_top": t["lineup_id"] in top_ids})

    return {"contests": contests, "contest_id": contest_id, "name": d["contest_name"], "entries": d["entries"],
            "n_candidates": d["n_candidates"], "n_shown": d["n_shown"], "weights": w, "lambda": lam,
            "method": d.get("method") or w.get("method") or "greedy", "bankroll": d.get("bankroll"), "book": d.get("book"),
            "chosen": {"cash": _mean([p["cash"] for p in picks]), "roi": _mean([p["roi"] for p in picks]), "n": len(picks)},
            "top": {"cash": _mean([p["cash"] for p in tops]), "roi": _mean([p["roi"] for p in tops]), "n": len(tops)},
            "overlap": overlap, "spread": d["spread"], "explained_variance": evr, "rooting": d.get("rooting"),
            "quality_svg": quality_svg, "map_svg": map_svg, "walk_svg": walk_svg, "rows": rows}
