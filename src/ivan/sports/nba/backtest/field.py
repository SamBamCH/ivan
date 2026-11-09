# src/ivan/sports/nba/backtest/field.py
"""
The field backtest: this season's real contest fields (standings.derive_contest, final exports only) against what the
field model says they should have looked like, rebuilt from the slate's last capture before lock the way the live path
builds it. One run answers "how far is our ownership from the real field, and in which direction".

Per derived contest:
  real       ownership counted from the lineups (copies included); the lineups' strength under the public projection
             relative to its optimal, duplicate share, salary left (the shape)
  model      the two-group field (field.model.simulate_field with data/nba/models/field_two_group.json) on the public
             projection rebuilt from the capture (public_projection.project on the capture's pool and props and the box
             scores before that day), at `field_size` entries; one field per slate, every contest of the slate compared to
             it, as the sim slices one field per run. Its ownership and shape.
  lock       the ownership the desk showed when the slate locked: the at_tip capture's ownership column (archive_at_lock,
             since 2026-10-23), when it was archived with ownership
  softmax    the ownership model fitted on real contests (field.model.predict, field_ownership.json), for reference

Metrics per contest (ownership): MAE over the players the public could roster, MAE over the real top 20, Spearman, how many
of the real top 10 the model has in its top 10, top-8 share of ownership (the chalk's concentration) real vs model.
Shape, real (subsampled to field_size) vs model: p10 / p50 / p90 of lineup strength relative to the optimal, share within
5 % / 10 % of it, duplicate share, max copies, salary left.
Across contests: calibration (players binned by model ownership -> mean real), bias by salary tier and by starter, the
largest misses. Output under data/nba/diagnostics/field_backtest/<stamp>/: contests.csv, players.csv, report.txt,
summary.json. Model fields are cached under data/nba/derived/field_backtest/ by capture and public projection.
"""
from __future__ import annotations

import hashlib
import json
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from ivan.core import archive
from ivan.sports.nba.names import normalize_name
from ivan.sports.nba.field import model as FM
from ivan.sports.nba.paths import DIAG_DIR, NBA_DATA_DIR
from ivan.sports.nba.field.solver import FieldPlayer
from ivan.sports.nba.field.public_field import OUT_STATUSES, POOL_MIN_PUB, field_params
from ivan.sports.nba.utils import _coerce_positions
from ivan.sports.nba.data_clean import standings as SD

Log = Callable[[str], None]
OUT_DIR = DIAG_DIR / "field_backtest"
CACHE_DIR = NBA_DATA_DIR / "derived" / "field_backtest"
SALARY_CAP = 50000
OWN_BINS = [0.0, 1.0, 3.0, 7.0, 15.0, 25.0, 40.0, 101.0]
TIERS = [0.0, 4.5, 6.5, 8.5, 99.0]
PARAM_KEYS = ("optimizer_share", "sigma_provider", "sigma_personal", "sigma_casual", "providers")


@dataclass
class SlateView:
    """A slate as the public saw it at its last capture before lock, with the public projection rebuilt on it."""
    capture_id: int
    slate: int
    lock: datetime
    players: pd.DataFrame        # id, name, key, team, opponent, position, salary, status, starter, gametime, out, pub, priced
    blobs: dict                  # the capture's raw rows: players, odds, game_lines
    n_games: int
    captured_at: Optional[datetime] = None
    odds_rows: int = 0           # the capture's props rows for this slate's players, fresh for their tips

    @property
    def capture_age_h(self) -> Optional[float]:
        """Hours between the capture and the lock: a capture from the day before saw none of the day's news."""
        return None if self.captured_at is None else (self.lock - self.captured_at).total_seconds() / 3600.0

    @property
    def priced_share(self) -> float:
        """Share of the players not ruled out whose points, rebounds and assists the books priced for that night."""
        avail = self.players[~self.players.out]
        return float(avail.priced.mean()) if len(avail) else 0.0

    @property
    def pub(self) -> Dict[int, float]:
        return {int(i): float(v) for i, v in zip(self.players.id, self.players.pub)}

    @property
    def salaries(self) -> Dict[int, int]:
        return {int(i): int(s or 0) for i, s in zip(self.players.id, self.players.salary)}


# ----------------------------------------------------------------------------- which contests
def final_contests(derived_dir: Path = SD.DERIVED_DIR, contest_ids: Optional[List[int]] = None) -> List[dict]:
    """The derived contests built from a final export (field.json + the folder), oldest capture first."""
    out = []
    for d in sorted(Path(derived_dir).iterdir()):
        if not d.name.isdigit() or not (d / "field.json").exists() or not (d / "lineups.parquet").exists():
            continue
        if not SD.derived_is_final(int(d.name), derived_dir):
            continue
        m = json.loads((d / "field.json").read_text())
        if m.get("capture_id") is None or (contest_ids and int(m["contest_id"]) not in set(contest_ids)):
            continue
        m["dir"] = d
        out.append(m)
    return sorted(out, key=lambda m: (m.get("capture_at") or "", m["contest_id"]))


# ----------------------------------------------------------------------------- the slate as the public saw it
def capture_view(session: Session, capture_id: int, history) -> SlateView:
    """The capture's pool with the public projection rebuilt on its props (tonight's rows only, market.fresh_rows) and the box
    scores before the slate's day; players ruled out project 0, as in public_field.public_projection."""
    from ivan.sports.nba.projection.market import fresh_rows
    from ivan.sports.nba.field.public_projection import project
    cap = session.get(archive.ArchivedSlate, int(capture_id))
    if cap is None:
        raise ValueError(f"no archive capture {capture_id}")
    blobs = archive.load_blobs(session, int(capture_id))
    df = pd.DataFrame(blobs.get("players", []))
    if df.empty:
        raise ValueError(f"capture {capture_id} has no players")
    for col in ("starter", "status", "opponent", "position"):
        if col not in df:
            df[col] = None
    df["key"] = df.name.map(normalize_name)
    df["status"] = df.status.fillna("").astype(str).str.upper().replace({"NONE": ""})
    df["out"] = df.status.isin(OUT_STATUSES)
    df["gametime"] = pd.to_datetime(df.gametime, errors="coerce")
    df["salary"] = pd.to_numeric(df.salary, errors="coerce").fillna(0).astype(int)
    odds = pd.DataFrame(blobs.get("odds", []))
    if len(odds):
        odds = fresh_rows(odds, dict(zip(df.key, df.gametime)))
        odds = odds[odds.player.isin(set(df.key))]
    pub = project(df[["key", "salary"]].rename(columns={"key": "player_name"}), odds, history, cap.lock.date())
    df["pub"] = np.where(df.out, 0.0, df.key.map(pub).astype(float).fillna(0.0))
    df["priced"] = df.key.isin(priced_players(odds))
    n_games = len({tuple(sorted([str(t).upper(), str(o).upper()])) for t, o in zip(df.team, df.opponent) if t and o}) or 1
    return SlateView(capture_id=int(capture_id), slate=int(cap.slate_key), lock=cap.lock, players=df.reset_index(drop=True), blobs=blobs, n_games=n_games,
                     captured_at=cap.captured_at, odds_rows=int(len(odds)))


def priced_players(odds: pd.DataFrame) -> set:
    """The players whose core markets (points, rebounds, assists) the rows price: the ones the replica projects from the props."""
    from ivan.sports.nba.projection.market import MARKET_TO_STAT
    from ivan.sports.nba.field.public_projection import CORE
    if odds is None or not len(odds):
        return set()
    o = odds.assign(stat=odds.market.map(MARKET_TO_STAT)).dropna(subset=["stat"])
    return {k for k, st in o.groupby("player").stat.agg(set).items() if all(c in st for c in CORE)}


def field_pool(view: SlateView) -> List[FieldPlayer]:
    """The field's player pool: everyone the public projects above POOL_MIN_PUB (public_field.public_pool on a capture)."""
    out = []
    for r in view.players.itertuples():
        if float(r.pub) < POOL_MIN_PUB or not r.salary or not r.position:
            continue
        team, opp = str(r.team or ""), r.opponent
        out.append(FieldPlayer(id=int(r.id), name=r.name, salary=int(r.salary), team=team, opponent=opp,
                               position=tuple(_coerce_positions(r.position)), fpts_mean=float(r.pub),
                               game_id="-".join(sorted([team.upper(), str(opp).upper()])) if team and opp else None,
                               gametime=r.gametime.to_pydatetime() if pd.notna(r.gametime) else None))
    return out


def public_optimal(pool: List[FieldPlayer]) -> float:
    """The optimal lineup's total under the public projection (the field's yardstick)."""
    from ivan.sports.nba.field.solver import generate_single_classic_lineup
    P = {p.id: p.fpts_mean for p in pool}
    ids = generate_single_classic_lineup(pool, base_scores=P, max_salary_per_team=SALARY_CAP)
    return float(sum(P[i] for i in ids)) if ids else float("nan")


# ----------------------------------------------------------------------------- the model's field
def field_key(pool: List[FieldPlayer], params: dict, seed: int) -> str:
    raw = json.dumps([[p.id, p.salary, list(p.position), round(p.fpts_mean, 1)] for p in pool] + [params.get(k) for k in PARAM_KEYS] + [int(seed)])
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


def model_field(view: SlateView, size: int, params: Optional[dict] = None, seed: int = 0, workers: int = 8,
                cache_dir: Path = CACHE_DIR, log: Log = print) -> List[tuple]:
    """The two-group field on the view's public projection at `size` entries (sorted id tuples, copies kept), as
    public_field.slate_field builds it live; cached by capture, public projection, parameters and seed."""
    params = params or field_params()
    pool = field_pool(view)
    if len(pool) < 8:
        raise ValueError(f"capture {view.capture_id}: the public projection covers fewer than 8 players")
    key = field_key(pool, params, seed)
    path = Path(cache_dir) / f"{view.capture_id}_{key}.json"
    if path.exists():
        cached = [tuple(x) for x in json.loads(path.read_text())]
        if len(cached) >= size:
            return cached[:size]
    t0 = time.time()
    old = FM.SIGMA_CASUAL; FM.SIGMA_CASUAL = float(params.get("sigma_casual", old))
    try:
        field = FM.simulate_field(pool, {p.id: p.fpts_mean for p in pool}, int(size), float(params["sigma_provider"]), float(params["sigma_personal"]),
                                  float(params["optimizer_share"]), seed=int(seed), workers=int(workers), providers=int(params.get("providers", 3)))
    finally:
        FM.SIGMA_CASUAL = old
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([list(x) for x in field]))
    log(f"  slate {view.slate}: model field of {len(field):,} lineups on {len(pool)} players in {time.time() - t0:.0f}s")
    return field


def ownership(field: List[tuple]) -> Dict[int, float]:
    """player id -> % of the field's entries holding him."""
    n = max(1, len(field))
    return {int(p): 100.0 * c / n for p, c in Counter(p for lu in field for p in lu).items()}


def at_lock_ownership(session: Session, slate: int) -> Optional[Dict[int, float]]:
    """The ownership the desk showed at the slate's lock: the first at_tip capture's ownership column, by player id; None when
    no at_tip capture of the slate carries ownership."""
    caps = sorted([c for c in archive.captures(session, "nba", source="at_tip") if str(c.slate_key) == str(int(slate))], key=lambda c: c.captured_at)
    for cap in caps[:1]:
        rows = archive.load_blobs(session, cap.id).get("players", [])
        own = {int(r["id"]): float(r["ownership"]) for r in rows if r.get("ownership") not in (None, 0) and r.get("id") is not None}
        if own:
            return own
    return None


def softmax_ownership(session: Session, view: SlateView, meta: dict, model: Optional[dict] = None) -> Optional[Dict[int, float]]:
    """The fitted ownership model (field.model.predict) on the capture, for this contest's size and fee; None without a model."""
    model = model or FM.load()
    if model is None:
        return None
    f = FM.features(view.blobs.get("players", []), view.blobs.get("odds", []), view.blobs.get("game_lines", []),
                    FM._history_at(session, view.slate, meta.get("capture_at") or view.lock))
    pred = FM.predict(model, f, int(meta.get("entries") or 1000), float(meta.get("entry_fee") or 1.0), view.n_games)
    return {int(i): float(p) for i, p in zip(f.id, pred)}


# ----------------------------------------------------------------------------- the real field
def real_lineups(contest_dir: Path) -> List[tuple]:
    """The contest's entries as sorted id tuples, copies repeated."""
    L = pd.read_parquet(Path(contest_dir) / "lineups.parquet")
    cols = [f"p{i}" for i in range(1, 9)]
    out = []
    for row, c in zip(L[cols].to_numpy(), L.copies.to_numpy()):
        key = tuple(int(x) for x in row)
        out.extend([key] * int(c))
    return out


# ----------------------------------------------------------------------------- comparing
def ownership_metrics(real: Dict[int, float], pred: Dict[int, float], ids: List[int]) -> dict:
    """field.model.metrics over `ids` plus the chalk's concentration (top-8 share) and the top-8 sum, real and predicted."""
    own = np.array([float(real.get(i, 0.0)) for i in ids]); p = np.array([float(pred.get(i, 0.0)) for i in ids])
    m = FM.metrics(own, p) if len(ids) > 2 else dict(mae=float(np.mean(np.abs(own - p))) if len(ids) else float("nan"))
    m["top8_real"] = float(np.sort(own)[-8:].sum()); m["top8_pred"] = float(np.sort(p)[-8:].sum())
    m["sum_pred"] = float(p.sum())
    return m


def shape(lineups: List[tuple], Q: Dict[int, float], q_opt: float, salaries: Dict[int, int], subsample: Optional[int] = None, seed: int = 0) -> dict:
    """field.model.field_shape (strength relative to the optimal, within 5/10 %, duplicate share) plus unique share, max copies
    and salary left, on `subsample` entries drawn without replacement when the field is bigger (so two fields compare at one size)."""
    L = list(lineups)
    if subsample and len(L) > subsample:
        rng = np.random.default_rng(seed); L = [L[i] for i in rng.choice(len(L), size=int(subsample), replace=False)]
    if not L:
        return dict(n=0)
    sh = FM.field_shape(L, Q, q_opt)
    c = Counter(L)
    left = np.array([SALARY_CAP - sum(salaries.get(i, 0) for i in lu) for lu in L], dtype=float)
    sh.update(n=len(L), unique_share=float(len(c) / len(L)), max_copies=int(max(c.values())),
              salary_left_mean=float(left.mean()), salary_full=float((left <= 0).mean()), salary_left_p90=float(np.quantile(left, 0.9)))
    return sh


def compare_contest(meta: dict, view: SlateView, preds: Dict[str, Optional[Dict[int, float]]], q_opt: float, field_size: int,
                    model_shape: Optional[dict], seed: int = 0) -> tuple[dict, List[dict]]:
    """One contest against the predictions: (the contest row, one row per player)."""
    real_own = {int(i): float(v) for i, v in FM.lineup_ownership(meta["dir"]).items()}
    P, salaries = view.pub, view.salaries
    avail = set(view.players.id[~view.players.out].astype(int))
    ids = sorted(avail | set(real_own))
    row = dict(contest_id=int(meta["contest_id"]), slate=view.slate, day=view.lock.date().isoformat(), lock=view.lock.isoformat(timespec="minutes"),
               name=meta.get("name"), entries=int(meta.get("entries") or 0), fee=float(meta.get("entry_fee") or 0), n_players=len(ids),
               capture_age_h=view.capture_age_h, odds_rows=view.odds_rows, priced_share=view.priced_share,
               real_unique_share_full=float(meta["unique"]) / max(1, int(meta.get("entries_matched") or 0)), real_max_copies_full=int(meta.get("max_copies") or 0))
    for label, pred in preds.items():
        if pred is None:
            continue
        row.update({f"{label}_{k}": v for k, v in ownership_metrics(real_own, pred, ids).items()})
    row.update({f"real_{k}": v for k, v in shape(real_lineups(meta["dir"]), P, q_opt, salaries, subsample=field_size, seed=seed).items()})
    if model_shape:
        row.update({f"model_{k}": v for k, v in model_shape.items()})
    by_id = view.players.set_index(view.players.id.astype(int))
    players = []
    for i in ids:
        p = by_id.loc[i] if i in by_id.index else None
        r = dict(contest_id=int(meta["contest_id"]), slate=view.slate, day=row["day"], id=int(i),
                 name=(p["name"] if p is not None else str(i)), team=(p["team"] if p is not None else None),
                 salary=(int(p["salary"]) if p is not None else None), position=(p["position"] if p is not None else None),
                 status=(p["status"] if p is not None else None), starter=(p["starter"] if p is not None else None),
                 pub=P.get(i), priced=(bool(p["priced"]) if p is not None else False), own_real=real_own.get(i, 0.0))
        for label, pred in preds.items():
            r[f"own_{label}"] = None if pred is None else float(pred.get(i, 0.0))
        players.append(r)
    return row, players


# ----------------------------------------------------------------------------- the run
def run(session: Session, field_size: int = 3000, workers: int = 8, seed: int = 0, since: Optional[str] = None, until: Optional[str] = None,
        contest_ids: Optional[List[int]] = None, with_model: bool = True, derived_dir: Path = SD.DERIVED_DIR, out_dir: Path = OUT_DIR,
        cache_dir: Path = CACHE_DIR, log: Log = print) -> dict:
    """Every final derived contest in the window compared to the model, the lock ownership and the softmax model. Writes
    contests.csv, players.csv, report.txt and summary.json under out_dir/<stamp>/ and returns them with the frames."""
    from ivan.sports.nba.projection.history import build_history, load_box_scores
    metas = final_contests(derived_dir, contest_ids)
    if not metas:
        raise ValueError("no derived contests from a final export; scrape and derive first (scripts/nba/standings.py)")
    history = build_history(load_box_scores(session))
    params = field_params()
    softmax_model = FM.load()
    by_cap: Dict[int, List[dict]] = {}
    for m in metas:
        by_cap.setdefault(int(m["capture_id"]), []).append(m)
    contest_rows, player_rows = [], []
    t0 = time.time()
    for cap_id, ms in by_cap.items():
        try:
            view = capture_view(session, cap_id, history)
        except ValueError as e:
            log(f"capture {cap_id}: skipped ({e})"); continue
        day = view.lock.date().isoformat()
        if (since and day < since) or (until and day > until):
            continue
        pool = field_pool(view)
        q_opt = public_optimal(pool) if len(pool) >= 8 else float("nan")
        own_model = model_shape = None
        if with_model and len(pool) >= 8:
            try:
                field = model_field(view, int(field_size), params, seed=seed, workers=workers, cache_dir=cache_dir, log=log)
                own_model = ownership(field)
                model_shape = shape(field, view.pub, q_opt, view.salaries)
            except Exception as e:   # noqa: BLE001 - the contests still compare against the lock and softmax predictions
                log(f"  slate {view.slate}: model field failed ({type(e).__name__}: {e})")
        own_lock = at_lock_ownership(session, view.slate)
        for m in ms:
            preds = {"model": own_model, "lock": own_lock, "softmax": softmax_ownership(session, view, m, softmax_model)}
            row, players = compare_contest(m, view, preds, q_opt, int(field_size), model_shape, seed=seed)
            contest_rows.append(row); player_rows.extend(players)
            log(f"  {day} {str(m.get('name'))[:44]:44} {row['entries']:>7,}e  " + "  ".join(
                f"{k} MAE {row[f'{k}_mae']:.2f}" for k in ("model", "lock", "softmax") if f"{k}_mae" in row))
    contests = pd.DataFrame(contest_rows); players = pd.DataFrame(player_rows)
    if contests.empty:
        raise ValueError("no contest in the window")
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    d = Path(out_dir) / stamp; d.mkdir(parents=True, exist_ok=True)
    contests.to_csv(d / "contests.csv", index=False); players.to_csv(d / "players.csv", index=False)
    text = report(contests, players, params, field_size)
    (d / "report.txt").write_text(text)
    summary = dict(stamp=stamp, field_size=int(field_size), seed=int(seed), params=params, contests=int(len(contests)), slates=int(contests.slate.nunique()),
                   days=[contests.day.min(), contests.day.max()], seconds=round(time.time() - t0),
                   means={c: float(contests[c].mean()) for c in contests.columns if contests[c].dtype.kind == "f"})
    (d / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    log(f"written {d} ({len(contests)} contests, {contests.slate.nunique()} slates, {summary['seconds']}s)")
    return dict(dir=d, contests=contests, players=players, report=text, summary=summary)


# ----------------------------------------------------------------------------- the report
def calibration(players: pd.DataFrame, col: str) -> pd.DataFrame:
    """Players binned by predicted ownership `col`: count, mean predicted, mean real, bias."""
    p = players.dropna(subset=[col])
    if p.empty:
        return pd.DataFrame()
    b = pd.cut(p[col], OWN_BINS, right=False, include_lowest=True)
    g = p.groupby(b, observed=True)
    out = pd.DataFrame(dict(n=g.size(), pred=g[col].mean(), real=g.own_real.mean()))
    out["bias"] = out.pred - out.real
    return out


def bias_by(players: pd.DataFrame, col: str, by: pd.Series, label: str) -> pd.DataFrame:
    p = players.dropna(subset=[col]).assign(_g=by.loc[players.dropna(subset=[col]).index])
    if p.empty:
        return pd.DataFrame()
    g = p.groupby("_g", observed=True)
    out = pd.DataFrame({"n": g.size(), "real": g.own_real.mean(), "pred": g[col].mean(), "bias": (g[col].mean() - g.own_real.mean()),
                        "mae": g.apply(lambda x: float(np.mean(np.abs(x[col] - x.own_real))), include_groups=False)})
    out.index.name = label
    return out


def report(contests: pd.DataFrame, players: pd.DataFrame, params: Optional[dict] = None, field_size: Optional[int] = None) -> str:
    params = params or {}
    lines = []
    w = lines.append
    w(f"NBA field backtest: {len(contests)} contests on {contests.slate.nunique()} slates, {contests.day.min()} .. {contests.day.max()}"
      + (f"; model field {field_size:,} entries" if field_size else ""))
    if params:
        w(f"two-group parameters: optimizer_share {params.get('optimizer_share')}, sigma_provider {params.get('sigma_provider')}, sigma_personal "
          f"{params.get('sigma_personal')}, sigma_casual {params.get('sigma_casual')}, providers {params.get('providers')} (fitted {params.get('fitted_at')}: {params.get('fitted_on', '')})")
    w("")
    if "priced_share" in contests:
        w("Inputs per slate (the capture the field is rebuilt from): hours before lock it was taken, props rows for its players, share of the available"
          " players the books priced (points, rebounds and assists); an unpriced player projects from history and salary alone")
        for r in contests.drop_duplicates("slate").itertuples():
            flag = "  <- no props: the whole field is the history replica" if not r.odds_rows else ("  <- a day-old capture" if (r.capture_age_h or 0) > 6 else "")
            w(f"  {r.day} slate {r.slate}  captured {_f(r.capture_age_h, 1):>5} h before lock  props rows {int(r.odds_rows):>7,}  priced {100 * r.priced_share:3.0f}%{flag}")
        w("")
    w("Ownership per contest (MAE in points of ownership; top20 = over the real field's 20 most-owned; rho = Spearman; top10 = real top-10 players the"
      " model also puts in its top 10; top8 = the eight most-owned players' summed ownership, real -> model)")
    cols = [c for c in ("model", "lock", "softmax") if f"{c}_mae" in contests]
    hdr = f"{'day':10} {'contest':40} {'entries':>8} {'fee':>6} " + " ".join(f"{c + ' MAE':>9} {c[:3] + ' t20':>8} {c[:3] + ' rho':>8}" for c in cols) + f" {'top10':>5} {'top8 real->model':>18}"
    w(hdr)
    for r in contests.itertuples():
        cells = []
        for c in cols:
            mae = getattr(r, f"{c}_mae", np.nan); t20 = getattr(r, f"{c}_mae_top20", np.nan); rho = getattr(r, f"{c}_spearman", np.nan)
            cells.append(f"{_f(mae):>9} {_f(t20):>8} {_f(rho):>8}")
        t10 = getattr(r, "model_top10_overlap", None)
        t8 = f"{_f(getattr(r, 'model_top8_real', np.nan), 0)} -> {_f(getattr(r, 'model_top8_pred', np.nan), 0)}" if "model_top8_real" in contests else ""
        w(f"{r.day:10} {str(r.name)[:40]:40} {r.entries:>8,} {r.fee:>6.2f} " + " ".join(cells) + f" {'' if t10 is None or pd.isna(t10) else int(t10):>5} {t8:>18}")
    w("")
    w("means over contests:")
    for c in cols:
        sub = contests.dropna(subset=[f"{c}_mae"])
        w(f"  {c:8} {len(sub):3} contests  MAE {sub[f'{c}_mae'].mean():.2f}  top-20 MAE {sub[f'{c}_mae_top20'].mean():.2f}  rho {sub[f'{c}_spearman'].mean():.3f}"
          + (f"  top-10 overlap {sub[f'{c}_top10_overlap'].mean():.1f}/10" if f"{c}_top10_overlap" in sub else "")
          + f"  top-8 share real {sub[f'{c}_top8_real'].mean():.0f} vs {sub[f'{c}_top8_pred'].mean():.0f}")
    if "priced_share" in contests and "model_mae" in contests:
        for label, sub in (("slates with props in the capture", contests[contests.odds_rows > 0]), ("slates without", contests[contests.odds_rows == 0])):
            sub = sub.dropna(subset=["model_mae"])
            if len(sub):
                w(f"  model on {label:32} {len(sub):3} contests  MAE {sub.model_mae.mean():.2f}  top-20 MAE {sub.model_mae_top20.mean():.2f}  rho {sub.model_spearman.mean():.3f}"
                  f"  top-8 share real {sub.model_top8_real.mean():.0f} vs {sub.model_top8_pred.mean():.0f}")
    w("")
    if "model_p50" in contests:
        w(f"Shape, real (subsampled to the model's size) vs model: lineup strength under the public projection relative to its optimal")
        w(f"{'day':10} {'contest':40} {'p10 r/m':>13} {'p50 r/m':>13} {'p90 r/m':>13} {'within10 r/m':>13} {'dup r/m':>13} {'maxcop r/m':>11} {'sal left r/m':>13}")
        for r in contests.itertuples():
            w(f"{r.day:10} {str(r.name)[:40]:40} {_pair(r, 'p10')} {_pair(r, 'p50')} {_pair(r, 'p90')} {_pair(r, 'within10')} {_pair(r, 'dup_share')} "
              f"{_pair(r, 'max_copies', 0, 11)} {_pair(r, 'salary_left_mean', 0)}")
        m = contests.mean(numeric_only=True)
        w(f"means: p10 {m.real_p10:.3f}/{m.model_p10:.3f}  p50 {m.real_p50:.3f}/{m.model_p50:.3f}  p90 {m.real_p90:.3f}/{m.model_p90:.3f}  within5 {m.real_within5:.2f}/{m.model_within5:.2f}"
          f"  within10 {m.real_within10:.2f}/{m.model_within10:.2f}  dup share {m.real_dup_share:.2f}/{m.model_dup_share:.2f}  unique {m.real_unique_share:.2f}/{m.model_unique_share:.2f}"
          f"  salary left {m.real_salary_left_mean:.0f}/{m.model_salary_left_mean:.0f}  full salary {m.real_salary_full:.2f}/{m.model_salary_full:.2f}")
        w("")
    for col, label in (("own_model", "model"), ("own_lock", "lock"), ("own_softmax", "softmax")):
        if col not in players or players[col].notna().sum() == 0:
            continue
        cal = calibration(players, col)
        if cal.empty:
            continue
        w(f"Calibration of the {label} ownership (players binned by predicted ownership; bias = predicted - real):")
        for b, r in cal.iterrows():
            w(f"  {str(b):14} n {int(r.n):5}  predicted {r.pred:5.1f}  real {r.real:5.1f}  bias {r.bias:+5.1f}")
        w("")
    if "own_model" in players and players.own_model.notna().any():
        p = players.dropna(subset=["salary"])
        tier = pd.cut(p.salary / 1000.0, TIERS, labels=["<4.5k", "4.5-6.5k", "6.5-8.5k", "8.5k+"])
        w("Model bias by salary tier (points of ownership):")
        for t, r in bias_by(p, "own_model", tier, "tier").iterrows():
            w(f"  {str(t):10} n {int(r.n):5}  real {r.real:5.1f}  model {r.pred:5.1f}  bias {r.bias:+5.1f}  MAE {r.mae:.2f}")
        if "priced" in p:
            w("Model bias by whether the books priced the player at the capture:")
            for t, r in bias_by(p, "own_model", p.priced.map({True: "priced", False: "unpriced"}), "priced").iterrows():
                w(f"  {str(t):10} n {int(r.n):5}  real {r.real:5.1f}  model {r.pred:5.1f}  bias {r.bias:+5.1f}  MAE {r.mae:.2f}")
            hot = p[p.own_real >= 30]
            if len(hot):
                w(f"  real chalk (30%+): {len(hot)} player-contests, {100 * hot.priced.mean():.0f}% priced; model MAE on them {np.mean(np.abs(hot.own_model - hot.own_real)):.1f}"
                  f" (priced {np.mean(np.abs(hot[hot.priced].own_model - hot[hot.priced].own_real)) if hot.priced.any() else float('nan'):.1f}, unpriced"
                  f" {np.mean(np.abs(hot[~hot.priced].own_model - hot[~hot.priced].own_real)) if (~hot.priced).any() else float('nan'):.1f})")
        st = p.starter.fillna("").astype(str).str.lower().isin(["confirmed", "expected", "true", "1", "s"]).map({True: "starter", False: "bench"})
        w("Model bias by starter label at the capture:")
        for t, r in bias_by(p, "own_model", st, "starter").iterrows():
            w(f"  {str(t):10} n {int(r.n):5}  real {r.real:5.1f}  model {r.pred:5.1f}  bias {r.bias:+5.1f}  MAE {r.mae:.2f}")
        real_bin = pd.cut(p.own_real, OWN_BINS, right=False, include_lowest=True)
        w("Model by REAL ownership (what the field actually did; bias = model - real):")
        for t, r in bias_by(p, "own_model", real_bin, "real").iterrows():
            w(f"  {str(t):14} n {int(r.n):5}  real {r.real:5.1f}  model {r.pred:5.1f}  bias {r.bias:+5.1f}  MAE {r.mae:.2f}")
        w("")
        big = p.assign(err=(p.own_model - p.own_real)).sort_values("err", key=np.abs, ascending=False).head(25)
        w("Largest misses (model - real), one line per player-contest:")
        for r in big.itertuples():
            w(f"  {r.day} {str(r.name)[:24]:24} {str(r.team or ''):4} ${(r.salary or 0) / 1000:4.1f}k pub {(r.pub or 0):5.1f} {'priced' if getattr(r, 'priced', False) else 'unpr. '} {str(r.starter or '')[:4]:4} {str(r.status or ''):3} "
              f"real {r.own_real:5.1f}  model {r.own_model:5.1f}  {r.err:+6.1f}" + (f"  lock {r.own_lock:5.1f}" if "own_lock" in p and pd.notna(r.own_lock) else "")
              + f"   {str(getattr(r, 'contest_id'))}")
    return "\n".join(lines) + "\n"


def _f(v, nd: int = 2) -> str:
    return "" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{nd}f}"


def _pair(r, key: str, nd: int = 2, width: int = 13) -> str:
    a, b = getattr(r, f"real_{key}", np.nan), getattr(r, f"model_{key}", np.nan)
    return f"{_f(a, nd)}/{_f(b, nd)}".rjust(width)
