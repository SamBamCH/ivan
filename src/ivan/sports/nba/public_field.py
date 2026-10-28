# src/ivan/sports/nba/public_field.py
"""
The public's side of the slate, kept apart from ours.

public projection   what the field is playing to: the Stokastic replica (public_projection.py) from the books' props and
                    public inputs (salary, status, history). Written to nba_player.pub_fpts. It changes when the pool,
                    the statuses or the props change, never when our minutes or rates are edited.
the field           the two-group field (field_model.simulate_field, data/nba/models/field_two_group.json) built on the
                    public projection, cached by a key made from the public projection alone (field_key), so an edit
                    to our projections cannot move it.
ownership           the field's exposure per player, written to nba_player.ownership.

refresh_public(session, history)   recompute all three; called by the pool refresh (ingest.store_slate), not by Publish.
slate_field(session, size)         the field for the sim and the late swap (the cached one when the key matches).
"""
from __future__ import annotations

import hashlib
import json
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from ivan.core.seed import seed_for
from ivan.core.utils import normalize_name
from ivan.sports.nba.models import NBAPlayer
from ivan.sports.nba.paths import DIAG_DIR, MODELS_DIR
from ivan.sports.nba.pre_lock.lineup_generation import FieldPlayer
from ivan.sports.nba.utils import _coerce_positions

Log = Callable[[str], None]
FIELD_PARAMS_JSON = MODELS_DIR / "field_two_group.json"
FIELD_PARAMS_DEFAULT = {"model": "two_group", "sigma_provider": 0.06, "sigma_personal": 0.05, "optimizer_share": 0.25,
                        "sigma_casual": 0.45, "providers": 3, "fitted_at": "2026-10-23"}
FIELD_CACHE_DIR = DIAG_DIR.parent / "derived" / "field_cache"
FIELD_CACHE_KEEP = 40
OWNERSHIP_FIELD = 3000          # entries in the field ownership is read from
POOL_MIN_PUB = 5.0              # public projection below this: not in the field's pool
OUT_STATUSES = {"OUT", "O"}


def field_params(path: Path = FIELD_PARAMS_JSON) -> dict:
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return dict(FIELD_PARAMS_DEFAULT)


# ----------------------------------------------------------------------------- the public projection
def public_projection(session: Session, history, on: Optional[datetime] = None) -> Dict[int, float]:
    """player id -> the public projection (0 for players ruled out)."""
    from ivan.sports.nba.public_projection import project
    players = session.query(NBAPlayer).all()
    if not players:
        return {}
    df = pd.DataFrame([dict(player_name=normalize_name(p.name), salary=p.salary or 0, id=int(p.id), out=(p.status or "").upper() in OUT_STATUSES) for p in players])
    odds = pd.read_sql("select player, market, sportsbook, side, point, price, last_update from odds", session.connection())
    odds = odds[odds.player.isin(set(df.player_name))]
    pub = project(df[["player_name", "salary"]], odds, history, (on or datetime.now()).date())
    return {int(r.id): (0.0 if r.out else float(pub.get(r.player_name, 0.0))) for r in df.itertuples()}


def field_key(session: Session) -> str:
    """What the field depends on: every player's id, salary, position and public projection. Our projections are not in it."""
    rows = session.query(NBAPlayer.id, NBAPlayer.salary, NBAPlayer.position, NBAPlayer.pub_fpts, NBAPlayer.draft_group_id).order_by(NBAPlayer.id).all()
    params = field_params()
    raw = json.dumps([[int(i), int(s or 0), str(p), round(float(f or 0.0), 1), g] for i, s, p, f, g in rows]
                     + [params.get(k) for k in ("optimizer_share", "sigma_provider", "sigma_personal", "sigma_casual", "providers")])
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


def public_pool(session: Session) -> List[FieldPlayer]:
    """The field's player pool: everyone the public projects above POOL_MIN_PUB, with the public projection as his score."""
    out = []
    for rp in session.query(NBAPlayer).all():
        pub = float(rp.pub_fpts or 0.0)
        if pub < POOL_MIN_PUB or rp.salary is None or not rp.position:
            continue
        team, opp = rp.team or "", rp.opponent
        out.append(FieldPlayer(id=int(rp.id), name=rp.name, salary=int(rp.salary), team=team, opponent=opp,
                               position=tuple(_coerce_positions(rp.position)), fpts_mean=pub,
                               game_id="-".join(sorted([team.upper(), opp.upper()])) if team and opp else None, gametime=rp.gametime))
    return out


# ----------------------------------------------------------------------------- the field
def slate_field(session: Session, size: int, log: Log = print) -> List[tuple]:
    """The two-group field on the public projection at `size` entries (sorted id tuples, copies kept), cached by field_key."""
    from ivan.sports.nba import field_model as FM
    key = field_key(session)
    path = FIELD_CACHE_DIR / f"{key}.json"
    if path.exists():
        cached = [tuple(x) for x in json.loads(path.read_text())]
        if len(cached) >= size:
            return cached[:size]
    players = public_pool(session)
    if len(players) < 8:
        raise ValueError("the public projection covers fewer than 8 players; refresh the pool (props and statuses) first")
    params = field_params()
    dg = next((g for (g,) in session.query(NBAPlayer.draft_group_id).limit(1).all()), 0)
    seed = seed_for(f"field/{dg}") or 0
    t0 = time.time()
    old = FM.SIGMA_CASUAL; FM.SIGMA_CASUAL = float(params.get("sigma_casual", old))
    try:
        field = FM.simulate_field(players, {p.id: p.fpts_mean for p in players}, int(size), float(params["sigma_provider"]),
                                  float(params["sigma_personal"]), float(params["optimizer_share"]), seed=int(seed), providers=int(params.get("providers", 3)))
    finally:
        FM.SIGMA_CASUAL = old
    rng = np.random.default_rng(int(seed) + 1); rng.shuffle(field)
    FIELD_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([list(x) for x in field]))
    for old_file in sorted(FIELD_CACHE_DIR.glob("*.json"), key=lambda f: f.stat().st_mtime)[:-FIELD_CACHE_KEEP]:
        old_file.unlink(missing_ok=True)
    log(f"public field: {len(field):,} lineups on the public projection in {time.time() - t0:.0f}s (key {key})")
    return field


def field_ownership(session: Session, size: int = OWNERSHIP_FIELD, log: Log = print) -> Dict[int, float]:
    field = slate_field(session, int(size), log=log)
    held = Counter(p for lu in field for p in lu)
    return {int(pid): 100.0 * c / max(1, len(field)) for pid, c in held.items()}


# ----------------------------------------------------------------------------- refresh
def ownership_field_size() -> int:
    """Entries in the field ownership is read from after a refresh: config field.ownership_entries, default OWNERSHIP_FIELD
    (3,000: about 45s on eight workers; 10,000 took 159s on 2026-10-28, too much to rebuild after every refresh). Set it to
    the sim's cap to have the generate step and the sim find their field cached."""
    try:
        from ivan.core.config import load_config
        return max(500, int(load_config("nba").get("field", {}).get("ownership_entries", OWNERSHIP_FIELD)))
    except Exception:   # noqa: BLE001
        return OWNERSHIP_FIELD


def refresh_public_projection(session: Session, history, log: Log = print, on: Optional[datetime] = None) -> dict:
    """
    The public projection from the pool and props as stored, written to nba_player.pub_fpts (a second or two). The
    field and ownership are built on it by refresh_field_ownership, which the callers run once the pool is committed.
    Returns {players, projected}.
    """
    pub = public_projection(session, history, on)
    # a hand override of the public number (the desk's pub column) replaces the replica's for that player: the field and
    # ownership are then built on it (the field key carries pub_fpts, so the field is re-sampled)
    try:
        from ivan.sports.nba.projection.live import read_overrides_full
        from ivan.core.utils import normalize_name
        ov = read_overrides_full()
    except Exception:   # noqa: BLE001
        ov = {}
    now = datetime.now()
    n_ov = 0
    for p in session.query(NBAPlayer).all():
        hand = (ov.get(normalize_name(p.name)) or {}).get("pub")
        if hand is not None:
            pub[int(p.id)] = float(hand); n_ov += 1
        p.pub_fpts = pub.get(int(p.id), 0.0); p.pub_updated_at = now
    if n_ov:
        log(f"public projection: {n_ov} hand override(s) applied")
    session.commit()
    n_pub = sum(1 for v in pub.values() if v > 0)
    log(f"public projection for {n_pub} players")
    return dict(players=len(pub), projected=n_pub)


def refresh_field_ownership(session: Session, log: Log = print, size: Optional[int] = None) -> dict:
    """
    The two-group field on the stored public projection (cached by its key: a rebuild only when a public number moved,
    about a minute at ownership_field_size entries) and every player's ownership from it, written to nba_player.ownership.
    Returns {owned, seconds}.
    """
    t0 = time.time()
    try:
        own = field_ownership(session, int(size or ownership_field_size()), log=log)
    except Exception as e:   # noqa: BLE001 - ownership is a by-product of the public projection
        session.rollback(); log(f"ownership not refreshed: {type(e).__name__}: {e}"); own = {}
    if own:
        for p in session.query(NBAPlayer).all():
            p.ownership = float(own.get(int(p.id), 0.0))
        session.commit()
    log(f"ownership for {len(own)} players")
    return dict(owned=len(own), seconds=round(time.time() - t0, 1))


def refresh_public(session: Session, history, log: Log = print, on: Optional[datetime] = None) -> dict:
    """
    The public projection, the field and ownership, from the pool and props as stored. Returns {players, projected, owned}.
    Safe to call often: the field is rebuilt only when its key changes. The pool refresh runs the two halves apart
    (ingest.run_ingest: the projection inside the pool's transaction, the field once it is committed).
    """
    out = refresh_public_projection(session, history, log=log, on=on)
    own = refresh_field_ownership(session, log=log)
    return dict(players=out["players"], projected=out["projected"], owned=own["owned"])
