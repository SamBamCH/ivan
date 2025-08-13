import re
import logging
from collections import Counter
from datetime import datetime, timedelta
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from sqlalchemy import func
from ivan.database.session import init_db, get_session
from ivan.sports.common.draftkings import build_requests_session, download_contest_dataframe

from ivan.core.utils import normalize_name
from ivan.sports.mlb.models import (
    HistoricalContest,
    ContestPayout,
    ActualContestEntry,
    SlateKDEModel,
    BatterProjection,
    PitcherProjection,
    ContestIngestStatus
)
from ivan.database.management import (
    select_contests_to_process,
    mark_success, mark_failure,
    window_start, within_window,
)

# regex to extract (position, name) pairs from a lineup string
_LINEUP_REGEX = re.compile(r"\b(1B|2B|3B|SS|C|OF|P)\s+(.+?)(?=(?:\s(?:1B|2B|3B|SS|C|OF|P)\s)|$)")
PITCHER_POS = {"SP", "RP", "P"}

MAX_ATTEMPTS = 3  # after this, mark permanently_failed

def _parse_lineup(cell: str) -> List[Tuple[str, str]]:
    """
    Parse a single 'Lineup' cell into a list of (position, player_name) tuples.
    """
    return _LINEUP_REGEX.findall(cell or "")


def _batch_fetch_latest(session, Model, key_attr: str, names: set, cutoff_dt: datetime) -> Dict[str, any]:
    """
    Retrieve the most recent projection record for each name in `names` before `cutoff_dt`.
    Returns a mapping name -> ORM instance.
    """
    rows = (
        session.query(Model)
               .filter(getattr(Model, key_attr).in_(names),
                       Model.created_at <= cutoff_dt)
               .order_by(getattr(Model, key_attr),
                         Model.created_at.desc())
               .all()
    )
    latest = {}
    for r in rows:
        k = getattr(r, key_attr)
        if k not in latest:
            latest[k] = r
    return latest


def parse_df(df: pd.DataFrame, contest_id: int) -> Tuple[
        List[List[Tuple[str, str]]],
        List[Dict],
        List[Dict]
    ]:
    """
    Given a raw standings DataFrame from DraftKings:
      - split off the last 4 columns (actual ownership info)
      - base_df contains everything else (Lineup, Position, Payout, etc.)
      - own_df contains the last 4 columns: Player, Roster Position, %Drafted, FPTS

    Returns:
      1) lineups: list of parsed lineup-lists [(pos,name),...]
      2) payouts: list of dicts for ContestPayout upsert
      3) actuals: list of dicts for ActualContestEntry upsert (including ownership)
    """
    # split into base and ownership dataframes
    base_df = df.iloc[:, :-4]
    own_df  = df.iloc[:, -4:].copy()

    # 1) parse all lineups from base_df
    lineups = [_parse_lineup(cell) for cell in base_df["Lineup"].astype(str)]

    # 2) build payout tiers using base_df
    payouts = []
    if "Position" in base_df.columns and "Payout" in base_df.columns:
        for payout_val, grp in base_df.groupby("Payout"):
            amt = float(str(payout_val).replace("$", "").replace(",", "")) or 0.0
            if amt <= 0.0:
                continue
            min_pos = int(grp["Position"].min())
            max_pos = int(grp["Position"].max())
            payouts.append({
                "contest_id": contest_id,
                "min_position": min_pos,
                "max_position": max_pos,
                "payout_amount": amt,
                "payout_text": payout_val,
                "currency": "USD"
            })

    # 3) collect actual entry results merging base_df + own_df
    actuals = []
    for idx, row in base_df.iterrows():
        # safe EntryId conversion (handle NaN)
        raw_id = row.get("EntryId")
        if pd.isna(raw_id):
            raw_id = row.get("Entry")
        if pd.isna(raw_id):
            entry_id = 0
        else:
            entry_id = int(raw_id)

        # safe rank conversion
        raw_rank = row.get("Position") or row.get("Rank")
        rank = int(raw_rank) if raw_rank is not None and not pd.isna(raw_rank) else 0

        # points and payout
        pts = float(row.get("Points") or row.get("FantasyPoints") or 0.0)
        payout_amt = float(str(row.get("Payout", "0")).replace("$", "").replace(",", "")) or 0.0

        # pull ownership info from own_df
        orow = own_df.iloc[idx]
        owner = orow.get("%Drafted") or orow.get("%Draft") or None
        fpts   = orow.get("FPTS") or None

        actuals.append({
            "contest_id":     contest_id,
            "entry_id":       entry_id,
            # lineup_id to resolve later
            "lineup":         row["Lineup"],
            "username":       row.get("User") or row.get("Username"),
            "rank":           rank,
            "fantasy_points": pts,
            "payout":         payout_amt,
            # include actual ownership fields
            "percent_drafted": owner,
            "actual_fpts":     fpts,
        })

    return lineups, payouts, actuals


def upsert_payout_changes(session, payouts: List[Dict]) -> None:
    """
    Upsert the ContestPayout tiers for a given contest.
    """
    for p in payouts:
        cp = ContestPayout(
            contest_id    = p["contest_id"],
            min_position  = p["min_position"],
            max_position  = p["max_position"],
            payout_amount = p["payout_amount"],
            payout_text   = p["payout_text"],
            currency      = p["currency"]
        )
        session.merge(cp)


def upsert_actuals(session, actuals: List[Dict]) -> None:
    """
    Upsert each entry's actual result into actual_contest_entry.
    NOTE: `lineup_id` resolution from the raw string must be handled later.
    """
    for a in actuals:
        ace = ActualContestEntry(
            contest_id     = a["contest_id"],
            entry_id       = a["entry_id"],
            lineup_id      = None,  # TODO: link to saved Lineup.id
            username       = a["username"],
            rank           = a["rank"],
            fantasy_points = a["fantasy_points"],
            payout         = a["payout"]
        )
        session.merge(ace)

def fit_slate_kde(
    session,
    hc: HistoricalContest,
    lineups: List[List[Tuple[str, str]]]
) -> None:
    """
    Given a HistoricalContest and its parsed lineup data,
    compute slate-level KDE bandwidths and fixed-bin histogram counts,
    then upsert a SlateKDEModel (only counts, no bin edges).
    """
    cid          = hc.contest_id
    contest_dt   = hc.date
    total_entries= hc.total_entries
    max_entries  = hc.multi_entry
    entry_fee    = hc.entry_fee

    # --- find cutoff timestamps for projections
    day_start = datetime(contest_dt.year, contest_dt.month, contest_dt.day)
    latest = lambda Model: (
        session.query(func.max(Model.created_at))
               .filter(Model.created_at <= contest_dt)
               .scalar()
    )
    pitcher_cutoff = latest(PitcherProjection)
    batter_cutoff  = latest(BatterProjection)
    cutoff_ts      = max(pitcher_cutoff, batter_cutoff)

    # --- batch fetch latest projections as of cutoff_ts
    p_names, b_names = set(), set()
    for lu in lineups:
        for pos, nm in lu:
            nn = normalize_name(nm)
            (p_names if pos in PITCHER_POS else b_names).add(nn)

    pmap = _batch_fetch_latest(session, PitcherProjection, "player", p_names, pitcher_cutoff)
    bmap = _batch_fetch_latest(session, BatterProjection,  "name",   b_names,  batter_cutoff)

    # --- build per-lineup projections & ownership lists
    projs, ownerships, teams, stack_types = [], [], set(), []
    owner_vals = [
        (getattr(rec, 'ownership', None) or getattr(rec, 'pown', 0.0)) / 100.0
        for rec in list(pmap.values()) + list(bmap.values())
    ]

    for lu in lineups:
        team_counts, vals, owns = Counter(), [], []
        for pos, nm in lu:
            nn  = normalize_name(nm)
            rec = pmap.get(nn) if pos in PITCHER_POS else bmap.get(nn)
            if not rec:
                continue
            if pos not in PITCHER_POS:
                team_counts[rec.team] += 1
            vals.append(rec.fpts)
            owns.append((getattr(rec, 'ownership', None) or getattr(rec, 'pown', 0.0)) / 100.0)
            teams.add(rec.team)

        if not vals:
            continue

        # simple “stack type” as e.g. "2-1"
        top_two = sorted(team_counts.values(), reverse=True)[:2]
        stack_types.append(f"{top_two[0] if top_two else 0}-{top_two[1] if len(top_two)>1 else 0}")

        projs.append(sum(vals))
        ownerships.append(np.mean(owns))

    if not projs:
        return

    # --- slate-level summary stats
    owner_vals.sort(reverse=True)
    top_owned    = owner_vals[0]
    top_5_owned  = np.mean(owner_vals[:5])   if len(owner_vals)>=5  else top_owned
    top_10_owned = np.mean(owner_vals[:10])  if len(owner_vals)>=10 else top_owned

    # payout % for 1st & top 5
    pool = entry_fee * total_entries
    sum_payout = lambda lo, hi: (
        session.query(func.sum(ContestPayout.payout_amount))
               .filter(ContestPayout.contest_id==cid,
                       ContestPayout.min_position.between(lo, hi))
               .scalar() or 0.0
    )
    first_pct = sum_payout(1, 1) / pool
    top5_pct  = sum_payout(1, 5) / pool

    # --- floor at 70, min-max scale projections to [0,1]
    MIN_PROJ = 70.0
    bounded = [max(p, MIN_PROJ) for p in projs]
    max_proj = float(max(bounded))
    projs_norm = [(p - MIN_PROJ) / (max_proj - MIN_PROJ) for p in bounded]

    # --- KDE bandwidths
    kde_own  = gaussian_kde(np.array(ownerships))
    kde_proj = gaussian_kde(np.array(projs_norm))

    # --- fixed-bin histograms (counts only)
    fixed_bins = np.linspace(0, 1, 21)  # 0.00,0.05,…,1.00
    hist_own_counts, _  = np.histogram(ownerships, bins=fixed_bins)
    hist_proj_counts, _ = np.histogram(projs_norm,   bins=fixed_bins)

    num_games = max(len(teams)//2, 1)

    model = SlateKDEModel(
        contest_id       = cid,
        contest_date     = contest_dt,
        num_games        = num_games,
        total_entries    = total_entries,
        max_entries      = max_entries,
        entry_fee        = entry_fee,
        top_owned        = top_owned,
        top_5_owned      = top_5_owned,
        top_10_owned     = top_10_owned,
        first_pct        = first_pct,
        top5_pct         = top5_pct,
        bw_own           = float(kde_own.factor),
        bw_proj          = float(kde_proj.factor),
        max_proj         = max_proj,
        # only counts now:
        hist_own_counts  = hist_own_counts.tolist(),
        hist_proj_counts = hist_proj_counts.tolist(),
        hist_stack_types = list(Counter(stack_types).keys()),
        hist_stack_probs = list(Counter(stack_types).values()),
    )
    session.merge(model)


# define once near imports
log = logging.getLogger(__name__)

def get_contest_data(profile_dir: str = r"C:\selenium\chrome-profile"):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger("get_contest_data")

    # init DB & HTTP session
    engine  = init_db("mlb")
    session = get_session(engine)
    sess    = build_requests_session(profile_dir)

    # only process contests in the last 3 days (up to now)
    now = datetime.utcnow()
    cutoff = now - timedelta(days=7)

    # skip contests we’ve already modeled
    modeled_subq = session.query(SlateKDEModel.contest_id)

    to_process = (
        session.query(HistoricalContest)
        .filter(
            HistoricalContest.date >= cutoff,
            HistoricalContest.date <= now,
            ~HistoricalContest.contest_id.in_(modeled_subq),
        )
        .all()
    )

    log.info("Need to process %d contests in window [%s → %s]", len(to_process), cutoff, now)

    for hc in to_process:
        # extra guard in case any rows have weird timestamps
        if not (cutoff <= hc.date <= now):
            continue

        cid, sched = hc.contest_id, hc.date
        log.info("→ Modeling contest %s @ %s", cid, sched)

        df = download_contest_dataframe(sess, cid)
        if df is None:
            log.warning("[warn] Contest %s: download returned None", cid)
            continue

        lineups, payouts, actuals = parse_df(df, cid)
        try:
            # only modeling for now
            # upsert_payout_changes(session, payouts)
            # upsert_actuals(session, actuals)
            fit_slate_kde(session, hc, lineups)
            session.commit()
            log.info("✔ Modeled %s", cid)
        except Exception:
            session.rollback()
            log.exception("✗ Failed modeling %s", cid)

    log.info("All done.")
