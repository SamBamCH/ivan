# ─── src/ivan/sports/mlb/historical/storage.py ─────────────────────────────────

from ivan.sports.mlb.models import ActualContestEntry, GeneralizedContestMetrics
from sqlalchemy.exc import IntegrityError

def write_payout_breakpoint(db, contest_id: int, entry_id: int, lineup_id: int, username: str, rank: int, fantasy_points: float, payout: float):
    """
    Insert one ContestPayout row when the payout tier changes.
    If it already exists (e.g. due to rerun), ignore the IntegrityError.
    """
    ce = ActualContestEntry(
        contest_id      = contest_id,
        entry_id        = entry_id,
        lineup_id       = lineup_id,
        username        = username,
        rank            = rank,
        fantasy_points  = fantasy_points,
        payout          = payout
    )
    try:
        db.add(ce)
        db.commit()
    except IntegrityError:
        db.rollback()
        # already inserted this payout tier—ignore

def write_generalized_metrics(db, contest_id: int, metrics_dict: dict):
    """
    Upsert a single GeneralizedContestMetrics row:
      { "P": [ ... ], "O": [ ... ], "S": [ ... ], "E": [ ... ] }
    """
    existing = db.query(GeneralizedContestMetrics).filter_by(contest_id=contest_id).one_or_none()
    if existing:
        existing.metrics_json = metrics_dict
        db.add(existing)
    else:
        gm = GeneralizedContestMetrics(
            contest_id   = contest_id,
            metrics_json = metrics_dict
        )
        db.add(gm)
    db.commit()
