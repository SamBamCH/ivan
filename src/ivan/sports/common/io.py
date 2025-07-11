import pandas as pd
import numpy as np
from collections import Counter
from sqlalchemy.orm import Session

from ivan.sports.mlb.models import Lineup, MLBPlayer

# fixed slot order for human reference (not used as column keys):
POSITION_ORDER = ["P","P","C","1B","2B","3B","SS","OF","OF","OF"]


def assign_players_to_slots(player_objs: list[MLBPlayer], slots: list[str]) -> list[MLBPlayer]:
    """
    Assign each of the 10 unique players exactly once to the fixed slot order.
    Raises if any player cannot fit or any slot left empty.
    """
    assignment: list[MLBPlayer | None] = [None] * len(slots)
    filled = [False] * len(slots)

    for p in player_objs:
        raw = p.position.split("/")
        norms = ["P" if x in ("SP","RP") else x for x in raw]
        placed = False
        for i, slot in enumerate(slots):
            if not filled[i] and slot in norms:
                assignment[i] = p
                filled[i] = True
                placed = True
                break
        if not placed:
            raise ValueError(f"Cannot place {p.name} ({p.id}) into any slot: {raw}")

    if not all(filled):
        missing = [slots[i] for i,u in enumerate(filled) if not u]
        raise ValueError(f"Slots unfilled: {missing}")

    return assignment  # type: ignore


def export_selected_lineups(
    session: Session,
    chosen_ids: list[int],
    salary_cap: int
) -> pd.DataFrame:
    rows = []
    lineups = session.query(Lineup).filter(Lineup.id.in_(chosen_ids)).all()
    all_pids = {pid for L in lineups for pid in L.player_ids_list}
    players  = {p.id: p for p in session.query(MLBPlayer).filter(MLBPlayer.id.in_(all_pids)).all()}

    for L in lineups:
        plist = L.player_ids_list
        if len(plist) != 10 or len(set(plist)) != 10:
            raise ValueError(f"Lineup {L.id} must have exactly 10 unique players")
        objs = [players[pid] for pid in plist]

        assigned = assign_players_to_slots(objs, POSITION_ORDER)
        slots_str = [f"{pl.name} ({pl.id})" for pl in assigned]

        fpts, payouts, rois, ranks = L.get_sim_arrays()
        payouts = np.array(payouts, dtype=float)
        rois    = np.array(rois,    dtype=float)
        cash_pct    = float((payouts > 0).mean())
        avg_roi_win = float(rois.mean())
        avg_proj    = float(np.mean([pl.fpts_projection for pl in assigned]))
        avg_own     = float(np.mean([pl.ownership       for pl in assigned]))

        hitter_ts = [pl.team for pl in assigned if pl.position not in ("SP","RP","P")]
        stacks = Counter(hitter_ts).most_common()
        primary_team, primary_count = stacks[0] if stacks else ("",0)
        secondary_team, secondary_count = stacks[1] if len(stacks)>1 else ("",0)

        opps = {pl.opponent for pl in assigned if pl.position in ("SP","RP","P")}
        print(opps)
        opp_hitters = sum(1 for pl in assigned if pl.team in opps and pl.position not in ("SP","RP","P"))

        total_sal = sum(pl.salary for pl in assigned)
        salary_rem = salary_cap - total_sal

        win_pct = 100.0 * np.mean([r==1 for r in ranks]) if ranks else 0.0

        row = {"lineup_id": L.id}
        # use index-based keys: slot_1 … slot_10
        for i, val in enumerate(slots_str, start=1):
            row[f"slot_{i}"] = val

        row.update({
            "primary_team": primary_team,
            "secondary_team": secondary_team,
            "primary_count": primary_count,
            "secondary_count": secondary_count,
            "opposing_hitters": opp_hitters,
            "salary_remaining": salary_rem,
            "avg_projection": avg_proj,
            "avg_ownership": avg_own,
            "cash_pct": cash_pct,
            "avg_roi_win": avg_roi_win,
            "win_pct": win_pct
        })
        rows.append(row)

    cols = ["lineup_id"] + [f"slot_{i}" for i in range(1,11)] + [
        "primary_team","secondary_team",
        "primary_count","secondary_count",
        "opposing_hitters","salary_remaining",
        "avg_projection","avg_ownership",
        "cash_pct","avg_roi_win","win_pct"
    ]
    return pd.DataFrame(rows, columns=cols)
