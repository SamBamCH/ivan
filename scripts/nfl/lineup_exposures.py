#!/usr/bin/env python3
"""
Enhanced lineup diagnostics & exposures for NFL (classic + showdown).

What this script does (superset of the original):
- Prints player exposure table (with optional position filter and top-N limit)
- Detects format per lineup (auto): showdown if any CPT (511) present; otherwise classic
- Computes showdown stack types (5-1 / 4-2 / 3-3) and CPT-majority/minority
- Computes classic QB stack types: "QB+X with BR Y" where
    X = # of same-team WR/TE (optionally counts RB if --include-rb-in-stacks)
    Y = # of bring-backs from a single opposing team (max among teams present)
- Summarizes duplicate trains (unique lineups, largest trains)

CLI examples:
  python lineup_exposures.py                          # default, full report
  python lineup_exposures.py --pos WR --top 30        # only WRs, show top 30 by exposure
  python lineup_exposures.py --pos QB --pos TE        # multiple position filters
  python lineup_exposures.py --no-dupes               # skip dupes summary
  python lineup_exposures.py --no-stacks              # skip stack summaries
  python lineup_exposures.py --include-rb-in-stacks   # RB counts toward WR/TE stack

Notes:
- Team fields on NFLPlayer: we try .team, .team_abbrev, or .TeamAbbrev (case-insensitive)
- Classic bring-back heuristic: choose the opponent team with the most skill players
- Showdown stack types ignore players with missing team

"""
from __future__ import annotations

from collections import Counter, defaultdict
import argparse
import json
from typing import Dict, Iterable, List, Optional, Tuple

from ivan.database.session import init_db, get_session
from ivan.sports.nfl.models import NFLPlayer, Lineup

# Roster slot IDs
QB, RB, WR, TE, DST = 66, 67, 68, 69, 71
CPT, FLEX = 511, 512

ROSTER_SLOT_MAP = {
    QB: "QB",
    RB: "RB",
    WR: "WR",
    TE: "TE",
    DST: "DST",
    CPT: "CPT",
    FLEX: "FLEX",
}

# Which slots should count as receivers for classic QB stacks
CLASSIC_PASS_CATCHERS = {WR, TE}


def _coerce_player_ids(val) -> List[int]:
    """Handle JSON column variations (already-parsed list vs stored as str)."""
    if val is None:
        return []
    if isinstance(val, list):
        return [int(x) for x in val]
    if isinstance(val, str):
        try:
            data = json.loads(val)
            if isinstance(data, list):
                return [int(x) for x in data]
        except Exception:
            return []
    return []


def _get_team_from_player(p: NFLPlayer) -> Optional[str]:
    # try multiple common attribute names
    for attr in ("team", "team_abbrev", "TeamAbbrev", "TEAM", "TEAMABBREV"):
        if hasattr(p, attr):
            val = getattr(p, attr)
            if val:
                return str(val).upper()
    return None


def _slot_label(slot_id: Optional[int]) -> str:
    if slot_id is None:
        return "—"
    return ROSTER_SLOT_MAP.get(slot_id, str(slot_id))


def _format_pct(v: float) -> str:
    return f"{v*100:6.2f}%"


def _detect_showdown_lineup(player_slot_ids: Iterable[int]) -> bool:
    # Heuristic: presence of CPT implies showdown
    return any(s == CPT for s in player_slot_ids)


def _summarize_dupes(lineups: List[Tuple[int, Tuple[int, ...]]]) -> Tuple[int, List[Tuple[int,int]]]:
    """Return (#unique_lineups, top_trains: [(size,count), ...]) where
    size = train size, count = how many trains of that size.
    """
    freq = Counter(key for _, key in lineups)
    unique = len(freq)
    trains_by_size = Counter(freq.values())  # size -> how many trains of that size
    top = sorted(trains_by_size.items(), key=lambda x: (-x[0], -x[1]))[:10]
    return unique, top


def _showdown_stack_type(team_list: List[Optional[str]], cpt_team: Optional[str]) -> Optional[str]:
    teams = [t for t in team_list if t]
    if not teams:
        return None
    counts = Counter(teams)
    top_two = sorted(counts.values(), reverse=True)[:2]
    if len(top_two) < 2:
        return None
    primary, secondary = top_two[0], top_two[1]
    label = f"{primary}-{secondary}"
    if cpt_team and counts:
        # determine whether CPT team is majority, minority, or tie
        cpt_ct = counts.get(cpt_team, 0)
        if cpt_ct == primary and list(counts.values()).count(primary) == 1:
            label += " (CPT-majority)"
        elif cpt_ct == secondary and list(counts.values()).count(secondary) == 1:
            label += " (CPT-minority)"
        else:
            label += " (CPT-tie/other)"
    return label


def _classic_qb_stack_type(player_objs: List[NFLPlayer], include_rb: bool) -> Optional[str]:
    # Identify QB
    qbs = [p for p in player_objs if getattr(p, 'roster_slot_id', None) == QB]
    if not qbs:
        return None
    qb = qbs[0]
    qb_team = _get_team_from_player(qb)
    if not qb_team:
        return None

    # Which slots count as receivers for the stack
    receivers = set(CLASSIC_PASS_CATCHERS)
    if include_rb:
        receivers.add(RB)

    teams = []
    opp_counts = Counter()
    same_team_receivers = 0

    for p in player_objs:
        slot = getattr(p, 'roster_slot_id', None)
        team = _get_team_from_player(p)
        if not team or p.id == qb.id:
            continue
        if team == qb_team and slot in receivers:
            same_team_receivers += 1
        elif team != qb_team and slot in receivers:
            opp_counts[team] += 1
        teams.append(team)

    bring_backs = 0
    if opp_counts:
        bring_backs = max(opp_counts.values())

    return f"QB+{same_team_receivers} with opp {bring_backs}"


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="NFL lineup diagnostics (classic + showdown)")
    ap.add_argument('--pos', action='append', choices=['QB','RB','WR','TE','DST','CPT','FLEX'],
                    help='Filter exposure table to these positions (repeatable).')
    ap.add_argument('--top', type=int, default=0, help='Show only top-N rows of exposure table.')
    ap.add_argument('--no-dupes', action='store_true', help='Skip duplicate trains summary.')
    ap.add_argument('--no-stacks', action='store_true', help='Skip stack summaries.')
    ap.add_argument('--include-rb-in-stacks', action='store_true',
                    help='For classic stacks, count RB as a pass-catcher toward the QB stack.')
    return ap


def main() -> None:
    args = build_arg_parser().parse_args()

    # 1) connect
    engine = init_db("nfl")
    session = get_session(engine)

    # 2) pull all lineups
    rows: List[Tuple[int, object]] = session.query(Lineup.id, Lineup.player_ids_list).all()
    total_lineups = len(rows)
    if total_lineups == 0:
        print("No lineups found.")
        return

    # 3) build lineup list with sorted-unique keys, and collect all player ids
    lineup_keys: List[Tuple[int, Tuple[int, ...]]] = []  # (lineup_id, sorted_tuple)
    all_ids: set[int] = set()
    for lid, plist in rows:
        ids = sorted(set(_coerce_player_ids(plist)))
        lineup_keys.append((lid, tuple(ids)))
        all_ids.update(ids)

    # 4) fetch player objects for all IDs we saw
    players: List[NFLPlayer] = (
        session.query(NFLPlayer)
               .filter(NFLPlayer.id.in_(list(all_ids)))
               .all()
    )
    id_to_player: Dict[int, NFLPlayer] = {p.id: p for p in players}

    # 5) Exposure counts
    counts = Counter()
    for _, ids in lineup_keys:
        for pid in ids:
            counts[pid] += 1

    # 6) Prepare exposure rows
    exposure_rows: List[Tuple[int,str,str,int,float]] = []
    for pid, c in counts.items():
        p = id_to_player.get(pid)
        name = getattr(p, 'name', f"Unknown ({pid})") if p else f"Unknown ({pid})"
        slot_id = getattr(p, 'roster_slot_id', None) if p else None
        slot_label = _slot_label(slot_id)
        exposure = c / total_lineups
        exposure_rows.append((pid, name, slot_label, c, exposure))

    # Optional position filter on exposure table
    if args.pos:
        pos_set = set(args.pos)
        exposure_rows = [r for r in exposure_rows if r[2] in pos_set]

    exposure_rows.sort(key=lambda r: (r[4], r[3], -r[0]), reverse=True)

    # 7) Print exposure table
    print(f"Total lineups: {total_lineups}")
    print(f"{'Player ID':>9}  {'Name':<32} {'Slot':<6} {'#Lineups':>8} {'Exposure':>9}")
    print("-" * 72)
    to_show = exposure_rows[:args.top] if args.top and args.top > 0 else exposure_rows
    for pid, name, slot_label, count, exp in to_show:
        print(f"{pid:>9}  {name:<32.32} {slot_label:<6} {count:>8} {_format_pct(exp)}")

    # 8) Duplicate trains summary
    if not args.no_dupes:
        unique, top_trains = _summarize_dupes(lineup_keys)
        print("\nDuplicates (train summary):")
        print(f"  Unique lineups: {unique} of {total_lineups}")
        if top_trains:
            print("  Top train sizes (size -> count of trains):")
            for size, how_many in top_trains:
                print(f"    {size:>2} -> {how_many}")
        else:
            print("  (No duplicate trains detected)")

    # 9) Stack summaries
    if not args.no_stacks:
        sd_counter = Counter()
        cl_counter = Counter()
        sd_unknown = 0
        cl_unknown = 0

        for _, ids in lineup_keys:
            objs = [id_to_player.get(pid) for pid in ids if id_to_player.get(pid) is not None]
            slot_ids = [getattr(p, 'roster_slot_id', None) for p in objs]
            is_showdown = _detect_showdown_lineup(slot_ids)

            if is_showdown:
                # showdown: stack type is team split + CPT majority/minority
                cpt_team = None
                team_list: List[Optional[str]] = []
                for p in objs:
                    team_list.append(_get_team_from_player(p))
                    if getattr(p, 'roster_slot_id', None) == CPT:
                        cpt_team = _get_team_from_player(p)
                label = _showdown_stack_type(team_list, cpt_team)
                if label:
                    sd_counter[label] += 1
                else:
                    sd_unknown += 1
            else:
                # classic: QB stack type
                label = _classic_qb_stack_type(objs, include_rb=args.include_rb_in_stacks)
                if label:
                    cl_counter[label] += 1
                else:
                    cl_unknown += 1

        # Print showdown stacks
        if sd_counter or sd_unknown:
            print("\nShowdown stack types:")
            total_sd = sum(sd_counter.values())
            if total_sd:
                for label, cnt in sd_counter.most_common():
                    pct = cnt / total_lineups
                    print(f"  {label:<22} {cnt:>6} {_format_pct(pct)}")
            if sd_unknown:
                print(f"  Unknown/undetermined: {sd_unknown}")

        # Print classic stacks
        if cl_counter or cl_unknown:
            print("\nClassic QB stack types:")
            total_cl = sum(cl_counter.values())
            if total_cl:
                for label, cnt in cl_counter.most_common():
                    pct = cnt / total_lineups
                    print(f"  {label:<22} {cnt:>6} {_format_pct(pct)}")
            if cl_unknown:
                print(f"  Unknown/undetermined: {cl_unknown}")


if __name__ == "__main__":
    main()
