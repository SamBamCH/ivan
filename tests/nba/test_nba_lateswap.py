"""Late swap (ivan.sports.nba.post_lock.lateswap): lock logic, candidates with locked slots, payout vs the field, slotting."""
from datetime import datetime, timedelta

import numpy as np

from ivan.sports.nba.post_lock import lateswap as LS
from ivan.sports.nba.post_lock.entry_parsing import OriginalEntry
from ivan.sports.nba.pre_lock.lineup_generation import SLOT_NAMES, FieldPlayer

NOW = datetime(2026, 10, 22, 19, 0)


def _fp(i, pos, tip_h, team="AAA", mean=30.0, salary=6000):
    return FieldPlayer(id=i, name=f"p {i}", salary=salary, team=team, opponent="BBB", position=(pos,), fpts_mean=mean, game_id="AAA-BBB",
                       gametime=NOW + timedelta(hours=tip_h))


def test_lock_and_resolve():
    assert LS.is_locked(_fp(1, "PG", -1), NOW) and not LS.is_locked(_fp(2, "PG", 1), NOW)
    names = {"luka doncic": 1, "jalen williams": 2, "jaylin williams": 3, "kenrich williams": 4}
    assert LS._resolve(names, "Luka Doncic") == 1 and LS._resolve(names, "Doncic") == 1
    assert LS._resolve(names, "Jal. Williams") == 2 and LS._resolve(names, "Williams") is None and LS._resolve(names, "") is None


def test_candidates_hold_locked_players_in_their_slots(monkeypatch):
    # a two-slot world: the solver is stubbed to echo the locked ids plus the best open players
    fps = {i: _fp(i, "PG" if i % 2 else "SG", -1 if i in (1, 2, 11, 12) else 1, mean=10.0 * i) for i in range(1, 13)}   # 11, 12: started, not ours
    calls = []

    def fake_solver(players, base_scores, locked_assignments=None, max_salary_per_team=None, **kw):
        calls.append(dict(locked_assignments))
        free = sorted((p for p in players if p.id not in locked_assignments), key=lambda p: -base_scores[p.id])[:8 - len(locked_assignments)]
        return list(locked_assignments) + [p.id for p in free]
    monkeypatch.setattr(LS, "generate_single_classic_lineup", fake_solver)
    e = OriginalEntry(entry_id="e1", contest_id=1, lineup_id=-1, slots={s: i for s, i in zip(SLOT_NAMES, range(1, 9))})
    worlds = np.random.default_rng(0).normal(50, 5, size=(40, 12)).astype(np.float32)
    cands = LS.candidates(e, fps, worlds, sorted(fps), NOW, n=3, rng=np.random.default_rng(1))
    assert cands[0] == list(range(1, 9)) and len(cands) <= 4
    assert e.locked_player_ids == {1, 2}
    assert all(c == {1: "PG", 2: "SG"} for c in calls)            # the two started games stay in DraftKings' slots
    for c in cands[1:]:
        assert {1, 2} <= set(c)                                   # our locked players kept
        assert not ({11, 12} & set(c))                            # started games we do not hold never enter


def test_slot_assignment_puts_late_games_in_flex():
    ps = [_fp(1, "PG", 3), _fp(2, "PG", 1), _fp(3, "C", 2)]
    out = LS._slot(ps, ["PG", "G", "UTIL"])
    assert [p.id for p in out] == [2, 1, 3]                       # the earliest PG takes PG, the later one G, the C goes to UTIL


def test_payout_rank_against_the_field():
    # 3 field entries, payouts 10 / 5 / 0; a candidate scoring between first and second ranks 2 in every world
    P = np.array([[10.0, 20.0, 30.0, 25.0]] * 4, dtype=np.float32)
    F = np.sort(P[:, [0, 1, 2]], axis=1)
    pay = np.zeros(5); pay[1], pay[2] = 10.0, 5.0
    s = P[:, 3]
    beaten = np.array([np.searchsorted(F[i], s[i], side="left") for i in range(4)])
    rank = 3 - beaten + 1
    assert (rank == 2).all() and pay[rank].mean() == 5.0


def test_entry_cells_after_lock():
    assert LS._pid("Keyonte George (44630968) (LOCKED)") == 44630968 and LS._is_locked_cell("Keyonte George (44630968) (LOCKED)")
    assert LS._pid("Ty Jerome (44630945)") == 44630945 and not LS._is_locked_cell("Ty Jerome (44630945)")
    assert LS._pid("Steven Adams (LOCKED)") is None and LS._cell_name("Steven Adams (LOCKED)") == "Steven Adams"


def test_exposure_before_and_after_the_swap():
    # two entries: the first swaps 3 -> 9, the second has a better candidate but is not improved (keeps its original)
    rep = {"contests": [{"entries": [
        dict(entry_id="1", original={"ids": [1, 2, 3]}, best={"ids": [1, 2, 9]}, improved=True, names={"1": "a", "2": "b", "3": "c", "9": "z"}),
        dict(entry_id="2", original={"ids": [1, 3, 4]}, best={"ids": [1, 3, 9]}, improved=False, names={"1": "a", "3": "c", "4": "d", "9": "z"}),
    ]}]}
    rows = {r["id"]: r for r in LS.exposure(rep)}
    assert rows[1] == dict(id=1, name="a", team="", before=2, after=2, delta=0, before_pct=100.0, after_pct=100.0)
    assert (rows[3]["before"], rows[3]["after"], rows[3]["delta"]) == (2, 1, -1)
    assert (rows[9]["before"], rows[9]["after"], rows[9]["delta"]) == (0, 1, 1)
    assert rows[4]["after"] == 1                                       # the unimproved entry keeps player 4
    assert [r["id"] for r in LS.exposure(rep)][0] == 1                 # sorted by entries after
    txt = LS.format_exposure(LS.exposure(rep), 2)
    assert "a" in txt.splitlines()[2] and "(100.0%)" in txt.splitlines()[2] and "-1" in txt and "+1" in txt
    assert len(LS.format_exposure(LS.exposure(rep), 2, changed_only=True).splitlines()) == 2 + 2 + 1   # head, rule, c and z (d stays at 1), footer
    assert LS.format_exposure([], 0) == "exposure: no entries"
