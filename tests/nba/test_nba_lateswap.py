"""Late swap (ivan.sports.nba.lateswap.swap): lock logic, candidates with locked slots, payout vs the field, slotting."""
from dataclasses import replace
from datetime import datetime, timedelta

import numpy as np

from ivan.sports.nba.lateswap import swap as LS
from ivan.sports.nba.lateswap.entry_parsing import OriginalEntry
from ivan.sports.nba.field.solver import SLOT_NAMES, FieldPlayer

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


# ----------------------------------------------------------------------------- the fetch apart from the swap (2026-10-28)
def test_export_is_current_until_the_next_tip():
    from datetime import datetime
    tips = [datetime(2026, 10, 27, 19, 0), datetime(2026, 10, 27, 20, 0), datetime(2026, 10, 27, 22, 0)]
    assert LS.export_is_current("20261027-190100", tips, datetime(2026, 10, 27, 19, 30))        # taken after the first tip, no tip since
    assert LS.export_is_current("20261027-190100", tips, datetime(2026, 10, 27, 19, 59))
    assert not LS.export_is_current("20261027-190100", tips, datetime(2026, 10, 27, 20, 1))     # the 20:00 game tipped since
    assert LS.export_is_current("20261027-200030", tips, datetime(2026, 10, 27, 21, 0))
    assert LS.export_is_current("20261027-120000", tips, datetime(2026, 10, 27, 18, 0))         # before the first tip any export is current
    assert not LS.export_is_current("garbage", tips, datetime(2026, 10, 27, 18, 0))
    assert LS.stamp_time("20261027-190100") == datetime(2026, 10, 27, 19, 1)


def test_field_export_auto_keeps_a_current_export_and_downloads_otherwise(monkeypatch):
    from datetime import datetime
    import pandas as pd
    stored = dict(df=pd.DataFrame({"a": [1]}), stamp="20261027-190100", origin="stored")
    downloads = []
    monkeypatch.setattr(LS, "stored_export", lambda cid: stored)
    monkeypatch.setattr(LS, "download_export", lambda http, cid, log=print: downloads.append(cid) or dict(df=pd.DataFrame({"a": [2]}), stamp="20261027-200100", origin="downloaded"))
    tips = [datetime(2026, 10, 27, 19, 0), datetime(2026, 10, 27, 20, 0)]
    got = LS.field_export(None, 1, source="auto", http_sess=object(), log=lambda m: None, now=datetime(2026, 10, 27, 19, 30), tips=tips)
    assert got["origin"] == "stored" and not downloads
    got = LS.field_export(None, 1, source="auto", http_sess=object(), log=lambda m: None, now=datetime(2026, 10, 27, 20, 30), tips=tips)
    assert got["origin"] == "downloaded" and downloads == [1]
    got = LS.field_export(None, 1, source="raw", http_sess=object(), log=lambda m: None, now=datetime(2026, 10, 27, 20, 30), tips=tips)
    assert got["origin"] == "stored" and downloads == [1]
    got = LS.field_export(None, 1, source="draftkings", http_sess=object(), log=lambda m: None, now=datetime(2026, 10, 27, 19, 30), tips=tips)
    assert got["origin"] == "downloaded" and downloads == [1, 1]
    got = LS.field_export(None, 1, source="auto", http_sess=None, log=lambda m: None, now=datetime(2026, 10, 27, 20, 30), tips=tips)
    assert got["origin"] == "stored"                                                           # no way to download: the stale one, with a warning


def test_lazy_http_opens_once_and_swallows_a_missing_login():
    calls = []
    def factory():
        calls.append(1); raise RuntimeError("DraftKings login needed")
    msgs = []
    h = LS.LazyHttp(factory, log=msgs.append)
    assert h.session() is None and h.session() is None and len(calls) == 1 and "login" in msgs[0]
    ok = LS.LazyHttp(lambda: "sess")
    assert ok.session() == "sess" and LS._http(ok) == "sess" and LS._http("plain") == "plain"


def test_sample_shared_takes_the_contest_size():
    field = [[i] * 8 for i in range(100)]
    shared = dict(field=field)
    s = LS.sample_shared(shared, 30, seed=1)
    assert len(s) == 30 and len({tuple(x) for x in s}) == 30 and all(x in field for x in s)
    assert LS.sample_shared(shared, 500, seed=1) == field
    assert LS.sample_shared(shared, 30, seed=1) == s                                           # seeded: the same sample for the same contest


# ----------------------------------------------------------------------------- unique candidates, unique entries (2026-11-12)
def _slate(open_extra=0, open_ids=()):
    """Eight players (ids 1-8, one per slot; started but for `open_ids`) plus open ones: a C (9) and `open_extra` more."""
    pos = ["PG", "SG", "SF", "PF", "C", "PG", "SF", "C"]
    fps = {i + 1: _fp(i + 1, p, 1 if i + 1 in open_ids else -1, team="AAA" if i % 2 else "BBB", salary=5000) for i, p in enumerate(pos)}
    for j in range(1 + open_extra):
        fps[9 + j] = replace(_fp(9 + j, "C", 1, team="CCC", salary=5000), game_id="CCC-DDD")     # the solver wants two games
        fps[1] = replace(fps[1], game_id="CCC-DDD")
    return fps


def test_the_solver_excludes_lineups():
    from ivan.sports.nba.field.solver import generate_single_classic_lineup
    fps = _slate(open_extra=1, open_ids=(5, 8))
    pool = list(fps.values()); sc = {p.id: 30.0 + p.id for p in pool}
    seen = []
    while True:
        ids = generate_single_classic_lineup(pool, base_scores=sc, max_salary_per_team=50000, exclude=seen)
        if not ids:
            break
        assert sorted(ids) not in seen
        seen.append(sorted(ids))
        assert len(seen) <= 6
    assert len(seen) == 6                                         # 2 of the 4 centres in C and UTIL: every pair once, then nothing


def test_candidates_are_distinct_and_stop_when_no_more_fit():
    import time
    # seven locked, the UTIL slot open with the original's player and two others: two more lineups exist, not ten
    fps = _slate(open_extra=1, open_ids=(8,))
    e = OriginalEntry(entry_id="e1", contest_id=1, lineup_id=-1, slots={s: i for s, i in zip(SLOT_NAMES, range(1, 9))})
    worlds = np.random.default_rng(0).normal(30, 5, size=(50, 10)).astype(np.float32)
    t0 = time.monotonic()
    cands = LS.candidates(e, fps, worlds, sorted(fps), NOW, n=10, rng=np.random.default_rng(1), seconds=30.0)
    assert time.monotonic() - t0 < 10.0                           # ended by the solver finding nothing, not by the limit
    assert sorted(map(LS.lineup_key, cands)) == [tuple(list(range(1, 8)) + [k]) for k in (8, 9, 10)]
    # asked for one, given one
    assert len(LS.candidates(e, fps, worlds, sorted(fps), NOW, n=1, rng=np.random.default_rng(1))) == 2


def test_candidates_give_up_after_the_time_limit(monkeypatch):
    import time
    fps = _slate(open_extra=1, open_ids=(8,))
    monkeypatch.setattr(LS, "generate_single_classic_lineup", lambda players, **kw: list(range(1, 8)) + [9])   # the same lineup every time
    e = OriginalEntry(entry_id="e1", contest_id=1, lineup_id=-1, slots={s: i for s, i in zip(SLOT_NAMES, range(1, 9))})
    worlds = np.zeros((5, 10), dtype=np.float32); said = []
    t0 = time.monotonic()
    cands = LS.candidates(e, fps, worlds, sorted(fps), NOW, n=10, rng=np.random.default_rng(1), seconds=0.2, log=said.append)
    assert 0.2 <= time.monotonic() - t0 < 5.0
    assert len(cands) == 2 and "1 of 10 candidates" in said[0]


def _row(ids, pay):
    return dict(ids=list(ids), exp_payout=float(pay))


def test_assign_unique_gives_every_entry_its_own_lineup():
    A, B, X, Y = [1, 2, 3], [1, 2, 4], [1, 2, 9], [1, 2, 8]
    # both entries' best is X: one takes it, the other its next best, whichever totals more
    chosen, stuck = LS.assign_unique({"a": [_row(A, 1.0), _row(X, 5.0), _row(Y, 4.0)], "b": [_row(B, 1.0), _row(X, 5.0), _row(Y, 2.0)]})
    assert not stuck and chosen["a"]["ids"] == Y and chosen["b"]["ids"] == X
    # a lineup is one whatever the slot order; an entry's best being another's original moves one of them
    chosen, stuck = LS.assign_unique({"a": [_row(A, 1.0), _row([4, 2, 1], 3.0)], "b": [_row(B, 3.0)]})
    assert not stuck and chosen["a"]["ids"] == A and chosen["b"]["ids"] == B
    # nothing gained: every entry keeps its original (ties go to the original)
    chosen, stuck = LS.assign_unique({"a": [_row(A, 0.0), _row(X, 0.0)], "b": [_row(B, 0.0), _row(Y, 0.0)]})
    assert chosen["a"]["ids"] == A and chosen["b"]["ids"] == B
    # originals that repeat: the one with somewhere to go moves, even at a loss
    chosen, stuck = LS.assign_unique({"a": [_row(A, 2.0)], "b": [_row(A, 2.0), _row(X, 1.0)]})
    assert not stuck and chosen["a"]["ids"] == A and chosen["b"]["ids"] == X
    # ... and with nowhere to go it is reported, the rest still unique
    said = []
    chosen, stuck = LS.assign_unique({"a": [_row(A, 2.0)], "b": [_row(A, 2.0)], "c": [_row(B, 1.0), _row(A, 9.0)]}, log=said.append)
    assert len(stuck) == 1 and stuck[0] in ("a", "b") and chosen["c"]["ids"] == B and "WARNING" in said[0]
    # 150 entries after the same lineups: 150 different lineups, the best ones taken
    rng = np.random.default_rng(3)
    shared = [_row([100 + k, 200 + k, 300 + k], p) for k, p in enumerate(rng.uniform(1, 9, size=160))]
    opts = {f"e{i}": [_row([i, 1000 + i, 2000 + i], 0.5)] + shared for i in range(150)}
    chosen, stuck = LS.assign_unique(opts)
    keys = [LS.lineup_key(r["ids"]) for r in chosen.values()]
    assert not stuck and len(chosen) == 150 and len(set(keys)) == 150
    assert sorted(r["exp_payout"] for r in chosen.values()) == sorted(r["exp_payout"] for r in shared)[10:]


def test_duplicate_entries_in_the_upload():
    import pandas as pd
    cells = ["a (11111)", "b (22222)", "c (33333)", "d (44444)", "e (55555)", "f (66666)", "g (77777)", "h (88888)"]
    rows = [["1", "c", "9", "$1"] + cells,
            ["2", "c", "9", "$1"] + cells[::-1][:7] + ["A (LOCKED)"],          # the same eight in other slots, one cell locked
            ["3", "c", "9", "$1"] + cells[:7] + ["z (99999)"],
            ["4", "d", "8", "$1"] + cells]                                    # another contest: not a duplicate
    df = pd.DataFrame(rows, columns=["Entry ID", "Contest Name", "Contest ID", "Entry Fee"] + list(SLOT_NAMES))
    assert LS.duplicate_entries(df) == {"9": ["2"]}
