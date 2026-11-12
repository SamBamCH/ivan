"""The NBA slate method: the sim's shared draw, the world-axis check, the choose run as one book (ivan.sports.nba.contest.choose + lineup_export)."""
from datetime import datetime

import numpy as np
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from ivan.sports.nba.contest import choose as CH
from ivan.sports.nba.contest import export as LE
from ivan.sports.nba.contest import simulate as SC
from ivan.sports.nba.models import ContestSimReport, HistoricalContest, Lineup, LineupSimulationResult, NBABase, NBAPlayer


@pytest.fixture
def session(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 't.sqlite'}", future=True)
    NBABase.metadata.create_all(engine)
    return sessionmaker(bind=engine)()


def seed_two_contests(session, worlds=60, draw="slate", seeds=(7, 7)):
    gt = datetime(2026, 11, 5, 19, 0)
    pos = ["PG", "SG", "SF", "PF", "C", "PG/SG", "SF/PF", "C", "PG", "SF"]
    for pid in range(1, 11):
        session.add(NBAPlayer(id=pid, name=f"player {pid}", team="BOS" if pid % 2 else "NYK", opponent="NYK" if pid % 2 else "BOS", position=pos[pid - 1],
                              roster_slot_id=1, gametime=gt, salary=4000 + 500 * pid, fpts_mean=20.0 + pid, fpts_std=8.0, minutes_proj=30.0))
    for cid, name, fee, n in ((100, "Double Up", 10.0, 1000), (200, "Millionaire", 20.0, 50000)):
        session.add(HistoricalContest(contest_id=cid, draft_group_id=1, name=name, date=gt, entry_fee=fee, contest_type="Classic", total_entries=n, multi_entry=20))
    rng = np.random.default_rng(0)
    combos = [[1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 3, 4, 5, 6, 7, 10], [9, 2, 3, 4, 5, 6, 7, 8], [1, 2, 10, 4, 5, 6, 7, 8], [9, 2, 3, 4, 8, 6, 7, 10], [1, 2, 3, 4, 5, 9, 7, 8]]
    for i, combo in enumerate(combos):
        session.add(Lineup(id=10 + i, player_ids_list=combo))
    session.flush()
    for i in range(len(combos)):
        hit = rng.random(worlds) < (0.5 if i < 3 else 0.05)                     # the first three cash often, the last three rarely
        for cid, name, fee, small, big in ((100, "Double Up", 10.0, 0.8, 0.8), (200, "Millionaire", 20.0, 0.1, 20.0)):
            roi = np.where(hit, small if i < 3 else big, -1.0)
            session.add(LineupSimulationResult(lineup_id=10 + i, contest_id=cid, contest_name=name, fpts=rng.normal(250, 20, worlds).tolist(),
                                               payout=(fee * (1 + roi)).tolist(), roi=roi.tolist(), rank=rng.integers(1, 1000, worlds).tolist()))
    for (cid, name), seed in zip(((100, "Double Up"), (200, "Millionaire")), seeds):
        session.add(ContestSimReport(contest_id=cid, contest_name=name, report={"seed": seed, "contest": {"id": cid, "name": name, "draw": draw}}))
    session.commit()


def test_shared_worlds_accepts_one_run_and_refuses_separate_draws(session):
    seed_two_contests(session)
    assert CH.shared_worlds(session, [100, 200]) == {"method": "montecarlo", "draw": "slate (seed 7)"}
    with pytest.raises(ValueError, match="no sim report"):
        CH.shared_worlds(session, [100, 999])
    rep = session.get(ContestSimReport, 200)
    rep.report = {**rep.report, "contest": {**rep.report["contest"], "draw": "contest"}}
    session.commit()
    with pytest.raises(ValueError, match="shared draw"):
        CH.shared_worlds(session, [100, 200])


def test_shared_worlds_refuses_two_seeds(session):
    seed_two_contests(session, seeds=(7, 8))
    with pytest.raises(ValueError, match="different draw seeds"):
        CH.shared_worlds(session, [100, 200])


def test_choose_slate_as_one_book_and_the_export(session, tmp_path, monkeypatch):
    seed_two_contests(session)
    with pytest.raises(ValueError, match="choose_slate"):
        CH.choose_lineup_ids(session, {"method": "slate"}, 2, 100)
    contests = [{"contest_id": 100, "contest_name": "Double Up", "entries": 2, "fee": 10.0},
                {"contest_id": 200, "contest_name": "Millionaire", "entries": 2, "fee": 20.0}]
    out = CH.choose_slate(session, contests, {"bankroll_dollars": 500.0})
    assert set(out["picks"]) == {100, 200} and len(out["picks"][100]) == 2 and len(out["picks"][200]) == 2
    assert set(out["picks"][100]) <= {10, 11, 12}                                  # the steady cashers go to the flat contest
    assert set(out["picks"][200]) & {13, 14, 15}                                   # the top-heavy one carries a tail lineup
    book = out["book"]
    assert book["bankroll_dollars"] == 500.0 and book["entries"] == 4 and book["fees"] == 60.0 and book["shared"]["draw"] == "slate (seed 7)"
    assert out["stats"][100]["book"]["method"] == "slate" and out["stats"][100]["viz"]["slate"] is book

    # the whole run: --method slate --bankroll-dollars is the one-call form; every file is written and the log carries the book
    f = tmp_path / "entries.csv"
    f.write_text("Entry ID,Contest Name,Contest ID,Entry Fee,PG,SG,SF,PF,C,G,F,UTIL,,Instructions\n"
                 "501,Double Up,100,$10,,,,,,,,,,1.\n502,Double Up,100,$10,,,,,,,,,,\n503,Millionaire,200,$20,,,,,,,,,,\n504,Millionaire,200,$20,,,,,,,,,,\n,,,,,,,,,,,,,\n")
    logs = []
    run, paths = LE.run_choose(session, f, tmp_path / "out" / "chosen_entries.csv",
                               LE.weights_resolver(flags={"method": "slate", "bankroll_dollars": 500.0}), log=logs.append, stamp="S")
    assert run.total == 4 and run.slate_book["entries"] == 4 and set(paths) == {"classic", "exposures", "lineups", "choose", "simroi"}
    assert any("slate book: 4 entries" in m for m in logs) and any("bankroll $500.00 (given)" in m for m in logs)
    assert run.slate_book["bankroll_from"] == {"bankroll": 500.0, "source": "given"}
    # nothing given: the slate method is the default and the bankroll is read from our results (ivan.core.bankroll)
    from ivan.core import bankroll as BK
    monkeypatch.setattr(BK, "current", lambda now=None: {"bankroll": 1234.5, "start": 1000.0, "since": None, "net": 234.5, "contests": 7,
                                                         "sports": {"nba": 234.5}, "errors": {}, "source": "results"})
    logs2 = []
    run2, _ = LE.run_choose(session, f, tmp_path / "out2" / "chosen_entries.csv", LE.weights_resolver(no_prompt=True), log=logs2.append, stamp="S")
    assert run2.total == 4 and run2.slate_book["bankroll_dollars"] == 1234.5 and run2.slate_book["bankroll_from"]["source"] == "results"
    assert any("bankroll $1,234.50 = $1,000.00 start + $234.50 from 7 scored contest(s) (NBA +$234.50)" in m for m in logs2)
    import json
    js = json.loads((tmp_path / "out" / "chosen_entries.choose.S.json").read_text())
    assert set(js) == {"100", "200"} and js["100"]["slate"]["entries"] == 4 and js["100"]["book"]["method"] == "slate"
    z = np.load(paths["simroi"])
    assert z["roi"].shape == (4, 60) and sorted(z["contest_id"].tolist()) == [100, 100, 200, 200]


def test_choose_slate_refuses_contests_simulated_apart(session):
    seed_two_contests(session, draw="contest")
    with pytest.raises(ValueError, match="shared draw"):
        CH.choose_slate(session, [{"contest_id": 100, "contest_name": "Double Up", "entries": 1, "fee": 10.0}], {})


def test_contest_knobs_take_slate_and_the_dollar_bankroll(tmp_path):
    p = tmp_path / "cw.json"
    cur = LE.set_contest_weights(123, {"method": "slate", "bankroll_dollars": "750"}, path=p)
    assert cur == {"method": "slate", "bankroll_dollars": 750.0}
    with pytest.raises(ValueError, match="method"):
        LE.set_contest_weights(123, {"method": "nope"}, path=p)


def test_player_samples_matrix_reads_the_shared_draw(monkeypatch):
    """With the run's draw handed in, no contest draws its own worlds: the matrix is the shared samples in the contest's column order."""
    raw = {1: np.array([1.0, 2.0, 3.0], dtype=np.float32), 2: np.array([10.0, 20.0, 30.0], dtype=np.float32)}
    monkeypatch.setattr(SC, "generate_player_samples", lambda *a, **k: pytest.fail("the contest drew its own worlds"))
    P, ns, got, missing = SC.player_samples_matrix(None, {2: 0, 1: 1, 3: 2}, 3, seed=None, raw=raw)
    assert ns == 3 and got is raw and missing == [3]
    assert P[:, 0].tolist() == [10.0, 20.0, 30.0] and P[:, 1].tolist() == [1.0, 2.0, 3.0] and (P[:, 2] == 0).all()


def test_slate_draw_is_one_call_to_the_sampler(monkeypatch):
    calls = []
    monkeypatch.setattr(SC, "generate_player_samples", lambda session, n_samples, randomness, seed: calls.append((n_samples, seed)) or {5: np.zeros(n_samples, dtype=np.float32)})
    monkeypatch.setattr(SC, "seed_for", lambda name, default=None: 42 if name == "sim/slate" else None)
    d = SC.slate_draw(None, 25, log=lambda m: None)
    assert calls == [(25, 42)] and d["n"] == 25 and d["seed"] == 42 and 5 in d["raw"]
