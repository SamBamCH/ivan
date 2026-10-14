"""The entries-file loop: every contest in the DK export is simulated and every entry row filled."""
from datetime import datetime
from pathlib import Path

import pandas as pd

from ivan.sports.mma.choose import choose_entries, export_entries, format_contests
from ivan.sports.mma.entries import BASE_COLS, contests_in, load_entries, parse_money
from ivan.sports.mma.lineup_generation import generate_candidates
from ivan.sports.mma.models import ContestPayout, HistoricalContest, LineupSimulationResult
from ivan.sports.mma.payouts import parse_payout_csv
from ivan.sports.mma.simulate_contest import resolve_contest, simulate_contest

DATA = Path(__file__).parent / "data"
ENTRIES = DATA / "entries.csv"


def test_load_entries_stops_at_the_player_pool():
    df = load_entries(ENTRIES)
    assert len(df) == 6
    assert list(df.columns[:4]) == BASE_COLS
    assert df["Contest ID"].tolist() == [900000001] * 4 + [900000002] * 2
    assert df["fee"].tolist() == [0.5] * 4 + [3.0] * 2
    assert parse_money("$1,000.00") == 1000.0 and parse_money("") == 0.0
    cs = contests_in(df)
    assert [(c["contest_id"], c["entries"], c["fee"]) for c in cs] == [(900000001, 4, 0.5), (900000002, 2, 3.0)]
    assert cs[0]["name"] == "UFC $20K mini-MAX [150 Max Entry]"


def _store_contest(session, cid: int, name: str, fee: float, total: int, payouts: Path):
    session.add(HistoricalContest(contest_id=cid, draft_group_id=1, name=name, date=datetime.now(), entry_fee=fee,
                                  contest_type="Classic", total_entries=total, multi_entry=150))
    for _, r in parse_payout_csv(payouts).iterrows():
        session.add(ContestPayout(contest_id=cid, min_position=int(r["Start Rank"]), max_position=int(r["End Rank"]),
                                  payout_amount=float(r["Payout"])))
    session.commit()


def test_entries_loop_fills_every_row(session, cfg, ingest, tmp_path, monkeypatch):
    monkeypatch.setenv("IVAN_SEED", "3")
    ingest(session, cfg)
    generate_candidates(session, cfg, n_candidates=40, n_jobs=1)
    entries = load_entries(ENTRIES)
    for c in contests_in(entries):
        _store_contest(session, c["contest_id"], c["name"], c["fee"], 47500, DATA / "payouts.csv")
        contest = resolve_contest(session, cfg, contest_id=c["contest_id"], contest_name=c["name"])
        assert contest["entry_fee"] == c["fee"] and contest["name"] == c["name"]
        simulate_contest(session, cfg, contest, field_size=200, n_sims=40)
    names = {r[0] for r in session.query(LineupSimulationResult.contest_name).distinct()}
    assert names == {c["name"] for c in contests_in(entries)}

    chosen = choose_entries(session, entries, cfg)
    up = chosen["upload"]
    assert list(up.columns) == BASE_COLS + ["F"] * 6
    assert len(up) == 6 and up["Entry ID"].tolist() == entries["Entry ID"].tolist()
    assert up["Contest ID"].tolist() == entries["Contest ID"].tolist()
    assert up.iloc[:, 4:].map(lambda s: "(" in s and s.endswith(")")).all().all()
    # six distinct fighters per row, no two rows of a contest identical
    rows = [tuple(sorted(r)) for r in up.iloc[:, 4:].itertuples(index=False)]
    assert all(len(set(r)) == 6 for r in rows) and len(set(rows[:4])) == 4
    assert [c["chosen"] for c in chosen["contests"]] == [4, 2] and not any(c["skipped"] for c in chosen["contests"])
    assert len(chosen["picks"]) == 6 and len(chosen["detail"]) == 6
    assert "Lineup ID" in chosen["detail"].columns and "Cash %" in chosen["detail"].columns
    assert "mini-MAX" in format_contests(chosen["contests"])

    paths = export_entries(session, chosen, out_dir=tmp_path, stamp="t")
    back = pd.read_csv(paths["upload"], dtype=str)
    assert list(back.columns) == BASE_COLS + ["F", "F.1", "F.2", "F.3", "F.4", "F.5"] and len(back) == 6
    assert (tmp_path / "chosen_entries.t.csv").exists() and paths["lineups"].exists() and paths["exposures"].exists()


def test_unsimulated_contest_is_skipped_not_fatal(session, cfg, ingest):
    ingest(session, cfg)
    entries = load_entries(ENTRIES)
    chosen = choose_entries(session, entries, cfg)
    assert chosen["upload"].empty and chosen["picks"] == []
    assert [c["skipped"] for c in chosen["contests"]] == ["not simulated", "not simulated"]
