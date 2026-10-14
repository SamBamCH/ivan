"""Placement, settlement and ledger logic of the tax backtest on a synthetic contest."""
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from ivan.sports.nfl import backtest_tax as bt
from ivan.sports.nfl.models import (ContestEntryResult, ContestFieldScores, ContestPayout, HistoricalContest, NFLBase,
                                    NFLDSTGame, NFLStat, ShowdownLineup, ShowdownPlayer)
from ivan.sports.nfl.placement import payout_vector, place, places_paid, prize_pool, remove_our_copies
from ivan.sports.nfl.tax import TaxConfig

TIERS = [{"min": 1, "max": 1, "amount": 100.0}, {"min": 2, "max": 3, "amount": 20.0}, {"min": 4, "max": 5, "amount": 5.0}]


def test_payout_vector_and_pool():
    pay = payout_vector(TIERS, 7)
    assert pay.tolist() == [100, 20, 20, 5, 5, 0, 0]
    assert prize_pool(TIERS) == 150
    assert places_paid(TIERS) == 5
    assert payout_vector(TIERS, 2).tolist() == [100, 20]      # table longer than the field


def test_place_ranks_and_tie_pooling():
    pay = payout_vector(TIERS, 10)
    field = [90, 80, 80, 50, 10]
    ranks, pays = place([95, 80, 20], field, pay)
    assert ranks.tolist() == [1, 3, 7]                          # 80 ties with two field 80s at ranks 3-5; 20 sits behind 50
    assert pays.tolist() == [100.0, (20 + 5 + 5) / 3, 0.0]


def test_place_with_pay_shorter_than_field():
    pay = payout_vector(TIERS, 3)
    ranks, pays = place([1.0], [50, 40, 30, 20], pay)
    assert ranks.tolist() == [5] and pays.tolist() == [0.0]


def test_remove_our_copies():
    field, removed = remove_our_copies([90, 80, 80, 10], [80, 80, 80, 7])
    assert removed == 2 and sorted(field.tolist()) == [10, 90]


def test_classify_format_and_kind():
    assert bt.classify_format("NFL $50K Double Up") == "cash"
    assert bt.classify_format("NFL Showdown $100K mini-MAX [150 Entry Max] (ATL @ NO)") == "gpp"
    assert bt.classify_format("NFL $10 Thing", 0.45) == "cash"
    assert bt.contest_kind("NFL Showdown $5K Dime Package", None) == "showdown"
    assert bt.contest_kind("NFL $750K Play-Action", "Classic") == "classic"


def test_parse_money():
    assert bt.parse_money("$0.50") == 0.5 and bt.parse_money("$1,250.00") == 1250 and bt.parse_money("") == 0
    assert bt.parse_money("($3.00)") == -3.0


# --------------------------------------------------------------------------- an end-to-end synthetic contest
LOCK = datetime(2026, 10, 6, 0, 15)            # UTC; a Monday-night showdown


@pytest.fixture
def session(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 't.sqlite'}", future=True)
    NFLBase.metadata.create_all(engine)
    s = sessionmaker(bind=engine)()
    s.add(HistoricalContest(contest_id=1, draft_group_id=9, name="NFL Showdown $1K Test [20 Entry Max] (A @ B)",
                            date=LOCK, entry_fee=1.0, contest_type="Showdown Captain Mode", total_entries=6, multi_entry=20))
    for t in TIERS:
        s.add(ContestPayout(contest_id=1, min_position=t["min"], max_position=t["max"], payout_amount=t["amount"]))
    # schedule week containing the lock, and the box scores
    from sqlalchemy import text
    s.execute(text("CREATE TABLE IF NOT EXISTS nfl_schedule (season INT, week INT, gameday TEXT)"))
    s.execute(text("INSERT INTO nfl_schedule VALUES (2026, 5, '2026-10-01'), (2026, 5, '2026-10-05')"))
    s.add(NFLStat(player_id="p1", player_display_name="qb one", position="QB", season=2026, week=5, team="A",
                  passing_yards=300, passing_tds=2))                        # 12 + 8 + 3 bonus = 23
    s.add(NFLStat(player_id="p2", player_display_name="wr two", position="WR", season=2026, week=5, team="B",
                  receptions=5, receiving_yards=50))                        # 10
    s.add(NFLDSTGame(season=2026, week=5, team="A", dst_points=4.0))
    s.commit()
    return s


def write_run(tmp_path, stamp, rows, simroi=None):
    p = tmp_path / f"chosen_entries.lineups.{stamp}.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    if simroi is not None:
        np.savez(tmp_path / f"chosen_entries.simroi.{stamp}.npz", **simroi)
    return p


def lineup_rows(entry_id, lineup_id, cpt, flex, fee="$1", cash=0.3, roi=0.1):
    base = dict(entry_id=entry_id, contest_name="NFL Showdown $1K Test [20 Entry Max] (A @ B)", contest_id=1,
                entry_fee=fee, format="showdown", lineup_id=lineup_id, cash_pct=cash, avg_roi=roi)
    rows = [{**base, "slot_no": 0, "slot": "CPT", "player_id": 10, "name": cpt[0], "team": cpt[1], "position": cpt[2], "salary": 10000}]
    for k, (n, tm, pos) in enumerate(flex, 1):
        rows.append({**base, "slot_no": k, "slot": "FLEX", "player_id": 10 + k, "name": n, "team": tm, "position": pos, "salary": 5000})
    return rows


def test_prelock_run_selection(tmp_path, session):
    before = write_run(tmp_path, "20261005-180000", lineup_rows(1, 1, ("qb one", "A", "QB"), [("wr two", "B", "WR")]))
    later_before = write_run(tmp_path, "20261005-190000", lineup_rows(1, 2, ("qb one", "A", "QB"), [("wr two", "B", "WR")]))
    after = write_run(tmp_path, "20261005-200000", lineup_rows(1, 3, ("qb one", "A", "QB"), [("wr two", "B", "WR")]))
    runs = bt.find_choose_runs([tmp_path])
    assert [r.stamp for r in runs] == ["20261005-180000", "20261005-190000", "20261005-200000"]
    info = bt.contest_info(session, [1])
    picked = bt.select_prelock_runs(runs, info)
    # lock is 00:15 UTC = 19:15 America/Chicago on the 5th; 19:00 local is before, 20:00 is after
    assert picked[1].stamp == "20261005-190000"


def test_reconstructed_settlement_from_field_scores(tmp_path, session):
    # our lineup: CPT qb one (23 * 1.5 = 34.5) + wr two (10) + DST A (4) = 48.5
    rows = lineup_rows(1, 1, ("qb one", "A", "QB"), [("wr two", "B", "WR"), ("a", "A", "DST")])
    # field (we are in it): 60, 48.5 (us), 48.5, 30, 20, 10  -> after removing our copy we tie at rank 2 with one other
    session.add(ContestFieldScores(contest_id=1, n_entries=6, scores=[60, 48.5, 48.5, 30, 20, 10]))
    session.commit()
    write_run(tmp_path, "20261005-180000", rows)
    entries, cov = bt.assemble(session, bt.find_choose_runs([tmp_path]))
    assert len(entries) == 1
    e = entries[0]
    assert e.source == "reconstructed" and e.points == pytest.approx(48.5)
    assert e.rank == 2 and e.payout == pytest.approx(20.0) and e.field_size == 6
    assert e.winnings == pytest.approx(19.0) and e.losses == 0
    assert cov.contests_settled == 1


def test_points_only_without_field(tmp_path, session):
    write_run(tmp_path, "20261005-180000", lineup_rows(1, 1, ("qb one", "A", "QB"), [("wr two", "B", "WR")]))
    entries, cov = bt.assemble(session, bt.find_choose_runs([tmp_path]))
    assert entries[0].source == "points_only" and entries[0].payout is None and entries[0].points == pytest.approx(44.5)
    assert cov.unsettled_contests and cov.contests_settled == 0


def test_no_actual_when_player_missing(tmp_path, session):
    # team Z has no box score at all -> unknown; a missing player on a team that played scored 0
    write_run(tmp_path, "20261005-180000", lineup_rows(1, 1, ("nobody", "Z", "QB"), [("wr two", "B", "WR")]))
    entries, _ = bt.assemble(session, bt.find_choose_runs([tmp_path]))
    assert entries[0].source == "no_actual" and "nobody" in entries[0].note
    write_run(tmp_path, "20261005-180100", lineup_rows(1, 1, ("qb one", "A", "QB"), [("inactive guy", "B", "WR")]))
    entries, _ = bt.assemble(session, bt.find_choose_runs([tmp_path]))
    assert entries[0].source == "points_only" and entries[0].points == pytest.approx(34.5) and "inactive guy" in entries[0].note


def test_name_suffix_fallback():
    from ivan.sports.nfl import actuals
    table = {("brian robinson", "WAS"): 12.5, ("DST", "WAS"): 7.0}
    assert actuals.lookup_points(table, "Brian Robinson Jr.", "WAS", "RB") == 12.5
    assert actuals.lookup_points(table, "Commanders", "WAS", "DST") == 7.0
    assert actuals.lookup_points(table, "Someone Else", "WAS", "RB") is None
    assert actuals.team_played(table, "WAS") and not actuals.team_played(table, "KC")


def test_field_from_showdown_lineups(session):
    session.add_all([ShowdownPlayer(contest_id=1, player_id=1, roster_slot_id=511, salary=1, position="QB", team="A", ownership_actual=1, fpts_actual=20.0),
                     ShowdownPlayer(contest_id=1, player_id=2, roster_slot_id=512, salary=1, position="WR", team="A", ownership_actual=1, fpts_actual=10.0),
                     ShowdownLineup(contest_id=1, cpt_player_id=1, flex_player_ids=[2], salary_remaining=0, dupe_count=3)])
    session.commit()
    f = bt.field_scores(session, 1)
    assert f.tolist() == [40.0, 40.0, 40.0]


def test_scraped_entry_and_dk_history_take_precedence(tmp_path, session):
    rows = lineup_rows(1, 1, ("qb one", "A", "QB"), [("wr two", "B", "WR")]) + \
        lineup_rows(2, 2, ("qb one", "A", "QB"), [("wr two", "B", "WR")])
    write_run(tmp_path, "20261005-180000", rows)
    session.add(ContestEntryResult(entry_id=2, contest_id=1, username="me", rank=1, points=44.5, payout=100.0))
    session.commit()
    hist = pd.DataFrame({"contest_id": [1, 7], "entry_id": [1, 99], "contest_name": ["t", "other contest"],
                         "game_type": ["Showdown Captain Mode", "Classic"], "sport": ["NFL", "NFL"],
                         "date": [pd.Timestamp("2026-10-05 19:15"), pd.Timestamp("2026-09-27 12:00")],
                         "place": [3, 500], "points": [44.5, 120.0], "entry_fee": [1.0, 3.0], "winnings": [20.0, 0.0],
                         "contest_entries": [6, 1000], "places_paid": [5, 200]})
    entries, cov = bt.assemble(session, bt.find_choose_runs([tmp_path]), dk_history=hist)
    by = {e.entry_id: e for e in entries}
    assert by[1].source == "dk_history" and by[1].payout == 20.0 and by[1].played
    assert by[2].source == "scraped_entry" and by[2].payout == 100.0
    assert by[99].source == "dk_history" and by[99].losses == 3.0 and cov.history_only_entries == 1
    assert by[99].lock is not None and by[99].tax_year == 2026
    played = [e for e in bt.assemble(session, bt.find_choose_runs([tmp_path]), dk_history=hist, played_only=True)[0]]
    assert {e.entry_id for e in played} == {1, 2, 99}


def test_load_dk_history_tolerant_columns(tmp_path):
    p = tmp_path / "hist.csv"
    p.write_text("Sport,Game_Type,Entry_Key,Entry,Contest_Key,Contest_Date_EST,Place,Points,Winnings_Non_Ticket,Winnings_Ticket,Contest_Entries,Entry_Fee,Prize_Pool,Places_Paid\n"
                 "NFL,Classic,5001,NFL $100K Thing,9001,2026-09-27 13:00:00,12,150.2,$25.00,$0.00,1000,$3.00,\"$100,000.00\",200\n"
                 "NBA,Classic,5002,NBA thing,9002,2026-10-20 19:00:00,1,300,$1.00,$0.00,10,$1.00,$10.00,3\n")
    df = bt.load_dk_history(p)
    assert len(df) == 1 and df.iloc[0].entry_id == 5001 and df.iloc[0].winnings == 25.0 and df.iloc[0].entry_fee == 3.0


def mk(entry_id, fee, payout, lock, cid=1, sim_roi=None, fmt="gpp"):
    return bt.EntryResult(contest_id=cid, contest_name=f"c{cid}", entry_id=entry_id, entry_fee=fee, lock=lock,
                          contest_type="showdown", format_class=fmt, payout=payout, source="dk_history", sim_roi=sim_roi)


def test_ledger_scenarios_formats_and_sweep():
    cfg = TaxConfig(marginal_rate=0.22, itemize="always")
    d0 = datetime(2026, 9, 14, 0, 15)
    entries = [mk(1, 10, 0, d0), mk(2, 10, 50, d0),                           # slate 1: W 40, L 10
               mk(3, 10, 0, d0 + timedelta(days=7)), mk(4, 10, 0, d0 + timedelta(days=7), fmt="cash"),   # slate 2: L 20
               mk(5, 10, None, d0 + timedelta(days=14))]                     # unsettled, ignored
    led = bt.ledger(entries, cfg, start_bankroll=5.0, scenarios=[cfg.with_(loss_cap_pct=1.0, label="repeal")])
    assert len(led) == 2
    last = led.iloc[-1]
    assert last.W == 40 and last.L == 30 and last.P == 10
    assert last.deductible == pytest.approx(27) and last.taxable_gambling == pytest.approx(13)
    assert last.est_tax == pytest.approx(0.22 * 13) and last["tax[repeal]"] == pytest.approx(0.22 * 10)
    assert last.bankroll == 15 and not last.tax_exceeds_bankroll
    sc = bt.scenario_table(entries, cfg)
    assert set(sc.scenario) >= {"as configured", "net DFS (1099 net)"}
    fm = bt.format_table(entries, cfg)
    assert set(fm.format_class) == {"cash", "gpp"}
    cash = fm[fm.format_class == "cash"].iloc[0]
    assert cash.L == 10 and cash.P == -10
    # the cash entry only adds losses: without it the deductible drops to 18 and taxable rises to 22, so it *saved* tax
    assert cash.tax_added == pytest.approx(0.22 * (13 - 22))
    sw = bt.volume_sweep(entries, cfg, 5.0, multipliers=(1, 2), n_boot=200, seed=1)
    assert list(sw.multiplier) == [1, 2] and sw.iloc[1].slates == 4
    assert sw.fees.iloc[1] == pytest.approx(2 * sw.fees.iloc[0], rel=0.2)


def test_sim_sweep_uses_traces():
    cfg = TaxConfig(marginal_rate=0.22, itemize="always")
    d0 = datetime(2026, 9, 14, 0, 15)
    roi = np.array([-1.0, -1.0, 9.0, -1.0], dtype=np.float32)      # cashes 1 in 4 for 10x
    entries = [mk(1, 10, 0, d0, sim_roi=roi), mk(2, 10, 0, d0 + timedelta(days=7), cid=2, sim_roi=roi)]
    sw = bt.volume_sweep(entries, cfg, 0.0, multipliers=(1,), n_boot=2000, seed=2, basis="sim")
    assert sw.iloc[0].basis == "sim"
    assert sw.iloc[0].pre_tax_profit == pytest.approx(2 * (0.25 * 90 - 0.75 * 10), abs=4)
    sva = bt.sim_vs_actual(entries)
    assert sva.sim_expected_profit.sum() == pytest.approx(2 * (0.25 * 100 - 10))
