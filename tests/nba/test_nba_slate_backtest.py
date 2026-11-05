"""slate_backtest: the view a slate opens in, ties sharing payouts, a final export scored as DraftKings did, a mid-slate one not."""
from datetime import date, datetime

import numpy as np

from ivan.sports.nba import slate_backtest as SB


def test_mode_for():
    now = datetime(2026, 10, 29, 18, 30)
    assert SB.mode_for(date(2026, 10, 28), datetime(2026, 10, 28, 18, 0), now) == "backtest"
    assert SB.mode_for(date(2026, 10, 29), datetime(2026, 10, 29, 18, 0), now) == "lateswap"
    assert SB.mode_for(date(2026, 10, 29), datetime(2026, 10, 29, 19, 0), now) == "prelock"
    assert SB.mode_for(date(2026, 10, 30), datetime(2026, 10, 30, 19, 0), now) == "prelock"


def test_ties_share_the_places_they_cover():
    pay = np.array([10.0, 6.0, 2.0, 0.0])
    rank, won = SB.ranked_payouts(np.array([300.0, 280.0, 280.0, 250.0]), pay)
    assert list(rank) == [1, 2, 2, 4] and list(won) == [10.0, 4.0, 4.0, 0.0]


FINAL = ("Rank,EntryId,EntryName,TimeRemaining,Points,Lineup,,Player,Roster Position,%Drafted,FPTS\n"
         "1,11,them (1/20),0,352.5,PG Ann One SG Bob Two SF Cal Three PF Dan Four C Ed Five G Fay Six F Gus Seven UTIL Hal Eight,,Ann One,PG,61.2%,70.5\n"
         "2,12,us,0,340.0,PG Ann One SG Bob Two SF Cal Three PF Dan Four C Ed Five G Fay Six F Gus Seven UTIL Hal Eight,,Bob Two,SG,55.0%,30.2\n"
         "2,13,us,0,340.0,PG Ann One SG Bob Two SF Cal Three PF Dan Four C Ed Five G Fay Six F Gus Seven UTIL Ian Nine,,Ian Nine,UTIL,12.0%,28.0\n"
         "4,14,them,0,301.0,PG Ian Nine SG Bob Two SF Gus Seven PF Dan Four C Hal Eight G Ann One F Cal Three UTIL Jo Ten,,,,,\n")
MID = FINAL.replace("UTIL Hal Eight,,Ann One", "UTIL LOCKED,,Ann One").replace("UTIL Hal Eight,,Bob Two", "UTIL LOCKED,,Bob Two").replace("UTIL Ian Nine,,Ian Nine", "UTIL LOCKED,,Ian Nine")
CONTEST = dict(contest_id=9, name="Big", entry_fee=5.0, total_entries=4)


def test_a_final_export_is_scored_as_draftkings_scored_it():
    from ivan.sports.nba.standings import parse_export
    ent, _ = parse_export(FINAL)
    r = SB.score_contest(CONTEST, {"12", "13"}, ent, np.array([10.0, 6.0, 2.0, 0.0]), datetime(2026, 11, 5, 10, 31))
    assert r["final"] and r["scored"].startswith("DraftKings' final export 11-05 10:31")
    assert r["found"] == 2 and r["fees"] == 10.0 and r["won"] == 8.0 and r["net"] == -2.0 and r["cashed"] == 2     # tied 2nd: (6 + 2) / 2 each
    assert [e["rank"] for e in r["entries"]] == [2, 2] and [e["points"] for e in r["entries"]] == [340.0, 340.0]
    assert r["entries"][0]["copies"] == 2 and r["unique"] == 3 and r["max_copies"] == 2 and r["winner"] == 352.5 and r["min_cash"] == 340.0
    assert r["best_rank"] == 2 and r["roi"] == -0.2


def test_a_mid_slate_export_is_not_scored():
    from ivan.sports.nba.standings import parse_export
    ent, _ = parse_export(MID)
    r = SB.score_contest(CONTEST, {"12", "13"}, ent, np.array([10.0, 6.0, 2.0, 0.0]), datetime(2026, 11, 4, 18, 6))
    assert not r["final"] and r["entries"] == [] and r["won"] is None and r["net"] is None and r["roi"] is None
    assert r["fees"] == 10.0 and r["field"] == 4 and r["locked"] == 3 and "mid-slate export 11-04 18:06" in r["scored"]


def test_summary_counts_only_final_contests():
    a = dict(final=True, name="A", contest_id=1, fees=10.0, won=8.0, found=2, cashed=2, ours=2, scored="final", claim=dict(roi=1.0))
    b = dict(final=False, name="B", contest_id=2, fees=3.0, won=None, found=0, cashed=0, ours=3, scored="mid-slate", claim=dict(roi=2.0))
    sm = SB.summarize([a, b])
    assert sm["contests"] == 1 and sm["fees"] == 10.0 and sm["net"] == -2.0 and sm["roi"] == -0.2 and sm["claimed_roi"] == 1.0
    assert not sm["final"] and sm["unscored"][0]["name"] == "B" and sm["fees_unscored"] == 3.0
    assert SB.summarize([a])["final"]


def test_roi_timeline(tmp_path):
    import json
    (tmp_path / "lateswap.20261028-185931.json").write_text(json.dumps({"when": "2026-10-28T18:57:13", "games_started": 2, "games_total": 3, "contests": [
        {"contest_id": "7", "entries": [{"entry_id": "a", "original": {"exp_payout": 1.0}, "best": {"exp_payout": 2.0}},
                                         {"entry_id": "zz", "original": {"exp_payout": 9.0}, "best": {"exp_payout": 9.0}}]}]}))
    results = [dict(contest_id=7, entry_fee=1.0, fees=1.0, won=0.5, entries=[dict(entry_id="a")],
                    claim=dict(roi=3.0, at=datetime(2026, 10, 28, 17, 50)))]
    tl = SB.roi_timeline(results, datetime(2026, 10, 28, 18, 0), date(2026, 10, 28), out_dir=tmp_path)
    assert [x["what"].split(":")[0] for x in tl] == ["the chooser's claim before lock", "late swap", "final"]
    assert tl[1]["roi"] == 0.0 and tl[1]["roi_swapped"] == 1.0 and tl[1]["entries"] == 1     # another player's entry is not ours
    assert tl[-1]["roi"] == -0.5
