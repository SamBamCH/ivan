"""slate_backtest: the view a slate opens in, ties sharing payouts, a final export scored as DraftKings did, a mid-slate one not; the flat
player table and one contest drilled into."""
from datetime import date, datetime

import numpy as np

from ivan.sports.nba.backtest import slate as SB


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
    from ivan.sports.nba.data_clean.standings import parse_export
    ent, _ = parse_export(FINAL)
    r = SB.score_contest(CONTEST, {"12", "13"}, ent, np.array([10.0, 6.0, 2.0, 0.0]), datetime(2026, 11, 5, 10, 31))
    assert r["final"] and r["scored"].startswith("DraftKings' final export 11-05 10:31")
    assert r["found"] == 2 and r["fees"] == 10.0 and r["won"] == 8.0 and r["net"] == -2.0 and r["cashed"] == 2     # tied 2nd: (6 + 2) / 2 each
    assert [e["rank"] for e in r["entries"]] == [2, 2] and [e["points"] for e in r["entries"]] == [340.0, 340.0]
    assert r["entries"][0]["copies"] == 2 and r["unique"] == 3 and r["max_copies"] == 2 and r["winner"] == 352.5 and r["min_cash"] == 340.0
    assert r["best_rank"] == 2 and r["roi"] == -0.2


def test_a_mid_slate_export_is_not_scored():
    from ivan.sports.nba.data_clean.standings import parse_export
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


def test_flat_table_adds_team_position_miss_and_value():
    players = {"BOS": [dict(name="ann one", salary=8000.0, fpts_proj=40.0, fpts_act=52.0, played=True, min_proj=34.0, min_act=36.0),
                       dict(name="bob two", salary=4000.0, fpts_proj=12.0, fpts_act=None, played=False, min_proj=10.0, min_act=None)],
               "NYK": [dict(name="cal three", salary=5000.0, fpts_proj=20.0, fpts_act=30.0, played=True, min_proj=28.0, min_act=30.0)]}
    rows = SB.flat_table(players, {"ann one": "PG/SG"})
    assert [r["name"] for r in rows] == ["ann one", "cal three", "bob two"]          # most points scored first
    ann = rows[0]
    assert ann["team"] == "BOS" and ann["pos"] == "PG/SG" and ann["delta"] == 12.0 and ann["value"] == 6.5
    assert rows[2]["delta"] is None and rows[2]["value"] is None and rows[2]["pos"] == ""


def test_payout_curve_rungs_end_at_the_last_paid_place():
    pay = np.array([100.0, 50.0, 20.0, 10.0, 10.0, 5.0, 5.0, 5.0, 0.0, 0.0])
    curve = SB.payout_curve(pay)
    assert [(c["place"], c["payout"]) for c in curve] == [(1, 100.0), (2, 50.0), (3, 20.0), (5, 10.0), (8, 5.0)]
    assert SB.payout_curve(np.zeros(4)) == []


def test_contest_detail_lists_our_entries_and_the_field(tmp_path):
    """One contest drilled into from the export alone: our entries with their copies, every owned player with our share of our entries
    there and his projection row, the payout curve; the money as score_contest gives it."""
    players = {"ann one": dict(team="BOS", pos="PG", salary=9000.0, own_proj=50.0, fpts_proj=60.0, fpts_act=70.5, played=True),
               "ian nine": dict(team="NYK", pos="C", salary=4000.0, own_proj=5.0, fpts_proj=20.0, fpts_act=28.0, played=True)}
    cd = SB.contest_detail(None, 1, 9, lock=datetime(2026, 11, 5, 19, 0), players=players, export=("20261105-103100", FINAL), contest=CONTEST,
                           pay=np.array([10.0, 6.0, 2.0, 0.0]), out_dir=tmp_path)
    c = cd["contest"]
    assert c["final"] and c["found"] == 0                      # no upload file names an Entry ID and no user name is given: nothing is ours
    assert cd["paid"] == 3 and [r["place"] for r in cd["curve"]] == [1, 2, 3]
    ann = next(f for f in cd["field"] if f["name"] == "ann one")
    assert ann["own"] == 61.2 and ann["team"] == "BOS" and ann["fpts_act"] == 70.5 and ann["own_proj"] == 50.0 and ann["exposure"] is None
    assert cd["field"][0]["name"] == "ann one"                 # by ownership
    cd = SB.contest_detail(None, 1, 9, players=players, export=("20261105-103100", FINAL), contest=CONTEST, pay=np.array([10.0, 6.0, 2.0, 0.0]),
                           out_dir=tmp_path, user="us")
    c = cd["contest"]
    assert c["found"] == 2 and c["won"] == 8.0 and cd["n_ours"] == 2 and [e["copies"] for e in c["entries"]] == [2, 1]
    ann = next(f for f in cd["field"] if f["name"] == "ann one"); ian = next(f for f in cd["field"] if f["name"] == "ian nine")
    assert ann["exposure"] == 1.0 and ian["exposure"] == 0.5
    mid = SB.contest_detail(None, 1, 9, players=players, export=("20261104-180600", MID), contest=CONTEST, pay=np.array([10.0, 6.0, 2.0, 0.0]), out_dir=tmp_path)
    assert not mid["contest"]["final"] and mid["curve"] == [] and mid["paid"] == 0
    none = SB.contest_detail(None, 1, 9, players=players, export=None, contest=CONTEST, out_dir=tmp_path)
    assert not none["contest"]["final"] and "no standings export" in none["contest"]["scored"]
