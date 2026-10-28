"""The Monte Carlo contest simulation (ivan.sports.nba.simulate_contest): the scoring chunk pays out exactly the pool, ties split; payouts scale; the report builds."""
import numpy as np

from ivan.sports.nba import simulate_contest as SC


def test_simulate_chunk_conserves_the_pool_and_splits_ties():
    rng = np.random.default_rng(0)
    n_sims, n_players = 20, 6
    P = rng.normal(10, 3, (n_sims, n_players)).astype(np.float32)
    lineup_ix = [[0, 1], [0, 1], [2, 3], [4, 5], [1, 2]]          # entries 0 and 1 are copies of one lineup
    payouts = {1: 100.0, 2: 50.0, 3: 20.0}
    part = SC._simulate_chunk(0, n_sims, lineup_ix, P, 10.0, payouts)
    assert part["scores"].shape == (n_sims, 5) and part["stats"]["sims_off"] == 0
    assert np.allclose(part["payouts"].sum(axis=1), 170.0)
    assert np.allclose(part["payouts"][:, 0], part["payouts"][:, 1])           # the copies tie and share
    assert np.allclose(part["rois"], (part["payouts"] - 10.0) / 10.0)
    assert (part["ranks"][:, 0] == part["ranks"][:, 1]).all() and part["ranks"].min() == 1


def test_scale_payouts_keeps_the_pool_proportional():
    tiers = [{"min_position": 1, "max_position": 1, "payout": 1000.0}, {"min_position": 2, "max_position": 10, "payout": 100.0},
             {"min_position": 11, "max_position": 100, "payout": 10.0}]
    scaled = SC.scale_payouts(tiers, num_field_lineups=1000, original_field_size=10000, entry_fee=5.0)
    pool = 1000 + 9 * 100 + 90 * 10
    assert abs(sum(scaled.values()) - pool / 10) < 1.0 and scaled[1] == max(scaled.values())


def test_contests_in_entries_one_row_per_contest(tmp_path):
    f = tmp_path / "entries.csv"
    f.write_text("Entry ID,Contest Name,Contest ID,Entry Fee,PG,SG,SF,PF,C,G,F,UTIL,,Instructions\n"
                 "1, Big ,100,$5,,,,,,,,,,1. Column A lists\n2,Big,100,$5,,,,,,,,,,2. more\n3,Small,200,$1,,,,,,,,,,\n,,,,,,,,,,,,,\n")
    c = SC.contests_in_entries(f)
    assert list(zip(c["Contest ID"], c["Contest Name"])) == [(100, "Big"), (200, "Small")]


def test_report_and_summary_build():
    n_sims, M = 5, 4
    scores = np.array([[10, 9, 8, 7]] * n_sims, dtype=np.float32)
    pay = np.array([[6, 3, 0, 0]] * n_sims, dtype=np.float32)
    players = {1: {"name": "a", "team": "X", "pos": "PG"}, 2: {"name": "b", "team": "Y", "pos": "C"}}
    report = SC.build_report(contest={"id": 1, "name": "T", "entry_fee": 1.0, "total_entries": 40, "field_cap": 4, "multi_entry": 1, "start": None, "candidates": 3},
                             field_meta={"sampler": "x"}, u_lids=[10, 11, 12], dupes=np.array([2, 1, 1]), pid_lists_by_lid={10: [1, 2], 11: [1], 12: [2]},
                             players=players, exposure={1: 3, 2: 3}, M=M, tiers=[{"min_position": 1, "max_position": 1, "payout": 60.0}],
                             scaled={1: 6.0, 2: 3.0}, pay_by_rank=np.array([6.0, 3.0, 0.0, 0.0]), paid_places=2, full_scores=scores, full_payouts=pay,
                             cash_pct=np.array([1.0, 1.0, 0.0]), avg_roi=np.array([3.5, -1.0, -1.0]), win_pct=np.array([1.0, 0, 0]),
                             sims_off=0, n_sims=n_sims, missing=[], seed=None, seconds=0.5)
    assert report["field"]["unique"] == 3 and report["payouts"]["paid_places"] == 2 and all(c["ok"] for c in report["checks"])
    assert "T (#1)" in SC.format_summary(report)


def test_pay_against_the_field_with_copies():
    import numpy as np
    from ivan.sports.nba.simulate_contest import pay_against
    field = np.array([100.0, 200.0, 300.0])                 # 3 field entries; with us, 4 places
    pay = np.array([0.0, 10.0, 5.0, 0.0, 0.0]); csum = np.cumsum(pay)
    p, r = pay_against(field, np.array([350.0, 250.0, 50.0]), np.array([0, 0, 0]), csum)
    assert list(r) == [1, 2, 4] and list(p) == [10.0, 5.0, 0.0]
    # our lineup is in the field twice (the 300 entry and one more copy): we tie for first with them and split 1st + 2nd + 3rd
    field2 = np.array([100.0, 300.0, 300.0]); p, r = pay_against(field2, np.array([300.0]), np.array([2]), csum)
    assert r[0] == 1 and abs(p[0] - 15.0 / 3) < 1e-9


def test_field_sizes_cap_less_our_entry():
    from ivan.sports.nba.simulate_contest import field_sizes
    assert field_sizes([14268, 71343, 2971, 1], 25000) == [14267, 24999, 2970, 1]
    assert field_sizes([500], 10000) == [499]
