"""The contest simulation (ivan.sports.nba.simulate_contest): payouts scale, the entries file parses, the field pays ties and copies."""
import numpy as np

from ivan.sports.nba import simulate_contest as SC




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




def test_pay_against_the_field_with_copies():
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
