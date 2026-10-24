"""The team minutes solve (ivan.sports.nba.projection.solve): budget, anchors, edits, position groups."""
import numpy as np

from ivan.sports.nba.projection.solve import Anchor, group_of, solve


def team(edit=None, out=None):
    A = [Anchor("c1", 32, 4, "big", source="book"), Anchor("c2", 14, 5, "big"), Anchor("pf", 30, 4, "big", source="book"),
         Anchor("pg1", 34, 4, "guard", source="book"), Anchor("pg2", 16, 5, "guard"), Anchor("sg", 30, 4, "guard", source="book"),
         Anchor("sf", 33, 4, "wing", source="book"), Anchor("w2", 24, 5, "wing"), Anchor("w3", 18, 5, "wing"),
         Anchor("d1", 4, 6, "wing", source="deep"), Anchor("d2", 3, 6, "big", source="deep"), Anchor("d3", 3, 6, "guard", source="deep")]
    if out:
        A = [a for a in A if a.name != out]
    if edit:
        for a in A:
            if a.name == edit[0]:
                a.mean, a.sd, a.source = edit[1], 0.05, "edit"
    return A


TARGETS = {"big": (78, 6), "guard": (82, 6), "wing": (81, 6)}


def test_groups_from_positions():
    assert group_of("PF/C") == "big" and group_of(["PG", "SG"]) == "guard" and group_of("SF") == "wing" and group_of("") == "wing"


def test_sums_to_the_budget_and_stays_near_the_books():
    sol = solve(team(), TARGETS)
    assert sol.ok and abs(sum(sol.minutes.values()) - 241) < 0.05
    assert all(abs(sol.minutes[n] - m) < 3 for n, m in (("c1", 32), ("pg1", 34), ("sf", 33)))      # priced starters barely move


def test_an_edit_holds_and_the_team_rebalances():
    sol = solve(team(edit=("pg1", 38)), TARGETS)
    assert abs(sol.minutes["pg1"] - 38) < 0.2 and abs(sum(sol.minutes.values()) - 241) < 0.05
    base = solve(team(), TARGETS).minutes
    assert sol.minutes["pg2"] < base["pg2"] and sol.minutes["sg"] <= base["sg"] + 0.1          # his own group gives the minutes up first


def test_an_absent_big_feeds_the_other_bigs_first():
    base = solve(team(), TARGETS).minutes
    sol = solve(team(out="c1"), TARGETS).minutes
    big_gain = (sol["c2"] - base["c2"]) + (sol["pf"] - base["pf"]) + (sol["d2"] - base["d2"])
    guard_gain = sum(sol[n] - base[n] for n in ("pg1", "pg2", "sg", "d3"))
    assert big_gain > guard_gain * 2 and sol["c2"] > base["c2"] + 5


def test_without_group_targets_it_is_a_plain_weighted_fill():
    sol = solve(team(out="c1"))
    assert sol.ok and abs(sum(sol.minutes.values()) - 241) < 0.05 and sol.group_target == {}


def test_caps_and_floors_are_respected():
    A = team(edit=("w3", 0)); sol = solve(A, TARGETS)
    assert sol.minutes["w3"] < 0.1 and max(sol.minutes.values()) <= 42.0 + 1e-6
