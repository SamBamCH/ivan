# tests/core/test_menu_picks.py
"""An explicit contest selection (the Backtest page) overrides the menu's regex items."""
from __future__ import annotations

import json

from ivan.core.replay import Menu, MenuItem, pick_contests

CONTESTS = [
    {"contest_id": 1, "name": "NFL $100K First Down [20 Entry Max]", "entry_fee": 1.0, "total_entries": 100000, "multi_entry": 20, "has_payouts": True},
    {"contest_id": 2, "name": "NFL $5K Huddle [Single Entry]", "entry_fee": 5.0, "total_entries": 1000, "multi_entry": 1, "has_payouts": True},
    {"contest_id": 3, "name": "NFL $2M Millionaire", "entry_fee": 20.0, "total_entries": 160000, "multi_entry": 150, "has_payouts": True},
]


def test_picks_win_over_items_and_cap_at_the_contest_max():
    menu = Menu(items=[MenuItem(match="First Down", entries=20)], picks={3: 500, 2: 5, 99: 1})
    picked = pick_contests(CONTESTS, menu, "classic")
    assert [(p["contest_id"], p["entries"], p["menu"]) for p in picked] == [(2, 1, "picked"), (3, 150, "picked")]


def test_load_reads_picks(tmp_path):
    p = tmp_path / "menu.json"
    p.write_text(json.dumps({"weights": {"cash_pct": 1.0}, "picks": [{"contest_id": 1, "entries": 7}], "contests": [{"match": "Huddle", "entries": 1}]}))
    menu = Menu.load(p)
    assert menu.picks == {1: 7} and menu.weights == {"cash_pct": 1.0} and menu.items[0].match == "Huddle"
    assert [p["contest_id"] for p in pick_contests(CONTESTS, menu, "classic")] == [1]
    menu.picks = {}
    assert [p["contest_id"] for p in pick_contests(CONTESTS, menu, "classic")] == [2]     # the regex items again
