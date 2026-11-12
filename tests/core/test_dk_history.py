# tests/core/test_dk_history.py
"""DraftKings' entry-history export (ivan.core.dk_history): parsed, stored by entry id, and laid over a sport's ledger slates."""
from __future__ import annotations

import os
from datetime import date, datetime

import pytest

from ivan.core import dk_history as H
from ivan.core import pnl

HEAD = "Sport,Game_Type,Entry_Key,Entry,Contest_Key,Contest_Date_EST,Place,Points,Winnings_Non_Ticket,Winnings_Ticket,Contest_Entries,Entry_Fee,Prize_Pool,Places_Paid\n"
ROWS = [
    'NFL,Showdown Captain Mode,9001,"NFL Showdown $3 Play-Action",101,2026-09-21 20:15:00,12,110.5,$10.00,$0.00,1000,$3.00,"$2,500.00",250\n',
    'NFL,Showdown Captain Mode,9002,"NFL Showdown $3 Play-Action",101,2026-09-21 20:15:00,400,80.0,$2.00,$0.00,1000,$3.00,"$2,500.00",250\n',
    'NFL,Showdown Captain Mode,9003,"NFL Showdown $3 Play-Action",101,2026-09-21 20:15:00,900,41.0,$0.00,$0.00,1000,$3.00,"$2,500.00",250\n',
    'NFL,Classic,9004,"NFL $1 First Down",102,2026-10-11 13:00:00,5000,90.0,$0.00,$0.00,20000,$1.00,"$15,000.00",4000\n',
    'NBA,Classic,9005,"NBA $5 Huddle",201,2026-11-04 19:00:00,3,300.0,$20.00,$5.00,500,$5.00,"$2,000.00",100\n',
    'MMA,Classic,9006,"UFC $2 Jab",301,2026-11-08 18:00:00,70,95.0,$0.00,$0.00,300,$2.00,$500.00,60\n',
    'NFL,Classic,8000,"NFL last season",50,2025-12-28 13:00:00,1,200.0,$100.00,$0.00,100,$10.00,$900.00,20\n',
]


@pytest.fixture
def store(tmp_path):
    export = tmp_path / "draftkings-contest-entry-history.csv"
    export.write_text(HEAD + "".join(ROWS))
    ts = datetime(2026, 11, 12, 9, 0).timestamp()
    os.utime(export, (ts, ts))
    s = tmp_path / "dk" / "entry_history.csv"
    meta = H.import_file(export, since=date(2026, 1, 1), store=s)
    assert (meta["rows"], meta["read_rows"], meta["new_rows"]) == (6, 6, 6) and meta["sports"] == {"NFL": 4, "NBA": 1, "MMA": 1}
    return s


def test_parse_reads_every_sport_and_adds_ticket_winnings(tmp_path):
    p = tmp_path / "h.csv"
    p.write_text(HEAD + "".join(ROWS))
    df = H.parse(p)
    assert len(df) == 7 and set(df.sport) == {"NFL", "NBA", "MMA"}
    assert df[df.entry_id == 9005].winnings.iloc[0] == 25.0 and df[df.entry_id == 9001].entry_fee.iloc[0] == 3.0
    assert H.parse_money("($3.00)") == -3.0 and H.parse_money("$1,250.00") == 1250.0


def test_import_keeps_the_year_and_a_later_export_replaces(store, tmp_path):
    assert H.sports(store) == ["mma", "nba", "nfl"] and len(H.entries("nfl", store)) == 4          # the 2025 entry is before `since`
    assert H.covered_until(store) == datetime(2026, 11, 12, 9, 0)
    later = tmp_path / "later.csv"                                                                  # the same entry, now paid; one new entry
    later.write_text(HEAD + ROWS[3].replace("$0.00,$0.00,20000", "$4.00,$0.00,20000")
                     + 'NFL,Classic,9007,"NFL $1 First Down",103,2026-11-15 13:00:00,9,150.0,$8.00,$0.00,20000,$1.00,"$15,000.00",4000\n')
    meta = H.import_file(later, since=date(2026, 1, 1), store=store)
    assert (meta["rows"], meta["new_rows"]) == (7, 1)
    df = H.entries("nfl", store)
    assert df[df.entry_id == 9004].winnings.iloc[0] == 4.0 and len(df) == 5


def test_contest_rows_are_ledger_rows(store):
    rows = H.contest_rows("nfl", store)
    c = rows[101]
    assert (c["found"], c["fees"], c["won"], c["net"], c["cashed"], c["best_rank"], c["field"]) == (3, 9.0, 12.0, 3.0, 2, 12, 1000)
    assert (c["gross_win"], c["gross_loss"]) == (7.0, 4.0) and c["final"] and c["day"] == "2026-09-21" and "entry history" in c["scored"]
    assert H.contest_rows("nba", store)[201]["won"] == 25.0 and H.contest_rows("golf", store) == {}


def test_merge_lays_the_history_over_a_sports_slates(store):
    scraped = dict(contest_id=102, name="NFL $1 First Down (scraped)", day="2026-10-11", final=True, detail=True, found=1, ours=1, fees=1.0, won=0.5,
                   net=-0.5, cashed=1, gross_win=0.0, gross_loss=0.5, best_rank=4999, best_points=90.0, field=20000)
    ghost = dict(contest_id=555, name="chosen for, never entered", day="2026-10-11", final=False, found=0, ours=20, fees=20.0, won=None)
    pending = dict(contest_id=556, name="played after the export", day="2026-11-12", final=False, found=0, ours=5, fees=5.0, won=None)
    slates = [dict(key="700", day="2026-10-11", complete=False, contests=[scraped, ghost]), dict(key="800", day="2026-11-12", complete=False, contests=[pending])]
    out = {s["key"]: s for s in H.merge("nfl", slates, slate_of={101: "600"}, store=store)}
    assert set(out) == {"700", "600", "800"}
    row = out["700"]["contests"][0]                                               # the history's money, the row's own slate, name and panel flag
    assert len(out["700"]["contests"]) == 1 and (row["won"], row["net"], row["name"], row["detail"]) == (0.0, -1.0, "NFL $1 First Down (scraped)", True)
    assert out["700"]["complete"] and row["best_rank"] == 4999
    assert out["600"]["day"] == "2026-09-21" and out["600"]["contests"][0]["contest_id"] == 101 and out["600"]["complete"]
    assert not out["800"]["complete"]                                             # not yet covered by an export: still unscored
    assert slates[0]["contests"][0]["won"] == 0.5                                 # the input is not changed
    led = pnl.roll_up(list(out.values()))
    assert led["overall"]["net"] == pytest.approx(2.0) and led["overall"]["fees_unscored"] == 5.0
    assert [s["key"] for s in H.merge("nfl", [], store=store)] == ["dk-2026-09-21", "dk-2026-10-11"]      # no draft group known: a slate per day


def test_ledger_of_a_sport_only_the_history_knows(store, tmp_path):
    led = H.ledger("mma", store)
    assert led["overall"]["net"] == -2.0 and led["overall"]["contests"] == 1 and list(led["days"]) == ["2026-11-08"]
    assert H.ledger("golf", store) is None and H.merge("nfl", [dict(key="1", day="x", complete=True, contests=[])], store=tmp_path / "none.csv") != []
