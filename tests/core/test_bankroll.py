"""The bankroll read from the results (ivan.core.bankroll): start + every sport's net, from `since` on; a given figure wins."""
import json

import pytest

from ivan.core import bankroll as BK


def _ledger(days: dict) -> dict:
    """{iso day: (net, scored contests)} -> the shape pnl.roll_up gives."""
    d = {day: {"net": net, "contests": n} for day, (net, n) in days.items()}
    return {"overall": {"net": sum(a["net"] for a in d.values()), "contests": sum(a["contests"] for a in d.values())}, "days": d}


LEDGERS = {"nba": _ledger({"2026-11-01": (150.0, 3), "2026-11-04": (-40.0, 2)}),
           "nfl": _ledger({"2026-10-12": (-300.0, 5), "2026-11-02": (25.5, 1)}),
           "mma": None,
           "mlb": _ledger({})}


def test_start_plus_every_sports_net():
    info = BK.from_ledgers(LEDGERS, {"start": 1000.0, "since": None, "sports": None})
    assert info["bankroll"] == pytest.approx(1000.0 + 110.0 - 274.5)
    assert info["sports"] == {"nba": pytest.approx(110.0), "nfl": pytest.approx(-274.5)}        # sports with no scored contest are left out
    assert info["contests"] == 11 and info["source"] == "results"
    line = BK.describe(info)
    assert "$835.50" in line and "NBA +$110.00" in line and "NFL -$274.50" in line and "11 scored" in line


def test_since_counts_only_the_days_from_it_on():
    info = BK.from_ledgers(LEDGERS, {"start": 500.0, "since": "2026-11-02", "sports": None})
    assert info["sports"] == {"nba": pytest.approx(-40.0), "nfl": pytest.approx(25.5)}
    assert info["bankroll"] == pytest.approx(485.5) and "since 2026-11-02" in BK.describe(info)


def test_config_file_and_defaults(tmp_path):
    assert BK.load_config(tmp_path / "none.json") == {"start": BK.DEFAULT_START, "since": None, "sports": None}
    p = tmp_path / "bankroll.json"
    p.write_text(json.dumps({"start": 2500, "since": "2026-11-01T00:00:00", "sports": ["NBA", " nfl "]}))
    assert BK.load_config(p) == {"start": 2500.0, "since": "2026-11-01", "sports": ["nba", "nfl"]}


def test_a_given_figure_wins_and_the_largest_of_several(monkeypatch):
    monkeypatch.setattr(BK, "current", lambda now=None, cfg=None: pytest.fail("a given bankroll must not read the results"))
    lines = []
    info = BK.for_slate([None, 300.0, "", 750.0], log=lines.append)
    assert info == {"bankroll": 750.0, "source": "given"}
    assert any("different bankroll_dollars" in l for l in lines) and any("(given)" in l for l in lines)


def test_blank_reads_the_results_and_grows_with_them(monkeypatch):
    cfg = {"start": 1000.0, "since": None, "sports": None}
    monkeypatch.setattr(BK, "current", lambda now=None, cfg_=None: BK.from_ledgers(LEDGERS, cfg))
    assert BK.for_slate([None, None])["bankroll"] == pytest.approx(835.5)
    better = dict(LEDGERS, nba=_ledger({"2026-11-01": (150.0, 3), "2026-11-04": (-40.0, 2), "2026-11-05": (400.0, 4)}))
    monkeypatch.setattr(BK, "current", lambda now=None, cfg_=None: BK.from_ledgers(better, cfg))
    assert BK.for_slate([])["bankroll"] == pytest.approx(1235.5)


def test_results_that_used_up_the_bankroll_refuse(monkeypatch):
    monkeypatch.setattr(BK, "current", lambda now=None, cfg_=None: BK.from_ledgers(LEDGERS, {"start": 100.0, "since": None, "sports": None}))
    with pytest.raises(ValueError, match="config/bankroll.json"):
        BK.for_slate([None])


def test_a_consumer_reads_no_results(monkeypatch):
    monkeypatch.setenv("IVAN_FEED_SSH", "user@laptop")
    monkeypatch.setattr(BK, "sport_ledger", lambda sport, now: pytest.fail("a consumer must not read the results"))
    info = BK.current(cfg={"start": 1000.0, "since": None, "sports": ["nba", "nfl"]})
    assert info["bankroll"] == 1000.0 and info["consumer"] and "reads no results" in BK.describe(info) and "NBA" not in BK.describe(info)


def test_an_unreadable_sport_is_named_not_hidden(monkeypatch):
    monkeypatch.delenv("IVAN_FEED_SSH", raising=False); monkeypatch.delenv("IVAN_FEED_URL", raising=False)
    def led(sport, now):
        if sport == "nfl":
            raise RuntimeError("no database")
        return LEDGERS[sport]
    monkeypatch.setattr(BK, "sport_ledger", led)
    info = BK.current(cfg={"start": 1000.0, "since": None, "sports": ["nba", "nfl"]})
    assert info["bankroll"] == pytest.approx(1110.0) and "nfl" in info["errors"]
    assert "NOT COUNTED" in BK.describe(info)
