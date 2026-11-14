"""Each fight's simulation results (fight_report.card_report) and the slate page's Fights section drawn from them."""
import os
import time

import pytest

from ivan.sports.mma.fight_report import card_report, format_card
from ivan.sports.mma.models import MMAFight, MMAPlayer


def test_card_report_summarises_every_fight(session, cfg, ingest):
    ingest(session, cfg)
    rep = card_report(session, cfg, n_sims=4000)
    assert len(rep["fights"]) == session.query(MMAFight).count() and rep["n_sims"] == 4000
    times = [f["gametime"] for f in rep["fights"]]
    assert times == sorted(times, reverse=True)                                    # the main event first
    assert rep["fights"][0]["rounds"] == 5 and [e["key"] for e in rep["fights"][0]["ends"]] == ["r1", "r2", "r3", "r4", "r5", "decision"]
    for f in rep["fights"]:
        a, b = f["fighters"]
        assert a["win"] + b["win"] == pytest.approx(1.0)
        assert sum(e["p1"] + e["p2"] for e in f["ends"]) == pytest.approx(1.0)      # the ways it ends partition the worlds
        assert sum(e["p1"] for e in f["ends"]) == pytest.approx(a["win"])
        assert a["win_finish"] + b["win_finish"] == pytest.approx(f["p_finish"])
        for s in (a, b):
            assert sum(s["hist"]) == pytest.approx(1.0) and len(s["hist"]) == rep["axis"]["xmax"] // rep["axis"]["bin"]
            assert s["p25"] <= s["p50"] <= s["p75"] <= s["p90"] and s["mean_win"] > s["mean_loss"]
            assert abs(s["mean"] - s["projection"]) < 0.08 * s["projection"] + 2    # the sim's mean is the stored projection
            assert "fpts" not in s
    # the same card reads the same twice
    again = card_report(session, cfg, n_sims=4000)
    assert again["fights"][0]["fighters"][0]["mean"] == rep["fights"][0]["fighters"][0]["mean"]
    text = format_card(rep)
    assert rep["fights"][0]["fighters"][0]["name"].title() in text and "ends (first / second fighter wins)" in text


def test_alerts_name_what_the_pool_lacks(session, cfg, ingest):
    ingest(session, cfg)
    assert card_report(session, cfg, n_sims=500)["alerts"] == []
    # prices blanked: every fight on the default finish chance, every fighter on the win bonus alone
    session.query(MMAFight).update({MMAFight.itd_odds: None})
    session.query(MMAPlayer).update({MMAPlayer.itd_odds: None, MMAPlayer.sig_strikes: None, MMAPlayer.takedowns: None})
    session.commit()
    rep = card_report(session, cfg, n_sims=500)
    texts = " | ".join(a["text"] for a in rep["alerts"])
    n = len(rep["fights"])
    assert f"{n} of {n} fights have no finish price" in texts and f"{2 * n} of {2 * n} fighters have no strikes or takedowns line" in texts
    assert all(a["level"] == "bad" for a in rep["alerts"]) and all(f["finish_source"] == "default" for f in rep["fights"])
    assert all(s["sig"] == 0 and "ITD" in s["blank"] for f in rep["fights"] for s in f["fighters"])


def test_inputs_saved_after_the_ingest_are_flagged(session, cfg, ingest, tmp_path):
    ingest(session, cfg)
    late = tmp_path / "player_odds.csv"
    late.write_text(open(cfg["paths"]["player_odds_csv"]).read())
    os.utime(late, (time.time() + 3600, time.time() + 3600))
    cfg["paths"]["player_odds_csv"] = str(late)
    alerts = card_report(session, cfg, n_sims=500)["alerts"]
    assert len(alerts) == 1 and alerts[0]["level"] == "bad" and "player_odds.csv was saved" in alerts[0]["text"] and "ingest.py again" in alerts[0]["text"]


def test_fights_section_renders(session, cfg, ingest):
    pytest.importorskip("fastapi")
    from ivan.web import frame
    from ivan.web.mma.fights import view
    ingest(session, cfg)
    rep = view(card_report(session, cfg, n_sims=2000))
    assert len(rep["fighters"]) == 2 * len(rep["fights"]) and rep["fighters"][0]["salary"] >= rep["fighters"][-1]["salary"]
    html = frame.templates.get_template("_mma_fights.html").render(fights=rep)
    f = rep["fights"][0]
    assert html.count('class="card fight"') == len(rep["fights"]) and f'id="fight-{f["id"]}"' in html and f["title"] in html
    assert html.count("<svg") == 2 * len(rep["fights"]) and "by a finish in round 1" in html and "Check the inputs" not in html
