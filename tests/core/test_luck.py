"""The luck test (ivan.core.validation.luck_summary): where the realized P/L sits in the sim's own spread."""
import numpy as np
import pandas as pd

from ivan.core import validation as V
from ivan.sports.nfl.validate import contest_luck


def _book(rng, n_entries, worlds, fee=5.0):
    traces = np.where(rng.random((n_entries, worlds)) < 0.25, 0.8, -1.0)
    traces[rng.random((n_entries, worlds)) < 0.005] = 40.0
    return traces, fee


def test_contest_luck_percentile_and_loss_probability():
    rng = np.random.default_rng(0)
    traces, fee = _book(rng, 20, 1000)
    ours = [{"entry_id": i, "sim_roi": traces[i]} for i in range(20)]
    pl = (fee * traces).sum(axis=0)
    rows = [{"entry_id": i, "entry_fee": fee, "actual_payout": fee + float(np.median(pl)) / 20} for i in range(20)]   # the median world
    lk = contest_luck(ours, rows)
    assert 0.4 <= lk["sim_pl_pctile"] <= 0.6 and 0 < lk["sim_p_loss"] < 1
    rows_lost = [{**r, "actual_payout": 0.0} for r in rows]
    assert contest_luck(ours, rows_lost)["sim_pl_pctile"] <= lk["sim_pl_pctile"]
    assert np.isnan(contest_luck(ours, [{**r, "actual_payout": np.nan} for r in rows])["sim_pl_pctile"])


def test_luck_summary_separates_variance_from_miscalibration():
    rng = np.random.default_rng(1)
    def frame(draw):
        rows = []
        for s in range(8):                                   # 8 slates, 2 contests each; the actual is one of the sim's own worlds (calibrated)...
            for c in range(2):
                traces, fee = _book(rng, 20, 1000)
                pl = (fee * traces).sum(axis=0)
                actual = draw(pl)
                for i in range(20):
                    rows.append({"contest_id": s * 10 + c, "slate": f"slate{s}", "entry_fee": fee, "actual_payout": fee + actual / 20,
                                 "sim_pl_pctile": float((pl <= actual).mean()), "sim_p_loss": float((pl < 0).mean()), "sim_pl_p50_roi": float(np.median(pl) / (20 * fee))})
        return pd.DataFrame(rows)
    calibrated = V.luck_summary(frame(lambda pl: float(rng.choice(pl))))
    assert calibrated["slates"] == 8 and calibrated["contests"] == 16 and len(calibrated["slate_pctiles"]) == 8
    assert calibrated["p_value"] is None or calibrated["p_value"] > 0.01
    broken = V.luck_summary(frame(lambda pl: float(np.min(pl)) - 1.0))   # ...or always below every world the sim imagined
    assert broken["mean_pctile"] == 0.0 and broken["direction"] == "below" and (broken["p_value"] is None or broken["p_value"] < 0.001)
    checks = V.build_checks(lineups={"settled": 320, "fees": 1600.0, "sim_roi": 2.0, "actual_roi": -0.3, "sim_cash_pct": 0.25, "actual_cash_rate": 0.24,
                                     "honest_roi": 0.4, "luck": broken})
    by = {c["name"]: c for c in checks}
    assert by["luck: realized P/L in the sim's own spread"]["status"] == "fail"
    assert "honest (cross-fit) +40%" in by["sim ROI vs actual ROI"]["detail"]
    assert by["sim ROI vs actual ROI"]["value"] == pytest.approx(0.7)


import pytest  # noqa: E402
