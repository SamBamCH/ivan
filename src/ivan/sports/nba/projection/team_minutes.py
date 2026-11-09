"""
The projection's game constants and the scratch rule.

data/nba/models/team_minutes.json (load_params, defaults when the file is missing): the final margin's sd around the
spread and the overtime rate that the projection's minutes band draws with (rotation.sample_worlds), and the scratch
probabilities (scratch_probability: by DK status, and for a bench player without a status by salary, the coach's
decision). role_of names a player's level from his minutes and the starter flag; tau_from_band turns the desk's hand
band into a spread. The team minutes draw that this module once held (a softmax split of the team's minutes by role)
was replaced by the rotation grid's worlds for the projection and by the game simulation for the sampler.
"""
from __future__ import annotations

import json
from typing import Optional

import numpy as np

from ivan.sports.nba.paths import MODELS_DIR

PARAMS_JSON = MODELS_DIR / "team_minutes.json"
ROLES = ("star", "starter", "rotation", "deep")
OT_MINUTES = 25.0
REGULATION = 240.0
DEFAULTS = {
    "margin_sd": 12.5,          # final margin around the spread
    "p_ot": 0.055,              # share of games with overtime (2025-26 regular season)
    "scratch": {"Q": 0.35, "GTD": 0.35, "D": 0.6, "P": 0.1},                        # P(out) by DK status
    # a bench player with no status who is projected to play but does not (coach's decision): by salary, measured on the
    # 2026-10-20/21 slates (109 such players: <=3.5k 38%, 3.5-4.5k 10%, 4.5-6k 9%, 6k+ none)
    "scratch_bench": {"3500": 0.35, "4500": 0.10, "6000": 0.08, "else": 0.0},
}


def role_of(mu: float, starter: Optional[bool]) -> str:
    if mu >= 28:
        return "star"
    if mu >= 18:
        return "starter" if starter else "rotation"
    if mu >= 9:
        return "rotation"
    return "deep"


def load_params(path=PARAMS_JSON) -> dict:
    p = json.loads(json.dumps(DEFAULTS))
    try:
        p.update(json.loads(path.read_text()))
    except (OSError, ValueError):
        pass
    return p


def tau_from_band(low: Optional[float], high: Optional[float]) -> Optional[float]:
    """The log-scale noise that puts a 10-90 band at (low, high) around their geometric middle; None when the band is not usable."""
    if low is None or high is None or low <= 0 or high <= low:
        return None
    return float((np.log(high) - np.log(low)) / (2 * 1.2816))




def scratch_probability(status: Optional[str], play_ewm: Optional[float] = None, mu: float = 0.0, params: Optional[dict] = None,
                        salary: Optional[int] = None, starter: Optional[bool] = None) -> float:
    """
    P(does not play) beyond what the allocation already knows: the DK injury
    status (Q, GTD, D, P), else for a bench player without a status the
    coach's-decision rate by salary (scratch_bench). The trailing DNP rate is
    not added: the allocation's baseline is already minutes x play rate.
    """
    P = params or load_params()
    st = (status or "").upper()
    if st in P.get("scratch", {}):
        return float(P["scratch"][st])
    if starter or salary is None or mu < 6:
        return 0.0
    sb = P.get("scratch_bench") or {}
    for cut in sorted(int(k) for k in sb if k != "else"):
        if int(salary) <= cut:
            return float(sb[str(cut)])
    return float(sb.get("else", 0.0))
