# src/ivan/sports/mma/odds.py
"""American-odds helpers shared by ingest and the fight simulator."""
from typing import Optional


def american_to_prob(odds: Optional[float]) -> float:
    """
    Implied probability of American odds, no de-vig. A missing or zero line
    is treated as a coin flip, which is how the original model handled a
    fighter or fight without a market: better than dropping them.
    """
    if odds is None or odds == 0:
        return 0.5
    odds = float(odds)
    if odds > 0:
        return 100.0 / (odds + 100.0)
    return -odds / (-odds + 100.0)


def to_float(value) -> Optional[float]:
    """Float from a CSV cell; None when empty or unparseable."""
    if value is None:
        return None
    s = str(value).strip().replace(",", "").replace("+", "")
    if s == "" or s.lower() == "nan":
        return None
    try:
        return float(s)
    except ValueError:
        return None


def win_split(odds_a: Optional[float], odds_b: Optional[float]) -> tuple[float, float]:
    """Each fighter's share of a market's implied probability, normalised to sum to 1."""
    pa, pb = american_to_prob(odds_a), american_to_prob(odds_b)
    total = pa + pb
    if total <= 0:
        return 0.5, 0.5
    return pa / total, pb / total
