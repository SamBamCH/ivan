# src/ivan/sports/mma/cards.py
"""
The card as the model saw it: every fighter's id, salary, fight, start and
win probability, and who is in a five-round fight. ingest stores one record
per run in the raw store (kind `card`, key = the DK draft group), so the
newest one is the card at the last ingest before lock.

The win probabilities come from hand-entered prices that are overwritten by
the next card, so this record is the only way to put a scraped field back
on the lineups the model could have predicted: standings.derive_card and
the duplication model (dupe_model.py) both read it.
"""
from __future__ import annotations

from typing import Iterable, Optional

from ivan.core import raw

KIND = "card"


def card_payload(draft_group_id: int, rows: Iterable[dict], main_event: Iterable[int], source: str = "ingest") -> dict:
    """rows need id, name, salary, fight_id, win_prob (gametime optional); main_event = ids of the five-round fighters."""
    main = {int(i) for i in main_event}
    return {"draft_group": int(draft_group_id), "source": source,
            "fighters": [{"id": int(r["id"]), "name": r["name"], "salary": int(r["salary"]), "fight_id": r["fight_id"],
                          "gametime": str(r["gametime"]) if r.get("gametime") is not None else None,
                          "win_prob": (float(r["win_prob"]) if r.get("win_prob") is not None else None),
                          "main_event": int(r["id"]) in main} for r in rows]}


def save_card(draft_group_id: int, rows: Iterable[dict], main_event: Iterable[int], source: str = "ingest"):
    """Store the card in the raw store; returns the artifact."""
    return raw.store("mma", KIND, int(draft_group_id), card_payload(draft_group_id, rows, main_event, source))


def load_card(draft_group_id: int) -> Optional[dict]:
    """
    The newest record of a card, or None: {"draft_group", "pool" (rows with
    id, name, salary, fight_id), "win_prob" {id: p}, "main_event" {ids}}.
    A fighter stored without a win probability reads as 0.5, as the sim does.
    """
    art = raw.latest("mma", KIND, int(draft_group_id))
    if art is None:
        return None
    return unpack(raw.load(art.path, "mma"))


def unpack(payload: dict) -> dict:
    fighters = payload["fighters"]
    return {"draft_group": int(payload["draft_group"]),
            "pool": [{"id": int(f["id"]), "name": f["name"], "salary": int(f["salary"]), "fight_id": f["fight_id"]} for f in fighters],
            "win_prob": {int(f["id"]): (float(f["win_prob"]) if f.get("win_prob") is not None else 0.5) for f in fighters},
            "main_event": {int(f["id"]) for f in fighters if f.get("main_event")}}


def stored_cards() -> list[int]:
    """Draft groups with a card record."""
    return sorted({int(a.key) for a in raw.index("mma") if a.kind == KIND})
