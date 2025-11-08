#!/usr/bin/env python3
"""
Simple ownership projection for NBA using field sampling.

This module provides a lightweight approximate ownership projection:

- Samples the fantasy field from existing `Lineup` rows using
  `sample_field.sample_field` (probability proportional to lineup projected fpts).
- Aggregates player exposures across sampled lineups to produce per-player
  ownership percentages (0-100).
- Writes the ownership back to `NBAPlayer.ownership`.

This is intentionally simple — a rough estimate for now.
"""

from typing import Dict, List
import logging
from collections import Counter
from sqlalchemy.orm import Session
import numpy as np

from ivan.sports.nba.models import NBAPlayer, Lineup
from ivan.sports.nba.sample_field import sample_field

logger = logging.getLogger(__name__)


def _coerce_player_ids(val) -> List[int]:
	"""Parse JSON-like player_ids_list from the DB into list[int]."""
	import json
	if val is None:
		return []
	if isinstance(val, list):
		return [int(x) for x in val]
	if isinstance(val, str):
		try:
			data = json.loads(val)
			if isinstance(data, list):
				return [int(x) for x in data]
		except Exception:
			return []
	# fallback: try to iterate
	try:
		return [int(x) for x in val]
	except Exception:
		return []


def _compute_exposures_from_counts(
	session: Session,
	counts_by_lineup: Dict[int, int],
	total_entries: int,
) -> Dict[int, float]:
	"""Convert lineup duplication counts to per-player ownership percentages (0-100).

	Args:
		session: SQLAlchemy session
		counts_by_lineup: mapping lineup_id -> count
		total_entries: size of simulated field

	Returns:
		mapping player_id -> ownership_pct (0-100)
	"""
	selected_lineup_ids = [lid for lid, c in counts_by_lineup.items() if c > 0]
	if not selected_lineup_ids:
		logger.warning("No lineups selected in field simulation")
		return {}

	# Batch query lineup -> player_ids to avoid SQL parameter limits
	BATCH_SIZE = 900
	player_exposure_counts = Counter()

	for i in range(0, len(selected_lineup_ids), BATCH_SIZE):
		batch = selected_lineup_ids[i : i + BATCH_SIZE]
		rows = (
			session.query(Lineup.id, Lineup.player_ids_list)
			.filter(Lineup.id.in_(batch))
			.all()
		)

		for lineup_id, player_ids_json in rows:
			player_ids = _coerce_player_ids(player_ids_json)
			dupe_count = counts_by_lineup.get(lineup_id, 0)
			for pid in player_ids:
				player_exposure_counts[pid] += dupe_count

	ownership_pct = {
		pid: (count / float(total_entries)) * 100.0
		for pid, count in player_exposure_counts.items()
	}

	return ownership_pct


def _write_ownership_to_db(session: Session, ownership_by_pid: Dict[int, float]) -> int:
	"""Write ownership percentages back to NBAPlayer.ownership.

	Returns number of updated rows.
	"""
	updated = 0

	# set all players to 0.0 first (players not on slate)
	try:
		session.query(NBAPlayer).update({NBAPlayer.ownership: 0.0}, synchronize_session=False)
	except Exception:
		# If NBAPlayer.ownership doesn't exist for some reason, raise a clear error
		raise

	for pid, pct in ownership_by_pid.items():
		result = (
			session.query(NBAPlayer)
			.filter(NBAPlayer.id == pid)
			.update({NBAPlayer.ownership: float(pct)})
		)
		updated += result

	session.commit()
	return updated


def project_ownership(
	session: Session,
	*,
	field_size: int = 10000,
	num_simulations: int = 1,
	aggregate_method: str = "mean",
	config: dict = None,
) -> Dict[int, float]:
	"""Project player ownership by sampling fields and aggregating exposures.

	Args:
		session: SQLAlchemy session
		field_size: number of entries per simulated field
		num_simulations: how many independent field samples to run
		aggregate_method: 'mean'|'median'|'max'
		config: passed through to `sample_field`

	Returns:
		mapping player_id -> ownership_pct (0-100)
	"""
	config = config or {}

	lineup_count = session.query(Lineup).count()
	if lineup_count == 0:
		logger.warning("No lineups found in database. Cannot project ownership.")
		logger.warning("Run 'python scripts/nba/generate_lineups.py' first.")
		return {}

	logger.info(f"[ownership] Found {lineup_count:,} lineups in DB; running {num_simulations} sim(s)")

	all_ownership = []

	for sim in range(num_simulations):
		logger.info(f"[ownership] Sampling field {sim + 1}/{num_simulations} (size={field_size})")
		counts = sample_field(session=session, field_size=field_size, cfg=config)
		ownership = _compute_exposures_from_counts(session=session, counts_by_lineup=counts, total_entries=field_size)
		all_ownership.append(ownership)

		if ownership:
			vals = list(ownership.values())
			logger.info(
				f"[ownership]   Sim {sim+1}: {len(ownership)} players, range [{min(vals):.2f}%, {max(vals):.2f}%], mean={np.mean(vals):.2f}%"
			)

	if not all_ownership:
		logger.error("All simulations failed or returned no ownership data")
		return {}

	# Aggregate
	all_player_ids = set()
	for d in all_ownership:
		all_player_ids.update(d.keys())

	aggregated: Dict[int, float] = {}
	for pid in all_player_ids:
		values = [d.get(pid, 0.0) for d in all_ownership]
		if aggregate_method == "median":
			aggregated[pid] = float(np.median(values))
		elif aggregate_method == "max":
			aggregated[pid] = float(np.max(values))
		else:
			aggregated[pid] = float(np.mean(values))

	# Persist
	updated = _write_ownership_to_db(session, aggregated)
	logger.info(f"[ownership] ✓ Updated ownership for {updated} players")

	# Log top owned
	top = sorted(aggregated.items(), key=lambda x: x[1], reverse=True)[:10]
	logger.info("[ownership] Top 10 projected ownership:")
	for pid, pct in top:
		p = session.query(NBAPlayer).get(pid)
		if p:
			logger.info(f"[ownership]   {p.name:30s} {pct:6.2f}%")

	return aggregated


def clear_ownership(session: Session) -> int:
	"""Set `NBAPlayer.ownership` to 0.0 for all players. Returns number updated."""
	result = session.query(NBAPlayer).update({NBAPlayer.ownership: 0.0}, synchronize_session=False)
	session.commit()
	logger.info(f"[ownership] Cleared ownership for {result} players")
	return int(result)

