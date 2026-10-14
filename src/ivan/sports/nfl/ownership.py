import json
from pathlib import Path
#!/usr/bin/env python3
"""
Ownership projection using field sampling simulations.

Strategy:
1. Detect slate type (showdown vs classic) from NFLPlayer roster_slot_id
2. Generate synthetic field using sample_sd_field or sample_classic_field
3. Aggregate player exposures across the simulated field
4. Write ownership percentages back to NFLPlayer table
"""

from typing import Dict, Optional
import logging
import numpy as np
from collections import Counter
from sqlalchemy.orm import Session

from ivan.core.utils import normalize_name
from ivan.core.seed import seed_for
from ivan.sports.nfl.models import NFLPlayer, ProjectionSnapshot

logger = logging.getLogger(__name__)


def _detect_slate_type(session: Session) -> str:
    """
    Detect slate type from roster_slot_id values.
    
    Returns:
        'showdown' if CPT/FLEX slots (511/512) present, else 'classic'
    """
    slots = set(
        session.query(NFLPlayer.roster_slot_id)
        .distinct()
        .filter(NFLPlayer.roster_slot_id.isnot(None))
        .all()
    )
    slot_ids = {s[0] for s in slots if s[0] is not None}
    
    # Showdown slots: CPT=511, FLEX=512
    if 511 in slot_ids or 512 in slot_ids:
        return 'showdown'
    return 'classic'


def _write_ownership_to_db(
    session: Session,
    ownership_by_pid: Dict[int, float],
) -> int:
    """
    Write ownership percentages to NFLPlayer table.

    Creates/updates 'ownership_proj' column (add to models.py if needed).

    Args:
        session: Database session
        ownership_by_pid: {player_id: ownership_pct (0-100)}

    Returns:
        Number of players updated
    """
    updated = 0

    # First, set all players to 0.0 ownership (those not in field)
    session.query(NFLPlayer).update(
        {NFLPlayer.ownership_proj: 0.0},
        synchronize_session=False
    )

    # Then update players that appeared in field
    for pid, own_pct in ownership_by_pid.items():
        result = (
            session.query(NFLPlayer)
            .filter(NFLPlayer.id == pid)
            .update({NFLPlayer.ownership_proj: float(own_pct)})
        )
        updated += result

    session.commit()
    return updated


def _estimate_ownership_from_lineups(
    session: Session,
    slate_type: str,
    num_lineups: int = 1000,
) -> Dict[int, float]:
    """
    Estimate player ownership by generating lineups and counting exposures.
    
    Simple and accurate approach:
    - Generate N lineups using actual lineup generation logic
    - Count how often each player appears
    - Return ownership percentages
    
    Args:
        session: Database session
        slate_type: 'showdown' or 'classic'
        num_lineups: Number of lineups to generate for sampling
    
    Returns:
        {player_id: ownership_pct} where ownership_pct is 0-100
    """
    from ivan.sports.nfl.models import NFLPlayer
    from ivan.sports.nfl.generate_lineups_logic import (
        generate_single_showdown_lineup_ilp,
        generate_single_classic_lineup,
        FieldPlayer,
    )
    import random
    
    # Get all active players
    db_players = session.query(NFLPlayer).filter(NFLPlayer.fpts_mean > 0).all()

    if not db_players:
        logger.warning("[ownership] No players with projections found")
        return {}

    # Convert to FieldPlayer format and collect stddevs
    field_players = []
    roster_slot_by_id = {}
    fpts_mean_by_id = {}
    fpts_std_by_id = {}

    for p in db_players:
        field_players.append(FieldPlayer(
            id=p.id,
            name=p.name,
            team=p.team or '',
            opponent=p.opponent,
            position=p.position or '',
            salary=p.salary,
            fpts_projection=p.fpts_mean,
        ))
        roster_slot_by_id[p.id] = p.roster_slot_id
        fpts_mean_by_id[p.id] = p.fpts_mean
        fpts_std_by_id[p.id] = p.fpts_std if hasattr(p, 'fpts_std') and p.fpts_std is not None else 0.0

    # Configurable CPT multiplier and extra randomness for CPT
    cpt_multiplier = 1.5
    cpt_extra_noise_std = 5.0  # Extra stddev for CPT position (tune as needed)

    # Generate lineups
    player_appearances = Counter()
    successful_lineups = 0

    logger.info(f"[ownership] Generating {num_lineups} lineups to estimate ownership (full sampling)...")

    import numpy as np
    rng = np.random.default_rng(seed_for("ownership/samples"))
    if seed_for("ownership") is not None:            # the ILP helpers below draw their cap drop from `random`
        import random
        random.seed(seed_for("ownership/ilp"))

    for i in range(num_lineups * 2):  # Try up to 2x to get num_lineups successful
        if successful_lineups >= num_lineups:
            break

        # Sample each player's fpts from their normal distribution
        base_scores = {}
        for p in field_players:
            mu = fpts_mean_by_id[p.id]
            sigma = fpts_std_by_id[p.id]
            is_cpt = slate_type == 'showdown' and roster_slot_by_id.get(p.id) == 511
            if is_cpt:
                # Inject extra randomness for CPT slot
                total_sigma = (sigma ** 2 + cpt_extra_noise_std ** 2) ** 0.5
                sample = float(rng.normal(mu, total_sigma))
                base_scores[p.id] = sample * cpt_multiplier
            else:
                sample = float(rng.normal(mu, sigma*0.50 if sigma > 0 else 1.0))
                base_scores[p.id] = sample

        # Generate lineup
        if slate_type == 'showdown':
            lineup_ids = generate_single_showdown_lineup_ilp(
                players=field_players,
                roster_slot_by_id=roster_slot_by_id,
                base_scores=base_scores,
            )
        else:
            lineup_ids = generate_single_classic_lineup(
                players=field_players,
                roster_slot_by_id=roster_slot_by_id,
                base_scores=base_scores,
            )

        if lineup_ids:
            for pid in lineup_ids:
                player_appearances[pid] += 1
            successful_lineups += 1

    if successful_lineups == 0:
        logger.warning("[ownership] No successful lineups generated")
        return {}

    logger.info(f"[ownership] Generated {successful_lineups} successful lineups")

    # Convert to ownership percentages
    ownership = {}
    for pid, count in player_appearances.items():
        ownership[pid] = (count / successful_lineups) * 100

    return ownership


def project_ownership(
    session: Session,
    *,
    config: dict = None,
    num_simulations: int = 1,
    aggregate_method: str = 'mean',
) -> Dict[int, float]:
    """
    Project player ownership by generating sample lineups and counting exposures.
    
    Strategy:
    1. Detect slate type (showdown/classic)
    2. Generate N lineups using actual lineup generation logic
    3. Count player appearances to calculate ownership
    4. Write ownership back to NFLPlayer.ownership_proj
    
    This approach is simple, accurate, and uses the same logic as lineup generation.
    
    Args:
        session: Database session
        config: Configuration dict (optional)
        num_simulations: Number of sampling runs to average (default 1)
        aggregate_method: How to combine sims ('mean', 'median', 'max')
    
    Returns:
        {player_id: ownership_pct} for all players with non-zero ownership
    """
    config = config or {}
    
    # 1. Detect slate type
    slate_type = _detect_slate_type(session)
    logger.info(f"[ownership] Detected slate type: {slate_type}")
    
    # 2. Generate lineups and count exposures
    own_cfg = config.get("ownership", {})
    num_lineups = int(own_cfg.get("num_lineups", config.get("ownership_num_lineups", 1000)))
    
    logger.info(f"[ownership] Generating {num_lineups} lineups to estimate ownership")
    
    all_ownership = []
    
    for sim_idx in range(num_simulations):
        logger.info(f"[ownership] Running simulation {sim_idx + 1}/{num_simulations}")
        
        try:
            ownership = _estimate_ownership_from_lineups(
                session=session,
                slate_type=slate_type,
                num_lineups=num_lineups,
            )
            
            all_ownership.append(ownership)
            
            # Log summary stats
            if ownership:
                values = list(ownership.values())
                logger.info(
                    f"[ownership]   Sim {sim_idx + 1}: "
                    f"{len(ownership)} players, "
                    f"ownership range [{min(values):.2f}%, {max(values):.2f}%], "
                    f"mean={np.mean(values):.2f}%"
                )
        
        except Exception as e:
            logger.error(f"[ownership] Simulation {sim_idx + 1} failed: {e}")
            continue
    
    if not all_ownership:
        logger.error("[ownership] All simulations failed")
        return {}
    
    # 4. Aggregate across simulations
    all_player_ids = set()
    for own_dict in all_ownership:
        all_player_ids.update(own_dict.keys())
    
    aggregated_ownership = {}
    
    for pid in all_player_ids:
        values = [own.get(pid, 0.0) for own in all_ownership]
        
        if aggregate_method == 'median':
            aggregated_ownership[pid] = float(np.median(values))
        elif aggregate_method == 'max':
            aggregated_ownership[pid] = float(np.max(values))
        else:  # mean (default)
            aggregated_ownership[pid] = float(np.mean(values))
    
    logger.info(
        f"[ownership] Aggregated {num_simulations} simulation(s) using {aggregate_method}"
    )
    
    # 5. Write to database
    updated = _write_ownership_to_db(session, aggregated_ownership)
    logger.info(f"[ownership] ✓ Updated ownership for {updated} players")
    
    # Log top owned players for validation
    top_owned = sorted(aggregated_ownership.items(), key=lambda x: x[1], reverse=True)[:10]
    logger.info("[ownership] Top 10 projected ownership:")
    for pid, own in top_owned:
        player = session.query(NFLPlayer).get(pid)
        if player:
            logger.info(f"[ownership]   {player.name:30s} {own:6.2f}%")
    
    return aggregated_ownership


def snapshot_projections(session: Session) -> int:
    """
    Snapshot every NFLPlayer's projection and ownership for this slate's date.
    Training (historical_contest.py) joins these to contest results later, so run it
    after distributions and ownership are final. Re-running replaces the same date.
    """
    players = session.query(NFLPlayer).all()
    if not players:
        logger.warning("[snapshot] no players in nfl_player; nothing to snapshot")
        return 0
    snapshot_date = min(p.gametime for p in players if p.gametime)
    records, names = [], set()
    for p in players:
        norm = normalize_name(p.name)
        names.add(norm)
        records.append({
            "id": p.id, "snapshot_date": snapshot_date, "player_name": p.name, "normalized_name": norm,
            "team": p.team, "opponent": p.opponent, "position": p.position, "salary": p.salary,
            "roster_slot_id": p.roster_slot_id,
            "fpts_mean": p.fpts_mean or 0.0, "fpts_median": p.fpts_p50 or 0.0, "fpts_std": p.fpts_std or 0.0,
            "fpts_std_model": p.fpts_std_model,
            "ownership_proj": p.ownership_proj or 0.0,
        })
    (session.query(ProjectionSnapshot)
            .filter(ProjectionSnapshot.snapshot_date == snapshot_date,
                    ProjectionSnapshot.normalized_name.in_(names))
            .delete(synchronize_session=False))
    session.bulk_insert_mappings(ProjectionSnapshot, records)
    session.commit()
    logger.info(f"[snapshot] wrote {len(records)} projection rows for {snapshot_date.date()}")
    return len(records)


def project_and_snapshot(
    session: Session,
    config: dict,
    *,
    num_lineups: Optional[int] = None,
    num_simulations: Optional[int] = None,
    snapshot: bool = True,
) -> Dict[int, float]:
    """
    The pre-lock ownership step: project ownership from field sampling using
    config["ownership"] (num_lineups, num_simulations), optionally overridden,
    then snapshot projections for training. Raises if nothing was projected so a
    silent all-zero ownership can't slip through to the simulator.
    """
    own_cfg = dict(config.get("ownership", {}))
    if num_lineups is not None:
        own_cfg["num_lineups"] = num_lineups
    sims = int(num_simulations if num_simulations is not None else own_cfg.get("num_simulations", 1))
    ownership = project_ownership(session, config={**config, "ownership": own_cfg}, num_simulations=sims)
    if not ownership:
        raise RuntimeError("ownership projection produced no players; see log above")
    mode = str(own_cfg.get("model", "learned")).lower()
    applied = False
    if mode == "learned":
        applied = apply_learned_ownership(session)
        ownership = {int(p.id): float(p.ownership_proj or 0.0) for p in session.query(NFLPlayer).all()} if applied else ownership
    if not applied and mode in ("learned", "sampler_calibrated") and (own_cfg.get("calibration") or {}).get("enabled", True):
        ownership = calibrate_ownership(session, ownership)
    if snapshot:
        snapshot_projections(session)
    return ownership


SLOT_TOTAL = {511: 100.0, 512: 500.0}        # showdown: one captain and five flex per lineup


def apply_learned_ownership(session: Session, model=None) -> bool:
    """
    Replace the sampled ownership with the learned model's prediction
    (ivan.sports.nfl.ownership_model: the sampler's projection plus points,
    salary, ranks, position and slot, fitted on scraped fields). Showdown
    slates only; returns False when no model file exists so the caller can
    fall back to the per-position calibration.
    """
    from ivan.sports.nfl import ownership_model as OM
    model = OM.load() if model is None else model
    if model is None:
        return False
    players = session.query(NFLPlayer).all()
    if not players or not {int(p.roster_slot_id) for p in players} <= set(SLOT_TOTAL):
        return False
    d = OM.slate_frame(session, players)
    pred = model.predict_frame(d)
    before = {int(p.id): float(p.ownership_proj or 0.0) for p in players}
    by_id = dict(zip(d.player_id.astype(int), pred))
    changed = []
    for p in players:
        new = round(float(by_id.get(int(p.id), before[int(p.id)])), 2)
        if abs(new - before[int(p.id)]) >= 2.0:
            changed.append((p.name, p.position, int(p.roster_slot_id), before[int(p.id)], new))
        p.ownership_proj = new
    session.commit()
    print(f"learned ownership model ({model.n_contests} contests, fitted {model.fitted_on}{', cutoff ' + model.cutoff if model.cutoff else ''}): "
          f"{len(changed)} moves of 2+ pts: " + "; ".join(f"{n} {pos} {'CPT' if slot == 511 else 'FLEX'} {b:.1f}->{a:.1f}"
                                                         for n, pos, slot, b, a in sorted(changed, key=lambda t: -abs(t[4] - t[3]))[:12]))
    return True


def load_ownership_calibration(model_dir=None) -> dict:
    from ivan.sports.nfl.paths import models_dir
    path = Path(models_dir() if model_dir is None else model_dir) / "ownership_calibration.json"
    try:
        return json.loads(path.read_text()) if path.exists() else {}
    except (OSError, ValueError):
        return {}


def calibrate_ownership(session: Session, ownership: Dict[int, float], cal: Optional[dict] = None) -> Dict[int, float]:
    """
    Map the field-sampling projection onto what the public actually does, per
    roster slot and position, with the lines fitted on scraped showdown
    contests (scripts/nfl/build_ownership_calibration.py): most positions
    on the projected ownership itself, DST and K on projected points (the
    public rosters them by projection, not by value). Each slot is then
    renormalised to its lineup total (captain 100, flex 500). Classic slates
    and slates without a calibration file are left alone.
    """
    cal = load_ownership_calibration() if cal is None else cal
    maps = (cal or {}).get("maps") or {}
    if not maps:
        return ownership
    players = session.query(NFLPlayer).all()
    slots = {int(p.roster_slot_id) for p in players}
    if not slots <= set(SLOT_TOTAL):
        return ownership
    before = {int(p.id): float(p.ownership_proj or 0.0) for p in players}
    raw_new: Dict[int, float] = {}
    for p in players:
        m = maps.get(f"{int(p.roster_slot_id)}:{p.position}")
        own = before[int(p.id)]
        if m is None:
            raw_new[int(p.id)] = own
            continue
        x = float(p.fpts_mean or 0.0) if m["on"] == "fpts_proj" else own
        raw_new[int(p.id)] = max(0.0, float(m["a"]) + float(m["b"]) * x)
    # renormalise per slot so the totals still add to one lineup's worth
    for slot, total in SLOT_TOTAL.items():
        ids = [int(p.id) for p in players if int(p.roster_slot_id) == slot]
        ssum = sum(raw_new[i] for i in ids)
        if ssum > 0:
            for i in ids:
                raw_new[i] = raw_new[i] * total / ssum
    changed = []
    for p in players:
        new = round(raw_new[int(p.id)], 2)
        if abs(new - before[int(p.id)]) >= 2.0:
            changed.append((p.name, p.position, int(p.roster_slot_id), before[int(p.id)], new))
        p.ownership_proj = new
    session.commit()
    if changed:
        print(f"ownership calibration ({cal.get('contests')} contests, fitted {cal.get('fitted_on')}): {len(changed)} moves of 2+ pts: "
              + "; ".join(f"{n} {pos} {'CPT' if slot == 511 else 'FLEX'} {b:.1f}->{a:.1f}" for n, pos, slot, b, a in sorted(changed, key=lambda t: -abs(t[4] - t[3]))[:12]))
    return {int(p.id): float(p.ownership_proj) for p in players}
