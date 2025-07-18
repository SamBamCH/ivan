import random
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict

from sqlalchemy import func
from ivan.database.session    import get_session
from ivan.core.utils          import get_config_value
from ivan.sports.lol.models   import LOLPlayer, FieldPlayer, Lineup

def generate_and_store_lol_field_lineups(
    session,
    config,
    num_lineups: int = None,
    batch_size:   int = 100
) -> int:
    # 1) how many?
    if num_lineups is None:
        num_lineups = get_config_value(config, "num_field_lineups", 10000)

    # 2) load latest LOLPlayer batch
    latest_upload = session.query(func.max(LOLPlayer.created_at)).scalar()
    if latest_upload is None:
        raise RuntimeError("No LOLPlayer rows found!")
    epsilon = timedelta(seconds=1)
    raws = (
        session.query(LOLPlayer)
               .filter(
                   LOLPlayer.odds.isnot(None),
                   LOLPlayer.created_at >= (latest_upload - epsilon)
               )
               .all()
    )

    # wrap into DTO
    players = [
        FieldPlayer(
            id              = p.id,
            name            = p.name,
            team            = p.team,
            opponent        = p.opponent,
            gametime            = p.gametime,
            salary          = p.salary,
            game_position   = p.game_position,
            roster_position = p.roster_position,
            odds            = p.odds
        )
        for p in raws
    ]
    print(f"Loaded {len(players)} players (with odds) from {latest_upload}")

    # 3) build implied-win probabilities from moneyline odds
    def ml_to_prob(odds: float) -> float:
        # negative odds (favorites): e.g. -150 → 150/(150+100)=0.60
        if odds < 0:
            return -odds / (-odds + 100)
        # positive odds (underdogs): e.g. +200 → 100/(200+100)=0.333
        return 100.0 / (odds + 100.0)

    probs    = np.array([ml_to_prob(p.odds) for p in players], dtype=float)
    salaries = np.array([p.salary           for p in players], dtype=float)

    # exponents to tune chalkiness
    alpha = get_config_value(config, "odds_alpha",       1.0)  # >1 peaks favorites, <1 flattens
    gamma = get_config_value(config, "salary_gamma",    0.2)  # tilt toward cheaper players

    raw_own = np.power(probs, alpha) * np.power(1.0 / salaries, gamma)
    if raw_own.sum() <= 0:
        raw_own = np.ones_like(raw_own)
    own_dist  = raw_own / raw_own.sum()
    target_own = { p: own_dist[i] for i,p in enumerate(players) }

    # 4) bucket by roster_position
    by_pos = defaultdict(list)
    for p in players:
        by_pos[p.roster_position].append(p)

    # 5) constraints from config
    min_salary = get_config_value(config, "min_lineup_salary", 48000)
    max_salary = get_config_value(config, "max_lineup_salary", 50000)
    min_games  = 2
    max_team   = 4
    stack_mix  = get_config_value(config, "stack_config_percentages", {
                     "4_3": 0.90, "4_x": 0.08, "3_3_1": 0.02
                 })

    def valid_stack(counts, stype):
        if stype == "4_3":   return 4 in counts and 3 in counts
        if stype == "4_x":   return counts[0] == 4
        if stype == "3_3_1": return counts[:3] == [3,3,1]
        return False

    # 6) sampling loop
    created = batch = attempts = 0
    while created < num_lineups:
        attempts += 1
        stype = random.choices(list(stack_mix), weights=list(stack_mix.values()), k=1)[0]

        # sample one CPT + one of each: TOP, JNG, MID, ADC, SUP, TEAM
        lineup = []
        for slot in ("CPT","TOP","JNG","MID","ADC","SUP","TEAM"):
            pool = by_pos[slot]
            w    = [target_own[p] for p in pool]
            pick = random.choices(pool, weights=w or None, k=1)[0]
            lineup.append(pick)

        # rejects
        if len({p.id for p in lineup}) < len(lineup):             # duplicates
            print("lineup failed due to duplicate players")
            continue
        sal = sum(p.salary for p in lineup)
        if sal < min_salary or sal > max_salary:                  # salary bounds
            print("lineup failed due to salary constraints")
            continue
        games = {p.gametime for p in lineup if p.gametime is not None}
        if len(games) < min_games:                                # distinct games
            print("lineup failed due to game constraint")
            continue
        counts = defaultdict(int)
        for p in lineup: counts[p.team]+=1
        if max(counts.values()) > max_team:                       # max per team
            print("lineup failed due to max player constraint")
            continue
        if not valid_stack(sorted(counts.values(),reverse=True), stype):
            print("lineup failed due to valid stack constraint")
            continue

        # passed → persist
        session.add(Lineup(player_ids_list=[p.id for p in lineup],
                           created_at=datetime.utcnow()))
        created += 1
        batch   += 1

        if batch >= batch_size:
            session.commit()
            batch = 0

        if attempts > num_lineups * 200:
            raise RuntimeError(
                f"Only generated {created}/{num_lineups} lineups after "
                f"{attempts} attempts—check constraints."
            )

    if batch:
        session.commit()

    print(f"✅ Generated {created} LoL lineups in {attempts} attempts.")
    return created
