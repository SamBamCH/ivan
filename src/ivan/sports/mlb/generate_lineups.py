import random
import numpy as np
from scipy.stats import nbinom
import pulp as plp
from datetime import datetime, timedelta
from sqlalchemy import func

from ivan.database.session        import get_session
from ivan.core.utils              import get_config_value
from ivan.sports.mlb.models       import MLBPlayer, Lineup, FieldPlayer
from ivan.sports.mlb.constraints  import Constraints

PITCHER_POS = {"SP", "RP", "P"}


def simulate_uncorrelated_fpts(players, config):
    """
    Sample one uncorrelated fantasy‐point outcome per player,
    but shrink (“disperse”) around the mean based on config.
    """
    # pull dispersion factors out of your config
    pitcher_disp = get_config_value(config, "lineup.pitcher_dispersion", 1.0)
    hitter_disp  = get_config_value(config, "lineup.hitter_dispersion", 1.0)
    simulated = {}
    for p in players:
        if any(pos in PITCHER_POS for pos in p.position):
            # draw raw, then compress toward the projection mean
            raw = np.random.normal(loc=p.fpts_projection, scale=p.stdv)
            simulated[p] = p.fpts_projection + (raw - p.fpts_projection) * pitcher_disp
        else:
            # negative‐binomial raw sample
            raw = nbinom.rvs(p.adj_r, p.adj_p)
            # mean of NB(n,p) = n*(1-p)/p
            mean = p.adj_r * (1 - p.adj_p) / p.adj_p
            # compress deviations toward the mean
            simulated[p] = mean + (raw - mean) * hitter_disp
    return simulated


def generate_single_lineup(
    players,
    config,
    max_retries=3,
    min_projection_sum=None,
    return_ids_only=False
):

    site    = 'dk'  # or config["site"]
    verbose = config.get("verbose", False)

    for attempt in range(1, max_retries + 1):
        simulated_fpts = simulate_uncorrelated_fpts(players, config)
        problem_name  = f"field_lineup_attempt{attempt}"
        problem       = plp.LpProblem(problem_name, plp.LpMaximize)

        # build your variables & constraints just like before
        lp_variables = {
            (pl, pos): plp.LpVariable(f"{pl.name}_{pos}_{pl.id}", cat=plp.LpBinary)
            for pl in players for pos in pl.position
        }
        constraints = Constraints(site, problem, players, lp_variables, config, verbose)
        constraints.add_static_constraints()
        constraints.add_team_stack_constraints()

        obj = plp.lpSum(
            simulated_fpts[pl] * lp_variables[(pl, pos)]
            for pl in players for pos in pl.position
        )
        problem += obj

        if min_projection_sum is not None:
            proj_expr = plp.lpSum(
                pl.fpts_projection * lp_variables[(pl, pos)]
                for pl in players for pos in pl.position
            )
            problem += proj_expr >= min_projection_sum, "MinProjectionConstraint"

        # 1) solve *with* logging turned on
        status_code = problem.solve(plp.GLPK(msg=False))
        status      = plp.LpStatus[status_code]

        # 2) if not Optimal, dump the LP for inspection
        if status != "Optimal":
            fn = f"debug_lineup_{attempt}.lp"
            problem.writeLP(fn)
            print(f"[DEBUG] Wrote LP to {fn} (open it or run `glpsol --check {fn}`)")
            # 3) you can now comment‐out blocks above (static vs. stack vs. projection)
            #    and repeat to see which block causes the infeasibility.
            continue

        # 4) success!
        chosen  = [pl for (pl, _), var in lp_variables.items() if var.varValue == 1]
        proj_sum = sum(pl.fpts_projection for pl in chosen)
        if return_ids_only:
            return [pl.id for pl in chosen]
        return chosen, proj_sum

    # everyone failed
    return None

def generate_and_store_field_lineups(
    session,
    config,
    num_lineups=None,
    batch_size=100
):
    """
    Loads only the FieldPlayer rows from the DB that came in on the most
    recent upload batch, then builds exactly num_lineups of them.
    """
    if num_lineups is None:
        num_lineups = 1000

    # find the most recent upload timestamp
    latest_upload = session.query(func.max(MLBPlayer.created_at)).scalar()
    if latest_upload is None:
        raise RuntimeError("No MLBPlayer rows found in database!")

    # set a tiny epsilon in case your DB timestamps differ by fractions of a second
    epsilon = timedelta(seconds=1)

    # load & wrap only the players from that batch (>= latest_upload - epsilon)
    raw = (
        session.query(MLBPlayer)
               .filter(
                   MLBPlayer.fpts_projection.isnot(None),
                   MLBPlayer.created_at >= (latest_upload - epsilon)
               )
               .all()
    )

    players = []
    for p in raw:
        pos_list = p.position if isinstance(p.position, (list, tuple)) else [p.position]
        players.append(FieldPlayer(
            id              = p.id,
            name            = p.name,
            salary          = p.salary,
            position        = pos_list,
            team            = p.team,
            opponent        = p.opponent,
            fpts_projection = p.fpts_projection,
            stdv            = p.stdv,
            adj_r           = p.adj_r,
            adj_p           = p.adj_p,
        ))

    print(f"initialized {len(players)} players from upload at {latest_upload}")

    created     = 0
    seed        = 0
    batch_count = 0

    while created < num_lineups:
        ids = generate_single_lineup(
            players, config,
            min_projection_sum=get_config_value(config, "lineup.min_fpts_threshold"),
            return_ids_only=True
        )
        seed += 1
        if not ids:
            print("infeasible, trying next seed…")
            continue

        session.add(Lineup(
            player_ids_list=ids,
            created_at=datetime.utcnow()
        ))
        created    += 1
        batch_count += 1

        # every batch_size lineups, flush & commit
        if batch_count >= batch_size:
            session.commit()
            batch_count = 0

    # commit any leftovers
    if batch_count > 0:
        session.commit()

    return created