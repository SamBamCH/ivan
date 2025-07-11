#!/usr/bin/env python
import os
import csv
import json
import pickle
from pathlib import Path
from datetime import datetime
from sqlalchemy import delete
import numpy as np

from ivan.database.session import init_db, get_session, Base
from ivan.sports.mlb.models import (
    BatterProjection,
    PitcherProjection,
    MLBPlayer,
    MLBStat
)
from ivan.sports.mlb.ingest import ingest_projections
from ivan.core.utils import parse_game_time, PROJECT_ROOT, normalize_name

CONFIG_PATH = os.path.join(PROJECT_ROOT, 'config', 'mlb.json')


def init_players(batch_ts: datetime):
    print(f'batch_ts: {batch_ts}')
    cfg    = json.load(open(CONFIG_PATH, encoding='utf-8'))
    engine = init_db('mlb', cfg.get('database_url'))
    Base.metadata.create_all(engine)
    sess   = get_session(engine)

    sess.execute(delete(MLBPlayer))
    sess.commit()

    model_path = Path(PROJECT_ROOT) / 'data' / 'mlb' / 'models' / 'nb_model.pkl'
    with open(model_path, 'rb') as mf:
        loaded = pickle.load(mf)
        hist_model = loaded[0] if isinstance(loaded, tuple) else loaded

    pid_fp = cfg['paths']['ids']
    missing_gameinfo = []
    missing_projections = []
    initialized_count = 0

    with open(pid_fp, encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            name   = normalize_name(row.get('Name', ''))
            raw_gi = row.get('Game Info', '')
            try:
                gametime = parse_game_time(raw_gi)
            except ValueError:
                missing_gameinfo.append((name, raw_gi))
                continue

            position = 'P' if row.get('Position') in ('SP', 'RP') else row.get('Position')
            team     = row.get('TeamAbbrev', '')
            salary   = int(row.get('Salary') or 0)
            id       = row.get('ID')

            p = MLBPlayer(
                id              = id,
                name            = name,
                team            = team,
                gametime        = gametime,
                salary          = salary,
                position        = position,
                created_at      = batch_ts
            )

            bp = sess.query(BatterProjection).filter(
                BatterProjection.name       == name,
                BatterProjection.team       == team,
                BatterProjection.created_at == batch_ts
            ).one_or_none()
            pp = sess.query(PitcherProjection).filter(
                PitcherProjection.player    == name,
                PitcherProjection.team      == team,
                PitcherProjection.created_at== batch_ts
            ).one_or_none()
            
            # assign non-P projections
            if position != 'P' and bp:
                p.opponent        = bp.opp_tm
                p.fpts_projection = bp.fpts
                p.ownership       = bp.pown
                p.lp              = bp.lp
                p.ipl             = bp.ipl
                p.doubles_2h      = bp.doubles_2h

            # assign P projections—and enforce std dev
            elif position == 'P' and pp:
                p.opponent        = pp.opp_tm
                p.fpts_projection = pp.fpts
                p.ownership       = pp.pown
                p.lp              = None
                p.ipl             = None
                p.doubles_2h      = pp.doubles_2h

                # calculate pitcher stdv from mlb_stats history
                hist = (
                    sess.query(MLBStat.dk_fpts)
                        .filter(
                            MLBStat.name.in_([name]),
                            MLBStat.position.in_(["SP","RP","P"])
                        )
                        .all()
                )
                vals = [v[0] for v in hist if v[0] is not None]
                if len(vals) < 2:
                    p.stdv = p.fpts_projection / 2
                else:
                    p.stdv = float(np.std(vals, ddof=1))
            else:
                missing_projections.append((name, position))
                continue

            # enrich & save
            p.enrich(sess, hist_model)
            sess.merge(p)
            initialized_count += 1

    sess.commit()

        # 8) Summary
    print(f"Skipped {len(missing_gameinfo)} rows due to unparsable Game Info.")
    print(f"Missing projections for {len(missing_projections)} names.")
    print(f"✅ Initialized {initialized_count} players into mlb_player table.")

    # ─── 9) Roster sanity check ────────────────────────────────────────────────
    from sqlalchemy import distinct
    teams = sess.query(distinct(MLBPlayer.team))\
                .filter(MLBPlayer.created_at == batch_ts)\
                .all()

    for (team,) in teams:
        roster = sess.query(MLBPlayer)\
                     .filter(
                         MLBPlayer.created_at == batch_ts,
                         MLBPlayer.team == team
                     )\
                     .all()

        batters  = [p for p in roster if p.position != 'P']
        pitchers = [p for p in roster if p.position == 'P']

        # if not exactly 9 batters + 1 pitcher, print out what we do have
        if len(batters) != 9 or len(pitchers) != 1:
            print(f"\n⚠️  Team {team!r} has {len(batters)} batters and {len(pitchers)} pitchers:")
            # sort batters by lineup position (lp), defaulting missing 'lp' to 0
            batters_sorted = sorted(batters, key=lambda p: p.lp or 0)
            for p in batters_sorted:
                print(f"    Batter ({p.lp}): {p.name}")
            # then list pitcher(s)
            for p in pitchers:
                print(f"    Pitcher:        {p.name}")

    print(f'Expected {len(teams) * 10} players based on {len(teams) / 2} games being on the slate.')


if __name__ == '__main__':
    batch_ts = ingest_projections()
    init_players(batch_ts)
