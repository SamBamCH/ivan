from ivan.database.session import init_db, get_session
from ivan.sports.common.draftkings import ingest_players
from ivan.sports.nba.models import NBAPlayer, NBAOdds, StartingLineups
from ivan.sports.common.odds import get_odds, parse_odds
from ivan.sports.nba.refactor.historical.data_fetch import fetch_starting_lineups
from ivan.sports.nba.refactor.pre_lock.fpts_fitting import fit_fpts_distributions
from ivan.sports.nba.refactor.cli import view_lineups
from datetime import datetime


import argparse, sys
from sqlalchemy import select, func


def parse_args():
    p = argparse.ArgumentParser(description="ingest and update player odds")
    p.add_argument("-p", action='store_true', help='Prelock flag, determines if we need to fetch DK payload or not.')
    return p.parse_args()

#MAIN ENTRY POINT
args = parse_args()
engine = init_db('nba')
session = get_session(engine)

if args.p:  # prelock
    # clear existing odds entirely (fresh slate)
    session.query(NBAOdds).delete(synchronize_session=False)
    #remove old NBAPlayers
    session.query(NBAPlayer).delete(synchronize_session=False)
    #ingest
    players = ingest_players('nba')
    for player in players:
        dbp = NBAPlayer(
            id=player.dk_id,
            name=player.name,
            team=player.team,
            opponent=player.opponent,
            position=player.position,
            roster_slot_id=player.roster_slot_id,
            gametime=player.gametime,
            salary=player.salary,
            status=player.status,
        )
        session.merge(dbp)
    
    session.commit()
    #get_odds
    api_key = '58699e4fe47856c9182db84884772536' #TODO: hide this
    relevant_teams = set(session.scalars(select(NBAPlayer.team).distinct()))
    odds = get_odds("basketball_nba", api_key, list(relevant_teams), markets="player_points,player_rebounds,player_assists,player_threes,player_blocks,player_steals,player_turnovers")
    #parse and upsert odds
    df_odds = parse_odds(odds)
    records = df_odds.to_dict(orient='records')
    session.bulk_insert_mappings(NBAOdds, records)
    session.commit()
    #fetch starting lineups
    fetch_starting_lineups(session)
    #remove out players
    session.query(NBAPlayer).filter(func.upper(NBAPlayer.status) == 'OUT').delete(synchronize_session=False)
    #remove players with no odds
    session.query(NBAPlayer).filter(
        ~NBAPlayer.name.in_(session.query(NBAOdds.player).distinct()),
        ~NBAPlayer.name.in_(session.query(StartingLineups.player_name).distinct())
    ).delete(synchronize_session=False)

    session.commit()

    #project players after cleaning out and useless players
    fit_fpts_distributions(session)
    view_lineups(session)
else:
    # POST-LOCK: Do NOT delete players. Do NOT ingest new players.
    # Only update odds and projections for UNLOCKED games (gametime > now).
    now = datetime.utcnow()

    # Determine unlocked players (whose games have not started)
    unlocked_players = session.query(NBAPlayer).filter(NBAPlayer.gametime > now).all()
    unlocked_player_names = [p.name for p in unlocked_players if p.name]
    unlocked_teams = {p.team for p in unlocked_players if p.team}

    # Refresh odds ONLY for unlocked players (preserve existing odds for locked players)
    if unlocked_teams:
        api_key = '58699e4fe47856c9182db84884772536'  # TODO: secure
        # Delete odds for unlocked players so we can upsert fresh ones
        if unlocked_player_names:
            session.query(NBAOdds).filter(NBAOdds.player.in_(unlocked_player_names)).delete(synchronize_session=False)
        odds = get_odds(
            sport_key="basketball_nba",
            api_key=api_key,
            regions=list(unlocked_teams),  # using teams as regions argument per existing helper pattern
            markets="player_points,player_rebounds,player_assists,player_threes,player_blocks,player_steals,player_turnovers"
        )
        df_odds = parse_odds(odds)
        if not df_odds.empty:
            records = df_odds.to_dict(orient='records')
            session.bulk_insert_mappings(NBAOdds, records)
        session.commit()

    # Update starting lineups (will include unlocked games only; locked games unchanged)
    fetch_starting_lineups(session)

    # IMPORTANT: Do NOT delete OUT players nor those without odds; preserve state.
    # Fit distributions ONLY for unlocked players by skipping started games inside fitting function.
    fit_fpts_distributions(session)
    view_lineups(session)


        


