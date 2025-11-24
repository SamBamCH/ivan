from ivan.database.session import init_db, get_session
from ivan.sports.common.odds import get_odds, parse_odds
from ivan.sports.nba.models import NBAPlayer, NBAOdds
from ivan.sports.nba.fpts_calculations import fit_fpts_distributions
from ivan.sports.nba.ownership import project_ownership
from ivan.sports.nba.refactor.historical.data_fetch import fetch_starting_lineups
from ivan.sports.nba.refactor.cli import view_lineups
from sqlalchemy import select
from rich.console import Console

console = Console()

engine = init_db('nba')
session = get_session(engine)

api_key = '58699e4fe47856c9182db84884772536'
relevant_teams = set(session.scalars(select(NBAPlayer.team).distinct()))
markets = (
        "player_points,player_rebounds,player_assists,player_threes,"
        "player_blocks,player_steals,player_turnovers"
    )

odds = get_odds("basketball_nba", api_key, list(relevant_teams), markets=markets)
df_odds = parse_odds(odds)
# This prevents stale odds from accumulating and allows clean re-runs
if not df_odds.empty:
    deleted = session.query(NBAOdds).delete(synchronize_session=False)
    console.print(f"[yellow]Cleared existing NBAOdds records[/yellow] count={deleted} (reason: updating to latest odds snapshot)")
    session.commit()

records = df_odds.to_dict(orient='records')
if not records:
    console.print("[red]No odds records parsed; skipping odds insert.[/red]")
else:
    session.bulk_insert_mappings(NBAOdds, records)
    session.commit()
    console.print(f"[green]Inserted updated odds records[/green] count={len(records)}")

fetch_starting_lineups(session)  # ensure starters loaded before fitting

fit_fpts_distributions(session, n_samples=10000)

project_ownership(session)

# Prompt for manual projections for any confirmed starters missing projections or records
view_lineups(session, prompt_missing=True)