import requests
from datetime import datetime
from dateutil import parser
import pprint
import pandas as pd

from ivan.core.utils import map_team_name_to_abbreviation, normalize_name

def _get_upcoming_events(sport, api_key):
    """
    Fetches all upcoming events for the specified sport.

    Returns:
        list: A list of upcoming events or None if the request fails.
    """
    url = f"https://api.the-odds-api.com/v4/sports/{sport}/events"
    params = {
        'apiKey': api_key,
        'dateFormat': 'iso',
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error fetching upcoming events: {response.status_code} - {response.text}")
        return None


def _get_event_odds(sport, api_key, event_id, markets=None):
    """
    Fetches odds for a specific event.

    Args:
        event_id (str): The ID of the event for which odds are being fetched.
        markets (str): The markets to include. If None, default markets are used.

    Returns:
        dict: A JSON response containing odds data or None if the request fails.
    """
    if markets is None:
        markets = (
            "player_pass_tds,player_pass_yds,player_pass_completions,"
            "player_pass_attempts,player_pass_interceptions,"
            "player_rush_yds,player_rush_attempts,"
            "player_receptions,player_reception_yds,player_anytime_td"
        )
    url = f"https://api.the-odds-api.com/v4/sports/{sport}/events/{event_id}/odds"
    params = {
        'apiKey': api_key,
        'regions': 'us',
        'markets': markets,
        'oddsFormat': 'american'
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error fetching event odds: {response.status_code} - {response.text}")
        return None

def get_odds(sport, api_key, relevant_teams):
    """
    Fetches odds only for each team’s next upcoming event.

    Returns:
        dict: mapping team_abbreviation -> odds JSON
    """
    # 1) grab all upcoming events
    events = _get_upcoming_events(sport, api_key)
    if not events:
        print("No upcoming events found.")
        return {}

    # 2) sort them by start time (earliest first)
    events = sorted(
        events,
        key=lambda ev: parser.isoparse(ev['commence_time'])
    )

    # 3) pick the first event for each relevant team
    next_event_by_team: dict[str, dict] = {}
    for ev in events:
        for side in ('home_team', 'away_team'):
            team_abbr = map_team_name_to_abbreviation(ev[side])
            if team_abbr in relevant_teams and team_abbr not in next_event_by_team:
                next_event_by_team[team_abbr] = ev

        # if we already have all teams covered, break early
        if len(next_event_by_team) == len(relevant_teams):
            break

    # 4) fetch odds just for those next events
    odds_by_team: dict[str, dict] = {}
    for team, ev in next_event_by_team.items():
        ev_id = ev['id']
        odds = _get_event_odds(sport, api_key, ev_id)
        odds_by_team[team] = odds

    return odds_by_team

def parse_odds(odds_by_team: dict) -> pd.DataFrame:
    """
    Flatten the nested odds JSON into a DataFrame with columns:
    player (pk), market, sportsbook, side, point, price, last_update
    """
    rows = []
    for _, ev in odds_by_team.items():
        if not ev or not ev.get("bookmakers"):
            continue

        for bm in ev["bookmakers"]:
            sportsbook = bm["key"]
            for market in bm["markets"]:
                market_key = market["key"]
                last_update = market["last_update"]
                for outcome in market["outcomes"]:
                    player_name = outcome["description"]
                    normalized = normalize_name(player_name)
                    side = outcome.get("name", "Over")
                    # default point for anytime TD bets
                    point = outcome.get("point", 0.5) if market_key == "player_anytime_td" else outcome.get("point")
                    price = outcome["price"]

                    rows.append({
                        "player": normalized,
                        "market": market_key,
                        "sportsbook": sportsbook,
                        "side": side,
                        "point": point,
                        "price": price,
                        "last_update": last_update
                    })
    df = pd.DataFrame(rows)
    return df
    

