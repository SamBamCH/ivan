import requests
from datetime import datetime
from dateutil import parser
import pprint
import pandas as pd

from ivan.core.utils import map_team_name_to_abbreviation, normalize_name

def get_game_odds(
    sport: str,
    api_key: str,
    regions: str = "us",
    markets: str = "h2h,spreads,totals",
    oddsFormat: str = "american",
    dateFormat: str = "iso",
    eventIds: str | None = None,
    bookmakers: str | None = None,
    commenceTimeFrom: str | None = None,
    commenceTimeTo: str | None = None,
    includeLinks: str | None = None,
    includeSids: str | None = None,
    includeBetLimits: str | None = None,
):
    """
    Calls /v4/sports/{sport}/odds with the provided params and returns JSON.
    """
    url = f"https://api.the-odds-api.com/v4/sports/{sport}/odds"
    params = {
        "apiKey": api_key,
        "regions": regions,
        "markets": markets,
        "oddsFormat": oddsFormat,
        "dateFormat": dateFormat,
    }
    # Optional filters
    if eventIds:          params["eventIds"] = eventIds
    if bookmakers:        params["bookmakers"] = bookmakers
    if commenceTimeFrom:  params["commenceTimeFrom"] = commenceTimeFrom
    if commenceTimeTo:    params["commenceTimeTo"] = commenceTimeTo
    if includeLinks:      params["includeLinks"] = includeLinks
    if includeSids:       params["includeSids"] = includeSids
    if includeBetLimits:  params["includeBetLimits"] = includeBetLimits

    resp = requests.get(url, params=params)
    if resp.status_code == 200:
        # You can access quota: resp.headers.get('x-requests-remaining'), etc.
        return resp.json(), resp.headers
    else:
        print(f"Error fetching game odds: {resp.status_code} - {resp.text}")
        return None, resp.headers if hasattr(resp, "headers") else {}

# --- NEW: flatten /odds response into per-outcome rows ---
def parse_game_odds(odds_json: list[dict]) -> pd.DataFrame:
    """
    Returns columns:
      game_id, sport_key, commence_time, home_team, away_team, home_abbr, away_abbr,
      bookmaker, market, last_update, outcome_name, side, point, price
    """
    rows: list[dict] = []
    if not odds_json:
        return pd.DataFrame(rows)

    for ev in odds_json:
        game_id       = ev.get("id")
        sport_key     = ev.get("sport_key")
        commence_iso  = ev.get("commence_time")
        commence_dt   = parser.isoparse(commence_iso) if commence_iso else None
        home_team     = ev.get("home_team")
        away_team     = ev.get("away_team")
        home_abbr     = map_team_name_to_abbreviation(home_team) if home_team else None
        away_abbr     = map_team_name_to_abbreviation(away_team) if away_team else None

        for bm in ev.get("bookmakers", []):
            bookmaker    = bm.get("key")
            for mkt in bm.get("markets", []):
                market       = mkt.get("key")     # 'h2h' | 'spreads' | 'totals' | ...
                last_update  = mkt.get("last_update")
                for out in mkt.get("outcomes", []):
                    outcome_name = out.get("name") or out.get("description")  # team for h2h/spreads; 'Over'/'Under' for totals
                    point        = out.get("point")
                    price        = out.get("price")

                    # helper side tag
                    side = None
                    if market == "totals":
                        if isinstance(outcome_name, str):
                            name_lc = outcome_name.lower()
                            if "over" in name_lc:  side = "over"
                            if "under" in name_lc: side = "under"
                    elif market in ("h2h", "spreads"):
                        if outcome_name == home_team:
                            side = "home"
                        elif outcome_name == away_team:
                            side = "away"

                    rows.append({
                        "game_id": game_id,
                        "sport_key": sport_key,
                        "commence_time": commence_dt,
                        "home_team": home_team,
                        "away_team": away_team,
                        "home_abbr": home_abbr,
                        "away_abbr": away_abbr,
                        "bookmaker": bookmaker,
                        "market": market,
                        "last_update": last_update,
                        "outcome_name": outcome_name,
                        "side": side,
                        "point": point,
                        "price": price,
                    })

    return pd.DataFrame(rows)

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

def get_odds(sport, api_key, relevant_teams, markets=None):
    """
    Fetches odds only for each team’s next upcoming event.

    Returns:
        dict: mapping team_abbreviation -> odds JSON
    """
    # 1) grab all upcoming events
    events = _get_upcoming_events(sport, api_key)
    if not events:
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
        odds = _get_event_odds(sport, api_key, ev_id, markets=markets)
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
    

