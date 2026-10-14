import requests

from ivan.core import raw
from dateutil import parser
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
    raw_sport=None,
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
        payload = resp.json()
        if raw_sport and payload:
            raw.store(raw_sport, "odds/lines", sport, payload, meta={"markets": markets})
        return payload, resp.headers
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

# The NFL player markets the ingest prices (the odds API bills per market, so the replay backfill asks for exactly these)
DEFAULT_PLAYER_MARKETS = (
    "player_pass_tds,player_pass_yds,player_pass_completions,"
    "player_pass_attempts,player_pass_interceptions,"
    "player_rush_yds,player_rush_attempts,"
    "player_receptions,player_reception_yds,player_anytime_td,"
    "player_defensive_interceptions,player_field_goals,player_kicking_points,player_kicking_points_alternate,"
    "player_field_goals_alternate,player_receptions_alternate,player_sacks,player_sacks_alternate,player_reception_yds_alternate,"
    "player_rush_tds_alternate,player_rush_yds_alternate,player_pass_tds_alternate,player_pass_yds_alternate"
)


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
    if response.status_code in (401, 403):
        # A dead or expired key is not transient; never let callers continue as if there were no games.
        raise RuntimeError(f"the-odds-api rejected the API key ({response.status_code}): {response.text[:200]}")
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
        markets = DEFAULT_PLAYER_MARKETS
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

def get_odds(sport, api_key, relevant_teams, markets=None, raw_sport=None):
    """
    Fetches odds only for each team’s next upcoming event. With raw_sport the
    events list and every event's odds response are kept as received (raw odds/events, odds/props).

    Returns:
        dict: mapping team_abbreviation -> odds JSON
    """
    # 1) grab all upcoming events
    events = _get_upcoming_events(sport, api_key)
    stamp = raw._stamp()
    if raw_sport and events:
        raw.store(raw_sport, "odds/events", sport, events, stamp=stamp)
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
        if raw_sport and odds:
            raw.store(raw_sport, "odds/props", ev_id, odds, stamp=stamp, meta={"team": team, "markets": markets, "commence_time": ev.get("commence_time")})
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
    



# --------------------------------------------------------------------------- historical (replay backfill)
HISTORICAL_BASE = "https://api.the-odds-api.com/v4/historical/sports"


def _historical_get(url: str, params: dict):
    """GET a historical endpoint; returns (data, snapshot timestamp, headers). Raises on a dead key."""
    resp = requests.get(url, params=params)
    if resp.status_code in (401, 403):
        raise RuntimeError(f"the-odds-api rejected the API key ({resp.status_code}): {resp.text[:200]}")
    if resp.status_code != 200:
        print(f"Error fetching {url}: {resp.status_code} - {resp.text[:200]}")
        return None, None, resp.headers
    body = resp.json()
    return body.get("data"), body.get("timestamp"), resp.headers


def get_historical_events(sport: str, api_key: str, at: str):
    """
    Events as they stood at `at` (ISO 8601 UTC, e.g. 2025-11-24T00:50:00Z):
    the snapshot the API has at or before that time. Costs 1 credit.
    Returns (events, snapshot_timestamp, headers).
    """
    return _historical_get(f"{HISTORICAL_BASE}/{sport}/events", {"apiKey": api_key, "date": at, "dateFormat": "iso"})


def get_historical_event_odds(sport: str, api_key: str, event_id: str, at: str, markets: str | None = None,
                              regions: str = "us", bookmakers: str | None = None):
    """
    One event's odds (player props included) as they stood at `at`. The API
    bills historical odds per market per region, so `markets` should be the
    list the live ingest asks for and nothing more. Returns
    (event_odds, snapshot_timestamp, headers); event_odds has the same shape
    as the live /events/{id}/odds response, so parse_odds() applies.
    """
    if markets is None:
        markets = DEFAULT_PLAYER_MARKETS
    params = {"apiKey": api_key, "regions": regions, "markets": markets, "oddsFormat": "american",
              "dateFormat": "iso", "date": at}
    if bookmakers:
        params["bookmakers"] = bookmakers
    return _historical_get(f"{HISTORICAL_BASE}/{sport}/events/{event_id}/odds", params)


def get_historical_game_odds(sport: str, api_key: str, at: str, markets: str = "h2h,spreads,totals", regions: str = "us"):
    """Every event's game lines as they stood at `at`; parse with parse_game_odds(). Returns (events, timestamp, headers)."""
    params = {"apiKey": api_key, "regions": regions, "markets": markets, "oddsFormat": "american",
              "dateFormat": "iso", "date": at}
    return _historical_get(f"{HISTORICAL_BASE}/{sport}/odds", params)


def credits_remaining(headers) -> tuple[int | None, int | None]:
    """(remaining, used) from the API's quota headers, or Nones."""
    try:
        return int(headers.get("x-requests-remaining")), int(headers.get("x-requests-used"))
    except (TypeError, ValueError, AttributeError):
        return None, None
