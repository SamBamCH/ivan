import requests
from datetime import datetime


def get_upcoming_events(sport, api_key):
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

def filter_events_by_first_event_date(events):
    """
    Filters events that occur on the same date as the first event.

    Args:
        events (list): A list of events.

    Returns:
        list: A list of events that occur on the same date as the first event.
    """
    if not events:
        return []

    # Get the date of the first event
    first_event_date = datetime.strptime(events[0]['commence_time'], '%Y-%m-%dT%H:%M:%SZ').date()

    # Filter events that occur on the same date as the first event
    filtered_events = [
        event for event in events
        if datetime.strptime(event['commence_time'], '%Y-%m-%dT%H:%M:%SZ').date() == first_event_date
    ]

    return filtered_events

def get_event_odds(sport, api_key, event_id, markets=None):
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

def get_odds(sport, api_key):
    """
    Fetches odds for the first event and other events on the same day.

    Returns:
        dict: A JSON response containing odds data for the events or None if any step fails.
    """
    # Get all upcoming events
    events = get_upcoming_events(sport, api_key)
    if not events:
        print("No upcoming events found.")
        return None

    # Filter events to include only those on the same date as the first event
    filtered_events = filter_events_by_first_event_date(events)
    if not filtered_events:
        print("No events found for the same date as the first event.")
        return None

    # Fetch odds for each filtered event
    odds_data = {}
    for event in filtered_events:
        event_id = event['id']
        event_odds = get_event_odds(sport, api_key, event_id)
        if event_odds:
            odds_data[event_id] = event_odds

    return odds_data