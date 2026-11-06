"""The slate's label in the bar and the picker: lock time · games · DK's slate title (ingest.slate_title, desk.Lobby.slate_label)."""
from ivan.sports.nba.ingest import slate_title
from ivan.sports.nba.desk import Lobby


def test_slate_title_strips_the_lobby_suffix():
    assert slate_title(" (Early Only)") == "Early Only"
    assert slate_title("(Turbo)") == "Turbo"
    assert slate_title(" (ORL @ PHX)") == "ORL @ PHX"
    assert slate_title(None) is None and slate_title("") is None and slate_title("  ") is None


def test_slate_label_reads_time_games_title():
    assert Lobby.slate_label("19:00", 8, "Early Only") == "19:00 · 8 games · Early Only"
    assert Lobby.slate_label("19:00", 8, None) == "19:00 · 8 games"
    assert Lobby.slate_label("21:00", 1, "ORL @ PHX", "archived") == "21:00 · 1 game · ORL @ PHX · archived"
    assert Lobby.slate_label("held", None) == "held"
