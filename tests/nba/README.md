# NBA Stats Fetcher

This script fetches NBA player box score data using the `nba_api` package and stores it in a SQLite database.

## API Used
**nba_api** - Free, no rate limits, official NBA.com data

## Installation
```bash
pip install nba_api
```

## Database Location
Data is stored in: `/data/nba/nba_history.sqlite`

## Usage Examples

### Fetch a specific date range
Edit `tests/nba/get_stats.py` and modify the main block:

```python
if __name__ == "__main__":
    # Example: Jan 1-7, 2024
    start = datetime(2024, 1, 1)
    end = datetime(2024, 1, 7)
    main(start, end, season="2023-24")
```

### Fetch entire 2024 calendar year
```python
if __name__ == "__main__":
    start = datetime(2024, 1, 1)
    end = datetime(2024, 12, 31)
    main(start, end, season="2023-24")
```

### Fetch the 2024-25 season (Oct 2024 - Apr 2025)
```python
if __name__ == "__main__":
    start = datetime(2024, 10, 1)
    end = datetime(2025, 4, 30)
    main(start, end, season="2024-25")
```

### Update with latest games (run daily during season)
```python
from datetime import datetime, timedelta

if __name__ == "__main__":
    # Fetch last 3 days to catch any games
    end = datetime.now()
    start = end - timedelta(days=3)
    main(start, end, season="2024-25")
```

## Box Score Data Included

- **Player Info**: player_id, player_name, team
- **Game Info**: game_id, game_date, opponent, is_home
- **Box Stats**: points, rebounds, assists, steals, blocks, turnovers, minutes
- **Shooting**: fgm, fga, fg_pct, fg3m, fg3a, fg3_pct, ftm, fta, ft_pct
- **Advanced**: plus_minus
- **Fantasy**: dk_fpts (DraftKings scoring)

## DraftKings Scoring Formula
- Points: 1.0
- 3-pointers made: 0.5
- Rebounds: 1.25
- Assists: 1.5
- Steals: 2.0
- Blocks: 2.0
- Turnovers: -0.5
- Double-double bonus: +1.5
- Triple-double bonus: +3.0 (total)

## Rate Limiting
The script includes a 0.6 second delay between requests (~100 req/min) to respect NBA API rate limits.

## Querying the Data

```python
from ivan.database.session import init_db, get_session
from ivan.sports.nba.models import NBAStat
from datetime import date

engine = init_db('nba')
session = get_session(engine)

# Get all stats for a specific player
lebron_stats = session.query(NBAStat).filter(
    NBAStat.player_name == 'LeBron James'
).all()

# Get stats for a specific date
jan_1_stats = session.query(NBAStat).filter(
    NBAStat.game_date == date(2024, 1, 1)
).all()

# Get high-scoring games (30+ points)
big_games = session.query(NBAStat).filter(
    NBAStat.points >= 30
).all()

session.close()
```

## Upsert Behavior
The script uses upsert logic - if a game+player combination already exists, it will update the record. This makes it safe to re-run for the same date range.
