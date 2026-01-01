from ivan.database.session import get_session, init_db
from ivan.sports.nfl.models import NFLPlayer
from ivan.sports.common.draftkings import ingest_players
from ivan.core.utils import normalize_name
import pandas as pd
#connect to DB
engine = init_db('nfl')
session = get_session(engine)
#clear old data
#session.query(NFLPlayer).delete(synchronize_session=False)
#ingest players from DK
#players = ingest_players('nfl', dedupe_by_name=False)
#for player in players:
#    dbp = NFLPlayer(
#        idplayer.dk_id,
#       name=player.name,
#        team=player.team,
#        opponent=player.opponent,
#        position=player.position,
#        roster_slot_id=player.roster_slot_id,
#        gametime=player.gametime,
#        salary=player.salary,
#        status=player.status
#    )
#    session.merge(dbp)

#session.commit()
#read stok csv
df = pd.read_csv('data/nfl/input/stok.csv')

#loop through rows of csv and append values to player objects

for idx, row in df.iterrows():
    raw_name = row.get('Player')
    norm_name = normalize_name(raw_name)
    players = session.query(NFLPlayer).filter(NFLPlayer.name == norm_name).all()
    slot_map = {66: None, 67: None, 68: None, 69: None, 71: None, 511: None, 512: None}
    for p in players:
        if p.roster_slot_id in slot_map:
            slot_map[p.roster_slot_id] = p

    projection = row.get('Projection')
    for slot_id in (66, 67, 68, 69, 71, 511, 512):
        player = slot_map.get(slot_id)
        if player is None:
            continue

        player.fpts_mean = row.get('Projection')
        player.fpts_std = row.get('Std Dev')
        player.fpts_dist_name = 'norm'
        player.fpts_dist_params = [row.get('Projection'), row.get('Std Dev')]
        if slot_id == 511:
            player.ownership_proj = row.get('CPT Ownership %')
        else:   
            player.ownership_proj = row.get('Ownership %')

            session.add(player)

session.commit()

session.query(NFLPlayer).filter(NFLPlayer.fpts_mean == 0).delete(synchronize_session=False)
session.commit()
