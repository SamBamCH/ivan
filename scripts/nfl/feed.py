#!/usr/bin/env python3
"""
scripts/nfl/feed.py
The NFL reference feed between two Ivans (ivan.core.feed, the command line in ivan.core.feed_cli). The server runs its
own slates on its own working tables (nfl_player, odds, projection_snapshot, lineup, lineup_simulation_results,
contest_sim_report, never touched here) and takes the laptop's reference tables (history, captures, settled contests,
the model pickles under data/nfl/models/) when its desk's "Refresh from laptop" is pressed. Nothing is pushed.
Producer (the laptop):
    scripts/nfl/feed.py publish [--quiet]       write data/feed/nfl/* (the reference bundle + models); overnight.py does this nightly,
                                                 and the consumer's pull asks for a fresh one (deploy/feed_serve.sh)
Consumer (the server):
    scripts/nfl/feed.py pull                    with IVAN_FEED_SSH=<user>@<host> in .env: ssh publish + rsync + apply (feed.pull_ssh)
    scripts/nfl/feed.py apply DIR               seed this database from bundles copied by hand
"""
from __future__ import annotations

from ivan.core import feed_cli

if __name__ == "__main__":
    raise SystemExit(feed_cli.main("nfl", doc=__doc__))
