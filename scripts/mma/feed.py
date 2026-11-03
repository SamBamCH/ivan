#!/usr/bin/env python3
"""
scripts/mma/feed.py
The MMA feed between two Ivans (ivan.core.feed, the command line in ivan.core.feed_cli): the whole card as the laptop
holds it (fights, fighters, contests, lineups, sims) with the hand-entered inputs (data/mma/input), the models and the
derived contest files. The server only looks at it: its slate page's "Refresh from laptop" pulls. Nothing is pushed.
Producer (the laptop):
    scripts/mma/feed.py publish [--quiet]       write data/feed/mma/*; the consumer's pull asks for a fresh one (deploy/feed_serve.sh)
Consumer (the server):
    scripts/mma/feed.py pull                    with IVAN_FEED_SSH=<user>@<host> in .env: ssh publish + rsync + apply (feed.pull_ssh)
    scripts/mma/feed.py apply DIR               seed this database from bundles copied by hand
"""
from __future__ import annotations

from ivan.core import feed_cli

if __name__ == "__main__":
    raise SystemExit(feed_cli.main("mma", doc=__doc__))
