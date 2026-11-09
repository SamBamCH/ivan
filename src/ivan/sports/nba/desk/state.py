"""
The Desk's state: the session, the LiveSlate, the projection store's frames, the header; rebuilt when the pool changes on disk.
"""
from __future__ import annotations

import threading
import time
from datetime import date, datetime, timedelta
from ivan.sports.nba.data_fetch import newsfeed as NF
from typing import Optional
import numpy as np
import pandas as pd
from dotenv import load_dotenv

from ivan.core import feed
from ivan.sports.nba.slate import autopilot
from ivan.core.utils import PROJECT_ROOT
from ivan.sports.nba.names import normalize_name
from ivan.sports.nba.utils import utc_to_local
from ivan.database.session import get_session, init_db
from ivan.sports.nba.ingest import ensure_schema, project_team
from ivan.sports.nba.projection import store as PSTORE
from ivan.sports.nba.models import Lineup, NBAMarketRate, NBAPlayer
from ivan.sports.nba.paths import NBA_DATA_DIR
from ivan.sports.nba.projection.engine import ProjectionParams
from ivan.sports.nba.projection.live import LiveSlate, OVERRIDE_KEYS, override_layers, read_overrides_full, write_overrides
from .lineups import backfill_lineup_slates
from ivan.sports.nba import slates as S


TEAM_BUDGET = 241.0


FOUL_SHORT_PF = 4          # a game with this many fouls...


FOUL_SHORT_SHARE = 0.8     # ...and minutes under this share of the player's recent average is marked foul-shortened


ADD_BACK_MINUTES = 6.0     # a player the rotation left off, added back by hand, starts at his history or this many minutes


STATE_FILE = NBA_DATA_DIR / "capture_state.json"


ROTATION_MIN = 15.0        # a scratch/questionable above this baseline is news


QUESTION_STATUSES = NF.QUESTION_STATUSES     # DK statuses that leave the player's night open (Q / GTD / D / P)


NEAR_TIP_MIN = 75          # minutes before tip when an unconfirmed lineup becomes a warning


MINUTES_DEV = 8.0          # model minutes this far from history is a role change


MARKET_GAP = 0.20          # market mean vs history-only projection, relative


def short_name(name: str) -> str:
    """'jaren jackson jr' -> 'jackson': the surname, for a lane segment's label."""
    toks = [t for t in str(name or "").split() if t not in ("jr", "sr", "ii", "iii", "iv")]
    return toks[-1] if toks else str(name or "")

load_dotenv(PROJECT_ROOT / ".env")                          # IVAN_FEED_SSH (a consumer: the laptop it pulls from) / IVAN_FEED_URL (older)
MEAN_MOVE = 0.05                                              # a market mean that moved by more than this re-projects the player's team (Desk.follow_odds)
ODDS_CHECK_S = 20.0                                           # how often the desk looks whether the odds table moved (Desk.follow_odds)
CONSUMER = feed.producer_source() is not None                 # a consumer pulls the laptop's pool on Refresh and never fetches DK, props or news itself
engine = init_db("nba")                                    # the history database (IVAN_NBA_DB_URL or data/nba/nba_history.sqlite); a slate's own is slates.engine(dg)
ensure_schema(engine)


class Lobby:
    """The slate index shared by every desk: the DraftKings lobby (fetched off the request path), the archive's slates and the
    slates this machine holds. One per process (LOBBY); reads the history database."""
    def __init__(self):
        self.session = get_session(engine)
        self.flash: list[str] = []
        self._lobby: tuple = (0.0, {})

    def lobby_slates(self) -> dict:
        """Today's classic slates in the DraftKings lobby, labelled with DK's start string; the loaded slate is always present.
        Fetched once at start, then refreshed in a background thread when older than two minutes: a page render never waits
        on DraftKings (2026-11-03: the header blocked up to 2.5 s on this call). Falls back to the slates the capture state
        knows when the lobby is unreachable."""
        now = time.time()
        if now - self._lobby[0] < 120:
            return self._lobby[1]
        if self._lobby[0] == 0.0:                              # never fetched: the picker needs it now
            self._lobby = (now, self._fetch_lobby())
            return self._lobby[1]
        th = getattr(self, "_lobby_thread", None)
        if th is None or not th.is_alive():
            def run():
                out = self._fetch_lobby()
                self._lobby = (time.time(), out or self._lobby[1])
            self._lobby_thread = threading.Thread(target=run, name="ivan-lobby", daemon=True); self._lobby_thread.start()
        return self._lobby[1]

    def _fetch_lobby(self) -> dict:
        out = {}
        try:
            from ivan.sports.nba.ingest import classic_slates
            for sl in classic_slates():                       # every day the lobby lists; the day picker places each on its day
                start = str(sl.get("start") or "")
                out[str(sl["id"])] = dict(label=f"{start} · {str(sl.get('name') or '').replace('NBA ', '')[:30]}", lock_dt=None, start=start, name=sl.get("name"),
                                          games=sl.get("games"), lock=sl.get("lock"), title=sl.get("title"))
        except Exception as e:   # noqa: BLE001 - the lobby is a convenience
            self.flash.append(f"could not list the lobby's slates ({type(e).__name__}); showing the known ones")
        return out

    def slate_games(self, key: str) -> Optional[int]:
        """How many games an archived slate had: the distinct team pairs of its capture's players (cached per slate key)."""
        cache = self.__dict__.setdefault("_slate_games", {})
        if key in cache:
            return cache[key]
        from ivan.core import archive
        try:
            caps = [c for c in archive.captures(self.session, "nba") if str(c.slate_key) == str(key)]
            cap = max(caps, key=lambda c: c.captured_at) if caps else None
            players = archive.load_blobs(self.session, cap.id).get("players", []) if cap else []
            pairs = {frozenset((str(r.get("team")), str(r.get("opponent")))) for r in players if r.get("team") and r.get("opponent")}
            cache[key] = len(pairs) or None
        except Exception:   # noqa: BLE001
            cache[key] = None
        return cache[key]

    @staticmethod
    def slate_label(when: str, games, title: Optional[str] = None, tail: str = "") -> str:
        """What the bar's slate dropdown and the picker card read: the lock time, the games, DK's slate title when it has one
        ("19:00 · 8 games · Early Only"); the main slate of the night has no title."""
        return " · ".join(x for x in [when, f"{games} game{'s' if games != 1 else ''}" if games else "", title or "", tail] if x)

    def slate_days(self, now: datetime) -> list:
        """
        [(date iso, date label, [slate, ...])] newest day first, each slate {id, label, source, loaded, games, title}: the DraftKings
        lobby's draft groups placed on their day (DK gives a weekday + time), then every slate with an archive capture (loadable from
        the capture: scripts/nba archive at ingest and at tip), and every slate this machine holds (a directory under
        data/nba/slates; `loaded` = held; its title from slate.json, written when the desk loads it from the lobby). The picker: a
        day, then its slates.
        """
        days: dict = {}
        seen = set()
        held = S.held_meta()
        wd = {"mon": 0, "tue": 1, "wed": 2, "thu": 3, "fri": 4, "sat": 5, "sun": 6}
        for key, v in self.lobby_slates().items():
            start = str(v.get("start") or v.get("label") or "")
            head = start.split()[0].lower()[:3] if start else ""
            if head in wd:
                d = now.date() + timedelta(days=(wd[head] - now.weekday()) % 7)
            elif head == "tom":
                d = now.date() + timedelta(days=1)
            else:
                d = now.date()
            lk = v.get("lock"); ng = v.get("games")
            if lk is not None:
                d = lk.date()
            when = f"{lk:%H:%M}" if lk is not None else (start.split(" ", 1)[1] if " " in start else start)
            title = v.get("title") or (held.get(int(key)) or {}).get("title")
            days.setdefault(d.isoformat(), []).append(dict(id=str(key), label=self.slate_label(when, ng, title),
                                                           source="lobby", sort=(lk.strftime("%H:%M") if lk is not None else start), loaded=int(key) in held,
                                                           lock=lk.isoformat(timespec="minutes") if lk is not None else None, contests=v.get("contests"), games=ng, title=title))
            seen.add(str(key))
        from sqlalchemy import text as _t
        for key, lock, src, n in self.session.execute(_t("select slate_key, min(lock), max(source), count(*) from archived_slate where sport = 'nba' group by slate_key")).all():
            if str(key) in seen:
                continue
            try:
                t = datetime.fromisoformat(str(lock))
            except (TypeError, ValueError):
                continue
            ng = self.slate_games(str(key))
            title = (held.get(int(key)) or {}).get("title")
            days.setdefault(t.date().isoformat(), []).append(dict(id=str(key), label=self.slate_label(f"{t:%H:%M}", ng, title, "archived"),
                                                                  source="archive", sort=t.strftime("%H:%M"), loaded=int(key) in held, lock=t.isoformat(timespec="minutes"), games=ng, title=title))
            seen.add(str(key))
        for hdg, m in held.items():                           # held slates the lobby no longer lists and the archive never saw
            if str(hdg) in seen:
                continue
            lock = S.meta_time(m, "lock")
            d = (lock or now).date().isoformat()
            ng = m.get("games")
            days.setdefault(d, []).append(dict(id=str(hdg), label=self.slate_label(m.get("label") or (f"{lock:%H:%M}" if lock else "held"), ng, m.get("title")),
                                               source="held", sort=(lock.strftime("%H:%M") if lock else ""), loaded=True,
                                               lock=lock.isoformat(timespec="minutes") if lock else None, games=ng, title=m.get("title")))
            seen.add(str(hdg))
        out = []
        for d in sorted(days, reverse=True):
            dd = date.fromisoformat(d)
            tag = "today" if dd == now.date() else ("tomorrow" if dd == now.date() + timedelta(days=1) else ("yesterday" if dd == now.date() - timedelta(days=1) else ""))
            out.append((d, f"{dd:%a %m-%d}" + (f" · {tag}" if tag else ""), sorted(days[d], key=lambda x: x["sort"])))
        return out



LOBBY = Lobby()
_HISTORY: dict = {}
_HISTORY_LOCK = threading.Lock()


class DeskBase:
    """One slate's desk: `dg` is its draft group, `self.session` its own database (data/nba/slates/<dg>/slate.sqlite with the
    history attached), `self.lock` what its routes hold. Every method expects the slate to be the current one (slates.use):
    the web sets it per request, a thread with `with slates.using(desk.dg)`."""
    def __init__(self, dg: int):
        self.dg = int(dg)
        self.lock = threading.RLock()
        self.engine = S.engine(self.dg)
        self.session = S.session(self.dg)
        self.slate: Optional[LiveSlate] = None
        self.pool_stamp = None
        self.overrides: dict[str, float] = {}
        self.proj: dict[str, pd.DataFrame] = {}
        self.rot: dict[str, dict] = {}
        self.hist_only: dict[str, pd.DataFrame] = {}
        self.flash: list[str] = []
        self.busy: Optional[str] = None                     # "loading slate 155581" while a background refresh runs; the board polls
        self._lobby: tuple = (0.0, {})
        with S.using(self.dg):
            self.rebuild()
            try:
                backfill_lineup_slates(self.session, log=self.flash.append)
            except Exception:   # noqa: BLE001 - the export does it again on demand
                self.session.rollback()

    def _stamp(self):
        """(latest projection time, player count, draft group, hash of the player ids, hash of the ownerships) of the pool on disk.
        A column query: it does not touch the player objects the page holds, so a stale page still renders from them. The
        ownerships are in it so a field build or a sim (they write nba_player.ownership, nothing else) reloads the desk too:
        the board's own column and the projections CSV export read the desk's players."""
        rows = self.session.query(NBAPlayer.id, NBAPlayer.fpts_updated_at, NBAPlayer.draft_group_id, NBAPlayer.ownership).all()
        return (max((r[1] for r in rows if r[1]), default=None), len(rows), rows[0][2] if rows else None,
                hash(frozenset(int(r[0]) for r in rows)), hash(frozenset((int(r[0]), r[3]) for r in rows)))

    def history(self):
        """(box, HistoryState), rebuilt only when nba_stats grew (the nightly box scores); one copy per process, every desk shares it."""
        n = self.session.execute(__import__("sqlalchemy").text("select count(*), max(game_date) from nba_stats")).one()
        key = (int(n[0]), str(n[1]))
        with _HISTORY_LOCK:
            if _HISTORY.get("key") != key:
                from ivan.sports.nba.projection.history import build_history, load_box_scores
                box = load_box_scores(self.session)
                _HISTORY.update(key=key, value=(box, build_history(box)))
            self._hist_key = key
            return _HISTORY["value"]

    def rebuild(self):
        self.session.expire_all()
        self.slate = LiveSlate(self.session, history=self.history())
        self.pool_stamp = self._stamp(); self.stale = False
        self.overrides = read_overrides_full()            # what the projection reads: the published layer (a consumer) under this machine's own
        self.local_overrides = read_overrides_full(layered=False) if S.overrides_csv().exists() else {}   # what this machine writes
        self.__dict__.pop("_layers_cache", None); self.__dict__.pop("_published_cache", None)
        self.proj, self.hist_only, self.rot = {}, {}, {}
        self._auto_pending = True                         # scenarios are matched once the rest of the rebuild is done (see end of rebuild)
        if getattr(self, "_calls_dg", None) != self.pool_stamp[2]:
            self._calls, self._calls_dg = {}, self.pool_stamp[2]   # typed calls belong to their slate
        hist_key = getattr(self, "_hist_key", None)
        if getattr(self, "_season_key", None) != hist_key:     # per-player season and recent-game tables: only when the box scores grew
            self.season = self.season_averages(self.slate.box)
            from ivan.sports.nba.data_fetch import pbp as PBP
            try:
                self.fouls = PBP.fouls_per_game(self.session)
            except Exception:   # noqa: BLE001 - no foul table yet
                self.fouls = {}
            self.recent = self.recent_games(self.slate.box, fouls=self.fouls)
            self._season_key = hist_key
        self.load_market_rows()
        self.load_frames()

        if getattr(self, "_auto_pending", False):
            self._auto_pending = False
            try:
                for m in self.auto_scenarios():
                    self.flash.append(m)
            except Exception as e:   # noqa: BLE001 - the pick is a convenience
                self.flash.append(f"scenario pick failed: {type(e).__name__}: {e}")

    @staticmethod

    def season_averages(box: pd.DataFrame) -> dict:
        """player -> (games, minutes per game, fantasy points per minute) this season, and the same over last season as a fallback."""
        today = datetime.now().date()
        start = date(today.year if today.month >= 10 else today.year - 1, 10, 1)
        out = {}
        for label, sub in (("season", box[(box.game_date >= start) & box.played]), ("last", box[(box.game_date < start) & box.played])):
            g = sub.groupby("player_name").agg(games=("minutes", "size"), minutes=("minutes", "mean"), fpts=("dk_fpts", "sum"), mins=("minutes", "sum"))
            for name, r in g.iterrows():
                out.setdefault(name, {})[label] = dict(games=int(r.games), minutes=float(r.minutes), fpm=float(r.fpts / r.mins) if r.mins > 0 else None)
        return out

    def consensus_odds(self) -> dict:
        """(player, stat) -> 'o-112 / u-108': the books' prices at the consensus line averaged in probability, the vig left in.
        Cached until the odds table or the market lines change (0.6 s over 150k rows otherwise, on every rebuild)."""
        from sqlalchemy import text as _t
        key = (tuple(self.session.execute(_t("select count(*), max(last_update) from odds")).one()),
               tuple(sorted((n, st, str(v.get("line"))) for n, d in self.market_rows.items() for st, v in d.items())))
        cache = getattr(self, "_consensus_cache", None)
        if cache is not None and cache[0] == key:
            return cache[1]
        out = self._consensus_odds()
        self._consensus_cache = (key, out)
        return out

    def load_market_rows(self) -> None:
        """player -> stat -> the stored market rate row (mean, line, books, rate, the minutes it was priced over) with the consensus
        prices; re-read after a refresh and after a hand starter call rebuilt the rates."""
        from ivan.sports.nba.models import NBAMarketRate
        self.market_rows = {}
        for r in self.session.query(NBAMarketRate).all():
            self.market_rows.setdefault(r.player_name, {})[r.stat] = dict(mean=float(r.market_mean), line=r.line, books=r.books, rate=float(r.rate), basis=float(r.basis_minutes),
                                                                           odds=None)
        for (name, stat), txt in self.consensus_odds().items():
            if name in self.market_rows and stat in self.market_rows[name]:
                self.market_rows[name][stat]["odds"] = txt

    def fresh_props(self) -> pd.DataFrame:
        """This slate's prop rows (market.fresh_odds over the pool's games) with stat, alt, posted (local) and stale (posted before the
        player's last downgrade, changes.status_cutoffs), cached on the odds table's stamp and the status stamps: built on the first
        hover after a change (about a second), then read per cell."""
        from ivan.sports.nba.slate import changes as CH
        from ivan.sports.nba.ingest import odds_stamp
        from ivan.sports.nba.projection.market import MARKET_TO_STAT, fresh_odds
        cutoffs = CH.status_cutoffs(self.session)
        key = (odds_stamp(self.session), self.pool_stamp[2], tuple(sorted((k, str(v)) for k, v in cutoffs.items())))
        cache = getattr(self, "_fresh_props_cache", None)
        if cache is not None and cache[0] == key:
            return cache[1]
        odds = fresh_odds(self.session, {normalize_name(p.name): p.gametime for p in self.slate.players}, cutoffs=cutoffs, mark=True)
        if not odds.empty:
            odds = odds[odds.market.isin(MARKET_TO_STAT)].copy()
            odds["stat"] = odds.market.map(MARKET_TO_STAT); odds["alt"] = odds.market.str.endswith("alternate")
            odds["posted"] = utc_to_local(odds["last_update"]) if odds["last_update"].notna().any() else pd.NaT
            odds["_side"] = odds.side.astype(str).str.lower()
            if "stale" not in odds.columns:
                odds["stale"] = False
        self._fresh_props_cache = (key, odds)
        return odds

    def prop_rows_for(self, name: str, stat: str) -> list:
        """Every (book, line) the books hold for the player's stat tonight: [{date, book, line, over, under, alt, stale}], the main
        lines first then the alternates, by book and line; `date` is when the book last posted it (local)."""
        odds = self.fresh_props()
        if odds.empty:
            return []
        g0 = odds[(odds.player == name) & (odds.stat == stat)]
        if g0.empty:
            return []
        amer = lambda prices: "" if prices.empty else f"{int(round(float(prices.iloc[0]))):+d}"   # noqa: E731
        out = []
        for (book, point, alt), g in g0.groupby(["sportsbook", "point", "alt"], sort=False):
            posted = g.posted.max()
            out.append(dict(date=None if pd.isna(posted) else posted.strftime("%m-%d %H:%M"), book=str(book), line=float(point),
                            over=amer(g[g._side == "over"].price), under=amer(g[g._side == "under"].price), alt=bool(alt), stale=bool(g.stale.any())))
        out.sort(key=lambda b: (b["alt"], b["book"], b["line"]))
        return out

    def follow_odds(self) -> None:
        """The odds table moved since this slate's market rates were built (a props fetch, for this slate or another of the day: the
        table is shared): rebuild the rates past the status cutoffs and re-project every team, so the projection follows the books
        rather than the snapshot the slate's last refresh took (ingest.ensure_market_rates). Checked at most every ODDS_CHECK_S
        seconds and once per odds stamp; a consumer has no raw store and skips."""
        if CONSUMER:
            return
        t = time.time()
        if t - getattr(self, "_odds_checked", 0.0) < ODDS_CHECK_S:
            return
        self._odds_checked = t
        from ivan.sports.nba.ingest import ensure_market_rates, odds_stamp
        stamp = odds_stamp(self.session)
        if stamp == getattr(self, "_odds_followed", None):
            return
        self._odds_followed = stamp
        try:
            old = {n: dict(v) for n, v in self.slate.market_means.items()}
            if ensure_market_rates(self.session, history=self.history(), log=lambda m: None):
                self.slate.reload_market(); self.load_market_rows()
                new = self.slate.market_means
                moved = {n for n in set(old) | set(new)
                         if set(old.get(n, {})) != set(new.get(n, {})) or any(abs(old[n][k] - new[n][k]) > MEAN_MOVE for k in old.get(n, {}) if k in new.get(n, {}))}
                teams = sorted({p.team for p in self.slate.players if normalize_name(p.name) in moved})
                for team in teams:                                # only the teams whose books moved: a team re-projects in about a second
                    self.reproject(team)
                if teams:
                    self.flash.append(f"the books moved for {len(moved)} player{'s' if len(moved) != 1 else ''}: market rates rebuilt, {', '.join(teams)} re-projected")
        except Exception as e:   # noqa: BLE001 - the desk keeps serving on the rates it has
            self.session.rollback()
            self.flash.append(f"market rates not followed: {type(e).__name__}: {e}")

    def _consensus_odds(self) -> dict:
        from ivan.sports.nba.slate import changes as CH
        from ivan.sports.nba.projection.market import MARKET_TO_STAT, american_to_prob, fresh_odds
        odds = fresh_odds(self.session, {normalize_name(p.name): p.gametime for p in self.slate.players}, cutoffs=CH.status_cutoffs(self.session))   # this slate's games, past the status cutoffs
        if odds.empty:
            return {}
        odds = odds[odds.market.isin(MARKET_TO_STAT) & ~odds.market.str.endswith("alternate")].copy()
        odds["stat"] = odds.market.map(MARKET_TO_STAT)
        out = {}
        for (name, stat), g in odds.groupby(["player", "stat"]):
            line = self.market_rows.get(name, {}).get(stat, {}).get("line")
            if line is None or pd.isna(line):
                continue
            at = g[np.isclose(g.point.astype(float), float(line))]
            parts = []
            for side, tag in (("over", "o"), ("under", "u")):
                prices = at[at.side.str.lower() == side].price.astype(float)
                if len(prices):
                    p_avg = float(np.mean([american_to_prob(x) for x in prices]))
                    amer = -100 * p_avg / (1 - p_avg) if p_avg >= 0.5 else 100 * (1 - p_avg) / p_avg
                    parts.append(f"{tag}{amer:+.0f}")
            if parts:
                out[(name, stat)] = " / ".join(parts)
        return out

    @staticmethod
    def recent_games(box: pd.DataFrame, n: int = 10, fouls: Optional[dict] = None) -> dict:
        """player -> his last n games in a box score, played or not (a DNP row shows the risk), newest first: (date, team, played,
        started, minutes, fppm, pf, foul_short). foul_short marks a game with FOUL_SHORT_PF fouls or more and minutes under
        FOUL_SHORT_SHARE of his average over the games he played: the minutes were probably cut by fouls, not by the coach's plan."""
        out = {}; fouls = fouls or {}
        sub = box.sort_values(["game_date", "game_id"])
        for name, g in sub.groupby("player_name"):
            tail = g.tail(n)
            played = tail[tail.played]
            avg = float(played.minutes.mean()) if len(played) else 0.0
            rows = []
            for r in tail.itertuples():
                pf = fouls.get((str(r.game_id), name))
                rows.append(dict(date=str(r.game_date), team=r.team, played=bool(r.played), started=bool(float(getattr(r, "starter", 0) or 0) > 0),
                                 minutes=float(r.minutes), fpts=float(r.dk_fpts),
                                 fpm=float(r.dk_fpts) / float(r.minutes) if r.minutes > 0 else None, pf=pf,
                                 foul_short=bool(pf is not None and pf >= FOUL_SHORT_PF and avg > 0 and float(r.minutes) < FOUL_SHORT_SHARE * avg)))
            out[name] = rows[::-1]
        return out

    def maybe_reload(self):
        """
        The pool changed on disk (a capture tick, a feed pull, a script, an edit from another desk process): reload it. A
        rebuild reads the projection store, so this costs milliseconds; a different set of players (another slate, a roster
        change) says so, new numbers for the same players load quietly.
        """
        self.follow_odds()
        now = self._stamp()
        if now == self.pool_stamp:
            return
        if now[1:4] != self.pool_stamp[1:4]:
            self.rebuild()
            self.flash.append("the player pool changed on disk (a refresh from another process changed the roster); reloaded")
        else:
            self.rebuild()                                    # new numbers (a projection, the ownership) for the same players: quietly

    def load_frames(self) -> None:
        """
        The desk's frames from the projection store (projection.store): what the last projection of each team wrote, whoever
        wrote it. A team with no record (a pool written before the store existed, a record from another version) is projected
        now, through the same path an edit takes, so the pool and the store agree afterwards.
        """
        recs = PSTORE.load_all(self.pool_stamp[2])
        self.proj, self.hist_only, self.rot = {}, {}, {}
        missing = []
        for t in self.slate.teams():
            rec = recs.get(t)
            if rec is None:
                missing.append(t); continue
            self.proj[t], self.hist_only[t], self.rot[t] = rec["frame"], rec["hist"], dict(rec.get("rot") or {})
        for t in missing:
            self.reproject(t)
        if missing:
            self.flash.append(f"projected {len(missing)} team(s) with no stored projection ({', '.join(missing[:6])}{'…' if len(missing) > 6 else ''})")

    def reproject(self, team: str):
        """
        Re-project one team and write it through: its nba_player rows, committed, and its store record (ingest.project_team,
        the same path a publish takes, seeded per team so the numbers are those a publish would give). The desk's frame is
        the record just written; nothing downstream can see a projection the desk does not show.
        """
        self.void_if_stale(team)
        rec = project_team(self.session, self.slate, team, self.overrides, ProjectionParams.from_config())
        self.session.commit()
        self.proj[team], self.hist_only[team], self.rot[team] = rec["frame"], rec["hist"], dict(rec.get("rot") or {})
        self.pool_stamp = self._stamp()                        # our own write is not "new data"
        try:
            autopilot.touch("publish")                         # the edit is in the pool the moment it is made: nothing is "unpublished"
        except OSError:
            pass

    def players(self, team: str) -> list[NBAPlayer]:
        return sorted([p for p in self.slate.players if p.team == team], key=lambda p: -(p.salary or 0))

    def void_if_stale(self, team: str) -> Optional[dict]:
        """A player the team's mapping had available is out now (a status, a scratch call): the mapping is void (ivan.sports.nba.slate.mapping):
        grid cleared, positive minutes pins lifted, the old mapping kept as a scenario assuming him in; the card says so."""
        from ivan.sports.nba.slate import mapping as MAP
        statuses = {normalize_name(p.name): (p.status or "") for p in self.players(team)}
        outs = MAP.stale_outs(team, statuses, self.overrides)
        if not outs:
            return None
        rep = MAP.void_mapping(team, outs, statuses.keys(), self.local_overrides, log=lambda m: None)
        self.write_local()
        self.flash.append(f"{team}: {', '.join(n.split()[-1].title() for n in outs)} now out: the mapping is void, the model's rotation is on the grid"
                          f" ({rep['pins']} minutes pin(s) lifted{'; the old mapping is scenario ' + repr(rep['label']) if rep['scenario'] else ''}): accept it (Hold all) or paint and hold")
        return rep

    def tip(self, team: str) -> Optional[datetime]:
        return min((p.gametime for p in self.players(team) if p.gametime), default=None)

    def market_means(self) -> dict[str, float]:
        """market fpts per priced player (sum of priced stats at DK weights; turnovers from history are small)"""
        out = {}
        rows = self.session.query(NBAMarketRate).all()
        from ivan.sports.nba.scoring import DK_WEIGHTS
        for r in rows:
            out[r.player_name] = out.get(r.player_name, 0.0) + DK_WEIGHTS.get(r.stat, 0.0) * float(r.market_mean)
        return out

    def rookie_of(self, name: str) -> Optional[dict]:
        """
        A rookie: his first NBA season is this one (ivan.sports.nba.data_fetch.experience, nba_api's list, fetched nightly); a name the
        list lacks is a rookie when no NBA game is on file before this season. {games: this season's games played} or None.
        """
        from ivan.sports.nba.data_fetch import experience as X
        if getattr(self, "_exp_stamp", None) != self.pool_stamp:
            self._first_years, self._exp_stamp = X.load(), self.pool_stamp
        sa = (getattr(self, "season", None) or {}).get(name) or {}
        team = next((p.team for p in self.slate.players if normalize_name(p.name) == name), None)
        if not X.is_rookie(name, X.season_start_year(), self._first_years, played_before=bool(sa.get("last")), team=team):
            return None
        return dict(games=int((sa.get("season") or {}).get("games", 0)))

    def draft_group(self) -> Optional[int]:
        return self.dg

    def notes(self) -> dict:
        """The loaded slate's player notes, not cleared (ivan.sports.nba.slate.notes): {player: row}."""
        from ivan.sports.nba.slate import notes as NOTES
        try:
            return NOTES.active(self.session, self.draft_group())
        except Exception:   # noqa: BLE001 - no table yet on an old database: no notes
            return {}

    def team_notes(self, team: str) -> list:
        names = {normalize_name(p.name): p.name for p in self.players(team)}
        return [dict(key=n, name=names[n], text=r.text, at=r.updated_at) for n, r in self.notes().items() if n in names]

    def pins(self) -> dict[str, float]:
        """player -> pinned minutes (the overrides' minutes column only)."""
        return {k: v["minutes"] for k, v in self.overrides.items() if v.get("minutes") is not None}

    def set_override(self, name: str, **fields) -> None:
        """Set or clear a player's pins in this machine's own layer (a value None clears that field here; on a consumer the
        published value then shows through); the merged view is refreshed. Persisted at once."""
        rec = dict(self.local_overrides.get(name, {}))
        for k, v in fields.items():
            rec[k] = v
        rec = {k: v for k, v in rec.items() if v is not None}
        if rec:
            self.local_overrides[name] = {k: rec.get(k) for k in OVERRIDE_KEYS}
        else:
            self.local_overrides.pop(name, None)
        self.write_local()

    def write_local(self) -> None:
        """Write this machine's own layer and refresh the merged view the projection reads."""
        write_overrides(self.local_overrides)
        self.overrides = read_overrides_full()
        self.__dict__.pop("_layers_cache", None); self.__dict__.pop("_published_cache", None)

    def _layer_of(self, name: str) -> dict:
        """{field: 'local' | 'published'} for the player's pins: what this machine typed vs what the producer published (a consumer)."""
        if S.published_pins() is None:
            return {}
        cache = self.__dict__.setdefault("_layers_cache", {})
        if name not in cache:
            pub, loc = override_layers(name)
            cache[name] = {**{k: "published" for k in pub}, **{k: "local" for k in loc}}
        return cache[name]

    def _published_of(self, name: str) -> dict:
        """The producer's published pins for the player (a consumer), {} elsewhere."""
        if S.published_pins() is None:
            return {}
        cache = self.__dict__.setdefault("_published_cache", {})
        if name not in cache:
            cache[name] = override_layers(name)[0]
        return cache[name]

    def freshness(self, now: datetime) -> list:
        """When each input was last updated, read from disk on every page (so it is current even while the page is stale):
        [(label, time or None, minutes ago or None, warn after minutes, what it is)]."""
        from sqlalchemy import text as _t
        q = lambda sql: self.session.execute(_t(sql)).scalar()          # noqa: E731
        props = None
        pdir = NBA_DATA_DIR / "raw" / "odds" / "props"
        if pdir.is_dir():                                               # the producer: the newest stored props fetch
            mt = [f.stat().st_mtime for d in pdir.iterdir() if d.is_dir() for f in d.iterdir()]
            props = datetime.fromtimestamp(max(mt)) if mt else None
        if props is None:                                               # a consumer has no raw store: the books' newest update
            last = q("select max(last_update) from odds")
            if last:
                props = utc_to_local(last)
        dg = next((p.draft_group_id for p in self.slate.players if p.draft_group_id), None)
        dk = q(f"select max(captured_at) from archived_slate where slate_key = '{int(dg)}'") if dg else None
        rows = [("props", props, 90, "the books' player props, last fetched"),
                ("DraftKings", dk, 30, "the DK pool: salaries, statuses, FPPG"),
                ("starters", q("select max(fetched_at) from starting_lineups"), 30, "Rotowire lineups, last read"),
                ("projected", q("select max(fpts_updated_at) from nba_player"), 30, "the projection, last calculated")]
        if CONSUMER:                                                    # what the laptop last published, as pulled (the Refresh button pulls)
            fs = feed.status("nba")
            prod = fs.get("producer") or {}
            rows.append(("published", fs.get("published_at"), 60,
                         "the laptop's last publish of the feed (" + (f"{prod.get('source')}, autopilot {'on' if prod.get('autopilot') else 'off'}; " if prod else "")
                         + (f"pulled here {str(fs.get('pulled_at'))[11:16]}" if fs.get("pulled_at") else "not pulled yet") + ")"))
        out = []
        for label, t, warn, what in rows:
            t = pd.Timestamp(t).to_pydatetime() if t is not None else None
            age = (now - t).total_seconds() / 60 if t is not None else None
            out.append(dict(label=label, at=t, age=age, warn=bool(age is not None and age > warn), what=what))
        return out

    def header(self, now: datetime) -> dict:
        ps = self.slate.players
        dg = self.dg
        lock = min((p.gametime for p in ps if p.gametime), default=None)
        odds_t = self.session.execute(__import__("sqlalchemy").text("select max(last_update) from odds")).scalar()
        meta = S.read_meta(self.dg)
        try:
            n_lineups = int(self.session.query(Lineup).filter(Lineup.draft_group_id == dg).count()) if dg is not None else 0
        except Exception:   # noqa: BLE001
            n_lineups = 0
        return dict(draft_group=self.dg, lock=lock, mins_to_lock=(lock - now).total_seconds() / 60 if lock else None, n_lineups=n_lineups,
                    pool_at=self.pool_stamp[0], n_players=len(ps), n_priced=len(self.slate.market),
                    starters_at=self.slate.starters_fetched_at, odds_at=odds_t, stale=getattr(self, "stale", False), fresh=self.freshness(now),
                    slate_days=LOBBY.slate_days(now), meta=meta, refreshed_at=S.meta_time(meta, "refreshed_at"), refreshed_by=meta.get("refreshed_by"),
                    pulled_at=S.meta_time(meta, "pulled_at"), published_at=S.meta_time(meta, "published_at"), mapped=self.mapped_summary(),
                    questions=self.questions_summary(now), news=self.news_status(now), team_states=self.team_states(now),
                    overrides=len(self.overrides), overrides_path=str(S.overrides_csv().relative_to(PROJECT_ROOT)),
                    autopilot=None if CONSUMER else autopilot.state(), feed=feed.status("nba") if CONSUMER else None)
