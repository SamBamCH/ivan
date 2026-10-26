"""
ivan.sports.nba.newsfeed

Player news from Rotowire's NBA news page (free, no login): every item is a headline about one player with the team,
the injury, the report's text and the source it cites (the tweet from Underdog, Shams or the beat writer). Fetched only
when asked (the desk's Check news button, its auto-check while on, scripts/nba/newsfeed.py); each fetch stores the page
raw (rotowire/news) and adds the headlines not seen before to nba_news with first_seen = now. Rotowire shows a date, not
a time, so first_seen is the news clock: minutes behind the tweet when the check runs every few minutes.

    check(session)                    fetch, store, return what is new
    latest(session, names, hours=48)  {player: [items newest first]} for the pool's players
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta
from typing import Dict, Iterable, List, Optional

import requests
from bs4 import BeautifulSoup
from sqlalchemy.orm import Session

from ivan.core import raw
from ivan.core.utils import normalize_name
from ivan.sports.nba.models import NBANews

NEWS_URL = "https://www.rotowire.com/basketball/news.php"
ROTOWIRE = "https://www.rotowire.com"
TEAM_ALIAS = {"PHO": "PHX", "GS": "GSW", "SA": "SAS", "NY": "NYK", "NO": "NOP", "UTAH": "UTA", "WSH": "WAS", "BKN": "BKN"}


def fetch_html(timeout: int = 15) -> str:
    return requests.get(NEWS_URL, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"}).text


def parse(html: str) -> List[dict]:
    """The page's items, newest first as Rotowire lists them."""
    soup = BeautifulSoup(html, "html.parser")
    out = []
    for it in soup.select("div.news-update"):
        head = it.select_one("a.news-update__headline")
        pl = it.select_one("a.news-update__player-link")
        if head is None or pl is None:
            continue
        href = head.get("href") or ""
        m = re.search(r"-(\d+)/?$", href)
        if not m:
            continue
        logo = it.select_one("img.news-update__logo")
        team = (logo.get("alt") or "").strip().upper() if logo else None
        team = TEAM_ALIAS.get(team, team) or None
        pos = it.select_one(".news-update__pos"); inj = it.select_one(".news-update__inj")
        ts = it.select_one(".news-update__timestamp"); body = it.select_one(".news-update__news")
        src = None
        if body is not None:
            a = body.find("a", href=True)
            src = a["href"] if a is not None else None
        out.append(dict(id=int(m.group(1)), player_raw=pl.get_text(" ", strip=True), player_name=normalize_name(pl.get_text(" ", strip=True)),
                        team=team, position=pos.get_text(strip=True) if pos else None, injury=inj.get_text(strip=True) if inj else None,
                        injured="is-injured" in (it.get("class") or []), headline=head.get_text(" ", strip=True),
                        text=body.get_text(" ", strip=True) if body else None, source_url=src,
                        url=(ROTOWIRE + href) if href.startswith("/") else href, date=ts.get_text(strip=True) if ts else None))
    return out


def store(session: Session, items: Iterable[dict], now: Optional[datetime] = None) -> List[NBANews]:
    """Add the items not stored yet (by Rotowire id) with first_seen = now; returns the new rows, newest first."""
    now = now or datetime.now()
    items = list(items)
    if not items:
        return []
    have = {int(i) for (i,) in session.query(NBANews.id).filter(NBANews.id.in_([int(x["id"]) for x in items])).all()}
    new = []
    for x in items:
        if int(x["id"]) in have:
            continue
        row = NBANews(first_seen=now, **x)
        session.add(row); new.append(row)
    session.commit()
    return sorted(new, key=lambda r: -r.id)


def check(session: Session, log=print) -> dict:
    """Fetch the page, keep it raw, store the new headlines. {new: [rows], total: n on the page, fetched_at}."""
    now = datetime.now()
    html = fetch_html()
    try:
        raw.store("nba", "rotowire/news", now.strftime("%Y-%m-%d"), html, ext="html", when=now)
    except Exception as e:   # noqa: BLE001 - the raw copy is a convenience
        log(f"raw store failed: {type(e).__name__}: {e}")
    items = parse(html)
    new = store(session, items, now=now)
    log(f"news: {len(items)} items on the page, {len(new)} new")
    return dict(new=new, total=len(items), fetched_at=now)


def latest(session: Session, names: Iterable[str], hours: float = 48.0, now: Optional[datetime] = None) -> Dict[str, List[NBANews]]:
    """{player: [items newest first]} for the named players, seen within `hours`."""
    now = now or datetime.now()
    names = [str(n) for n in names]
    if not names:
        return {}
    out: Dict[str, List[NBANews]] = {}
    for i in range(0, len(names), 500):
        q = session.query(NBANews).filter(NBANews.player_name.in_(names[i:i + 500]), NBANews.first_seen >= now - timedelta(hours=hours))
        for r in q.all():
            out.setdefault(r.player_name, []).append(r)
    for k in out:
        out[k].sort(key=lambda r: (r.first_seen, r.id), reverse=True)
    return out


def age(ts: datetime, now: Optional[datetime] = None) -> str:
    now = now or datetime.now()
    s = max(0.0, (now - ts).total_seconds())
    if s < 90:
        return "just now"
    if s < 3600:
        return f"{int(s // 60)}m ago"
    if s < 36 * 3600:
        return f"{s / 3600:.0f}h ago"
    return f"{ts:%m-%d %H:%M}"
