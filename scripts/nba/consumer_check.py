#!/usr/bin/env python3
"""
scripts/nba/consumer_check.py

Drive the consumer's desk (the server's /nba, the brother's) from the producer and check the round trip, so a
"does it work for you?" question has an answer without a browser on the other machine.

    scripts/nba/consumer_check.py                      the public path (Cloudflare tunnel): https://nba.prattflix.com/nba
    scripts/nba/consumer_check.py --tailnet            the tailnet path: http://ivan.prattflix.com/nba
    scripts/nba/consumer_check.py --url http://100.102.143.78/nba
    scripts/nba/consumer_check.py --refresh            also press Refresh there and show what the desk said
    scripts/nba/consumer_check.py --cookie TOKEN       use this login cookie instead of minting one over ssh

The login: the desk's cookie is signed with the secret in the server's data/.desk_secret, so a token minted
there (ssh <host> venv/bin/python -c 'from ivan.web import auth; print(auth.token())') logs in without the
password. Nothing is typed, stored or sent but that token.

Checks (PASS / FAIL / INFO):
  login      no cookie -> redirected to /login; the cookie -> the page
  page       the shell renders; the header's feed badge (laptop autopilot, who published, when)
  board      the board partial renders with team cards; the players, late swap and exposures views answer
  csv        /export/projections.csv: 200, Cache-Control no-store (the Cloudflare cache fix of 2026-10-26),
             cf-cache-status not HIT, the stamp in the filename, the rows; the projection matches the bundle the
             laptop published (same stamp on both machines), with the differences counted
  lineups    /export/lineups.csv answers
  feed       the server's pull state vs this machine's feed manifest: same live stamp = the brother has what you published
  refresh    (--refresh) POST /refresh, then the board's flash: "nothing new ..." or "the published projection is in"
"""
from __future__ import annotations

import argparse
import csv
import http.cookiejar
import io
import json
import re
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from ivan.core.utils import PROJECT_ROOT

PUBLIC = "https://nba.prattflix.com/nba"
TAILNET = "http://ivan.prattflix.com/nba"
COOKIE = "ivan_desk"


class Report:
    def __init__(self):
        self.rows: list[tuple[str, str, str]] = []

    def add(self, name: str, ok: bool | None, note: str) -> None:
        self.rows.append((name, "PASS" if ok else ("INFO" if ok is None else "FAIL"), note))
        print(f"{self.rows[-1][1]:4} {name:9} {note}")

    @property
    def failed(self) -> bool:
        return any(r[1] == "FAIL" for r in self.rows)


def ssh(host: str, cmd: str, timeout: int = 30) -> str:
    r = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", host, cmd], capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError((r.stderr or r.stdout).strip().splitlines()[-1] if (r.stderr or r.stdout).strip() else f"ssh exit {r.returncode}")
    return r.stdout.strip()


def mint_cookie(host: str, root: str) -> str:
    return ssh(host, f"cd {root} && venv/bin/python -c 'from ivan.web import auth; print(auth.token())'")


class Client:
    """A browser with one cookie: GET/POST with redirects followed; the response headers kept."""

    def __init__(self, base: str, cookie: str | None):
        self.base = base.rstrip("/")
        self.jar = http.cookiejar.CookieJar()
        self.opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(self.jar))
        self.cookie = cookie

    def _req(self, path: str, data: dict | None = None, cookie: bool = True, timeout: int = 60):
        url = self.base + path
        body = urllib.parse.urlencode(data).encode() if data is not None else None
        req = urllib.request.Request(url, data=body, method="POST" if data is not None else "GET")
        if cookie and self.cookie:
            req.add_header("Cookie", f"{COOKIE}={self.cookie}")
        req.add_header("User-Agent", "ivan-consumer-check")
        try:
            with self.opener.open(req, timeout=timeout) as r:
                return r.status, dict(r.headers), r.read(), r.geturl()
        except urllib.error.HTTPError as e:
            return e.code, dict(e.headers), e.read(), e.geturl()

    def get(self, path: str, **kw):
        return self._req(path, None, **kw)

    def post(self, path: str, data: dict, **kw):
        return self._req(path, data, **kw)


def text(b: bytes) -> str:
    return b.decode("utf-8", errors="replace")


def strip_tags(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", s)).strip()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default=None, help=f"the desk's base URL (default {PUBLIC})")
    ap.add_argument("--tailnet", action="store_true", help=f"use {TAILNET}")
    ap.add_argument("--host", default="ivan-server", help="ssh alias of the server (to mint the cookie and read the pull state)")
    ap.add_argument("--remote-root", default="/opt/ivan/app")
    ap.add_argument("--cookie", default=None, help="a login cookie value; default: minted over ssh")
    ap.add_argument("--refresh", action="store_true", help="press Refresh on the consumer and report what it said")
    args = ap.parse_args()
    base = args.url or (TAILNET if args.tailnet else PUBLIC)
    rep = Report()
    print(f"consumer desk: {base}")

    # the login
    try:
        cookie = args.cookie or mint_cookie(args.host, args.remote_root)
    except Exception as e:   # noqa: BLE001
        rep.add("login", False, f"could not mint a cookie on {args.host}: {e}")
        return 1
    c = Client(base, cookie)
    st, hdr, body, url = c.get("/", cookie=False)
    rep.add("login", url.endswith("/login") or st in (401, 403), f"without a cookie: {st} -> {url.replace(base, '')}")
    st, hdr, body, url = c.get("/")
    page = text(body)
    ok = st == 200 and "NBA desk" in page and not url.endswith("/login")
    rep.add("login", ok, f"with the cookie: {st} {url.replace(base, '') or '/'}")
    if not ok:
        return 1

    # the page and its header
    m = re.search(r'<span class="news-sum muted" title="the laptop publishes[^"]*">(.*?)</span>', page, re.S)
    badge = strip_tags(m.group(1)) if m else None
    rep.add("page", badge is not None, f"feed badge: {badge or 'absent (the server runs code without it, or it is not a consumer)'}")
    fresh = re.findall(r'<span class="f[^"]*" title="([^"]*)">', page)
    pub = [t.replace("&#39;", "'") for t in fresh if "last publish" in t]
    rep.add("page", None, "freshness: " + ("; ".join(pub) if pub else "no 'published' entry"))
    stale = "new data · load" in page
    rep.add("page", None, "stale badge shown (a pull landed since the page was built)" if stale else "page is current with its pool")

    # the board and the views
    st, hdr, body, url = c.get("/board?view=games")
    board = text(body)
    games = len(re.findall(r'<div class="gm', board))
    teams = re.findall(r'board\?view=games&(?:amp;)?team=([A-Z]{2,4})', board)
    rep.add("board", st == 200 and games > 0, f"{st}, {games} games in the menu, {len(set(teams))} team links")
    if teams:                                                # the games view shows one team card at a time: open the first
        st, hdr, body, url = c.get(f"/board?view=games&team={teams[0]}")
        card = text(body)
        rows = len(re.findall(r'<tr class="p\b|<tr data-player|<tr class="row', card)) or card.count("<tr")
        rep.add("board", st == 200 and f'id="card-{teams[0]}"' in card, f"team card {teams[0]}: {st}, {rows} rows" + (" · says published" if "published" in card else ""))
    for view in ("players", "lateswap", "exposures"):
        st, hdr, body, url = c.get(f"/board?view={view}")
        rep.add("board", st == 200, f"view {view}: {st}, {len(body) // 1024} KB")

    # the CSV exports
    st, hdr, body, url = c.get("/export/projections.csv")
    cc = hdr.get("Cache-Control", hdr.get("cache-control", ""))
    cf = hdr.get("cf-cache-status", hdr.get("CF-Cache-Status", ""))
    fn = re.search(r'filename="([^"]+)"', hdr.get("Content-Disposition", hdr.get("content-disposition", "")) or "")
    rows = list(csv.reader(io.StringIO(text(body)))) if st == 200 else []
    rep.add("csv", st == 200 and len(rows) > 1, f"projections.csv: {st}, {max(0, len(rows) - 1)} rows, file {fn.group(1) if fn else '?'}")
    rep.add("csv", "no-store" in cc, f"Cache-Control: {cc or 'none'}" + (f" · cf-cache-status: {cf}" if cf else " · not through Cloudflare"))
    if cf:
        rep.add("csv", cf.upper() != "HIT", f"Cloudflare served it {'from its cache' if cf.upper() == 'HIT' else 'from the origin'}")
    st2, hdr2, body2, _ = c.get("/export/lineups.csv")
    rep.add("lineups", st2 == 200, f"lineups.csv: {st2}, {len(body2.splitlines())} lines")

    # the feed: what the server applied vs what this machine published
    try:
        remote = json.loads(ssh(args.host, f"cat {args.remote_root}/data/feed-state/nba.json"))
        local = json.loads((PROJECT_ROOT / "data" / "feed" / "nba" / "manifest.json").read_text())
        rs, ls = (remote.get("live") or {}).get("stamp"), (local.get("live") or {}).get("stamp")
        rep.add("feed", rs == ls, f"server applied live {rs} (pulled {(remote.get('live') or {}).get('pulled_at')}); laptop published {ls}"
                + ("" if rs == ls else " -> the server has not pulled the latest bundle yet"))
        prod = local.get("producer") or {}
        rep.add("feed", None, f"laptop's manifest: published {local.get('published_at')}, by {prod.get('source', '?')}, autopilot {'on' if prod.get('autopilot') else 'off'}")
    except Exception as e:   # noqa: BLE001
        rep.add("feed", False, f"could not compare the feed state: {e}")
        rs = ls = None

    # the projection on the consumer vs this machine's pool, when both hold the same bundle
    if rows and rs == ls:
        try:
            from ivan.database.session import get_session, init_db
            from ivan.sports.nba.models import NBAPlayer
            from ivan.core.utils import normalize_name
            sess = get_session(init_db("nba"))
            mine = {normalize_name(p.name): (p.fpts_mean, p.draft_group_id) for p in sess.query(NBAPlayer).all()}
            hdrs = rows[0]; i_name, i_fpts, i_team = hdrs.index("name"), hdrs.index("fpts"), hdrs.index("team")
            diffs, n, dgs = [], 0, set()
            for r in rows[1:]:
                k = normalize_name(r[i_name])
                if k in mine and r[i_fpts] not in ("", None) and mine[k][0] is not None:
                    n += 1; d = abs(float(r[i_fpts]) - float(mine[k][0]))
                    if d > 0.5:
                        diffs.append((r[i_name], r[i_team], float(r[i_fpts]), float(mine[k][0])))
            same_slate = len({v[1] for v in mine.values()} & {None} ) == 0
            rep.add("csv", None if diffs else True, f"{n} players compared with this machine's pool; {len(diffs)} differ by more than 0.5 fpts"
                    + (f" (his own edits or a newer local projection): " + ", ".join(f"{a} {c:.1f} vs {d:.1f}" for a, b, c, d in diffs[:4]) if diffs else ""))
        except Exception as e:   # noqa: BLE001
            rep.add("csv", None, f"projection comparison skipped: {e}")

    # press Refresh there
    if args.refresh:
        st, hdr, body, url = c.post("/refresh", {"what": "dk"})
        st, hdr, body, url = c.get("/board?view=games")
        said = list(dict.fromkeys(strip_tags(x) for x in re.findall(r'<div class="flash">(.*?)</div>', text(body), re.S)))
        rep.add("refresh", bool(said), "the desk said: " + (" | ".join(said[:3]) if said else "nothing visible (no flash found on the board)"))
        st, hdr, body, url = c.get("/")
        m = re.search(r'<span class="news-sum muted" title="the laptop publishes[^"]*">(.*?)</span>', text(body), re.S)
        rep.add("refresh", m is not None, "header after the pull: " + (strip_tags(m.group(1)) if m else "no feed badge"))

    print()
    print(f"{'all checks passed' if not rep.failed else 'FAILURES above'} · {base}")
    return 1 if rep.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
