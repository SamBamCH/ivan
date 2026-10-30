#!/usr/bin/env python3
"""
scripts/nba/consumer_check.py

Drive the consumer's desk (the server's /nba, the brother's) from the producer and check the round trip and the
files he will actually use, so a "does it work for you?" question has an answer without a browser on the other machine.

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
  page       the shell renders; the header's feed badge (laptop autopilot, who published, when); the loaded slate
  board      the games menu renders, a team card opens; the players, late swap and exposures views answer
  feed       the server's pull state vs this machine's feed manifest: same live stamp = he has what you published
  csv        /export/projections.csv: 200, Cache-Control no-store (the Cloudflare cache fix of 2026-10-26),
             cf-cache-status not HIT, the stamp in the filename is the pool's projection time, the columns, every pooled
             player has a projection, the row count is the pool; when both machines hold the same bundle the fpts of the
             teams he did not edit equal this machine's pool (his edited teams are listed, their differences expected)
  lineups    /export/lineups.csv and ?ids=1: the slate in the filename is the loaded slate (else the export fell back to
             an older slate with lineups: he has not generated for this one), eight slots, eight distinct players per
             row, every player in the pool, salary under the cap, slot eligibility by position, the ids match the pool
  lateswap   the view renders; the newest late-swap upload CSV it offers downloads and has the entry columns, eight
             slots and an id in every cell (ids of the loaded slate when the file is from today)
  refresh    (--refresh) POST /refresh, then the board's flash and the header afterwards
"""
from __future__ import annotations

import argparse
import csv
import http.cookiejar
import io
import json
import re
import subprocess
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime

from ivan.core.utils import PROJECT_ROOT, normalize_name

PUBLIC = "https://nba.prattflix.com/nba"
TAILNET = "http://ivan.prattflix.com/nba"
COOKIE = "ivan"
SLOTS = ("PG", "SG", "SF", "PF", "C", "G", "F", "UTIL")
ELIGIBLE = {"PG": {"PG"}, "SG": {"SG"}, "SF": {"SF"}, "PF": {"PF"}, "C": {"C"}, "G": {"PG", "SG"}, "F": {"SF", "PF"}, "UTIL": {"PG", "SG", "SF", "PF", "C"}}
SALARY_CAP = 50000


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
        out = (r.stderr or r.stdout).strip()
        raise RuntimeError(out.splitlines()[-1] if out else f"ssh exit {r.returncode}")
    return r.stdout.strip()


def mint_cookie(host: str, root: str) -> str:
    # PYTHONPATH=src: that tree's own code and data root, even where the venv's editable install points at another checkout (the sandbox)
    return ssh(host, f"cd {root} && PYTHONPATH={root}/src venv/bin/python -c 'from ivan.web import auth; print(auth.token())'")


class Client:
    """A browser with one cookie: GET/POST with redirects followed; the response headers kept."""

    def __init__(self, base: str, cookie: str | None):
        self.base = base.rstrip("/")
        self.opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))
        self.cookie = cookie

    def _req(self, path: str, data: dict | None = None, cookie: bool = True, timeout: int = 90):
        body = urllib.parse.urlencode(data).encode() if data is not None else None
        req = urllib.request.Request(self.base + path, data=body, method="POST" if data is not None else "GET")
        if cookie and self.cookie:
            req.add_header("Cookie", f"{COOKIE}={self.cookie}")
        req.add_header("User-Agent", "ivan-consumer-check")
        try:
            with self.opener.open(req, timeout=timeout) as r:
                return r.status, {k.lower(): v for k, v in r.headers.items()}, r.read(), r.geturl()
        except urllib.error.HTTPError as e:
            return e.code, {k.lower(): v for k, v in e.headers.items()}, e.read(), e.geturl()

    def get(self, path: str, **kw):
        return self._req(path, None, **kw)

    def post(self, path: str, data: dict, **kw):
        return self._req(path, data, **kw)


def text(b: bytes) -> str:
    return b.decode("utf-8", errors="replace")


def strip_tags(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", s)).strip().replace("&#39;", "'").replace("&amp;", "&")


def rows_of(b: bytes) -> list[list[str]]:
    return list(csv.reader(io.StringIO(text(b))))


def filename(hdr: dict) -> str:
    m = re.search(r'filename="([^"]+)"', hdr.get("content-disposition", "") or "")
    return m.group(1) if m else ""


def check_projections(rep: Report, rows: list[list[str]], fname: str, page: str, mine: dict | None, his_teams: set | None) -> dict:
    """The projections CSV as he would open it. Returns {normalized name: row dict} for the lineups check."""
    need = ["name", "dk_id", "team", "position", "salary", "status", "minutes", "fpts", "fpts_sd", "public_fpts", "ownership", "in_pool"]
    hdr = rows[0] if rows else []
    missing = [c for c in need if c not in hdr]
    rep.add("csv", not missing, f"columns: {len(hdr)}" + (f", missing {missing}" if missing else ", all the expected ones present"))
    if missing:
        return {}
    ix = {c: hdr.index(c) for c in hdr}
    recs = {normalize_name(r[ix["name"]]): {c: r[i] for c, i in ix.items()} for r in rows[1:] if len(r) == len(hdr)}
    pooled = [r for r in recs.values() if r["in_pool"] == "1"]
    blank = [r["name"] for r in pooled if r["fpts"] in ("", None)]
    rep.add("csv", not blank, f"{len(recs)} players, {len(pooled)} in the pool with minutes" + (f"; {len(blank)} pooled without a projection: {blank[:4]}" if blank else ", every one projected"))
    # the stamp in the filename is the pool's projection time shown in the header (pool HH:MM in the brand's hover)
    m_pool = re.search(r"pool (\d\d:\d\d)", page)
    m_file = re.search(r"_(\d{8})-(\d{4})\.csv$", fname)
    if m_pool and m_file:
        hhmm = m_file.group(2)[:2] + ":" + m_file.group(2)[2:]
        rep.add("csv", hhmm == m_pool.group(1), f"file stamp {m_file.group(1)}-{hhmm} vs the desk's pool time {m_pool.group(1)}" + ("" if hhmm == m_pool.group(1) else " -> the export is not the pool the page shows"))
    if mine is not None:
        same_n = len(recs) == len(mine)
        rep.add("csv", same_n, f"row count {len(recs)} vs this machine's pool {len(mine)}" + ("" if same_n else " -> not the same pool"))
        diffs, edited, n = [], [], 0
        for k, r in recs.items():
            if k not in mine or r["fpts"] in ("", None) or mine[k][0] is None:
                continue
            n += 1
            d = abs(float(r["fpts"]) - float(mine[k][0]))
            if d > 0.05:
                (edited if his_teams is not None and r["team"] in his_teams else diffs).append((r["name"], r["team"], float(r["fpts"]), float(mine[k][0])))
        by_team = sorted({t for _, t, _, _ in diffs})
        rep.add("csv", not diffs, f"{n} projections compared with this machine's pool: {len(diffs)} differ on teams he did not edit"
                + (f" ({', '.join(by_team)}): " + ", ".join(f"{a} {c:.1f} vs {d:.1f}" for a, _, c, d in diffs[:4]) if diffs else ""))
        if edited:
            rep.add("csv", None, f"{len(edited)} differ on his edited teams ({', '.join(sorted({t for _, t, _, _ in edited}))}), as expected: "
                    + ", ".join(f"{a} {c:.1f} vs {d:.1f}" for a, _, c, d in edited[:4]))
    return recs


def check_lineups(rep: Report, c: Client, recs: dict, slate: str | None) -> None:
    st, hdr, body, _ = c.get("/export/lineups.csv")
    fn = filename(hdr)
    rows = rows_of(body) if st == 200 else []
    if st != 200 or len(rows) < 2 or (rows and rows[0] and rows[0][0].startswith("no generated lineups")):
        note = rows[0][0] if rows and rows[0] else f"{st}, {len(rows)} rows"
        rep.add("lineups", None if st == 200 else False, f"lineups.csv ({fn or 'no file'}): {note}")
        return
    m = re.search(r"nba_lineups_(\d+)_", fn)
    dg = m.group(1) if m else None
    if slate and dg:
        rep.add("lineups", dg == slate, f"{fn}: slate {dg} vs the loaded slate {slate}" + ("" if dg == slate else " -> the export fell back to an older slate with lineups; he has not generated for this one"))
    rep.add("lineups", tuple(rows[0]) == SLOTS, f"header {rows[0]}" + ("" if tuple(rows[0]) == SLOTS else f" (expected {list(SLOTS)})"))
    body_rows = [r for r in rows[1:] if any(r)]
    bad_len = [i for i, r in enumerate(body_rows, 2) if len(r) != 8 or any(not x for x in r)]
    dupes = [i for i, r in enumerate(body_rows, 2) if len(set(r)) != len(r)]
    rep.add("lineups", not bad_len and not dupes, f"{len(body_rows)} lineups, eight filled slots each" + (f"; {len(bad_len)} rows short: lines {bad_len[:5]}" if bad_len else "") + (f"; {len(dupes)} rows with a repeated player: lines {dupes[:5]}" if dupes else ""))
    if recs:
        unknown, over, wrong_slot, n = set(), [], [], 0
        for i, r in enumerate(body_rows, 2):
            if len(r) != 8:
                continue
            n += 1
            sal = 0
            for slot, name in zip(SLOTS, r):
                rec = recs.get(normalize_name(name))
                if rec is None:
                    unknown.add(name); continue
                sal += int(float(rec["salary"] or 0))
                pos = set((rec["position"] or "").replace("-", "/").split("/"))
                if pos and not (pos & ELIGIBLE[slot]):
                    wrong_slot.append((i, slot, name, rec["position"]))
            if sal > SALARY_CAP:
                over.append((i, sal))
        rep.add("lineups", not unknown, f"players in the pool: {'all' if not unknown else str(len(unknown)) + ' not in the projections CSV: ' + ', '.join(sorted(unknown)[:5])}")
        rep.add("lineups", not over, f"salary under {SALARY_CAP}: {'every lineup' if not over else str(len(over)) + ' over, e.g. line ' + str(over[0][0]) + ' at ' + str(over[0][1])}")
        rep.add("lineups", not wrong_slot, "slot eligibility: " + ("every slot holds an eligible position" if not wrong_slot else f"{len(wrong_slot)} wrong, e.g. line {wrong_slot[0][0]} {wrong_slot[0][1]} = {wrong_slot[0][2]} ({wrong_slot[0][3]})"))
        st2, hdr2, body2, _ = c.get("/export/lineups.csv?ids=1")
        idrows = [r for r in rows_of(body2)[1:] if any(r)] if st2 == 200 else []
        ids = {r["dk_id"] for r in recs.values() if r.get("dk_id")}
        bad_ids = {x for r in idrows for x in r if x not in ids}
        rep.add("lineups", st2 == 200 and len(idrows) == len(body_rows) and not bad_ids,
                f"ids variant: {st2}, {len(idrows)} rows" + (f", {len(bad_ids)} ids not in the pool: {sorted(bad_ids)[:5]}" if bad_ids else ", every id in the pool"))


def check_lateswap(rep: Report, c: Client, recs: dict, slate: str | None) -> None:
    st, hdr, body, _ = c.get("/board?view=lateswap")
    view = text(body)
    rep.add("lateswap", st == 200 and ("late swap" in view.lower() or "standings" in view.lower()), f"view: {st}, {len(body) // 1024} KB")
    links = re.findall(r'/download/(lateswap[^"\']+\.csv)', view)
    if not links:
        rep.add("lateswap", None, "no late-swap upload CSV offered yet (none run on the consumer); after lock, run it and re-check")
        return
    name = sorted(set(links))[-1]
    st, hdr, body, _ = c.get(f"/download/{urllib.parse.quote(name)}")
    rows = rows_of(body) if st == 200 else []
    if st != 200 or len(rows) < 2:
        rep.add("lateswap", False, f"{name}: {st}, {len(rows)} rows")
        return
    h = rows[0]
    slots = [h.index(s) for s in SLOTS if s in h]
    has_entry = any(x.lower().startswith("entry id") for x in h)
    rep.add("lateswap", len(slots) == 8 and has_entry, f"{name}: {len(rows) - 1} entries, columns {h[:4]}... with {len(slots)} slot columns" + ("" if has_entry else ", no Entry ID column"))
    cells = [r[i] for r in rows[1:] for i in slots if len(r) > max(slots)]
    with_id = [re.search(r"\((\d+)\)", x) for x in cells]
    empty = sum(1 for x in cells if not x.strip())
    rep.add("lateswap", empty == 0 and all(with_id), f"{len(cells)} slot cells: {empty} empty, {sum(1 for m in with_id if m is None)} without a player id")
    m = re.search(r"lateswap\.(\d{8})", name)
    today = m and m.group(1) == datetime.now().strftime("%Y%m%d")
    if recs and today:
        ids = {r["dk_id"] for r in recs.values() if r.get("dk_id")}
        foreign = {m.group(1) for m in with_id if m and m.group(1) not in ids}
        rep.add("lateswap", not foreign, f"today's file: {'every player is in the loaded slate' if not foreign else str(len(foreign)) + ' ids not in the loaded slate ' + str(slate)}")
    elif m:
        rep.add("lateswap", None, f"the file is from {m.group(1)}, not today: its players belong to that slate")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default=None, help=f"the desk's base URL (default {PUBLIC})")
    ap.add_argument("--tailnet", action="store_true", help=f"use {TAILNET}")
    ap.add_argument("--host", default="ivan-server", help="ssh alias of the server (to mint the cookie and read its files)")
    ap.add_argument("--remote-root", default="/opt/ivan/app")
    ap.add_argument("--cookie", default=None, help="a login cookie value; default: minted over ssh")
    ap.add_argument("--refresh", action="store_true", help="press Refresh on the consumer and report what it said")
    ap.add_argument("--sandbox", action="store_true", help="the sandbox consumer on the server (deploy/sandbox_server.sh) through the ssh tunnel at 127.0.0.1:8002")
    args = ap.parse_args()
    if args.sandbox:
        args.url = args.url or "http://127.0.0.1:8002/nba"
        args.remote_root = "/opt/ivan/sandbox/app" if args.remote_root == "/opt/ivan/app" else args.remote_root
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

    # press Refresh there first, so every check below sees what he would see after pressing it
    if args.refresh:
        c.post("/refresh", {"what": "dk"})
        st, hdr, body, url = c.get("/board?view=games")
        said = list(dict.fromkeys(strip_tags(x) for x in re.findall(r'<div class="flash">(.*?)</div>', text(body), re.S)))
        rep.add("refresh", bool(said) and not any("failed" in s for s in said), "the desk said: " + (" | ".join(said[:3]) if said else "nothing visible (no flash found on the board)"))
        st, hdr, body, url = c.get("/")
        page = text(body)

    # the page and its header
    m = re.search(r'<span class="news-sum muted" title="the laptop publishes[^"]*">(.*?)</span>', page, re.S)
    badge = strip_tags(m.group(1)) if m else None
    rep.add("page", badge is not None, f"feed badge: {badge or 'absent (the server runs code without it, or it is not a consumer)'}")
    fresh = [t for t in re.findall(r'<span class="f[^"]*" title="([^"]*)">', page) if "last publish" in t]
    rep.add("page", None, "freshness: " + ("; ".join(strip_tags(t) for t in fresh) if fresh else "no 'published' entry"))
    m = re.search(r'title="slate (\d+) locks ([^·"]*)', page)
    slate = m.group(1) if m else None
    rep.add("page", slate is not None, f"loaded slate {slate} locks {m.group(2).strip() if m else '?'}" if m else "no slate loaded")
    rep.add("page", None, "stale badge shown (a pull landed since the page was built)" if "new data · load" in page else "page is current with its pool")

    # the board and the views
    st, hdr, body, url = c.get("/board?view=games")
    board = text(body)
    games = len(re.findall(r'<div class="gm', board))
    teams = re.findall(r'board\?view=games&(?:amp;)?team=([A-Z]{2,4})', board)
    rep.add("board", st == 200 and games > 0, f"{st}, {games} games in the menu, {len(set(teams))} team links")
    if teams:                                                # the games view shows one team card at a time: open the first
        st, hdr, body, url = c.get(f"/board?view=games&team={teams[0]}")
        card = text(body)
        rep.add("board", st == 200 and f'id="card-{teams[0]}"' in card, f"team card {teams[0]}: {st}, {card.count('<tr')} rows" + (" · says published" if "published" in card else ""))
    for view in ("players", "exposures"):
        st, hdr, body, url = c.get(f"/board?view={view}")
        rep.add("board", st == 200, f"view {view}: {st}, {len(body) // 1024} KB")

    # the feed: what the server applied vs what this machine published; his own edited teams
    mine = his_teams = None
    rs = ls = None
    try:
        remote = json.loads(ssh(args.host, f"cat {args.remote_root}/data/feed-state/nba.json"))
        local = json.loads((PROJECT_ROOT / "data" / "feed" / "nba" / "manifest.json").read_text())
        rs, ls = (remote.get("live") or {}).get("stamp"), (local.get("live") or {}).get("stamp")
        rep.add("feed", rs == ls, f"server applied live {rs} (pulled {(remote.get('live') or {}).get('pulled_at')}); laptop published {ls}"
                + ("" if rs == ls else " -> the server has not pulled the latest bundle yet (its timer runs every 5 minutes; his Refresh pulls at once)"))
        prod = local.get("producer") or {}
        rep.add("feed", None, f"laptop's manifest: published {local.get('published_at')}, by {prod.get('source', '?')}, autopilot {'on' if prod.get('autopilot') else 'off'}")
    except Exception as e:   # noqa: BLE001
        rep.add("feed", False, f"could not compare the feed state: {e}")
    try:
        his = ssh(args.host, f"cd {args.remote_root}/data/nba/input && (cat minutes_overrides.csv 2>/dev/null; echo ==; ls rotations 2>/dev/null)")
        ov, _, rot = his.partition("==")
        his_names = [normalize_name(r[0]) for r in csv.reader(io.StringIO(ov.strip())) if r and r[0] and r[0].lower() not in ("player", "name")]
        his_teams = {p.replace(".json", "") for p in rot.split() if p.endswith(".json")}
        his_edit_names = set(his_names)
    except Exception as e:   # noqa: BLE001
        his_edit_names = set()
        rep.add("feed", None, f"his own edits unknown: {e}")
    if rs == ls and rs:
        try:
            from ivan.database.session import get_session, init_db
            from ivan.sports.nba.models import NBAPlayer
            sess = get_session(init_db("nba"))
            mine = {normalize_name(p.name): (p.fpts_mean, p.team) for p in sess.query(NBAPlayer).all()}
            his_teams = (his_teams or set()) | {mine[n][1] for n in his_edit_names if n in mine}
            if his_teams:
                rep.add("feed", None, f"his own edits on: {', '.join(sorted(his_teams))}")
        except Exception as e:   # noqa: BLE001
            rep.add("feed", None, f"this machine's pool not readable: {e}")

    # the CSV exports
    st, hdr, body, url = c.get("/export/projections.csv")
    cc, cf, fn = hdr.get("cache-control", ""), hdr.get("cf-cache-status", ""), filename(hdr)
    rows = rows_of(body) if st == 200 else []
    rep.add("csv", st == 200 and len(rows) > 1, f"projections.csv: {st}, {max(0, len(rows) - 1)} rows, file {fn or '?'}")
    rep.add("csv", "no-store" in cc, f"Cache-Control: {cc or 'none'}" + (f" · cf-cache-status: {cf}" if cf else " · not through Cloudflare"))
    if cf:
        rep.add("csv", cf.upper() != "HIT", f"Cloudflare served it {'from its cache' if cf.upper() == 'HIT' else 'from the origin'}")
    recs = check_projections(rep, rows, fn, page, mine, his_teams) if rows else {}
    check_lineups(rep, c, recs, slate)
    check_lateswap(rep, c, recs, slate)

    print()
    print(f"{'all checks passed' if not rep.failed else 'FAILURES above'} · {base}")
    return 1 if rep.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
