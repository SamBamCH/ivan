"""
The slate's news table and the team states (mapped / open question / changed) the chips and needs-you read.
"""
from __future__ import annotations

from datetime import datetime
from ivan.sports.nba import newsfeed as NF
from typing import Optional


class NewsMixin:
    def news_table(self, now: Optional[datetime] = None) -> dict:
        """
        The slate's news table (newsfeed.slate_news, the same lists scripts/nba/newsfeed.py slate prints): the open
        questions, the players OUT tonight who played their team's last game, and the standing outs, each with the latest
        headline on file; this desk adds its calls (a pin or a starter call), the model's minutes, its starter calls and each
        team's mapped time. Built once per request: the same `now` object gets the same table (the menu, the team states,
        the header and every card's questions read it).
        """
        now = now or datetime.now()
        memo = getattr(self, "_news_memo", None)
        if memo is not None and memo[0] is now:
            return memo[1]
        calls, minutes, mapped, starters = {}, {}, {}, {}
        for t in self.slate.teams():
            ms = self.mapped_state(t)
            if ms.get("state") in ("mapped", "changed") and ms.get("saved_at"):
                try:
                    mapped[t] = datetime.fromisoformat(str(ms["saved_at"]))
                except ValueError:
                    pass
            proj = self.proj.get(t)
            if proj is not None and "minutes" in proj:
                minutes.update({str(n): float(v) for n, v in proj["minutes"].items()})
            starters[t] = set(self.starters_of(t) or [])
        for n, ov in self.overrides.items():
            if ov.get("minutes") is not None:
                calls[n] = "scratched" if ov["minutes"] <= 0 else f"pinned {ov['minutes']:g} min"
            elif ov.get("starter") is not None:
                calls[n] = "starts" if ov["starter"] > 0 else "benched"
        tbl = NF.slate_news(self.session, self.slate, now=now, calls=calls, minutes=minutes, mapped_at=mapped, starters=starters)
        for q in tbl["questionable"]:
            q["answered"] = q["call"] is not None
        self._news_memo = (now, tbl)
        return tbl

    def questions(self, team: str, now: Optional[datetime] = None) -> list:
        """
        The team's rows of the news table's questions (DK status Q / GTD / D / P, not ruled out): who they are, what they
        usually play, whether you have made a call (a minutes pin or a starter call answers the question), and the latest
        news on file with its age and whether it arrived after the team was mapped. Open questions first, then the most
        minutes at stake. No questions = good to go at lock, barring a surprise lineup.
        """
        now = now or datetime.now()
        qs = [dict(q) for q in self.news_table(now)["questionable"] if q["team"] == team]
        qs.sort(key=lambda q: (q["answered"], -(q["hist"] or 0)))
        return qs

    def questions_summary(self, now: Optional[datetime] = None) -> dict:
        now = now or datetime.now()
        teams = sorted({p.team for p in self.slate.players if p.team})
        by = {t: self.questions(t, now) for t in teams}
        by = {t: q for t, q in by.items() if q}
        return dict(n=sum(len(q) for q in by.values()), open=sum(1 for q in by.values() for x in q if not x["answered"]),
                    with_news=sum(1 for q in by.values() for x in q if x["news"]), new=sum(1 for q in by.values() for x in q if x["news"] and x["news"]["new"]),
                    teams={t: [f"{x['raw'].split()[-1]} {x['status']}" + (" ✓" if x["answered"] else "") for x in q] for t, q in by.items()})

    def news_status(self, now: Optional[datetime] = None) -> dict:
        now = now or datetime.now()
        pool = [p.name for p in self.slate.players]
        recent = NF.latest(self.session, pool, hours=24, now=now)
        n = sum(len(v) for v in recent.values())
        last = getattr(self, "news_checked_at", None)
        return dict(checked_at=last, checked=NF.age(last, now) if last else None, auto=bool(getattr(self, "news_auto", False)), n_slate=n,
                    newest=max((v[0].first_seen for v in recent.values()), default=None))

    def team_states(self, now: Optional[datetime] = None) -> list:
        """
        One chip per team on the slate: green = mapped and no open question; yellow = an open question (a questionable
        player who could shake the rotation), mapped or not; red = mapped but the inputs moved since; grey = unmapped
        with nothing open. The header's one-line read of the slate.
        """
        now = now or datetime.now()
        out = []
        for t in sorted({p.team for p in self.slate.players if p.team}):
            ms = self.mapped_state(t); qs = self.questions(t, now)
            open_q = [q for q in qs if not q["answered"]]
            if ms["state"] == "changed":
                colour, why = "changed", "mapped, but changed since: " + ", ".join(ms["diff"])
            elif open_q:
                colour, why = "open", f"{len(open_q)} open question(s): " + ", ".join(f"{q['raw']} {q['status']}" for q in open_q) + (" · mapped" if ms["state"] == "mapped" else " · unmapped")
            elif ms["state"] == "mapped":
                colour, why = "ready", "mapped, no open questions"
            else:
                colour, why = "unmapped", "unmapped" + (f" · {', '.join(ms['diff'])}" if ms.get("diff") else "")
            out.append(dict(team=t, state=ms["state"], colour=colour, why=why, open=len(open_q), new=sum(1 for q in open_q if q["news"] and q["news"]["new"])))
        return out
