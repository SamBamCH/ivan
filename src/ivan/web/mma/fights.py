# src/ivan/web/mma/fights.py
"""
The slate page's Fights section: ivan.sports.mma.fight_report.card_report (scripts/mma/fights.py prints the same) with
two drawings per fight, both dependency-free SVG:

    ends_svg    one bar the width of the fight: the first fighter's wins from the left (blue), the second's from the
                right (orange), each split by round with the decision (lighter) in the middle
    dist_svg    the two fighters' fantasy points, the first's bars up and the second's down from one axis; the axis and
                the bar scale are the card's, so one fight's picture compares with the next
"""
from __future__ import annotations

from ivan.sports.mma.fight import DECISION
from ivan.web.charts import _esc

A, A_LIGHT, B, B_LIGHT = "#1f5fbf", "#a9c4ee", "#d97706", "#f3cf9a"       # first fighter: finish, decision; second fighter likewise
INK, MUTED, LINE = "#1b1b1b", "#6b6b6b", "#e2e2e2"
FONT = "font-family:system-ui,sans-serif"


def ends_svg(fight: dict, width: int = 620, height: int = 46) -> str:
    """Who wins and how. Segments left to right: first fighter R1..Rn, decision | second fighter decision, Rn..R1."""
    a, b = fight["fighters"]
    segs = [(e, "p1", A_LIGHT if e["key"] == DECISION else A, a) for e in fight["ends"]]
    segs += [(e, "p2", B_LIGHT if e["key"] == DECISION else B, b) for e in reversed(fight["ends"])]
    top, bar_h, gap = 16, 22, 2
    out = [f'<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" role="img" style="width:100%;max-width:{width}px;height:auto;{FONT}">',
           f'<text x="0" y="11" font-size="11" fill="{INK}"><tspan font-weight="600">{_esc(a["name"].title())}</tspan> wins {a["win"]:.0%}</text>',
           f'<text x="{width}" y="11" font-size="11" text-anchor="end" fill="{INK}"><tspan font-weight="600">{_esc(b["name"].title())}</tspan> wins {b["win"]:.0%}</text>']
    x = 0.0
    for e, side, colour, who in segs:
        p = e[side]
        w = p * width
        if w <= 0:
            continue
        draw_w = max(w - gap, 0.6)
        how = "by decision" if e["key"] == DECISION else f"by a finish in round {e['key'][1:]}"
        out.append(f'<g><title>{_esc(who["name"].title())} {how}: {p:.1%} of worlds</title>'
                   f'<rect x="{x + gap / 2:.1f}" y="{top}" width="{draw_w:.1f}" height="{bar_h}" rx="2" fill="{colour}"/>')
        if draw_w >= 40:
            dark = e["key"] != DECISION
            out.append(f'<text x="{x + w / 2:.1f}" y="{top + 15}" font-size="10.5" text-anchor="middle" fill="{"#fff" if dark else INK}">{e["label"]} {p:.0%}</text>')
        elif draw_w >= 20:
            out.append(f'<text x="{x + w / 2:.1f}" y="{top + 15}" font-size="10.5" text-anchor="middle" fill="{"#fff" if e["key"] != DECISION else INK}">{e["label"]}</text>')
        out.append("</g>")
        x += w
    out.append("</svg>")
    return "".join(out)


def dist_svg(fight: dict, axis: dict, width: int = 620, height: int = 168) -> str:
    """The two fighters' fantasy points per `axis["bin"]` points: the first up, the second down, one x axis between them."""
    a, b = fight["fighters"]
    left, right, mid_h = 34, 8, 16                       # the middle band holds the x labels
    pw = width - left - right
    half = (height - mid_h) / 2 - 4
    y_up, y_dn = 4 + half, 4 + half + mid_h              # the two baselines
    step, xmax, ymax = axis["bin"], axis["xmax"], max(axis["ymax"], 1e-9)
    n = len(a["hist"])
    bw = pw / n

    def px(v: float) -> float:
        return left + min(max(v, 0.0), xmax) / xmax * pw

    out = [f'<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" role="img" style="width:100%;max-width:{width}px;height:auto;{FONT}">']
    for frac in (0.5, 1.0):                              # two quiet gridlines a side, labelled once
        for y in (y_up - frac * half, y_dn + frac * half):
            out.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width - right}" y2="{y:.1f}" stroke="{LINE}"/>')
            out.append(f'<text x="{left - 4}" y="{y + 3.5:.1f}" font-size="9.5" text-anchor="end" fill="{MUTED}">{frac * ymax:.0%}</text>')
    tick = 20 if xmax <= 160 else 50
    for v in range(0, xmax + 1, tick):
        out.append(f'<text x="{px(v):.1f}" y="{y_up + mid_h - 4.5:.1f}" font-size="9.5" text-anchor="middle" fill="{MUTED}">{v}{"+" if v == xmax else ""}</text>')
    for s, colour, base, sign in ((a, A, y_up, -1), (b, B, y_dn, 1)):
        for i, share in enumerate(s["hist"]):
            if share <= 0:
                continue
            h = max(share / ymax * half, 1.0)
            x0 = left + i * bw + 1
            y0 = base - h if sign < 0 else base
            lo, last = i * step, i == n - 1
            rng_text = f"{lo}+ points" if last else f"{lo} to {lo + step} points"
            out.append(f'<g><title>{_esc(s["name"].title())}: {rng_text} in {share:.1%} of worlds</title>'
                       f'<rect x="{x0:.1f}" y="{y0:.1f}" width="{bw - 2:.1f}" height="{h:.1f}" rx="2" fill="{colour}"/></g>')
        out.append(f'<line x1="{left}" y1="{base}" x2="{width - right}" y2="{base}" stroke="{MUTED}"/>')
        mx = px(s["mean"])
        y_far = base + sign * half
        out.append(f'<g><title>{_esc(s["name"].title())}: mean {s["mean"]:.1f} points</title>'
                   f'<line x1="{mx:.1f}" y1="{base}" x2="{mx:.1f}" y2="{y_far:.1f}" stroke="{INK}" stroke-width="1.5" stroke-dasharray="3 3"/></g>')
        label_y = 14 if sign < 0 else height - 6
        out.append(f'<text x="{width - right}" y="{label_y}" font-size="11" text-anchor="end" fill="{INK}">'
                   f'<tspan fill="{colour}">■</tspan> <tspan font-weight="600">{_esc(s["name"].title())}</tspan> mean {s["mean"]:.1f}</text>')
    out.append("</svg>")
    return "".join(out)


def view(report: dict) -> dict:
    """The report with each fight's drawings, and the card's fighters in one list (salary order) for the table at the top."""
    fighters = []
    for f in report["fights"]:
        f["ends_svg"], f["dist_svg"] = ends_svg(f), dist_svg(f, report["axis"])
        a, b = f["fighters"]
        f["title"] = f"{a['name'].title()} vs {b['name'].title()}"
        for s, opp in ((a, b), (b, a)):
            fighters.append(dict(s, fight=f["id"], opponent=opp["name"]))
    report["fighters"] = sorted(fighters, key=lambda s: -s["salary"])
    return report
