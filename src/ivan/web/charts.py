# src/ivan/web/charts.py
"""
Tiny dependency-free SVG charts for the web UI (the app vendors everything;
no chart library). Coordinates are computed here so templates only |safe them.
"""
from __future__ import annotations

import math
from typing import Callable, Iterable, Optional

ACCENT, TOP, GREY, INK, MUTED, LINE = "#1f5fbf", "#d97706", "#8a8a8a", "#1b1b1b", "#6b6b6b", "#e2e2e2"


def _ticks(lo: float, hi: float, n: int = 5) -> list[float]:
    span = (hi - lo) or 1.0
    raw = span / n
    mag = 10 ** math.floor(math.log10(raw))
    step = mag
    for m in (1, 2, 2.5, 5, 10):
        step = m * mag
        if span / step <= n:
            break
    start = math.ceil(lo / step) * step
    return [round(start + k * step, 10) for k in range(int((hi - start) / step) + 1)]


def _esc(s: str) -> str:
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


class _Frame:
    """Data -> pixel mapping with padded extents and a drawn axis frame."""

    def __init__(self, xs, ys, width, height, left=58, right=16, top=16, bottom=44, pad=0.05):
        self.w, self.h, self.left, self.right, self.top, self.bottom = width, height, left, right, top, bottom
        xs, ys = list(xs) or [0.0, 1.0], list(ys) or [0.0, 1.0]
        xlo, xhi, ylo, yhi = min(xs), max(xs), min(ys), max(ys)
        xspan, yspan = (xhi - xlo) or 1.0, (yhi - ylo) or 1.0
        self.xlo, self.xhi = xlo - pad * xspan, xhi + pad * xspan
        self.ylo, self.yhi = ylo - pad * yspan, yhi + pad * yspan
        self.pw, self.ph = width - left - right, height - top - bottom

    def px(self, x: float) -> float:
        return self.left + (x - self.xlo) / (self.xhi - self.xlo) * self.pw

    def py(self, y: float) -> float:
        return self.top + self.ph - (y - self.ylo) / (self.yhi - self.ylo) * self.ph

    def axes(self, xlabel: str, ylabel: str, xfmt: Callable, yfmt: Callable) -> list[str]:
        out = [f'<rect x="{self.left}" y="{self.top}" width="{self.pw}" height="{self.ph}" fill="#fff" stroke="{LINE}"/>']
        for t in _ticks(self.xlo, self.xhi):
            if self.xlo <= t <= self.xhi:
                x = self.px(t)
                out.append(f'<line x1="{x:.1f}" y1="{self.top}" x2="{x:.1f}" y2="{self.top + self.ph}" stroke="{LINE}"/>')
                out.append(f'<text x="{x:.1f}" y="{self.top + self.ph + 14}" font-size="10" text-anchor="middle" fill="{MUTED}">{_esc(xfmt(t))}</text>')
        for t in _ticks(self.ylo, self.yhi):
            if self.ylo <= t <= self.yhi:
                y = self.py(t)
                out.append(f'<line x1="{self.left}" y1="{y:.1f}" x2="{self.left + self.pw}" y2="{y:.1f}" stroke="{LINE}"/>')
                out.append(f'<text x="{self.left - 5}" y="{y + 3.5:.1f}" font-size="10" text-anchor="end" fill="{MUTED}">{_esc(yfmt(t))}</text>')
        out.append(f'<text x="{self.left + self.pw / 2:.1f}" y="{self.h - 8}" font-size="11" text-anchor="middle" fill="{INK}">{_esc(xlabel)}</text>')
        out.append(f'<text transform="translate(13,{self.top + self.ph / 2:.1f}) rotate(-90)" font-size="11" text-anchor="middle" fill="{INK}">{_esc(ylabel)}</text>')
        return out


def scatter_svg(points: Iterable[dict], *, xlabel: str, ylabel: str, xfmt: Callable = lambda v: f"{v:g}",
                yfmt: Callable = lambda v: f"{v:g}", width: int = 560, height: int = 400,
                iso_lines: Optional[list] = None, legend: Optional[list] = None) -> str:
    """
    points: {x, y, kind: cand|pick|top, shade: 0..1, label, title}. Candidates
    are grey (darker = higher shade), quality-only picks orange rings, picks
    blue with their pick number. iso_lines: {x1,y1,x2,y2,label} in data units.
    """
    points = list(points)
    f = _Frame([p["x"] for p in points], [p["y"] for p in points], width, height)
    parts = [f'<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:{width}px;height:auto;font-family:system-ui,sans-serif">']
    parts += f.axes(xlabel, ylabel, xfmt, yfmt)
    for ln in iso_lines or []:
        parts.append(f'<line x1="{f.px(ln["x1"]):.1f}" y1="{f.py(ln["y1"]):.1f}" x2="{f.px(ln["x2"]):.1f}" y2="{f.py(ln["y2"]):.1f}" '
                     f'stroke="{TOP}" stroke-dasharray="4 4" stroke-width="1" opacity="0.8"/>')
        if ln.get("label"):
            mx, my = (f.px(ln["x1"]) + f.px(ln["x2"])) / 2, (f.py(ln["y1"]) + f.py(ln["y2"])) / 2
            parts.append(f'<text x="{mx + 4:.1f}" y="{my - 4:.1f}" font-size="9" fill="{TOP}">{_esc(ln["label"])}</text>')
    for kind, r in (("cand", 3.0), ("top", 6.0), ("pick", 6.5)):
        for p in points:
            if p.get("kind", "cand") != kind:
                continue
            x, y, t = f.px(p["x"]), f.py(p["y"]), _esc(p.get("title", ""))
            if kind == "cand":
                parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="{GREY}" opacity="{0.2 + 0.6 * float(p.get("shade", 0.5)):.2f}"><title>{t}</title></circle>')
            elif kind == "top":
                parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="none" stroke="{TOP}" stroke-width="1.8"><title>{t}</title></circle>')
            else:
                parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="{ACCENT}" stroke="#fff" stroke-width="1"><title>{t}</title></circle>')
                if p.get("label"):
                    parts.append(f'<text x="{x:.1f}" y="{y + 3:.1f}" font-size="8" font-weight="700" text-anchor="middle" fill="#fff" pointer-events="none">{_esc(p["label"])}</text>')
    if legend:
        y0 = f.top + 10
        for i, (kind, text) in enumerate(legend):
            y = y0 + i * 14
            x = f.left + f.pw - 150
            if kind == "cand":
                parts.append(f'<circle cx="{x}" cy="{y}" r="3" fill="{GREY}" opacity="0.6"/>')
            elif kind == "top":
                parts.append(f'<circle cx="{x}" cy="{y}" r="5" fill="none" stroke="{TOP}" stroke-width="1.8"/>')
            elif kind == "pick":
                parts.append(f'<circle cx="{x}" cy="{y}" r="5" fill="{ACCENT}"/>')
            else:
                parts.append(f'<line x1="{x - 6}" y1="{y}" x2="{x + 6}" y2="{y}" stroke="{TOP}" stroke-dasharray="4 4"/>')
            parts.append(f'<text x="{x + 10}" y="{y + 3.5}" font-size="10" fill="{INK}">{_esc(text)}</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def bars_svg(trace: list[dict], lam: float, quality_only: list[float], *, width: int = 1000, height: int = 300) -> str:
    """
    One stacked bar per pick: quality (dark) plus lambda x diversity (light) =
    the score that won the pick. The dashed line is the quality of the k-th
    best lineup by quality alone, so the gap to the dark bar is what the k-th
    pick gave up for diversity.
    """
    n = len(trace)
    if n == 0:
        return ""
    top = max([t["score"] for t in trace] + quality_only[:n] + [0.0]) * 1.08
    f = _Frame([0.5, n + 0.5], [0.0, top], width, height, pad=0.0)
    parts = [f'<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:{width}px;height:auto;font-family:system-ui,sans-serif">']
    parts += f.axes("pick number (the order the greedy chooser took them)", "score", lambda v: f"{v:.0f}", lambda v: f"{v:.2f}")
    bw = max(2.0, f.pw / n * 0.7)
    for t in trace:
        x = f.px(t["pick"]) - bw / 2
        y_q, y_s = f.py(t["q"]), f.py(t["score"])
        tip = _esc(f"pick {t['pick']}: lineup {t['lineup_id']} · quality {t['q']:.3f} + {lam:g} x diversity {t['div']:.3f} = {t['score']:.3f}")
        parts.append(f'<rect x="{x:.1f}" y="{y_q:.1f}" width="{bw:.1f}" height="{f.py(0) - y_q:.1f}" fill="{ACCENT}"><title>{tip}</title></rect>')
        parts.append(f'<rect x="{x:.1f}" y="{y_s:.1f}" width="{bw:.1f}" height="{max(0.0, y_q - y_s):.1f}" fill="{ACCENT}" opacity="0.35"><title>{tip}</title></rect>')
    if quality_only:
        pts = " ".join(f"{f.px(k + 1):.1f},{f.py(v):.1f}" for k, v in enumerate(quality_only[:n]))
        parts.append(f'<polyline points="{pts}" fill="none" stroke="{TOP}" stroke-width="1.5" stroke-dasharray="5 4"/>')
    x, y = f.left + 10, f.top + 12
    parts.append(f'<rect x="{x}" y="{y - 6}" width="10" height="10" fill="{ACCENT}"/><text x="{x + 14}" y="{y + 3}" font-size="10">quality of the pick</text>')
    parts.append(f'<rect x="{x + 140}" y="{y - 6}" width="10" height="10" fill="{ACCENT}" opacity="0.35"/><text x="{x + 154}" y="{y + 3}" font-size="10">+ lambda x diversity</text>')
    parts.append(f'<line x1="{x + 300}" y1="{y}" x2="{x + 316}" y2="{y}" stroke="{TOP}" stroke-dasharray="5 4" stroke-width="1.5"/><text x="{x + 320}" y="{y + 3}" font-size="10">k-th best by quality alone</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def iso_quality_lines(w1: float, w2: float, cash_range: list, roi_range: list, levels=(0.25, 0.5, 0.75)) -> list[dict]:
    """
    Lines of equal quality on the cash/ROI plane. Quality is w1*cash' + w2*roi'
    with both features normalised to [0, 1]; each line is where that equals
    level x (w1 + w2). Returned in data units, clipped to the normalised square.
    """
    out = []
    total = w1 + w2
    if total <= 0:
        return out
    (clo, chi), (rlo, rhi) = cash_range, roi_range
    for lv in levels:
        c = lv * total
        cands = []
        for xn in (0.0, 1.0):                        # crossings with the vertical edges
            if w2 > 0:
                yn = (c - w1 * xn) / w2
                if 0.0 <= yn <= 1.0:
                    cands.append((xn, yn))
        for yn in (0.0, 1.0):                        # crossings with the horizontal edges
            if w1 > 0:
                xn = (c - w2 * yn) / w1
                if 0.0 <= xn <= 1.0:
                    cands.append((xn, yn))
        cands = sorted(set((round(a, 6), round(b, 6)) for a, b in cands))
        if len(cands) >= 2:
            (x1, y1), (x2, y2) = cands[0], cands[-1]
            out.append({"x1": clo + x1 * (chi - clo), "y1": rlo + y1 * (rhi - rlo),
                        "x2": clo + x2 * (chi - clo), "y2": rlo + y2 * (rhi - rlo), "label": f"quality {c:.2f}"})
    return out
