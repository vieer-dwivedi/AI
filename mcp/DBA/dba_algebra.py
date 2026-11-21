"""DBA A.5: period algebra on exact intervals. Windows are half-open [start, end), Fractions of a day from birth.
Every list-returning search has a `truncated` flag; nothing enumerates the whole 9^6 tree by default.
"""
from dataclasses import dataclass
from fractions import Fraction

from dba_core import Period, to_fraction
import dba_vimshottari as v


@dataclass(frozen=True)
class Window:
    start: Fraction
    end: Fraction
    sources: tuple = ()          # lord paths (or tags) that produced it

    @property
    def duration(self):
        return self.end - self.start


def _norm(items):
    out = []
    for it in items:
        s, e = it.start, it.end
        if e > s:
            out.append((s, e, (getattr(it, "path", None),) if hasattr(it, "path") else tuple(getattr(it, "sources", ()))))
    return sorted(out, key=lambda x: (x[0], x[1]))


def merge_adjacent(items):
    """Merge overlapping or touching intervals into windows (keeps the source paths)."""
    out = []
    for s, e, src in _norm(items):
        if out and s <= out[-1].end:
            last = out[-1]
            out[-1] = Window(last.start, max(last.end, e), last.sources + src)
        else:
            out.append(Window(s, e, src))
    return out


def union(*lists):
    return merge_adjacent([w for l in lists for w in l])


def intersect(a, b):
    A, B = merge_adjacent(a), merge_adjacent(b)
    out, i, j = [], 0, 0
    while i < len(A) and j < len(B):
        s, e = max(A[i].start, B[j].start), min(A[i].end, B[j].end)
        if s < e:
            out.append(Window(s, e, A[i].sources + B[j].sources))
        if A[i].end < B[j].end:
            i += 1
        else:
            j += 1
    return out


def gaps(items, lo, hi):
    lo, hi = to_fraction(lo), to_fraction(hi)
    out, cur = [], lo
    for w in merge_adjacent(items):
        if w.end <= lo or w.start >= hi:
            continue
        if w.start > cur:
            out.append(Window(cur, min(w.start, hi)))
        cur = max(cur, w.end)
    if cur < hi:
        out.append(Window(cur, hi))
    return out


def coverage_fraction(items, lo, hi):
    """Exact share of [lo, hi) covered by the windows."""
    lo, hi = to_fraction(lo), to_fraction(hi)
    covered = sum(min(w.end, hi) - max(w.start, lo) for w in merge_adjacent(items) if w.end > lo and w.start < hi)
    return covered / (hi - lo)


def find_periods(moon_deg, predicate, level, t_from, t_to, year_days=v.DEFAULT_YEAR_MODE, max_results=10000):
    """All level-`level` periods overlapping [t_from, t_to) whose Period satisfies predicate(period).

    Children of non-overlapping parents are never generated. Returns {"periods", "truncated", "searched"}.
    """
    if not 1 <= level <= len(v.LEVEL_NAMES):
        raise ValueError("level must be 1..6")
    lo, hi = to_fraction(t_from), to_fraction(t_to)
    yd = v.year_days_from(year_days)
    cycles = int(hi / (v.TOTAL_YEARS * v.nominal_year_days(yd))) + 2
    found, truncated = [], False

    def walk(p):
        nonlocal truncated
        if truncated or p.end <= lo or p.start >= hi:
            return
        if p.level == level:
            if predicate(p):
                if len(found) >= max_results:
                    truncated = True
                    return
                found.append(p)
            return
        for c in v.children(p):
            walk(c)

    for m in v.mahadashas(moon_deg, yd, count=9 * cycles):
        walk(m)
        if truncated or m.start >= hi:
            break
    return {"periods": found, "truncated": truncated, "searched": (lo, hi)}
