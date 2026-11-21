"""DBA core: exact period arithmetic (no floats in boundaries).

Times are Fractions of a day measured from the birth instant (birth = 0).
A boundary belongs to the NEW period: a period is the half-open interval [start, end).
Rendering to a date happens only at the edge (render()), rounded to the nearest second.
"""
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from fractions import Fraction
import math

SECONDS_PER_DAY = 86400


def to_fraction(x):
    """Exact Fraction from int / str ('365.25') / float (exact binary value) / Fraction."""
    if isinstance(x, Fraction):
        return x
    if isinstance(x, bool):
        raise TypeError("bool is not a number")
    if isinstance(x, int):
        return Fraction(x)
    if isinstance(x, str):
        return Fraction(Decimal(x.strip()))
    return Fraction(x)


@dataclass(frozen=True)
class Period:
    path: tuple          # lord names from Maha down, e.g. ("Venus", "Sun")
    start: Fraction      # days from birth, inclusive
    end: Fraction        # days from birth, exclusive
    # Only for non-linear year modes (true_solar_arc): dasha-year coordinates and the time map.
    # Excluded from equality/hash so fixed-year behaviour is unchanged.
    tau_start: Fraction = field(default=None, compare=False)
    tau_end: Fraction = field(default=None, compare=False)
    tmap: object = field(default=None, compare=False, repr=False)

    @property
    def level(self):
        return len(self.path)

    @property
    def lord(self):
        return self.path[-1]

    @property
    def duration(self):
        return self.end - self.start

    def contains(self, t):
        return self.start <= t < self.end


def render(birth: datetime, t) -> datetime:
    """Birth datetime + t days, rounded to the nearest second (half up)."""
    seconds = math.floor(to_fraction(t) * SECONDS_PER_DAY + Fraction(1, 2))
    return birth + timedelta(seconds=seconds)


def is_contiguous(periods):
    return all(a.end == b.start for a, b in zip(periods, periods[1:]))


def find_running(periods, t):
    """The period of a sorted, contiguous list that contains t (start inclusive), or None."""
    t = to_fraction(t)
    for p in periods:
        if p.start <= t < p.end:
            return p
    return None
