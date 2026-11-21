"""DBA A.2: local civil time -> UTC with explicit rules. Facts and warnings only.
Returns utc_used, offset used, tz_rule_used, warnings. Never guesses silently.
"""
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

UTC = ZoneInfo("UTC")


def _offset(naive, tz, fold):
    return naive.replace(tzinfo=tz, fold=fold).utcoffset()


def to_utc(local_naive, tz_name=None, longitude_deg=None, ambiguous="earlier", nonexistent="shift_forward"):
    """local_naive: naive datetime (wall clock). Exactly one of tz_name / longitude_deg (local mean time).

    ambiguous  : 'earlier' (first occurrence, DST) | 'later' | 'reject'
    nonexistent: 'shift_forward' (apply the pre-gap offset, i.e. wall time + gap) | 'reject'
    """
    warnings = []
    if (tz_name is None) == (longitude_deg is None):
        raise ValueError("give exactly one of tz_name or longitude_deg")
    if tz_name is None:
        off = timedelta(seconds=round(float(longitude_deg) / 15 * 3600))
        return {"utc_used": local_naive - off, "offset_hours": off.total_seconds() / 3600,
                "tz_rule_used": "LMT longitude %.4f" % float(longitude_deg), "warnings": warnings}
    tz = ZoneInfo(tz_name)
    o0, o1 = _offset(local_naive, tz, 0), _offset(local_naive, tz, 1)
    rule = "tzdata %s" % tz_name
    if o0 != o1:
        # Ambiguous (clocks went back) or nonexistent (clocks went forward)
        round_trip = local_naive.replace(tzinfo=tz, fold=0).astimezone(UTC).astimezone(tz).replace(tzinfo=None)
        if round_trip != local_naive:
            if nonexistent == "reject":
                raise ValueError("nonexistent local time %s in %s" % (local_naive, tz_name))
            off = o0
            warnings.append("nonexistent local time (DST gap); used offset before the gap")
            rule += " (gap, shift_forward)"
        else:
            if ambiguous == "reject":
                raise ValueError("ambiguous local time %s in %s" % (local_naive, tz_name))
            fold = 0 if ambiguous == "earlier" else 1
            off = _offset(local_naive, tz, fold)
            warnings.append("ambiguous local time (clocks went back); used the %s occurrence" % ambiguous)
            rule += " (fold=%d)" % fold
    else:
        off = o0
    name = local_naive.replace(tzinfo=tz).tzname()
    if name == "LMT" or (tz_name in ("Asia/Kolkata", "Asia/Calcutta") and local_naive < datetime(1906, 1, 1)):
        warnings.append("pre-1906 or LMT zone data: India had local mean times; verify the source of the birth time")
    return {"utc_used": local_naive - off, "offset_hours": off.total_seconds() / 3600,
            "tz_rule_used": rule, "warnings": warnings}
