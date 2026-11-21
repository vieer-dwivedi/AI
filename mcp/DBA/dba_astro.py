"""DBA A.3: ayanamsa and node options on top of Swiss Ephemeris (pyswisseph, already installed).
Facts only. Returns the option used with every value.
"""
from datetime import datetime
from fractions import Fraction

import swisseph as swe

import dba_vimshottari as v

AYANAMSAS = {
    "LAHIRI": "SIDM_LAHIRI",
    "KP_NEW": "SIDM_KRISHNAMURTI_VP291",
    "KP_OLD": "SIDM_KRISHNAMURTI",
    "RAMAN": "SIDM_RAMAN",
    "FAGAN_BRADLEY": "SIDM_FAGAN_BRADLEY",
    "TRUE_CITRA": "SIDM_TRUE_CITRA",
}
DEFAULT_AYANAMSA = "LAHIRI"


def _mode(name):
    key = name.upper()
    if key not in AYANAMSAS:
        raise ValueError("unknown ayanamsa %r; options: %s" % (name, ", ".join(AYANAMSAS)))
    return getattr(swe, AYANAMSAS[key]), key


def julian_day_ut(utc: datetime):
    return swe.julday(utc.year, utc.month, utc.day,
                      utc.hour + utc.minute / 60.0 + utc.second / 3600.0 + utc.microsecond / 3.6e9, swe.GREG_CAL)


def moon(utc: datetime, ayanamsa=DEFAULT_AYANAMSA):
    mode, key = _mode(ayanamsa)
    swe.set_sid_mode(mode)
    jd = julian_day_ut(utc)
    res = swe.calc_ut(jd, swe.MOON, swe.FLG_SWIEPH | swe.FLG_SIDEREAL | swe.FLG_SPEED)[0]
    nak, i, frac = v.moon_start(Fraction(res[0]))
    return {"longitude_deg": res[0], "speed_deg_per_day": res[3], "ayanamsa": key,
            "ayanamsa_deg": swe.get_ayanamsa_ut(jd), "nakshatra_index": nak,
            "maha_lord": v.LORDS[i], "fraction_elapsed": float(frac)}


def rahu_ketu(utc: datetime, ayanamsa=DEFAULT_AYANAMSA, node="mean"):
    if node not in ("mean", "true"):
        raise ValueError("node must be 'mean' or 'true'")
    mode, key = _mode(ayanamsa)
    swe.set_sid_mode(mode)
    body = swe.MEAN_NODE if node == "mean" else swe.TRUE_NODE
    lon = swe.calc_ut(julian_day_ut(utc), body, swe.FLG_SWIEPH | swe.FLG_SIDEREAL)[0][0]
    return {"Rahu": lon, "Ketu": (lon + 180) % 360, "node": node, "ayanamsa": key}


def ayanamsa_shift_table(utc: datetime, year_mode="julian", names=None, baseline=DEFAULT_AYANAMSA):
    """Per ayanamsa: Moon longitude, first Maha lord, and the shift of every Maha boundary versus `baseline` (days)."""
    names = list(names or AYANAMSAS)
    base = moon(utc, baseline)
    base_ms = v.mahadashas(Fraction(base["longitude_deg"]), year_mode)
    rows = []
    for n in names:
        m = moon(utc, n)
        ms = v.mahadashas(Fraction(m["longitude_deg"]), year_mode)
        same = ms[0].lord == base_ms[0].lord
        rows.append({"ayanamsa": m["ayanamsa"], "moon_deg": m["longitude_deg"],
                     "diff_deg_vs_baseline": m["longitude_deg"] - base["longitude_deg"],
                     "maha_lord": m["maha_lord"], "first_maha_lord_changes": not same,
                     "boundary_shift_days": float(ms[0].end - base_ms[0].end) if same else None})
    return rows
