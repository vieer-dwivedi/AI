"""Transit engine, part 4: Tajaka / Varshaphal (annual, monthly, sixty-hour charts), Muntha, lord of the year,
Tajaka yogas and the 36 Sahams.

Builds on `transit_engine`. PyJHora conventions (see progress.md): `annual_chart(years=N)` is the N-th year of life, so
age A (completed years) uses years=A+1; `monthly_chart(years, months=M)` has M=1 at the solar return itself;
`lord_of_the_month(years_from_dob=A, months_from_dob=M-1)`.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

import transit_engine as E
from state import BirthState
from transit_engine import NAKSHATRAS, PLANETS, SIGNS, TransitError

SIGN_LORDS = [2, 5, 3, 1, 0, 3, 5, 2, 4, 6, 6, 4]
ITHASALA_TYPES = {1: "Varthamaana (present)", 2: "Poorna (complete, within 1 degree)", 3: "Bhavishya (future)"}
MAX_AGE = 120


def _age_arg(state: BirthState, age: Optional[int], year: Optional[int]) -> int:
    """Completed years of age whose birthday solar return is wanted."""
    if age is not None:
        if not 0 <= age <= MAX_AGE:
            raise TransitError(f"`age` must be between 0 and {MAX_AGE}.")
        return int(age)
    dob = state.date_of_birth
    if year is not None:
        if not dob.year <= year <= dob.year + MAX_AGE:
            raise TransitError(f"`year` must be between {dob.year} and {dob.year + MAX_AGE}.")
        return int(year - dob.year)
    now = E.now_local(state)
    return max(0, now.year - dob.year - ((now.month, now.day) < (dob.month, dob.day)))


def _chart_rows(chart: list[Any], natal_lagna_sign: Optional[int] = None) -> dict[str, Any]:
    """PyJHora chart ([['L',(sign,deg)],[0,(sign,deg)],...]) -> lagna + planet dicts with houses from that chart's lagna."""
    lagna_lon = None
    lons: dict[int, float] = {}
    for row in chart:
        ident, (sign, deg) = row[0], row[1]
        lon = (int(sign) * 30.0 + float(deg)) % 360.0
        if ident == "L":
            lagna_lon = lon
        else:
            lons[int(ident)] = lon
    if lagna_lon is None or len(lons) < 9:
        raise TransitError("Unexpected Tajaka chart format from PyJHora.")
    lagna_sign = int(lagna_lon // 30)
    planets = []
    for p in range(9):
        d = E.describe_longitude(lons[p])
        planets.append({"planet": PLANETS[p], "sign": d["sign"], "degree_in_sign": d["degree_in_sign"],
                        "nakshatra": d["nakshatra"], "pada": d["pada"],
                        "house_from_chart_lagna": E._house(lagna_sign, d["sign_index"])})
    return {"lagna": {k: v for k, v in E.describe_longitude(lagna_lon).items()
                      if k in ("sign", "degree_in_sign", "nakshatra", "pada")},
            "lagna_sign_index": lagna_sign, "planets": planets, "lons": lons}


def _fmt_pravesh(stamp: Any) -> str:
    """PyJHora's [(y,m,d), 'H:M:S'-ish] -> ISO string (local time at the birth place)."""
    (y, m, d), hms = stamp
    if isinstance(hms, (tuple, list)):
        hh, mm, ss = int(hms[0]), int(hms[1]), int(float(hms[2]))
    else:
        parts = str(hms).replace("h", ":").replace("m", ":").replace("s", "").split(":")
        hh, mm, ss = (int(float(x)) for x in (parts + ["0", "0"])[:3])
    return f"{int(y):04d}-{int(m):02d}-{int(d):02d}T{hh:02d}:{mm:02d}:{ss:02d}"


def _yoga_pairs(rows: list[tuple]) -> list[dict[str, Any]]:
    return [{"planets": [PLANETS[r[0]], PLANETS[r[1]]], **({"type": ITHASALA_TYPES.get(r[2], str(r[2]))} if len(r) > 2 else {})}
            for r in rows]


def _tajaka_yogas(b: Any, ctx: Any, chart: list[Any], annual_jd: float) -> dict[str, Any]:
    """Run PyJHora's Tajaka yoga finders on an annual chart; each one independently guarded."""
    ty = b.tajaka_yoga

    out: dict[str, Any] = {}
    errors: dict[str, str] = {}

    def run(name: str, fn: Any, *args: Any) -> Any:
        try:
            return E._quiet(fn, *args)
        except Exception as exc:  # PyJHora helpers are uneven; report instead of failing the whole tool
            errors[name] = f"{type(exc).__name__}: {exc}"
            return None

    ith = run("ithasala", ty.get_ithasala_yoga_planet_pairs, chart)
    if ith is not None:
        out["ithasala"] = _yoga_pairs(ith)
    ees = run("eesarpha", ty.get_eesarpha_yoga_planet_pairs, chart)
    if ees is not None:
        out["eesarpha"] = _yoga_pairs(ees)
    kam = run("kamboola", ty.get_kamboola_yoga_planet_pairs, chart)
    if kam is not None:
        present, kpairs, _ = kam
        out["kamboola"] = {"present": bool(present), "moon_ithasala_pairs": _yoga_pairs(kpairs)}
    rad = run("radda", ty.get_radda_yoga_planet_pairs, chart)
    if rad is not None:
        out["radda"] = _yoga_pairs(rad)
    nak = run("nakta", ty.get_nakta_yoga_planet_triples, chart)
    if nak is not None:
        out["nakta"] = [[PLANETS[x] for x in t] for t in nak]
    yam = run("yamaya", ty.get_yamaya_yoga_planet_triples, chart)
    if yam is not None:
        out["yamaya"] = [[PLANETS[x] for x in t] for t in yam]
    duh = run("duhphali_kutta", ty.get_duhphali_kutta_yoga_planet_pairs, annual_jd, ctx.place)
    if duh is not None:
        out["duhphali_kutta"] = _yoga_pairs([tuple(x) for x in duh])
    # Induvara / Ishkavala: PyJHora's versions are unreliable ("TODO Not Working"), so computed here.
    rows = _chart_rows(chart)
    houses = {E._house(rows["lagna_sign_index"], int(E.describe_longitude(rows["lons"][p])["sign_index"])) for p in range(7)}
    apoklima = {3, 6, 9, 12}
    out["ishkavala"] = bool(houses) and houses.isdisjoint(apoklima)
    out["induvara"] = bool(houses) and houses <= apoklima
    out["notes"] = ("Ithasala types: 1 Varthamaana, 2 Poorna, 3 Bhavishya. Ishkavala = Sun..Saturn only in kendras/panapharas "
                    "from the annual lagna; Induvara = only in apoklimas (own computation). Khallasara is not implemented "
                    "in PyJHora and Manahoo is skipped (its PyJHora code mixes up Mars/Saturn list indices).")
    if errors:
        out["errors"] = errors
    return out


def get_varshaphal(state: BirthState, age: Optional[int] = None, year: Optional[int] = None,
                   include_yogas: bool = True) -> dict[str, Any]:
    """Tajaka annual chart (Varsha Pravesh): the solar-return moment, annual lagna and planets, Muntha, lord of the
    year and Tajaka yogas. `age` = completed years (default: current age); or give a calendar `year`."""
    with E._calc(state) as (b, ctx):
        a = _age_arg(state, age, year)
        years = a + 1
        chart, stamp = E._quiet(b.tajaka.annual_chart, ctx.jd, ctx.place, 1, years)
        annual_jd = E._quiet(b.drik.next_solar_date, ctx.jd, ctx.place, years=years)
        rows = _chart_rows(chart)
        nat = E._natal(state, b, ctx)
        natal_lagna_sign = int(E.describe_longitude(nat.lagna)["sign_index"])
        muntha = (natal_lagna_sign + a) % 12
        try:
            lord_year = E._quiet(b.tajaka.lord_of_the_year, ctx.jd, ctx.place, a)
            lord_year_name = PLANETS[int(lord_year)]
        except Exception as exc:
            lord_year_name = f"unavailable ({type(exc).__name__})"
        out: dict[str, Any] = {
            "state": state.summary(), "age_completed": a, "year_of_life": years,
            "solar_return_local": _fmt_pravesh(stamp),
            "annual_lagna": rows["lagna"], "planets": rows["planets"],
            "muntha": {"sign": SIGNS[muntha], "house_from_annual_lagna": E._house(rows["lagna_sign_index"], muntha),
                       "lord": PLANETS[SIGN_LORDS[muntha]],
                       "rule": "Muntha = natal lagna sign advanced by one sign per completed year."},
            "lord_of_the_year": lord_year_name,
            "natal_lagna": SIGNS[natal_lagna_sign],
        }
        if include_yogas:
            out["tajaka_yogas"] = _tajaka_yogas(b, ctx, chart, annual_jd)
        return out


def get_monthly_chart(state: BirthState, age: Optional[int] = None, year: Optional[int] = None, month: int = 1,
                      sixty_hour: Optional[int] = None) -> dict[str, Any]:
    """Tajaka monthly chart (Maasa Pravesh): `month` 1..12 of the year of life (1 = the solar return itself), with the
    lord of the month. Optional `sixty_hour` 1..12 gives the 60-hour (2.5-day) chart inside that month."""
    if not 1 <= month <= 12:
        raise TransitError("`month` must be 1..12 (1 = the month that starts at the solar return).")
    if sixty_hour is not None and not 1 <= sixty_hour <= 12:
        raise TransitError("`sixty_hour` must be 1..12 (each is 2.5 days).")
    with E._calc(state) as (b, ctx):
        a = _age_arg(state, age, year)
        years = a + 1
        if sixty_hour is None:
            chart, stamp = E._quiet(b.tajaka.monthly_chart, ctx.jd, ctx.place, 1, years, month)
            kind = "monthly"
        else:
            chart, stamp = E._quiet(b.tajaka.sixty_hour_chart, ctx.jd, ctx.place, 1, years, month, sixty_hour)
            kind = "sixty_hour"
        rows = _chart_rows(chart)
        try:
            lm = E._quiet(b.tajaka.lord_of_the_month, ctx.jd, ctx.place, a, month - 1)
            lord_month = PLANETS[int(lm)]
        except Exception as exc:
            lord_month = f"unavailable ({type(exc).__name__})"
        return {"state": state.summary(), "kind": kind, "age_completed": a, "month_of_year": month,
                "sixty_hour_index": sixty_hour, "chart_moment_local": _fmt_pravesh(stamp),
                "lagna": rows["lagna"], "planets": rows["planets"], "lord_of_the_month": lord_month,
                "note": "The lord of the month is for the monthly chart; it is not recomputed for the 60-hour chart."}


def get_sahams(state: BirthState, age: Optional[int] = None, year: Optional[int] = None) -> dict[str, Any]:
    """The 36 Sahams (Arabic parts) of the natal chart (no age/year) or of the Varsha (annual) chart for that age/year."""
    with E._calc(state) as (b, ctx):
        if age is None and year is None:
            jd, label = ctx.jd, "natal"
        else:
            a = _age_arg(state, age, year)
            jd = E._quiet(b.drik.next_solar_date, ctx.jd, ctx.place, years=a + 1)
            label = f"annual chart at age {a}"
        names = sorted(n for n in dir(b.saham) if n.endswith("_saham") and not n.startswith("_"))
        rows, errors = [], {}
        for n in names:
            try:
                lon = float(E._quiet(getattr(b.saham, n), jd, ctx.place))
            except Exception as exc:
                errors[n] = f"{type(exc).__name__}: {exc}"
                continue
            d = E.describe_longitude(lon)
            rows.append({"saham": n.replace("_saham", "").replace("_", " ").title(), "sign": d["sign"],
                         "degree_in_sign": d["degree_in_sign"], "nakshatra": d["nakshatra"]})
        out = {"state": state.summary(), "chart": label, "count": len(rows), "sahams": rows}
        if errors:
            out["errors"] = errors
        return out
