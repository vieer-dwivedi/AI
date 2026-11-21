import sys
from datetime import datetime
sys.path.insert(0, "mcp/dasha")
import dasha_engine as e
from state import BirthState

st = BirthState(date_of_birth="1990-05-15", time_of_birth="10:30", latitude=19.07, longitude=72.87, timezone="Asia/Kolkata")
birth_local = datetime(1990, 5, 15, 10, 30, 0)
pj = e.get_mahadashas(st)["mahadashas"]
print("first rows:", [(r["lord"], r["start"], r["end"], r["duration_days"]) for r in pj[:2]])

b = e.get_basis(st)
frac = b["moon"]["fraction_traversed"]
bal_years = (1 - frac) * 6
print("fraction traversed:", frac, " balance years:", bal_years, " classical:", b["dasha_basis"]["classical_vimsottari_balance_years"])

first_end = datetime.fromisoformat(pj[0]["end"])
gap_days = (first_end - birth_local).total_seconds() / 86400
print("PyJHora: days from birth to end of first MD:", round(gap_days, 4))
for yd in (365.25, 365.2425, 365.24219, 365.256364):
    print("expected yd=%s: %.4f d -> PyJHora minus expected %+.4f d" % (yd, bal_years * yd, gap_days - bal_years * yd))
print("implied balance years at 365.2425:", gap_days / 365.2425)
print("implied fraction traversed:", 1 - gap_days / 365.2425 / 6)
