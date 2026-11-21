import sys, json
sys.path.insert(0, "mcp/dasha")
import dasha_engine as e
from state import BirthState
st = BirthState(date_of_birth="1990-05-15", time_of_birth="10:30", latitude=19.07, longitude=72.87, timezone="Asia/Kolkata")
b = e.get_basis(st)
print(json.dumps(b, indent=1, default=str)[:3000])
import io, contextlib
with contextlib.redirect_stdout(io.StringIO()):
    from jhora import const
    from jhora.panchanga import drik
print("ayanamsa mode default:", getattr(drik, "_ayanamsa_mode", None), getattr(const, "_DEFAULT_AYANAMSA_MODE", None))
