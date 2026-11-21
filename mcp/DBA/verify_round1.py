import sys, io, contextlib, importlib
from fractions import Fraction
with contextlib.redirect_stdout(io.StringIO()):
    from jhora import const
    mods = {}
    for name in ["ashtottari","chathuraaseethi_sama","dwadasottari","dwisatpathi","panchottari",
                 "sataatbika","shastihayani","shattrimsa_sama","shodasottari","yogini"]:
        mods[name] = importlib.import_module("jhora.horoscope.dhasa.graha."+name)
P = {0:"Sun",1:"Moon",2:"Mars",3:"Mercury",4:"Jupiter",5:"Venus",6:"Saturn",7:"Rahu",8:"Ketu"}
print("== PyJHora Vimshottari order/years")
print([(P[k], v) for k, v in zip(const.vimsottari_adhipati_list, [const.vimsottari_dict[k] for k in const.vimsottari_adhipati_list])], sum(const.vimsottari_dict.values()))
for name, m in mods.items():
    for attr in ("dhasa_adhipathi_list","dhasa_adhipathi_star_count","ashtottari_adhipathi_dict_seed","yogini_dict","dhasa_adhipathi_dict","yogini_adhipathi_list"):
        if hasattr(m, attr):
            d = getattr(m, attr)
            try:
                if name == "ashtottari":
                    items = [(P.get(k,k), v) for k, v in d.items()]
                else:
                    items = [(P.get(k,k) if isinstance(k,int) else k, v) for k, v in d.items()]
            except Exception as e:
                items = repr(d)[:300]
            print(f"== {name}.{attr}: {items}")
            try:
                vals = list(d.values())
                if vals and isinstance(vals[0], (int,float)):
                    print("   sum =", sum(vals))
                elif vals and isinstance(vals[0], (list,tuple)):
                    print("   (structured values)")
            except Exception:
                pass
    for attr in ("human_life_span_for_ashtottari_dhasa",):
        if hasattr(m, attr): print("   life span:", getattr(m, attr))
print()
print("== Year constants")
for n in ("sidereal_year","tropical_year","savana_year","lunar_year","average_gregorian_year","nakshathra_year"):
    print(n, getattr(const, n))
print()
print("== Year-length effect (days) of using X instead of 365.25, at age 20/60/120 and for a 19y Saturn MD")
for n, y in (("tropical 365.24219",365.24219),("sidereal 365.256364",365.256364),("gregorian 365.2425",365.2425),("savana 360",360.0),("lunar 354.367",354.36707)):
    print(f"{n:22s}", [round((y-365.25)*a,2) for a in (20,60,120)], "Saturn19y:", round((y-365.25)*19,2))
print()
print("== 15-minute birth-time shift of whole timeline (Moon 33'/hr avg, nak = 800')")
yrs = {"Ketu":7,"Venus":20,"Sun":6,"Moon":10,"Mars":7,"Rahu":18,"Jupiter":16,"Saturn":19,"Mercury":17}
for k, v in yrs.items():
    print(k, v, round(33.0*15/60/800*v*365.25, 1))
print()
print("== ayanamsa effects on Venus MD (Lahiri-Raman 1d26m46.7s; Lahiri-KPold 5m49.0s)")
for lab, deg in (("Lahiri-Raman", 1+26/60+46.7/3600),("Lahiri-KPold",5/60+49.0/3600)):
    frac = deg/(360/27)
    print(lab, round(deg,4), "deg =", round(frac*100,2), "% nak ->", round(frac*20,3), "y =", round(frac*20*365.25,1), "d (Venus 20y)")
print()
print("== counts", 9**6, "periods at level 6 per 120y; levels:", [9**k for k in range(1,7)])
print("== 360 vs 365.25 for 18y:", 18*365.25, 18*360, 18*5.25, " whole 120y:", 120*5.25)
print("== 3 arcsec Moon -> Venus MD days:", 3/60/800*20*365.25)
