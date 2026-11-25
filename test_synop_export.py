"""Проверка на разбора и публикуването на SYNOP без мрежа: python test_synop_export.py"""
import datetime as dt, os, tempfile
import src.output.obs_export as ox
import src.ingest.synop as sy

sy.SYNOP_STATIONS["15015"] = {}                 # румънска станция от истинска проба
T = dt.datetime(2026, 11, 25, 12, tzinfo=dt.timezone.utc)
RAW = "AAXX 25121 15015 01570 70502 11001 21031 39637 42831 57012 69901 70272 84570 333 41998 55302="
r = sy.build_report({"wmo": "15015", "time": T, "raw": RAW})
assert (r["t"], r["td"], r["wind_kt"], r["wind_dir"], r["vis_m"], r["ww"], r["n_oktas"], r["h"]) == \
       (-0.1, -3.1, 4, 50, 20000, 2, 7, 5), r
fog = sy.build_report({"wmo": "15015", "time": T, "raw": "AAXX 25061 15015 02994 80000 10005 20005 74500 333 5="})
assert (fog["vis_m"], fog["wind_kt"], fog["wind_dir"], fog["ww"], fog["wx"]) == (1000, 0, None, 45, ["FG"]), fog
assert sy.build_report({"wmo": "15015", "time": T, "raw": "AAXX 25121 15015 NIL="})["t"] is None

d = tempfile.mkdtemp()
ox.OBS_DIR, ox.SYNOP_CACHE = os.path.join(d, "obs"), os.path.join(d, "c", "s.json")
rec = lambda m: {"wmo": "15015", "time": T + dt.timedelta(hours=m), "raw": RAW}
calls = []
def ok(now): calls.append(now); return [rec(0), rec(3)]
def bad(now): calls.append(now); raise RuntimeError("503")
now = T + dt.timedelta(hours=4)
o = ox.export_synop(now, ok);                              assert len(o["reports"]) == 2 and len(calls) == 1
o = ox.export_synop(now + dt.timedelta(minutes=20), ok);   assert len(calls) == 1          # още не е време
o = ox.export_synop(now + dt.timedelta(minutes=55), bad);   assert len(calls) == 2           # опит, неуспех → кеш
assert len(o["reports"]) == 2 and o["fetched_at"] == "2026-11-25T16:00:00Z", o["fetched_at"]
o = ox.export_synop(now + dt.timedelta(minutes=70), ok);   assert len(calls) == 2           # пак не е време (опитът е в 16:55)
o = ox.export_synop(now + dt.timedelta(hours=9), bad)                                         # >12 ч от съобщенията
assert o is None or all(x["time"] >= "2026-11-25T13:00:00Z" for x in o["reports"])
print("OK")
