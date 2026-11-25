"""Проверка на разбора и публикуването на METAR без мрежа: python test_metar_export.py"""
import datetime as dt, json, os, tempfile
import src.output.obs_export as ox
from src.ingest.metar import build_report

T = dt.datetime(2026, 11, 25, 6, 0, tzinfo=dt.timezone.utc)
R = lambda ic, raw, m=0: {"icao": ic, "raw": raw, "time": T + dt.timedelta(minutes=m)}
recs = [R("LBSF", "METAR LBSF 250600Z 00000KT 0300 R27/0600N FG VV001 M01/M01 Q1030 NOSIG="),
        R("LBBG", "METAR LBBG 250630Z VRB02KT 9999 BKN012 OVC030 03/01 Q1020 TEMPO 3000 -RA=", 30)]

b = build_report(recs[0])
assert (b["t"], b["td"], b["wind_kt"], b["wind_dir"], b["vis_m"], b["rvr_m"], b["wx"], b["vv_ft"]) == \
       (-1, -1, 0, None, 300, 600, ["FG"], 100), b
b = build_report(recs[1])
assert (b["wind_dir"], b["vis_m"], b["cloud_base_ft"], b["wx"]) == (None, 10000, 1200, []), b

d = tempfile.mkdtemp()
ox.OBS_DIR, ox.CACHE = os.path.join(d, "obs"), os.path.join(d, "cache", "m.json")
now = T + dt.timedelta(hours=1)
o = ox.export_metar(recs, now)
assert [r["id"] for r in o["reports"]] == ["LBSF", "LBBG"] and o["fetched_at"].endswith("07:00:00Z")
# неуспешно изтегляне 40 мин по-късно: същите съобщения, fetched_at остава старото
o = ox.export_metar(None, now + dt.timedelta(minutes=40))
assert len(o["reports"]) == 2 and o["fetched_at"] == "2026-11-25T07:00:00Z" and o["generated_at"].endswith("07:40:00Z")
# след 3 ч старите отпадат
assert ox.export_metar(None, T + dt.timedelta(hours=3, minutes=20))["reports"][0]["id"] == "LBBG"
# без кеш и без данни — не пише
ox.CACHE = os.path.join(d, "none.json"); os.remove(os.path.join(d, "obs", "metar_latest.json"))
assert ox.export_metar(None, now) is None and not os.path.exists(os.path.join(d, "obs", "metar_latest.json"))
print("OK")
