"""
Наблюдения за други проекти: docs/obs/metar_latest.json
=======================================================
Формат: docs/formats/obs_metar.md (съгласуван с fogNowcast, docs/obs_metar_synop.md).

METAR/SPECI за петте летища за последните WINDOW_H часа, най-старото първо.
Ако изтеглянето се провали, публикува се последното добро (кеш data/cache/metar_last.json,
ползва се и между пусковете чрез actions/cache) с оригиналните им "time";
"fetched_at" е кога последно е успяло изтеглянето, "generated_at" — кога е сглобен файлът.
Без кеш и без данни файлът не се пише (по-добре 404, отколкото празен списък, който
изглежда като „няма съобщения“).
"""

import os, sys, json, logging
import datetime as dt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from config.settings import BASE_DIR

logger = logging.getLogger("output.obs")

WINDOW_H = 3
OBS_DIR = os.path.join(BASE_DIR, "docs", "obs")
CACHE = os.path.join(BASE_DIR, "data", "cache", "metar_last.json")
_Z = "%Y-%m-%dT%H:%M:%SZ"


def _load_cache():
    try:
        with open(CACHE, encoding="utf-8") as f:
            c = json.load(f)
        recs = [{"icao": r["icao"], "raw": r["raw"],
                 "time": dt.datetime.strptime(r["time"], _Z).replace(tzinfo=dt.timezone.utc)}
                for r in c["records"]]
        return recs, dt.datetime.strptime(c["fetched_at"], _Z).replace(tzinfo=dt.timezone.utc)
    except Exception:
        return [], None


def export_metar(records, now=None):
    """records: списъкът от fetch_awc или None при неуспешно изтегляне."""
    from src.ingest.metar import build_report
    now = now or dt.datetime.now(dt.timezone.utc)
    cached, cached_fetch = _load_cache()
    if records is None:
        fetched_at, src_note = cached_fetch, "кеш"
    else:
        fetched_at, src_note = now, "AWC"
    merged = {(r["icao"], r["time"], r["raw"]): r for r in cached + (records or [])}
    lo = now - dt.timedelta(hours=WINDOW_H)
    recs = sorted((r for r in merged.values() if r["time"] >= lo), key=lambda r: (r["time"], r["icao"]))
    if fetched_at is None or (records is None and not recs):
        logger.warning("METAR: няма нито нови, нито кеширани данни — obs/metar_latest.json не се пише")
        return None

    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    with open(CACHE, "w", encoding="utf-8") as f:
        json.dump({"fetched_at": fetched_at.strftime(_Z),
                   "records": [{"icao": r["icao"], "raw": r["raw"], "time": r["time"].strftime(_Z)}
                               for r in recs]}, f, ensure_ascii=False)

    out = {"generated_at": now.strftime(_Z), "fetched_at": fetched_at.strftime(_Z),
           "source": "awc", "reports": [build_report(r) for r in recs]}
    os.makedirs(OBS_DIR, exist_ok=True)
    with open(os.path.join(OBS_DIR, "metar_latest.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    logger.info(f"METAR → docs/obs/metar_latest.json: {len(recs)} съобщения ({src_note}, "
                f"изтеглени {fetched_at:%H:%M} UTC)")
    return out


# ── SYNOP: obs/synop_latest.json ────────────────────────────────────────────
SYNOP_CACHE = os.path.join(BASE_DIR, "data", "cache", "synop_last.json")
SYNOP_WINDOW_H = 12
SYNOP_MIN_INTERVAL_MIN = 50      # най-много една заявка към Ogimet на толкова минути (≈ 1 на час)


def _load_synop_cache():
    try:
        with open(SYNOP_CACHE, encoding="utf-8") as f:
            c = json.load(f)
        P = lambda x: dt.datetime.strptime(x, _Z).replace(tzinfo=dt.timezone.utc) if x else None
        recs = [{"wmo": r["wmo"], "raw": r["raw"], "time": P(r["time"])} for r in c["records"]]
        return recs, P(c.get("fetched_at")), P(c.get("attempted_at"))
    except Exception:
        return [], None, None


def export_synop(now=None, fetch=None):
    """Тегли от Ogimet най-много веднъж на SYNOP_MIN_INTERVAL_MIN минути (между пусковете
    решава кешът), публикува последните SYNOP_WINDOW_H часа. При пропусната/неуспешна заявка
    се публикуват кешираните съобщения с оригиналните им "time"; "fetched_at" е кога
    последно е успяло изтеглянето. Без кеш и без данни файлът не се пише.
    fetch(now) → списък за build_report (по подразбиране ingest.synop.fetch_synop)."""
    from src.ingest.synop import fetch_synop, build_report
    now = now or dt.datetime.now(dt.timezone.utc)
    fetch = fetch or fetch_synop
    cached, fetched_at, attempted_at = _load_synop_cache()
    new, status = [], "кеш"
    if attempted_at is None or now - attempted_at >= dt.timedelta(minutes=SYNOP_MIN_INTERVAL_MIN):
        attempted_at = now
        try:
            new = fetch(now)
            fetched_at, status = now, "Ogimet"
        except Exception as e:
            logger.warning(f"SYNOP: изтеглянето не успя ({str(e)[:100]}) — публикувам кеша")
    else:
        status = "кеш (още не е време за нова заявка)"
    merged = {(r["wmo"], r["time"]): r for r in cached + new}      # новото замества кеширано със същия час
    lo = now - dt.timedelta(hours=SYNOP_WINDOW_H)
    recs = sorted((r for r in merged.values() if r["time"] >= lo), key=lambda r: (r["time"], r["wmo"]))

    os.makedirs(os.path.dirname(SYNOP_CACHE), exist_ok=True)
    F = lambda t: t.strftime(_Z) if t else None
    with open(SYNOP_CACHE, "w", encoding="utf-8") as f:
        json.dump({"fetched_at": F(fetched_at), "attempted_at": F(attempted_at),
                   "records": [{"wmo": r["wmo"], "raw": r["raw"], "time": F(r["time"])} for r in recs]},
                  f, ensure_ascii=False)
    if fetched_at is None or not recs:
        logger.warning("SYNOP: няма данни и няма кеш — obs/synop_latest.json не се пише")
        return None

    out = {"generated_at": F(now), "fetched_at": F(fetched_at), "source": "ogimet",
           "reports": [build_report(r) for r in recs]}
    os.makedirs(OBS_DIR, exist_ok=True)
    with open(os.path.join(OBS_DIR, "synop_latest.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    logger.info(f"SYNOP → docs/obs/synop_latest.json: {len(recs)} съобщения ({status}, "
                f"изтеглени {fetched_at:%H:%M} UTC)")
    return out
