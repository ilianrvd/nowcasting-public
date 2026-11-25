"""
METAR за летищата на страницата на клуба
========================================
AWC Data API (aviationweather.gov), последните 2 ч. Взима се само нужното
за статуса: явленията с валеж/TS и дали трендът (TEMPO/BECMG) има валеж.
Подробният парсер е в airport-nowcast; тук нарочно е минимален.

fetch_awc(icaos, hours=3) → всички съобщения (METAR/SPECI) за прозореца, най-старото първо
latest_per_airport(records) → {icao: {"raw", "time", "wx", "precip", "shower", "ts", "trend_precip"}}
(последният METAR/SPECI за летището; липсващо летище → няма ключ) — за статуса на страницата
fetch_metars(icaos) = двете заедно (запазено за съвместимост)

Публикуваното obs/metar_latest.json (формат: docs/formats/obs_metar.md) се строи от
build_report(record): полетата се четат от самото съобщение (raw), не от AWC JSON —
AWC дава видимост в мили, а t/td и облаците там са закръглени/преработени.
"""

import re
import logging
import datetime as dt

import requests

from config.settings import USER_AGENT

logger = logging.getLogger("ingest.metar")

URL = "https://aviationweather.gov/api/data/metar"
_PRECIP = ("DZ", "RA", "SN", "SG", "PL", "GR", "GS", "UP")
_WX = re.compile(r"^(\+|-|VC)?(MI|BC|PR|DR|BL|SH|TS|FZ)?((?:DZ|RA|SN|SG|PL|GR|GS|UP|BR|FG|FU|VA|DU|SA|HZ|PO|SQ|FC|SS|DS)*)$")


def _is_wx(tok):
    m = _WX.match(tok)
    return bool(m and m.group(3)) or tok in ("TS", "-TS", "+TS")


def _precip_groups(groups):
    """Групите с валеж или TS, без VC (BR, FG и т.н. отпадат)."""
    return [g for g in groups if not g.startswith("VC") and (any(p in g for p in _PRECIP) or "TS" in g)]


def parse(raw: str) -> dict:
    """Явленията от основната част и дали трендът има валеж/TS."""
    toks = raw.replace("=", "").split()
    main, trend, part = [], [], "main"
    for tok in toks[1:]:
        if tok == "RMK":
            break
        if tok in ("TEMPO", "BECMG"):
            part = "trend"
        elif tok == "NOSIG":
            part = "nosig"
        elif part == "main" and _is_wx(tok) and not tok.startswith("RE"):
            main.append(tok)
        elif part == "trend" and _is_wx(tok):
            trend.append(tok)
    wx = _precip_groups(main)
    return {"raw": raw, "wx": wx,
            "precip": bool(wx),
            "shower": any("SH" in g or "TS" in g for g in wx),
            "ts": any("TS" in g for g in wx),
            "trend_precip": bool(_precip_groups(trend))}


def fetch_awc(icaos, hours=3, timeout=20) -> list:
    """AWC Data API → [{"icao", "raw", "time"(UTC)}], най-старото първо; без дубликати."""
    r = requests.get(URL, params={"ids": ",".join(icaos), "format": "json", "hours": hours},
                     timeout=timeout, headers={"User-Agent": USER_AGENT})
    r.raise_for_status()
    seen, out = set(), []
    for rec in (r.json() if r.content else []):
        ic, raw, t = rec.get("icaoId"), rec.get("rawOb"), rec.get("obsTime")
        if ic not in icaos or not raw or not isinstance(t, (int, float)):
            continue
        t = dt.datetime.fromtimestamp(t, tz=dt.timezone.utc)
        if (ic, t, raw) in seen:
            continue
        seen.add((ic, t, raw))
        out.append({"icao": ic, "raw": raw, "time": t})
    out.sort(key=lambda x: (x["time"], x["icao"]))
    return out


def latest_per_airport(records) -> dict:
    out = {}
    for rec in records:
        ic, t = rec["icao"], rec["time"]
        if ic in out and out[ic]["time"] >= t:
            continue
        out[ic] = {**parse(rec["raw"]), "time": t}
    logger.info("METAR: " + (" · ".join(f"{ic} {m['time']:%H%M}Z {' '.join(m['wx']) or '—'}"
                                         + (" (TEMPO/BECMG с валеж)" if m["trend_precip"] else "")
                                         for ic, m in sorted(out.items())) or "няма"))
    return out


def fetch_metars(icaos, hours=3, timeout=20) -> dict:
    return latest_per_airport(fetch_awc(icaos, hours, timeout))


# ── Пълен разбор за публикуване (obs/metar_latest.json) ─────────────────────
_KT_PER_MS = 1.943844
_WIND = re.compile(r"^(VRB|\d{3}|///)(\d{2,3})(?:G(\d{2,3}))?(KT|MPS|KMH)$")
_VIS = re.compile(r"^(\d{4})(?:[NSEW]{1,2})?$")
_RVR = re.compile(r"^R\d{2}[LCR]?/([PM]?)(\d{4})(?:V[PM]?(\d{4}))?(?:[UDN])?$")
_TEMP = re.compile(r"^(M?\d{2})/(M?\d{2})?$")
_CLOUD = re.compile(r"^(FEW|SCT|BKN|OVC)(\d{3})")
_VV = re.compile(r"^VV(\d{3})$")
_WXTOK = re.compile(r"^(\+|-|VC)?(MI|BC|PR|DR|BL|SH|TS|FZ)?(DZ|RA|SN|SG|PL|GR|GS|UP|BR|FG|FU|VA|DU|SA|HZ|PO|SQ|FC|SS|DS)*$")


def _t(tok):
    return -int(tok[1:]) if tok.startswith("M") else int(tok)


def build_report(rec: dict) -> dict:
    """Запис от fetch_awc → запис за obs/metar_latest.json.
    Правила: вятърът е в kt; wind_dir е null при VRB и при щил; vis_m е преобладаващата
    видимост (CAVOK = 10000, 9999 = 10000); rvr_m е най-ниската RVR; wx са групите на
    явленията от основната част така, както са в съобщението; cloud_base_ft е основата на
    най-ниската BKN/OVC (таван), null ако няма; vv_ft — вертикалната видимост, null ако няма.
    Липсващо/нечетимо поле → null."""
    raw = rec["raw"]
    toks = raw.replace("=", "").split()
    out = {"id": rec["icao"], "time": rec["time"].strftime("%Y-%m-%dT%H:%M:%SZ"),
           "t": None, "td": None, "wind_kt": None, "wind_dir": None, "vis_m": None,
           "rvr_m": None, "wx": [], "cloud_base_ft": None, "vv_ft": None, "raw": raw}
    rvrs, ceil, in_main = [], [], True
    for i, tok in enumerate(toks):
        if tok == "RMK" or tok in ("TEMPO", "BECMG", "NOSIG"):
            if tok == "RMK" or tok in ("TEMPO", "BECMG"):
                in_main = False
        m = _WIND.match(tok) if in_main and out["wind_kt"] is None else None
        if m:
            d, sp, _g, u = m.groups()
            sp = int(sp)
            kt = sp if u == "KT" else sp * _KT_PER_MS if u == "MPS" else sp / 1.852
            out["wind_kt"] = int(round(kt))
            out["wind_dir"] = int(d) if d.isdigit() and not (sp == 0 and int(d) == 0) else None
            continue
        if not in_main or i == 0:
            continue
        if tok == "CAVOK":
            out["vis_m"] = 10000
        elif out["vis_m"] is None and _VIS.match(tok):
            v = int(_VIS.match(tok).group(1))
            out["vis_m"] = 10000 if v == 9999 else v
        elif _RVR.match(tok):
            rvrs.append(int(_RVR.match(tok).group(2)))
        elif _TEMP.match(tok) and out["t"] is None:
            a, b = _TEMP.match(tok).groups()
            out["t"] = _t(a)
            out["td"] = _t(b) if b else None
        elif _CLOUD.match(tok):
            cov, h = _CLOUD.match(tok).groups()
            if cov in ("BKN", "OVC"):
                ceil.append(int(h) * 100)
        elif _VV.match(tok):
            out["vv_ft"] = int(_VV.match(tok).group(1)) * 100
        elif tok not in ("METAR", "SPECI", "AUTO", "COR", "NSC", "NCD", "SKC", "CLR", "NSW") \
                and not tok.startswith("RE") and _WXTOK.match(tok) \
                and (_WXTOK.match(tok).group(2) or _WXTOK.match(tok).group(3)):
            out["wx"].append(tok)
    out["rvr_m"] = min(rvrs) if rvrs else None
    out["cloud_base_ft"] = min(ceil) if ceil else None
    return out
