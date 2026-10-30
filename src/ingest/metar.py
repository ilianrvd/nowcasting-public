"""
METAR за летищата на страницата на клуба
========================================
AWC Data API (aviationweather.gov), последните 2 ч. Взима се само нужното
за статуса: явленията с валеж/TS и дали трендът (TEMPO/BECMG) има валеж.
Подробният парсер е в airport-nowcast; тук нарочно е минимален.

fetch_metars(icaos) → {icao: {"raw", "time", "wx", "precip", "shower", "ts", "trend_precip"}}
(последният METAR/SPECI за летището; липсващо летище → няма ключ)
"""

import re
import logging
import datetime as dt

import requests

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


def fetch_metars(icaos, hours=2, timeout=20) -> dict:
    r = requests.get(URL, params={"ids": ",".join(icaos), "format": "json", "hours": hours},
                     timeout=timeout, headers={"User-Agent": "nowcasting-public"})
    r.raise_for_status()
    out = {}
    for rec in (r.json() if r.content else []):
        ic, raw, t = rec.get("icaoId"), rec.get("rawOb"), rec.get("obsTime")
        if ic not in icaos or not raw or not isinstance(t, (int, float)):
            continue
        t = dt.datetime.fromtimestamp(t, tz=dt.timezone.utc)
        if ic in out and out[ic]["time"] >= t:
            continue
        out[ic] = {**parse(raw), "time": t}
    logger.info("METAR: " + (" · ".join(f"{ic} {m['time']:%H%M}Z {' '.join(m['wx']) or '—'}"
                                         + (" (TEMPO/BECMG с валеж)" if m["trend_precip"] else "")
                                         for ic, m in sorted(out.items())) or "няма"))
    return out
