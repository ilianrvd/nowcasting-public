"""
SYNOP от Ogimet (getsynop) за българските станции около летищата
================================================================
Една заявка за блок 15 (България + Румъния) за последните WINDOW_H часа; от отговора се
пазят само станциите от settings.SYNOP_STATIONS. Българските станции предават на 3 часа.
Правила за заявките: общият USER_AGENT, най-много една заявка на MIN_INTERVAL_MIN минути
(виж src/output/obs_export.py), не повече от една заявка на пуск.

fetch_synop(now) → [{"wmo", "time"(UTC), "raw"}], най-старото първо; изключение при грешка.
build_report(rec) → запис за obs/synop_latest.json (формат: docs/formats/obs_synop.md).
Полетата се четат от самото съобщение (AAXX, секция 1); не се декодира нищо, което не е
сигурно — нечетимо/липсващо поле → null.
"""

import re
import logging
import datetime as dt

import requests

from config.settings import USER_AGENT, SYNOP_STATIONS

logger = logging.getLogger("ingest.synop")

URL = "https://www.ogimet.com/cgi-bin/getsynop"
BLOCK = "15"
WINDOW_H = 12
_LINE = re.compile(r"^(\d{5}),(\d{4}),(\d{2}),(\d{2}),(\d{2}),(\d{2}),(.*)$")
_KT_PER_MS = 1.943844


def fetch_synop(now: dt.datetime, timeout=60) -> list:
    begin = (now - dt.timedelta(hours=WINDOW_H)).strftime("%Y%m%d%H00")
    r = requests.get(URL, params={"block": BLOCK, "begin": begin, "end": now.strftime("%Y%m%d%H%M")},
                     timeout=timeout, headers={"User-Agent": USER_AGENT})
    r.raise_for_status()
    text = r.text.strip()
    out, bad = {}, 0
    for ln in text.splitlines():
        m = _LINE.match(ln.strip())
        if not m:
            bad += 1
            continue
        wmo, y, mo, d, h, mi, raw = m.groups()
        if wmo not in SYNOP_STATIONS:
            continue
        t = dt.datetime(int(y), int(mo), int(d), int(h), int(mi), tzinfo=dt.timezone.utc)
        out[(wmo, t)] = {"wmo": wmo, "time": t, "raw": raw.strip()}
    if text and bad and not out and bad >= len(text.splitlines()):
        raise ValueError(f"Ogimet: отговорът не прилича на SYNOP (напр. {text[:80]!r})")
    recs = sorted(out.values(), key=lambda x: (x["time"], x["wmo"]))
    logger.info(f"SYNOP: {len(recs)} съобщения за {len({r['wmo'] for r in recs})} станции")
    return recs


# ── Разбор на секция 1 ───────────────────────────────────────────────────────
def _vis_m(vv: str):
    if not vv.isdigit():
        return None
    v = int(vv)
    if v == 0:
        return 50                      # < 100 m
    if v <= 50:
        return v * 100
    if 56 <= v <= 80:
        return (v - 50) * 1000
    if 81 <= v <= 88:
        return (35 + 5 * (v - 81)) * 1000
    if v == 89:
        return 70000
    return {90: 25, 91: 50, 92: 200, 93: 500, 94: 1000, 95: 2000, 96: 4000,
            97: 10000, 98: 20000, 99: 50000}.get(v)


def _temp(g: str):
    """1sTTT / 2sTdTdTd → °C; None ако е с // или е влажност (sn 9)."""
    if len(g) != 5 or "/" in g[2:] or g[1] not in "01":
        return None
    v = int(g[2:]) / 10.0
    return -v if g[1] == "1" else v


def _wx(ww):
    """Явление в текст по ww (само мъгла/мъгла-мараня/валеж/гръмотевици; иначе None).
    Минало явление (20–29) не е текущо → не се дава."""
    if ww is None:
        return None
    if ww == 10:
        return "BR"
    if ww in (11, 12):
        return "BCFG" if ww == 11 else "MIFG"
    if ww == 40:
        return "VCFG"
    if ww == 41:
        return "BCFG"
    if 42 <= ww <= 47:
        return "FG"
    if ww in (48, 49):
        return "FZFG"
    if 50 <= ww <= 55:
        return "DZ"
    if ww in (56, 57):
        return "FZDZ"
    if 58 <= ww <= 65:
        return "RA"
    if ww in (66, 67):
        return "FZRA"
    if 68 <= ww <= 69:
        return "RASN"
    if 70 <= ww <= 75:
        return "SN"
    if ww == 79:
        return "PL"
    if 80 <= ww <= 82:
        return "SHRA"
    if 83 <= ww <= 86:
        return "SHSN" if ww >= 85 else "SHRASN"
    if 87 <= ww <= 90:
        return "GR"
    if 91 <= ww <= 99:
        return "TS"
    return None


def build_report(rec: dict) -> dict:
    raw = rec["raw"].rstrip("= ")
    out = {"id": f"SYNOP:{rec['wmo']}", "time": rec["time"].strftime("%Y-%m-%dT%H:%M:%SZ"),
           "t": None, "td": None, "wind_kt": None, "wind_dir": None, "vis_m": None,
           "ww": None, "wx": [], "n_oktas": None, "h": None, "cloud_base_ft": None, "raw": raw}
    toks = raw.split()
    if len(toks) < 5 or toks[0] != "AAXX" or toks[2] != rec["wmo"]:
        return out                                     # NIL или нечетимо — само идентификацията
    iw = toks[1][4:5]
    g = toks[3:]
    n = next((i for i, t in enumerate(g) if t in ("333", "444", "555")), len(g))
    sec1 = g[:n]
    if sec1 and len(sec1[0]) == 5:                     # iR iX h VV
        h, vv = sec1[0][2], sec1[0][3:5]
        out["h"] = int(h) if h.isdigit() else None
        out["vis_m"] = _vis_m(vv)
    if len(sec1) > 1 and len(sec1[1]) == 5:            # N dd ff
        N, dd, ff = sec1[1][0], sec1[1][1:3], sec1[1][3:5]
        out["n_oktas"] = int(N) if N.isdigit() and N != "9" else None
        speed = None
        if ff.isdigit():
            speed = int(ff)
            if speed == 99 and len(sec1) > 2 and sec1[2].startswith("00") and sec1[2][2:].isdigit():
                speed = int(sec1[2][2:])
        if speed is not None and iw in "0134":
            kt = speed * _KT_PER_MS if iw in "01" else float(speed)
            out["wind_kt"] = int(round(kt))
            if dd.isdigit() and int(dd) <= 36 and not (speed == 0 and dd == "00"):
                out["wind_dir"] = int(dd) * 10
    for t in sec1[2:]:
        if len(t) != 5:
            continue
        if t[0] == "1" and out["t"] is None:
            out["t"] = _temp(t)
        elif t[0] == "2" and out["td"] is None:
            out["td"] = _temp(t)
        elif t[0] == "7" and out["ww"] is None and t[1:3].isdigit():
            out["ww"] = int(t[1:3])
    x = _wx(out["ww"])
    out["wx"] = [x] if x else []
    return out
