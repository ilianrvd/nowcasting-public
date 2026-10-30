"""
Експорт за публичната страница (docs/embed/)
============================================
Подготвя данните за интерактивната карта (Leaflet) и таблицата за стартовете:

  docs/embed/data/f_XX.png   — прозрачни кадри в Web Mercator (EPSG:3857), за
                               L.imageOverlay върху подложка; цветове по dBZ
  docs/embed/data/frames.json — списък кадри (час UTC, водещо време, източник),
                               граници, движение по региони, мълнии, свежест
  docs/embed/data/sites.json  — за всеки старт: състояние 🟢/🟡/🔴 сега и до +3 ч

Правило за стартовете (радиус около старта):
  🔴 конвекция: мълнии (LI, последните 15 мин) в R_CONV, или ядро ≥ 45 dBZ
               с площ ≥ CONV_MIN_PX пиксела (≈ km²) в R_CONV
  🟡 валеж:     ехо ≥ 20 dBZ (≈0.6 mm/h) в ≥ RAIN_MIN_FRAC от кръга R_RAIN
Иска се площ, не единичен пиксел — шум и отражения от терена (румънския
composite) иначе палят цветове от нищо. Причината се пише в sites.json ("why").
  🟢 иначе
Летищата (тип „летище“, ICAO в скоби в името): METAR с валеж ги прави поне 🟡
(TS → 🔴) в началото — виж settings.METAR_SITES и _metar_leads.
Прогнозните кадри са от blend-а (радар → H40B → ICON); ≥45 dBZ там идва от
усилването с мълнии или от LPI ядрата на ICON, т.е. отразява конвекция.
"""

import os, re, sys, csv, json, logging, shutil
import datetime as dt
from zoneinfo import ZoneInfo
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from config.settings import BASE_DIR, WIND, METAR_SITES

logger = logging.getLogger("output.web")

OUT_DIR = os.path.join(BASE_DIR, "docs", "embed", "data")
SITES_CSV = os.path.join(BASE_DIR, "config", "sites.csv")
TZ = ZoneInfo("Europe/Sofia")

R_CONV_KM = 20.0
R_RAIN_KM = 10.0
DBZ_RAIN, DBZ_CONV = 20.0, 45.0
CONV_MIN_PX = 4          # 🔴: поне толкова пиксела ≥45 dBZ (≈ 4 km² ядро)
RAIN_MIN_FRAC = 0.05     # 🟡: поне 5% от кръга 10 km (≈ 16 km²) с ≥20 dBZ
DBZ_SHOW = 7.0           # на картата: под това не се рисува (≈0.1 mm/h; радарът така или иначе е от 15 dBZ)
MIN_BLOB_PX = 4          # изолирани петна под този размер се махат (само за картата)

# Прогнозни кадри (минути след наблюдението)
LEADS_NOW = [10, 20, 30, 40, 50, 60]
LEADS_BLEND = [90, 120, 150, 180, 240, 300, 360]
SITE_LEADS = [0, 30, 60, 90, 120, 180]

# Цветове за страницата: светло синьо = много слаб (7–15 dBZ, само сателит/модел),
# синьо = слаб, зелено = умерен, жълто/оранжево = силен, червено/виолетово = много силен / градушка
LEVELS = [7, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65]
COLORS = ['#8FD3FF', '#4AA8F0', '#1F78D6', '#00C800', '#009000', '#FFFF00', '#E7C000',
          '#FF9000', '#FF0000', '#C80000', '#FF00FF', '#9955C9']
ALPHA = [220, 225, 225]  # сините — почти плътни, иначе върху сивата основа не се виждат; останалите 210


def _rgba_table():
    tab = np.zeros((len(COLORS), 4), dtype=np.uint8)
    for i, c in enumerate(COLORS):
        tab[i, :3] = [int(c[k:k + 2], 16) for k in (1, 3, 5)]
        tab[i, 3] = ALPHA[i] if i < len(ALPHA) else 210
    return tab


_RGBA = _rgba_table()


def _merc_y(lat):
    return np.log(np.tan(np.pi / 4 + np.radians(lat) / 2))


def _to_mercator_rows(field, lat):
    """Пренарежда редовете (възходяща lat) към равномерна стъпка в Mercator y,
    отгоре надолу — както очаква L.imageOverlay."""
    y = _merc_y(lat)
    ny = len(lat)
    y_out = np.linspace(y[-1], y[0], ny)                 # север → юг
    idx = np.clip(np.searchsorted(y, y_out), 1, ny - 1)
    nearer = np.where(np.abs(y[idx - 1] - y_out) < np.abs(y[idx] - y_out), idx - 1, idx)
    return field[nearer, :]


def _clean(dbz):
    """Поле за страницата: под DBZ_SHOW → NaN, изолирани петна < MIN_BLOB_PX → NaN.
    Ползва се и за картата, и за таблицата, за да показват едно и също."""
    from scipy.ndimage import label
    out = np.where(np.isfinite(dbz) & (dbz >= DBZ_SHOW), dbz, np.nan).astype(np.float32)
    lab, n = label(np.isfinite(out))
    if n:
        size = np.bincount(lab.ravel())
        out[(lab > 0) & (size[lab] < MIN_BLOB_PX)] = np.nan
    return out


def _save_png(dbz, lat, path, nodata=None, model=None):
    from PIL import Image
    d = _to_mercator_rows(dbz, lat)          # dbz вече е почистено (_clean)
    img = np.zeros(d.shape + (4,), dtype=np.uint8)
    ok = np.isfinite(d)
    if model is not None:                    # лека рядка щриховка: зоната е по модел (ICON + мълнии)
        md = _to_mercator_rows(model.astype(np.uint8), lat).astype(bool)
        yy, xx = np.indices(md.shape)
        img[md & (((yy + xx) % 14) < 1)] = (110, 118, 128, 110)
    if nodata is not None:                   # сива щриховка там, където нямаме данни
        nd = _to_mercator_rows(nodata.astype(np.uint8), lat).astype(bool) & ~ok
        yy, xx = np.indices(nd.shape)
        stripe = ((yy + xx) % 10) < 3
        img[nd & stripe] = (110, 118, 128, 120)
        img[nd & ~stripe] = (110, 118, 128, 35)
    k = np.clip(np.digitize(np.where(ok, d, 0), LEVELS) - 1, 0, len(COLORS) - 1)
    img[ok] = _RGBA[k[ok]]
    Image.fromarray(img, "RGBA").save(path, optimize=True)


def _local(t):
    return t.astimezone(TZ).strftime("%H:%M")


def load_sites():
    sites = []
    if not os.path.exists(SITES_CSV):
        return sites
    with open(SITES_CSV, encoding="utf-8") as f:
        rows = [r for r in f if not r.startswith("#")]
    for r in csv.DictReader(rows):
        try:
            sites.append({"name": r["name"], "region": r["region"],
                          "lat": float(r["lat"]), "lon": float(r["lon"]),
                          "dirs": r["dirs"], "type": r["type"], "dh_m": r["dh_m"],
                          "acc": r["acc"], "note": r["note"]})
        except (KeyError, ValueError):
            continue
    return sites


def _disk_masks(sites, lat, lon):
    """За всеки старт: индекси на пикселите в R_CONV и R_RAIN (за бързина — по веднъж)."""
    lon2d, lat2d = np.meshgrid(lon, lat)
    out = []
    for s in sites:
        dy = (lat2d - s["lat"]) * 111.0
        dx = (lon2d - s["lon"]) * 111.0 * np.cos(np.radians(s["lat"]))
        d = np.hypot(dx, dy)
        out.append((np.nonzero(d <= R_CONV_KM), np.nonzero(d <= R_RAIN_KM)))
    return out


def _status(dbz, ltg, masks, nodata=None):
    """Състояние на един старт за едно поле: ('red'/'yellow'/'green'/'nodata', причина).
    'nodata': нищо не се вижда, но над половината от кръга е без радар и без сателит."""
    conv_idx, rain_idx = masks
    v = np.nan_to_num(dbz[conv_idx], nan=0.0)
    r = np.nan_to_num(dbz[rain_idx], nan=0.0)
    n_ltg = int((ltg[conv_idx] >= 1).sum()) if ltg is not None else 0
    n45 = int((v >= DBZ_CONV).sum())
    frac20 = float((r >= DBZ_RAIN).mean()) if r.size else 0.0
    why = f"max {v.max():.0f} dBZ, ≥45: {n45} px, ≥20: {frac20:.0%}" + (f", мълнии {n_ltg} px" if n_ltg else "")
    if n_ltg > 0 or n45 >= CONV_MIN_PX:
        return "red", why
    if frac20 >= RAIN_MIN_FRAC:
        return "yellow", why
    if nodata is not None:
        nd = float(nodata[rain_idx].mean()) if rain_idx[0].size else 0.0
        if nd > 0.5:
            return "nodata", why + f", без данни {nd:.0%}"
    return "green", why


# ── METAR за летищата ─────────────────────────────────────
_RANK = {None: -1, "nodata": 0, "green": 0, "yellow": 1, "red": 2}
_ICAO = re.compile(r"\(([A-Z]{4})\)")


def site_icao(s):
    """ICAO на старт от тип „летище“ (от името, напр. „Летище София (LBSF)“), иначе None."""
    m = _ICAO.search(s.get("name", ""))
    return m.group(1) if m and "летище" in s.get("type", "").lower() else None


def _metar_leads(mt, t0, leads, fields, idx20, cfg=METAR_SITES):
    """
    METAR с валеж → (статус, текст) или None за всяко водещо време.
    „Следа“: при +0 има облак до track_km → важи, докато движението държи облак
    там (до track_max_min); при SH/TS спира с облака, при обложен валеж — не
    по-рано от задържането. „Сляпо“: няма облак → задържане hold_h / trend_hold_h.
    """
    def near(f):
        return f is not None and bool((np.nan_to_num(f[idx20], nan=-99.0) >= cfg["echo_dbz"]).any())

    track = near(fields.get(0))
    hold_h = cfg["trend_hold_h"] if mt["trend_precip"] else cfg["hold_h"]
    txt = f"METAR {mt['time']:%H:%M} UTC: {' '.join(mt['wx'])}"
    out = []
    for m in leads:
        age_h = (t0 + dt.timedelta(minutes=m) - mt["time"]).total_seconds() / 3600
        if track and m <= cfg["track_max_min"] and near(fields.get(m)):
            ok = True
        elif track and mt["shower"]:
            ok = False                        # конвективният облак е излязъл
        else:
            ok = age_h <= hold_h
        red = mt["ts"] and m <= cfg["ts_red_max_min"]          # TS: червено само в началото
        out.append(("red" if red else "yellow", txt) if ok else None)
    return out, track


def _apply_metar(st, why, mlist):
    """Статусът не пада под METAR; причината от METAR отива отпред."""
    for i, mw in enumerate(mlist):
        if mw is None or st[i] is None:
            continue
        c, w = mw
        if _RANK[c] > _RANK[st[i]]:
            st[i], why[i] = c, f"{w} · {why[i]}" if why[i] else w
        else:
            why[i] = f"{why[i]} · {w}" if why[i] else w


# ── Вятър за стартовете ───────────────────────────────────
_DIR = {"С": 0, "СИ": 45, "И": 90, "ЮИ": 135, "Ю": 180, "ЮЗ": 225, "З": 270, "СЗ": 315}
_DIR_NAME = ["С", "ССИ", "СИ", "ИСИ", "И", "ИЮИ", "ЮИ", "ЮЮИ",
             "Ю", "ЮЮЗ", "ЮЗ", "ЗЮЗ", "З", "ЗСЗ", "СЗ", "ССЗ"]


def _arcs(dirs: str):
    """Посоки за старт от таблицата → дъги (от, до по часовника). None = не се оценява.
    „И-Ю-З“ = от изток през юг до запад; „СИ, Ю“ = две отделни посоки."""
    if not dirs or "всички" in dirs:
        return None
    arcs = []
    for tok in dirs.split(","):
        angs = [_DIR[p.strip()] for p in tok.split("-") if p.strip() in _DIR]
        if len(angs) == 1:
            arcs.append((angs[0], angs[0]))
        for a, b in zip(angs, angs[1:]):          # по по-краткия път
            arcs.append((a, b) if (b - a) % 360 <= 180 else (b, a))
    return arcs or None


def _rel_dir(x: float, arcs) -> str:
    """'in' — в сектора (± dir_tol), 'back' — далеч от средата на всеки сектор, иначе 'cross'."""
    tol, back = WIND["dir_tol_deg"], WIND["back_deg"]
    d = lambda p, q: abs((p - q + 180) % 360 - 180)
    all_back = True
    for a, b in arcs:
        span = (b - a) % 360
        if (x - a) % 360 <= span or min(d(x, a), d(x, b)) <= tol:
            return "in"
        if d(x, (a + span / 2) % 360) <= back:
            all_back = False
    return "back" if all_back else "cross"


def _wind10_class(dfrom, spd, gust, arcs):
    """червено: над red_ms или пориви над gust_red_ms; жълто: страничен / в гръб; иначе зелено."""
    W = WIND
    spd = round(spd)                       # класът — по показаното цяло число m/s
    g = round(gust) if gust is not None else 0
    r = _rel_dir(dfrom, arcs) if arcs else "in"
    rel = "" if arcs is None else {"in": "в посоката", "cross": "страничен", "back": "в гръб"}[r]
    if spd > W["red_ms"] or g > W["gust_red_ms"]:
        return "red", rel
    if spd <= W["calm_ms"]:
        return "green", "тихо"
    if r != "in":
        return "yellow", rel
    return "green", rel


def _wind850_class(spd):
    spd = round(spd)
    if spd > WIND["h_red_ms"]:
        return "red"
    return "yellow" if spd > WIND["h_green_max_ms"] else "green"


def _site_winds(sites, winds, grid, t_list):
    """Вятър за всеки старт и всеки момент от t_list: (w10, w850) списъци по стартове."""
    glat, glon = grid
    tw = np.array([t.timestamp() for t in winds["times"]])
    out10, out850 = [], []
    for s in sites:
        iy = int(np.argmin(np.abs(glat - s["lat"])))
        ix = int(np.argmin(np.abs(glon - s["lon"])))
        arcs = _arcs(s.get("dirs", ""))
        r10, r850 = [], []
        for t in t_list:
            k = int(np.argmin(np.abs(tw - t.timestamp())))                 # най-близкият час
            kg = int(np.searchsorted(tw, t.timestamp() - 1))               # часът, който съдържа t
            kg = min(kg, len(tw) - 1)
            if winds.get("u10") is not None:
                u, v = float(winds["u10"][k, iy, ix]), float(winds["v10"][k, iy, ix])
                spd = float(np.hypot(u, v))
                dfrom = (np.degrees(np.arctan2(-u, -v)) + 360) % 360
                gust = float(winds["gust"][kg, iy, ix]) if winds.get("gust") is not None else None
                cls, rel = _wind10_class(dfrom, spd, gust, arcs)
                r10.append({"d": int(round(dfrom)), "s": int(round(spd)),
                            "g": None if gust is None else int(round(gust)), "c": cls,
                            "t": f"от {_DIR_NAME[int((dfrom + 11.25) // 22.5) % 16]} {spd:.0f} m/s"
                                 + ("" if gust is None else f", пориви {gust:.0f}") + (f" — {rel}" if rel else "")})
            else:
                r10.append(None)
            if winds.get("u850") is not None:
                u, v = float(winds["u850"][k, iy, ix]), float(winds["v850"][k, iy, ix])
                spd = float(np.hypot(u, v))
                dfrom = (np.degrees(np.arctan2(-u, -v)) + 360) % 360
                r850.append({"d": int(round(dfrom)), "s": int(round(spd)), "c": _wind850_class(spd),
                             "t": f"850 hPa: от {_DIR_NAME[int((dfrom + 11.25) // 22.5) % 16]} {spd:.0f} m/s"})
            else:
                r850.append(None)
        out10.append(r10); out850.append(r850)
    return out10, out850


def _lightning_points(li, lat, lon, step=0.1):
    """Мълниите (LI, 15 мин) събрани в клетки по 0.1° — за символи на картата."""
    if li is None or not (li >= 1).any():
        return []
    ys, xs = np.nonzero(li >= 1)
    cells = {}
    for y, x in zip(ys, xs):
        key = (round(lat[y] / step), round(lon[x] / step))
        cells[key] = max(cells.get(key, 0), float(li[y, x]))
    return [[k[0] * step, k[1] * step, int(v)] for k, v in cells.items()]


def export_web(forecast, blended=None, blend_times=None, winds=None, wind_grid=None, metars=None):
    comp = forecast["last_composite"]
    lat, lon = comp["lat"], comp["lon"]
    t0 = comp["timestamp"]
    now = dt.datetime.now(dt.timezone.utc)

    if os.path.isdir(OUT_DIR):
        shutil.rmtree(OUT_DIR)
    os.makedirs(OUT_DIR, exist_ok=True)

    # ── Кадри: (водещо време, поле, източник) ─────────────
    obs = comp.get("dbz_obs_plot", comp.get("dbz_radar", comp["dbz"]))
    frames = [(0, obs, "наблюдение")]
    fc_dbz = forecast.get("forecast_dbz_plot", forecast["forecast_dbz"])
    fc_times = forecast["timestamps"]

    def from_blend(m):
        if blended is None:
            return None
        k = int(np.argmin([abs((bt - t0).total_seconds() / 60 - m) for bt in blend_times]))
        return blended[k] if abs((blend_times[k] - t0).total_seconds() / 60 - m) <= 3 else None

    def from_nowcast(m):
        k = int(np.argmin([abs((ft - t0).total_seconds() / 60 - m) for ft in fc_times]))
        return fc_dbz[k] if abs((fc_times[k] - t0).total_seconds() / 60 - m) <= 3 else None

    method = forecast.get("method", "радар")
    for m in LEADS_NOW:
        f = from_blend(m)
        frames.append((m, f if f is not None else from_nowcast(m), f"прогноза ({method})"))
    for m in LEADS_BLEND:
        f = from_blend(m)
        if f is not None:
            frames.append((m, f, "радар → модел" if m <= 150 else "модел ICON"))
    frames = [(m, _clean(f), src) for m, f, src in frames if f is not None]

    # „Няма данни“: без радар и без сателит, докато ICON още не е поел (тегло < 50%).
    # Така празното място не изглежда като „сухо“, а валежът на модела не изскача на +10 мин.
    from src.blend.blending import blend_weights
    covered = comp.get("coverage")
    if covered is None:
        covered = np.isfinite(comp["dbz"])
    nd_zone = None if (comp.get("sat_on_map") or comp.get("fill_src")) else ~np.asarray(covered, dtype=bool)
    horizon = (fc_times[-1] - t0).total_seconds() / 60 if fc_times else 0.0

    def nodata_for(m):
        if nd_zone is None or not nd_zone.any():
            return None
        if m == 0:
            return nd_zone
        return nd_zone if blend_weights(m, radar_horizon=horizon)[1] < 0.5 else None

    nodata = [nodata_for(m) for m, _, _ in frames]

    # Резерв ICON + мълнии: същата зона и същият прозорец, но с данни (по модел) → лека щриховка
    fill_src = comp.get("fill_src")
    md_zone = ~np.asarray(covered, dtype=bool) if fill_src else None
    fill_label = {"icon_ltg": "модел ICON, коригиран по мълнии", "icon": "модел ICON"}.get(fill_src)

    def model_for(m):
        if md_zone is None or not md_zone.any():
            return None
        return md_zone if (m == 0 or blend_weights(m, radar_horizon=horizon)[1] < 0.5) else None

    model = [model_for(m) for m, _, _ in frames]
    if model[0] is not None:
        logger.info(f"  Без радар и сателит: {md_zone.mean():.0%} от домейна → {fill_label} "
                    f"до +{max(m for (m, _, _), n in zip(frames, model) if n is not None)} мин")
    if nodata[0] is not None:
        logger.info(f"  Без радар и сателит: {nd_zone.mean():.0%} от домейна → „няма данни“ "
                    f"до +{max(m for (m, _, _), n in zip(frames, nodata) if n is not None)} мин")

    meta_frames = []
    for i, (m, f, src) in enumerate(frames):
        name = f"f_{i:02d}.png"
        _save_png(f, lat, os.path.join(OUT_DIR, name), nodata[i], model[i])
        t = t0 + dt.timedelta(minutes=m)
        meta_frames.append({"file": name, "lead_min": m, "time_utc": t.strftime("%Y-%m-%dT%H:%M:%SZ"),
                            "time_local": _local(t), "source": src,
                            "nodata": nodata[i] is not None, "model": model[i] is not None})

    # ── Стартове ──────────────────────────────────────────
    sites = load_sites()
    li = comp.get("li_raw")
    if sites:
        masks = _disk_masks(sites, lat, lon)
        by_lead = {m: (f, nd) for (m, f, _), nd in zip(frames, nodata)}
        site_out = []
        for s, mk in zip(sites, masks):
            st, why = [], []
            for m in SITE_LEADS:
                if m not in by_lead:
                    st.append(None); why.append(None)
                    continue
                f, nd = by_lead[m]
                c, w = _status(f, li if m == 0 else None, mk, nd)
                st.append(c); why.append(w)
            # за точките на картата: състояние за всеки кадър (вкл. +4…+6 ч)
            sw = [_status(f, li if m == 0 else None, mk, nd) for (m, f, _), nd in zip(frames, nodata)]
            rec = {**s, "status": st, "why": why,
                   "status_frames": [c for c, _ in sw], "why_frames": [w for _, w in sw]}
            ic = site_icao(s)
            mt = (metars or {}).get(ic) if ic and METAR_SITES.get("enabled", True) else None
            if mt and mt["precip"] and (now - mt["time"]).total_seconds() <= METAR_SITES["stale_min"] * 60:
                fields = {m: f for m, f, _ in frames}
                ml, track = _metar_leads(mt, t0, SITE_LEADS, fields, mk[0])
                _apply_metar(rec["status"], rec["why"], ml)
                mf, _ = _metar_leads(mt, t0, [m for m, _, _ in frames], fields, mk[0])
                _apply_metar(rec["status_frames"], rec["why_frames"], mf)
                rec["metar"] = {"time": mt["time"].strftime("%Y-%m-%dT%H:%M:%SZ"),
                                "wx": mt["wx"], "mode": "следа" if track else "сляпо"}
                upto = max((m for m, x in zip(SITE_LEADS, ml) if x), default=None)
                logger.info(f"  METAR {ic}: {' '.join(mt['wx'])} ({'следа' if track else 'сляпо'})"
                            + (f" → {'червено/жълто' if mt['ts'] else 'жълто'} до +{upto} мин" if upto is not None
                               else " → не важи за таблицата"))
            if md_zone is not None:              # кръгът на старта е предимно в зоната „по модел“
                in_md = float(md_zone[mk[1]].mean()) > 0.5 if mk[1][0].size else False
                if in_md:
                    rec["model_frames"] = [md is not None for md in model]
                    rec["model_leads"] = [model_for(m) is not None for m in SITE_LEADS]
            site_out.append(rec)

        # Вятър (ICON): за колоните на таблицата и за всеки кадър на картата
        if winds is not None and wind_grid is not None:
            try:
                t_tab = [t0 + dt.timedelta(minutes=m) for m in SITE_LEADS]
                t_fr = [t0 + dt.timedelta(minutes=m) for m, _, _ in frames]
                w10, w850 = _site_winds(sites, winds, wind_grid, t_tab)
                f10, f850 = _site_winds(sites, winds, wind_grid, t_fr)
                for so, a, b, fa, fb in zip(site_out, w10, w850, f10, f850):
                    so["w10"], so["w850"] = a, b
                    so["w10_frames"] = [x["c"] if x else None for x in fa]
                    so["w850_frames"] = [x["c"] if x else None for x in fb]
                    so["wtxt_frames"] = ["\n".join(y["t"] for y in (p, q) if y) for p, q in zip(fa, fb)]
                logger.info(f"  Вятър за стартовете: {len(site_out)} × {len(t_tab)} часа")
            except Exception as e:
                logger.warning(f"  Вятър за стартовете: {e}")
        with open(os.path.join(OUT_DIR, "sites.json"), "w", encoding="utf-8") as fo:
            json.dump({"leads_min": SITE_LEADS,
                       "frame_leads_min": [m for m, _, _ in frames],
                       "leads_local": [_local(t0 + dt.timedelta(minutes=m)) for m in SITE_LEADS],
                       "radius_conv_km": R_CONV_KM, "radius_rain_km": R_RAIN_KM,
                       "wind_thr": {k: v for k, v in WIND.items() if k.endswith("_ms")},
                       "sites": site_out}, fo, ensure_ascii=False)
        n_red = sum(1 for s in site_out if "red" in s["status"][:4])
        logger.info(f"  Стартове: {len(site_out)}, с 🔴 до +90 мин: {n_red}")
        for s in site_out:                      # за диагностика: защо е червено
            for m, c, w in zip(SITE_LEADS, s["status"], s["why"]):
                if c == "red":
                    logger.info(f"    🔴 {s['name']} +{m} мин: {w}")

    # ── frames.json ───────────────────────────────────────
    mi = comp.get("motion_info") or {}
    meta = {
        "generated_utc": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "observation_utc": t0.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "observation_local": _local(t0),
        "method": method,
        "bounds": [[float(lat[0]), float(lon[0])], [float(lat[-1]), float(lon[-1])]],
        "frames": meta_frames,
        "motion": [{"region": r["name"], "dir": r["dir_txt"], "ms": round(r["kmh"] / 3.6),
                    "src": r["src"]} for r in mi.get("regions", [])],
        "lightning": _lightning_points(li, lat, lon),
        "sources": comp.get("sources", []),
        "satellite": bool(comp.get("sat_on_map")),
        "fill_label": fill_label,
    }
    with open(os.path.join(OUT_DIR, "frames.json"), "w", encoding="utf-8") as fo:
        json.dump(meta, fo, ensure_ascii=False)
    logger.info(f"Web експорт: {len(meta_frames)} кадъра → {OUT_DIR}")
    return OUT_DIR
