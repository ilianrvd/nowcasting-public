"""
Nowcast: ANVIL (основен) / S-PROG (резерв)
==========================================
Optical flow (Lucas-Kanade) + каскадно разлагане + semi-Lagrangian екстраполация.
ANVIL (Pulkkinen et al. 2020) — локален ARI(p,1) по каскадните нива: пази
малките конвективни ядра и тенденцията растеж/разпад. Иска p+2 кадъра;
при по-малко (или грешка) → S-PROG. Изборът: settings.NOWCAST["method"].

Входът е хомогенна серия от composite.py на равномерна ос (10 мин).
Timestep-ът е реалният интервал между кадрите — S-PROG и LK приемат,
че една стъпка = един интервал на входа.

Движение: LK от радара БЕЗ усилването от мълниите, само ехо ≥ motion_min_dbz;
после проверка по региони срещу вятъра на 700 hPa от ICON (check_motion_vs_wind).
В серията се записва logs/debug/series_*.npz за разбор.
"""

import os, sys, logging
import datetime as dt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from config.settings import NOWCAST, DOMAIN

logger = logging.getLogger("nowcast")

HORIZON_MIN = 150        # радарен хоризонт на прогнозата
R_MAX = (10 ** (70 / 10) / 200) ** (1 / 1.6)   # 70 dBZ — таван за ANVIL


def dbz_to_r(dbz, a=200.0, b=1.6):
    return (10.0 ** (dbz / 10.0) / a) ** (1.0 / b)

def r_to_dbz(R, a=200.0, b=1.6):
    R = np.clip(R, 0.001, None)
    return 10.0 * np.log10(a * R ** b)


def compute_motion(composites):
    """
    Optical flow (Lucas-Kanade) от поредица composites.
    NaN = извън радарното покритие (маска за LK); 0 dBZ = сухо в покритието.
    Ползва радара БЕЗ усилването от мълниите (dbz_radar) — усилването е за
    интензивността и не бива да създава фалшиво движение (12.10: 36 срещу 16 KT).
    Само ехо ≥ motion_min_dbz — слабите петна не се следят.
    Връща V в px за една стъпка на входа.
    """
    if len(composites) < 3:
        logger.error("Нужни са поне 3 composites!")
        return None

    D = np.stack([c.get("dbz_radar", c["dbz"]) for c in composites])
    thr = NOWCAST.get("motion_min_dbz", 15.0)
    D = np.where(np.isfinite(D) & (D < thr), 0.0, D)
    R = np.where(D > 0, dbz_to_r(D), 0.0)
    R[np.isnan(D)] = np.nan
    R = np.ma.masked_invalid(R)

    from pysteps.motion.lucaskanade import dense_lucaskanade
    return dense_lucaskanade(R, fd_kwargs={"buffer_mask": 15})


def extrapolate(field, V, n_steps):
    """Чиста semi-Lagrangian екстраполация (резерв, ако S-PROG гръмне)."""
    from pysteps.extrapolation.semilagrangian import extrapolate as sl_ext
    return sl_ext(field, V, n_steps)

COMPASS16 = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
             "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]


def motion_summary(V, last_comp, ts):
    """
    Средно движение на ехото в SIGMET стил (посока НАКЪДЕ).
    pySTEPS: V[0] = x (изток), V[1] = y (по редовете; тук lat нараства → север).
    Векторна средна само върху пикселите с ехо в последния кадър.
    """
    vx, vy = V[0], V[1]
    speed = np.hypot(vx, vy)
    echo = np.nan_to_num(last_comp.get("dbz_radar", last_comp["dbz"]), nan=0.0) \
        >= NOWCAST.get("motion_min_dbz", 15.0)
    sel = echo & (speed > 0)
    if sel.sum() < 50:
        sel = speed > 0
    if not sel.any():
        return {"dir_deg": 0.0, "dir_txt": "STNR", "kmh": 0.0, "kt": 0.0, "n_px": 0}
    mx, my = float(vx[sel].mean()), float(vy[sel].mean())
    kmh = float(np.hypot(mx, my)) * DOMAIN["resolution_km"] * 60.0 / ts
    deg = (np.degrees(np.arctan2(mx, my)) + 360.0) % 360.0
    txt = COMPASS16[int((deg + 11.25) // 22.5) % 16] if kmh >= 5 else "STNR"
    return {"dir_deg": deg, "dir_txt": txt, "kmh": kmh,
            "kt": kmh / 1.852, "n_px": int(sel.sum())}



REGIONS = {   # (име, условие върху lat/lon) — същите като зоните на радарите
    "север":  lambda la, lo: la >= 43.0,
    "ЮИ":     lambda la, lo: (la < 43.0) & (lo >= 26.0),
    "ЮЗ":     lambda la, lo: (la < 43.0) & (lo < 26.0),
}


def check_motion_vs_wind(V, last_comp, ts, wind700):
    """
    Сравнява движението от LK с вятъра на 700 hPa от ICON ПО РЕГИОНИ
    (север / ЮИ / ЮЗ), върху пикселите с ехо във всеки регион.
      - регион с достатъчно ехо и LK близо до ICON  → остава LK;
      - регион с голямо отклонение (посока/скорост) → ICON;
      - регион без ехо (напр. ЮЗ без радар)          → ICON
        (там LK е само интерполация отдалече, а H40B се мести с това V).
    Преходът между регионите е плавен (Гаусово тегло), за да няма шев във V.
    Връща (V, описание на източника).
    """
    from scipy.interpolate import RegularGridInterpolator
    from scipy.ndimage import gaussian_filter
    cfg = NOWCAST["wind_check"]
    lat, lon = last_comp["lat"], last_comp["lon"]
    lon2d, lat2d = np.meshgrid(lon, lat)
    pts = np.column_stack([lat2d.ravel(), lon2d.ravel()])

    def rg(f):
        it = RegularGridInterpolator((wind700["lat"], wind700["lon"]), f,
                                     bounds_error=False, fill_value=None)
        return it(pts).reshape(lat2d.shape)

    k = cfg.get("speed_factor", 1.0)
    u, v = k * rg(wind700["u"]), k * rg(wind700["v"])               # m/s
    px_per_ms = ts * 60.0 / 1000.0 / DOMAIN["resolution_km"]        # m/s → px/стъпка
    V_icon = np.stack([u * px_per_ms, v * px_per_ms]).astype(np.float32)

    echo = np.nan_to_num(last_comp.get("dbz_radar", last_comp["dbz"]), nan=0.0) \
        >= NOWCAST.get("motion_min_dbz", 15.0)
    to_kt = DOMAIN["resolution_km"] * 60.0 / ts / 1.852
    deg = lambda a: (np.degrees(np.arctan2(a[0], a[1])) + 360) % 360

    w_icon = np.zeros(echo.shape, dtype=np.float32)      # 1 = ICON, 0 = LK
    parts, used, reg_src = [], [], {}
    for name, cond in REGIONS.items():
        reg = cond(lat2d, lon2d)
        sel = reg & echo
        n = int(sel.sum())
        few = n < cfg.get("min_echo_px", 50)
        m = reg if few else sel
        ic = np.array([V_icon[0][m].mean(), V_icon[1][m].mean()])
        s_ic = np.hypot(*ic) * to_kt
        if few:
            w_icon[reg] = 1.0
            parts.append(f"{name}: без ехо → ICON {s_ic:.0f}KT/{deg(ic):.0f}°")
            used.append("ICON"); reg_src[name] = "ICON"
            continue
        lk = np.array([V[0][sel].mean(), V[1][sel].mean()])
        s_lk = np.hypot(*lk) * to_kt
        ddir = abs((deg(lk) - deg(ic) + 180) % 360 - 180)
        ratio = s_lk / max(s_ic, 1e-3)
        bad_dir = (s_lk >= cfg["min_speed_kt"] and s_ic >= cfg["min_speed_kt"]
                   and ddir > cfg["max_dir_diff_deg"])
        bad_spd = ((ratio > cfg["max_speed_ratio"] or ratio < 1 / cfg["max_speed_ratio"])
                   and abs(s_lk - s_ic) >= cfg["min_speed_diff_kt"])
        txt = (f"{name}: LK {s_lk:.0f}KT/{deg(lk):.0f}° vs ICON {s_ic:.0f}KT/{deg(ic):.0f}° "
               f"({ddir:.0f}°, ×{ratio:.1f}, {n} px)")
        if bad_dir or bad_spd:
            w_icon[reg] = 1.0
            why = "/".join(w for w, b in (("посока", bad_dir), ("скорост", bad_spd)) if b)
            parts.append(f"{txt} → ICON ({why})")
            used.append("ICON"); reg_src[name] = "ICON"
        else:
            parts.append(f"{txt} → LK")
            used.append("LK"); reg_src[name] = "LK"

    sigma = cfg.get("blend_sigma_px", 25)
    if 0 < w_icon.mean() < 1 and sigma > 0:
        w_icon = np.clip(gaussian_filter(w_icon, sigma), 0.0, 1.0)
    V_out = ((1.0 - w_icon) * V + w_icon * V_icon).astype(np.float32)

    logger.info(f"  Проверка с ICON 700 hPa ×{k:g} ({wind700['time']:%H} UTC): " + " | ".join(parts))
    if all(u_ == "LK" for u_ in used):
        src = "радар (LK)"
    elif all(u_ == "ICON" for u_ in used):
        src = "ICON 700 hPa"
    else:
        src = "LK + ICON 700 hPa (по региони)"
    return V_out, src, reg_src


def motion_by_region(V, last_comp, ts, reg_src=None):
    """
    Движението по региони за заглавието на картата:
    средно върху ехото в региона (или върху целия регион, ако ехо няма).
    """
    lon2d, lat2d = np.meshgrid(last_comp["lon"], last_comp["lat"])
    echo = np.nan_to_num(last_comp.get("dbz_radar", last_comp["dbz"]), nan=0.0) \
        >= NOWCAST.get("motion_min_dbz", 15.0)
    to_kmh = DOMAIN["resolution_km"] * 60.0 / ts
    out = []
    for name, cond in REGIONS.items():
        reg = cond(lat2d, lon2d)
        sel = reg & echo if (reg & echo).sum() >= 50 else reg
        mx, my = float(V[0][sel].mean()), float(V[1][sel].mean())
        kmh = float(np.hypot(mx, my)) * to_kmh
        deg = (np.degrees(np.arctan2(mx, my)) + 360.0) % 360.0
        txt = COMPASS16[int((deg + 11.25) // 22.5) % 16] if kmh >= 5 else "STNR"
        out.append({"name": name, "dir_txt": txt, "kt": kmh / 1.852, "kmh": kmh,
                    "src": (reg_src or {}).get(name, "LK")})
    return out

def _dump_series(composites, V, keep_hours=12):
    """Записва входната серия и V в logs/debug/ — за разбор на странно движение."""
    try:
        from config.settings import LOG_DIR
        d = os.path.join(LOG_DIR, "debug")
        os.makedirs(d, exist_ok=True)
        t0 = composites[-1]["timestamp"]
        path = os.path.join(d, f"series_{t0:%Y%m%d_%H%M}.npz")
        np.savez_compressed(
            path,
            dbz=np.stack([c.get("dbz_radar", c["dbz"]) for c in composites]).astype(np.float16),
            times=np.array([c["timestamp"].strftime("%Y-%m-%dT%H:%M:%SZ") for c in composites]),
            sources=np.array([",".join(c.get("sources", [])) for c in composites]),
            lat=composites[-1]["lat"], lon=composites[-1]["lon"],
            V=V.astype(np.float32))
        cutoff = dt.datetime.now().timestamp() - keep_hours * 3600
        for f in os.listdir(d):
            fp = os.path.join(d, f)
            if f.startswith("series_") and os.path.getmtime(fp) < cutoff:
                os.remove(fp)
        logger.info(f"  Серията записана: {path}")
    except Exception as e:
        logger.warning(f"  Записът на серията не стана: {e}")

def run_sprog(composites, n_leadtimes=None, n_cascade_levels=None, wind700=None):
    """
    S-PROG nowcasting.
    Returns: dict с forecast_dbz, timestamps, motion, last_composite
    """
    if n_cascade_levels is None:
        n_cascade_levels = NOWCAST["n_cascade_levels"]

    if len(composites) < 3:
        logger.error("S-PROG: нужни са поне 3 composites")
        return None

    # Реален timestep от входа (composite.py гарантира равномерна ос)
    dts = [(composites[i + 1]["timestamp"] - composites[i]["timestamp"]
            ).total_seconds() / 60 for i in range(len(composites) - 1)]
    ts = max(1, round(float(np.median(dts))))
    if max(abs(d - ts) for d in dts) > 1.0:
        logger.warning(f"  Неравномерни интервали {[round(d, 1) for d in dts]} "
                       f"— движението ще е неточно")

    if n_leadtimes is None:
        n_leadtimes = max(1, HORIZON_MIN // ts)

    logger.info(f"Nowcast: интервали {[round(d, 1) for d in dts]} мин → "
                f"{n_leadtimes}×{ts} мин = {n_leadtimes * ts} мин")

    V = compute_motion(composites)
    if V is None:
        return None

    _dump_series(composites, V)
    motion_src, reg_src = "радар (LK)", None
    if wind700 is not None:
        try:
            V, motion_src, reg_src = check_motion_vs_wind(V, composites[-1], ts, wind700)
        except Exception as e:
            logger.warning(f"  Проверката с ICON не мина: {e}")

    motion = motion_summary(V, composites[-1], ts)
    motion["source"] = motion_src
    motion["regions"] = motion_by_region(V, composites[-1], ts, reg_src)
    logger.info("  Движение по региони: " + " | ".join(
        f"{r['name']} {r['dir_txt']} {r['kt']:.0f}KT ({r['src']})" for r in motion["regions"]))
    composites[-1]["motion_info"] = motion
    logger.info(f"Движение [{motion_src}]: MOV {motion['dir_txt']} {motion['kt']:.0f}KT "
                f"({motion['kmh']:.0f} km/h), към {motion['dir_deg']:.0f}°, "
                f"по {motion['n_px']} px с ехо")

    frames = [np.nan_to_num(c["dbz"], nan=0.0) for c in composites]
    R = dbz_to_r(np.stack(frames))
    R[R < 0.1] = 0.0

    method = NOWCAST.get("method", "sprog")
    ac = NOWCAST.get("anvil", {})
    need = ac.get("ar_order", 2) + 2                    # ANVIL (ARI) иска p+2 кадъра
    if method == "anvil" and len(composites) < need:
        logger.warning(f"ANVIL иска поне {need} кадъра, има {len(composites)} — S-PROG")
        method = "sprog"

    R_fc = None
    if method == "anvil":
        try:
            from pysteps.nowcasts.anvil import forecast as anvil_fc
            logger.info(f"ANVIL: {len(composites)} кадъра, локален AR в радиус "
                        f"{ac.get('ar_window_radius', 25)} px")
            R_fc = anvil_fc(R[-need:], V, n_leadtimes,
                            n_cascade_levels=n_cascade_levels,
                            ar_order=ac.get("ar_order", 2),
                            ar_window_radius=ac.get("ar_window_radius", 25),
                            num_workers=ac.get("num_workers", 1))
            R_fc = np.clip(np.nan_to_num(R_fc, nan=0.0), 0.0, R_MAX)
        except Exception as e:
            logger.warning(f"ANVIL: {e} — S-PROG")
            method, R_fc = "sprog", None

    if R_fc is None:
        try:
            from pysteps.nowcasts.sprog import forecast as sprog_fc
            from pysteps.utils.transformation import dB_transform
            R_log, _ = dB_transform(R, threshold=0.1, zerovalue=-15.0)
            R_fc = sprog_fc(R_log[-3:], V, n_leadtimes,
                            n_cascade_levels=n_cascade_levels,
                            precip_thr=-10.0)
            R_fc, _ = dB_transform(R_fc, inverse=True)
        except Exception as e:
            logger.warning(f"S-PROG: {e} — semilagrangian fallback")
            method = "extrapolation"
            R_fc = extrapolate(R[-1], V, n_leadtimes)

    fc_dbz = r_to_dbz(np.clip(R_fc, 0.01, None))
    fc_dbz[R_fc < 0.1] = np.nan
    ref_time = composites[-1]["timestamp"]
    fc_times = [ref_time + dt.timedelta(minutes=ts * (i + 1))
                for i in range(n_leadtimes)]
    return {
        "forecast_dbz": fc_dbz,
        "timestamps": fc_times,
        "motion": V,
        "last_composite": composites[-1],
        "method": {"anvil": "ANVIL", "sprog": "S-PROG"}.get(method, "екстраполация"),
    }
