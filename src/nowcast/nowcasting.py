"""
Nowcast: S-PROG екстраполация
==============================
Optical flow + cascade decomposition + semi-Lagrangian extrapolation.

Входът е хомогенна серия от composite.py на равномерна ос (10 мин).
Timestep-ът е реалният интервал между кадрите — S-PROG и LK приемат,
че една стъпка = един интервал на входа.
"""

import os, sys, logging
import datetime as dt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from config.settings import NOWCAST, DOMAIN

logger = logging.getLogger("nowcast")

HORIZON_MIN = 150        # радарен хоризонт на прогнозата


def dbz_to_r(dbz, a=200.0, b=1.6):
    return (10.0 ** (dbz / 10.0) / a) ** (1.0 / b)

def r_to_dbz(R, a=200.0, b=1.6):
    R = np.clip(R, 0.001, None)
    return 10.0 * np.log10(a * R ** b)


def compute_motion(composites, raw=True):
    """
    Optical flow (Lucas-Kanade) от поредица composites.
    NaN = извън радарното покритие (маска за LK); 0 dBZ = сухо в покритието.
    raw=True: движението се смята от радара БЕЗ усилването от мълниите
    (усилването е за интензивността, не трябва да създава фалшиво движение).
    Връща V в px за една стъпка на входа.
    """
    if len(composites) < 3:
        logger.error("Нужни са поне 3 composites!")
        return None

    key = (lambda c: c.get("dbz_radar", c["dbz"])) if raw else (lambda c: c["dbz"])
    D = np.stack([key(c) for c in composites])
    thr = NOWCAST.get("motion_min_dbz", 15.0)       # слабото ехо не се следи
    D = np.where(np.isfinite(D) & (D < thr), 0.0, D)
    R = np.where(D > 0, dbz_to_r(D), 0.0)
    R[np.isnan(D)] = np.nan
    R = np.ma.masked_invalid(R)

    try:
        from pysteps.motion.lucaskanade import dense_lucaskanade
    except ImportError as e:
        logger.error(f"pySTEPS не е инсталиран — задължителен е за optical flow: {e}")
        raise

    V = dense_lucaskanade(R, fd_kwargs={"buffer_mask": 15})
    return V


def extrapolate(field, V, n_steps):
    """Semi-Lagrangian екстраполация."""
    try:
        from pysteps.extrapolation.semilagrangian import extrapolate as sl_ext
        return sl_ext(field, V, n_steps)
    except ImportError:
        pass

    ny, nx = field.shape
    result = np.zeros((n_steps, ny, nx))
    for step in range(n_steps):
        y = np.arange(ny)[:, None] - V[1] * (step + 1)
        x = np.arange(nx)[None, :] - V[0] * (step + 1)
        yi = np.clip(np.round(y).astype(int), 0, ny - 1)
        xi = np.clip(np.round(x).astype(int), 0, nx - 1)
        result[step] = field[yi, xi]
    return result

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



def check_motion_vs_wind(V, last_comp, ts, wind700):
    """
    Сравнява средното движение от LK с очакваното движение от ICON
    (вятърът на 700 hPa × speed_factor ≈ 0.7) върху пикселите с ехо.
    При голямо отклонение (посока и/или скорост) връща V от ICON (също ×0.7).
    Връща (V, източник).
    """
    from scipy.interpolate import RegularGridInterpolator
    cfg = NOWCAST["wind_check"]
    lat, lon = last_comp["lat"], last_comp["lon"]
    lon2d, lat2d = np.meshgrid(lon, lat)
    pts = np.column_stack([lat2d.ravel(), lon2d.ravel()])

    def rg(f):
        it = RegularGridInterpolator((wind700["lat"], wind700["lon"]), f,
                                     bounds_error=False, fill_value=None)
        return it(pts).reshape(lat2d.shape)

    k = cfg.get("speed_factor", 1.0)                                # движение ≈ 0.7 × вятъра
    u, v = k * rg(wind700["u"]), k * rg(wind700["v"])               # m/s
    px_per_ms = ts * 60.0 / 1000.0 / DOMAIN["resolution_km"]        # m/s → px/стъпка
    V_icon = np.stack([u * px_per_ms, v * px_per_ms]).astype(np.float32)

    echo = np.nan_to_num(last_comp.get("dbz_radar", last_comp["dbz"]), nan=0.0) \
        >= NOWCAST.get("motion_min_dbz", 15.0)
    if echo.sum() < 50:
        echo = np.ones_like(echo)
    to_kt = DOMAIN["resolution_km"] * 60.0 / ts / 1.852
    lk = np.array([V[0][echo].mean(), V[1][echo].mean()])
    ic = np.array([V_icon[0][echo].mean(), V_icon[1][echo].mean()])
    s_lk, s_ic = np.hypot(*lk) * to_kt, np.hypot(*ic) * to_kt
    d_lk = (np.degrees(np.arctan2(lk[0], lk[1])) + 360) % 360
    d_ic = (np.degrees(np.arctan2(ic[0], ic[1])) + 360) % 360
    ddir = abs((d_lk - d_ic + 180) % 360 - 180)

    bad_dir = (s_lk >= cfg["min_speed_kt"] and s_ic >= cfg["min_speed_kt"]
               and ddir > cfg["max_dir_diff_deg"])
    ratio = s_lk / max(s_ic, 1e-3)
    bad_spd = ((ratio > cfg["max_speed_ratio"] or ratio < 1 / cfg["max_speed_ratio"])
               and abs(s_lk - s_ic) >= cfg["min_speed_diff_kt"])

    msg = (f"Проверка с ICON 700 hPa ({wind700['time']:%H} UTC): "
           f"LK {s_lk:.0f}KT към {d_lk:.0f}° | ICON×{k:g} {s_ic:.0f}KT към {d_ic:.0f}° | "
           f"разлика {ddir:.0f}°, отношение {ratio:.1f}")
    if bad_dir or bad_spd:
        why = " и ".join(w for w, b in (("посока", bad_dir), ("скорост", bad_spd)) if b)
        logger.warning(f"  {msg} → голямо отклонение ({why}) — движението е от ICON")
        return V_icon, "ICON 700 hPa"
    logger.info(f"  {msg} → ок, остава LK")
    return V, "радар (LK)"

def _log_motion_by_region(V, last_comp, ts):
    """Диагностика: движението поотделно по региони (север RO / ЮИ резерв)."""
    lat2d, lon2d = np.meshgrid(last_comp["lat"], last_comp["lon"], indexing="ij")
    regions = {"север (≥43°N)": lat2d >= 43.0,
               "ЮИ (<43°N, ≥26°E)": (lat2d < 43.0) & (lon2d >= 26.0)}
    echo = np.nan_to_num(last_comp.get("dbz_radar", last_comp["dbz"]), nan=0.0) >= 15.0
    parts = []
    for name, reg in regions.items():
        sel = reg & echo
        n = int(sel.sum())
        if n < 50:
            parts.append(f"{name}: малко ехо ({n} px)")
            continue
        vx, vy = V[0][sel], V[1][sel]
        kt = lambda a, b: float(np.hypot(a, b)) * DOMAIN["resolution_km"] * 60.0 / ts / 1.852
        deg = (np.degrees(np.arctan2(vx.mean(), vy.mean())) + 360.0) % 360.0
        sp = np.hypot(vx, vy) * DOMAIN["resolution_km"] * 60.0 / ts / 1.852
        parts.append(f"{name}: {kt(vx.mean(), vy.mean()):.0f}KT към {deg:.0f}° "
                     f"(скорости p10–p90 {np.percentile(sp, 10):.0f}–{np.percentile(sp, 90):.0f}KT, {n} px)")
    logger.info("  По региони: " + " | ".join(parts))


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

    logger.info(f"S-PROG: интервали {[round(d, 1) for d in dts]} мин → "
                f"{n_leadtimes}×{ts} мин = {n_leadtimes * ts} мин")

    V = compute_motion(composites)
    if V is None:
        return None

    if any("dbz_radar" in c for c in composites):      # диагностика: колко променя усилването
        V_b = compute_motion(composites, raw=False)
        if V_b is not None:
            mb = motion_summary(V_b, composites[-1], ts)
            logger.info(f"  (за сравнение, от усилените кадри: MOV {mb['dir_txt']} "
                        f"{mb['kt']:.0f}KT — не се ползва)")

    _log_motion_by_region(V, composites[-1], ts)
    _dump_series(composites, V)
    motion_src = "радар (LK)"
    if wind700 is not None:
        try:
            V, motion_src = check_motion_vs_wind(V, composites[-1], ts, wind700)
        except Exception as e:
            logger.warning(f"  Проверката с ICON не мина: {e}")

    motion = motion_summary(V, composites[-1], ts)
    motion["source"] = motion_src
    composites[-1]["motion_info"] = motion
    logger.info(f"Движение [{motion_src}]: MOV {motion['dir_txt']} {motion['kt']:.0f}KT "
                f"({motion['kmh']:.0f} km/h), към {motion['dir_deg']:.0f}°, "
                f"по {motion['n_px']} px с ехо")

    frames = [np.nan_to_num(c["dbz"], nan=0.0) for c in composites]
    R = dbz_to_r(np.stack(frames))
    R[R < 0.1] = 0.0

    try:
        from pysteps.nowcasts.sprog import forecast as sprog_fc
        from pysteps.utils.transformation import dB_transform
        R_log, _ = dB_transform(R, threshold=0.1, zerovalue=-15.0)
        R_fc = sprog_fc(R_log[-3:], V, n_leadtimes,
                        n_cascade_levels=n_cascade_levels,
                        precip_thr=-10.0)
        R_fc, _ = dB_transform(R_fc, inverse=True)
    except (ImportError, Exception) as e:
        logger.warning(f"S-PROG: {e} — semilagrangian fallback")
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
    }
