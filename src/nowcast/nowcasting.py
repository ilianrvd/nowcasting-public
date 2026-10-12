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
    R = dbz_to_r(np.stack([key(c) for c in composites]))
    R = np.ma.masked_invalid(R)
    R[R < 0.05] = 0.0

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
    echo = np.nan_to_num(last_comp["dbz"], nan=0.0) >= 15.0
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

def run_sprog(composites, n_leadtimes=None, n_cascade_levels=None):
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

    motion = motion_summary(V, composites[-1], ts)
    composites[-1]["motion_info"] = motion
    logger.info(f"Optical flow (радар без усилване): MOV {motion['dir_txt']} {motion['kt']:.0f}KT "
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
