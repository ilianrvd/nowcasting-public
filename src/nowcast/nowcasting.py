"""
Nowcast: S-PROG екстраполация
==============================
Optical flow + cascade decomposition + semi-Lagrangian extrapolation.
"""

import os, sys, logging
import datetime as dt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from config.settings import NOWCAST, DOMAIN

logger = logging.getLogger("nowcast")


def dbz_to_r(dbz, a=200.0, b=1.6):
    return (10.0 ** (dbz / 10.0) / a) ** (1.0 / b)

def r_to_dbz(R, a=200.0, b=1.6):
    R = np.clip(R, 0.001, None)
    return 10.0 * np.log10(a * R ** b)


def compute_motion(composites):
    """Optical flow от поредица composites."""
    if len(composites) < 2:
        logger.error("Нужни са поне 2 composites!")
        return None

    frames = []
    for c in composites:
        d = c["dbz"].copy()
        d[np.isnan(d)] = 0.0
        frames.append(d)

    precip = np.stack(frames)
    R = dbz_to_r(precip)
    R[R < 0.1] = 0.0

    try:
        from pysteps.motion.lucaskanade import dense_lucaskanade
        V = dense_lucaskanade(R)
        speed = np.sqrt(V[0]**2 + V[1]**2)
        ms = np.nanmean(speed[speed > 0]) if np.any(speed > 0) else 0
        logger.info(f"Optical flow: {ms:.1f} px/step")
        return V
    except ImportError:
        logger.warning("pySTEPS липсва — OpenCV fallback")

    try:
        import cv2
        f1 = np.clip(frames[-2] / 70 * 255, 0, 255).astype(np.uint8)
        f2 = np.clip(frames[-1] / 70 * 255, 0, 255).astype(np.uint8)
        flow = cv2.calcOpticalFlowFarneback(f1, f2, None,
            0.5, 5, 15, 3, 5, 1.2, 0)
        return np.stack([flow[..., 1], flow[..., 0]])
    except ImportError:
        logger.error("Нито pySTEPS, нито OpenCV!")
        return None


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
        y = np.arange(ny)[:, None] - V[0] * (step + 1)
        x = np.arange(nx)[None, :] - V[1] * (step + 1)
        yi = np.clip(np.round(y).astype(int), 0, ny - 1)
        xi = np.clip(np.round(x).astype(int), 0, nx - 1)
        result[step] = field[yi, xi]
    return result


def run_sprog(composites, n_leadtimes=None, n_cascade_levels=None):
    """
    S-PROG nowcasting.
    Returns: dict с forecast_dbz, timestamps, motion, last_composite
    """
    if n_cascade_levels is None:
        n_cascade_levels = NOWCAST["n_cascade_levels"]

    # Изчисли реалния timestep от composite timestamps
    if len(composites) >= 2:
        real_dt = (composites[-1]["timestamp"] - composites[-2]["timestamp"]).total_seconds() / 60
        ts = round(real_dt)
        if ts < 1:
            ts = NOWCAST["timestep_min"]
    else:
        ts = NOWCAST["timestep_min"]

    # Покрий поне 60 минути
    if n_leadtimes is None:
        n_leadtimes = max(NOWCAST["n_leadtimes"], 60 // ts)

    logger.info(f"S-PROG: {n_leadtimes}×{ts}min = {n_leadtimes * ts}min")
 
    V = compute_motion(composites)
    if V is None:
        return None

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

    logger.info(f"Forecast: max={np.nanmax(fc_dbz):.1f} dBZ")

    return {
        "forecast_dbz": fc_dbz,
        "timestamps": fc_times,
        "motion": V,
        "last_composite": composites[-1],
    }


def enhance_with_lightning(forecast, ltg_density, ltg_lat, ltg_lon,
                           boost_dbz=10.0):
    """Усилва прогнозата където има мълнии но слаб/никакъв radar echo."""
    fc_dbz = forecast["forecast_dbz"].copy()
    comp_lat = forecast["last_composite"]["lat"]
    comp_lon = forecast["last_composite"]["lon"]

    if ltg_lat[0] > ltg_lat[-1]:
        ltg_lat = ltg_lat[::-1]
        ltg_density = ltg_density[::-1, :]

    interp = RegularGridInterpolator(
        (ltg_lat, ltg_lon), ltg_density,
        method="nearest", bounds_error=False, fill_value=0.0)
    lon2d, lat2d = np.meshgrid(comp_lon, comp_lat)
    ltg = interp(np.column_stack([lat2d.ravel(), lon2d.ravel()])
                 ).reshape(len(comp_lat), len(comp_lon))

    mask = ltg > 0.5
    cnt = 0
    for t in range(fc_dbz.shape[0]):
        low = np.isnan(fc_dbz[t]) | (fc_dbz[t] < 30)
        enh = mask & low
        fc_dbz[t][enh] = np.maximum(
            np.nan_to_num(fc_dbz[t][enh], nan=0), 30 + boost_dbz * ltg[enh] / ltg.max())
        cnt += enh.sum()

    if cnt > 0:
        logger.info(f"Lightning enhancement: {cnt} px")

    forecast["forecast_dbz"] = fc_dbz
    return forecast


# Нужен за enhance_with_lightning
from scipy.interpolate import RegularGridInterpolator
