"""
Blending: Radar nowcast + ICON-EU
==================================
0-30 мин  → 100% радар
30-60 мин → 90% радар + 10% ICON
60-90 мин → 70% радар + 30% ICON
...
180+ мин  → 100% ICON

Същата blend_weights схема от nowcasting системата на ДП РВД.
"""

import os, sys, logging
import datetime as dt
import numpy as np
from scipy.interpolate import RegularGridInterpolator

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from config.settings import BLEND, ICON

logger = logging.getLogger("blend")


def precip_to_dbz(precip_mmh, a=None, b=None):
    """mm/h → dBZ (Marshall-Palmer Z-R)."""
    if a is None: a = ICON["zr_a"]
    if b is None: b = ICON["zr_b"]
    R = np.clip(precip_mmh, 0.01, None)
    Z = a * R ** b
    dbz = 10.0 * np.log10(Z)
    dbz[precip_mmh < 0.1] = np.nan
    return dbz.astype(np.float32)


def blend_weights(minutes: float) -> tuple[float, float]:
    """
    Теглата за blending radar/NWP.
    По-бавен преход — радарът доминира до 90 мин.
    (Идентична с blending.py от ДП РВД системата)
    """
    if minutes <= 30:
        return 1.0, 0.0
    elif minutes <= 60:
        w = (minutes - 30) / 30.0
        return 1.0 - w * 0.1, w * 0.1
    elif minutes <= 90:
        w = (minutes - 60) / 30.0
        return 0.9 - w * 0.2, 0.1 + w * 0.2
    elif minutes <= 120:
        w = (minutes - 90) / 30.0
        return 0.7 - w * 0.2, 0.3 + w * 0.2
    elif minutes <= 180:
        w = (minutes - 120) / 60.0
        return 0.5 - w * 0.3, 0.5 + w * 0.3
    elif minutes <= 360:
        w = (minutes - 180) / 180.0
        return 0.2 - w * 0.2, 0.8 + w * 0.2
    else:
        return 0.0, 1.0


def interpolate_icon(target_time: dt.datetime,
                     icon_data: dict,
                     target_lat: np.ndarray,
                     target_lon: np.ndarray) -> np.ndarray:
    """
    Интерполира ICON precipitation grid до целевия момент и grid.
    Конвертира mm → dBZ чрез Z-R.
    """
    valid_times = icon_data["valid_times"]
    precip = icon_data["precipitation_mm"]   # (T, nlat, nlon) или (T,)
    icon_lat = icon_data.get("lat")
    icon_lon = icon_data.get("lon")

    # Намери двата съседни часа
    t0_idx, t1_idx = None, None
    for i, vt in enumerate(valid_times[:-1]):
        if vt <= target_time <= valid_times[i + 1]:
            t0_idx, t1_idx = i, i + 1
            break

    if t0_idx is None:
        # Извън обхвата — вземи последния
        t0_idx = len(valid_times) - 1
        t1_idx = t0_idx

    # Времева интерполация
    if t0_idx == t1_idx:
        w = 0.0
    else:
        dt_total = (valid_times[t1_idx] - valid_times[t0_idx]).total_seconds()
        dt_target = (target_time - valid_times[t0_idx]).total_seconds()
        w = dt_target / dt_total if dt_total > 0 else 0.0

    if precip.ndim == 1:
        # Единична точка — uniform grid
        p_interp = precip[t0_idx] * (1 - w) + precip[t1_idx] * w
        dbz_icon = precip_to_dbz(np.full((len(target_lat), len(target_lon)),
                                         p_interp))
        return dbz_icon

    # 3D grid — пространствена + времева интерполация
    p0 = precip[t0_idx]
    p1 = precip[t1_idx]
    p_interp = p0 * (1 - w) + p1 * w

    if icon_lat is not None and icon_lon is not None:
        interp = RegularGridInterpolator(
            (icon_lat, icon_lon), p_interp,
            method="linear", bounds_error=False, fill_value=0.0)
        lon2d, lat2d = np.meshgrid(target_lon, target_lat)
        pts = np.column_stack([lat2d.ravel(), lon2d.ravel()])
        p_on_grid = interp(pts).reshape(len(target_lat), len(target_lon))
    else:
        p_on_grid = np.full((len(target_lat), len(target_lon)),
                            p_interp.mean())

    return precip_to_dbz(p_on_grid)


def blend_nowcast_icon(forecast_dbz: np.ndarray,
                       forecast_times: list[dt.datetime],
                       icon_data: dict,
                       target_lat: np.ndarray,
                       target_lon: np.ndarray,
                       timestep_min: int = 5) -> np.ndarray:
    """
    Блендва S-PROG nowcast с ICON прогноза.

    Parameters
    ----------
    forecast_dbz : (n_steps, ny, nx) — radar nowcast
    forecast_times : list of datetime за всяка стъпка
    icon_data : dict от fetch_icon_grid
    target_lat, target_lon : координати на grid-а

    Returns
    -------
    blended : (n_total, ny, nx) — 0 до 6h
    blend_times : list of datetime
    """
    ref_time = forecast_times[0] - dt.timedelta(minutes=timestep_min)
    n_total = BLEND["n_steps"]
    ny, nx = len(target_lat), len(target_lon)

    blended = np.full((n_total, ny, nx), np.nan, dtype=np.float32)
    blend_times = []

    n_radar = forecast_dbz.shape[0]

    for step in range(n_total):
        minutes = (step + 1) * timestep_min
        target_time = ref_time + dt.timedelta(minutes=minutes)
        blend_times.append(target_time)

        rw, iw = blend_weights(minutes)

        # Radar component
        if step < n_radar and rw > 0:
            radar = forecast_dbz[step].copy()
        else:
            radar = np.full((ny, nx), np.nan)
            rw = 0.0
            iw = 1.0

        # ICON component
        if iw > 0 and icon_data is not None:
            icon = interpolate_icon(target_time, icon_data,
                                    target_lat, target_lon)
        else:
            icon = np.full((ny, nx), np.nan)

        # Blend — MAX approach за dBZ (по-добро от линейно средно)
        r_valid = ~np.isnan(radar)
        i_valid = ~np.isnan(icon)
        both = r_valid & i_valid

        result = np.full((ny, nx), np.nan)
        result[r_valid & ~i_valid] = radar[r_valid & ~i_valid]
        result[i_valid & ~r_valid] = icon[i_valid & ~r_valid]
        result[both] = rw * radar[both] + iw * icon[both]

        blended[step] = result

        if step % 12 == 0 or step == n_total - 1:
            logger.info(f"  +{minutes:3d} мин: R:{rw:.0%} I:{iw:.0%} "
                        f"max={np.nanmax(result):.1f} dBZ")

    return blended, blend_times
