"""
Blending: Radar nowcast + ICON-EU
==================================
АРХИТЕКТУРА (v2 — развързани времеви оси):

Blend мрежата е ФИКСИРАНА: 72 стъпки × 5 мин = 360 мин (6 часа).
Тя не зависи от радарния timestep.

За всяка целева минута T:
  - Радар: избира се S-PROG кадърът чийто timestamp е най-близо
    до T (по време, не по индекс). Ако няма кадър в рамките на
    толеранса → радарен принос 0.
  - ICON: времева интерполация между часовите стъпки (по време).
  - Смесване в Z-space: Z = rw*Z_radar + iw*Z_icon (физически
    коректно, dBZ е логаритмична скала).

Така радарният timestep (5, 10, 14 мин — какъвто е реално) и
blend оста са напълно независими. Етикетите +30/+60/... мин на
картите винаги отговарят на реални минути.

ICON ползва "showers" (конвективен валеж).
"""

import os, sys, logging
import datetime as dt
import numpy as np
from scipy.interpolate import RegularGridInterpolator

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from config.settings import BLEND, ICON

logger = logging.getLogger("blend")


# ────────────────────────────────────────────────────────────
# Конверсии
# ────────────────────────────────────────────────────────────
def dbz_to_z(dbz: np.ndarray) -> np.ndarray:
    """dBZ → Z (linear). NaN → 0 (няма ехо)."""
    d = np.nan_to_num(dbz, nan=-999.0)
    return np.where(d > -900, 10.0 ** (d / 10.0), 0.0)


def z_to_dbz(z: np.ndarray, min_z: float = 1e-3) -> np.ndarray:
    """Z → dBZ. Под min_z → NaN (няма ехо)."""
    dbz = np.full(z.shape, np.nan, dtype=np.float32)
    valid = z > min_z
    dbz[valid] = 10.0 * np.log10(z[valid])
    return dbz


def precip_to_dbz(precip_mmh, a=None, b=None):
    """mm/h → dBZ (Marshall-Palmer Z-R)."""
    if a is None: a = ICON["zr_a"]
    if b is None: b = ICON["zr_b"]
    R = np.clip(precip_mmh, 0.01, None)
    Z = a * R ** b
    dbz = 10.0 * np.log10(Z)
    dbz[precip_mmh < 0.1] = np.nan
    return dbz.astype(np.float32)


# ────────────────────────────────────────────────────────────
# Тегла
# ────────────────────────────────────────────────────────────
def blend_weights(minutes: float) -> tuple[float, float]:
    """Тегла (radar, icon) като функция на прогн. хоризонт в минути."""
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


# ────────────────────────────────────────────────────────────
# ICON интерполация (по време + пространство)
# ────────────────────────────────────────────────────────────
def interpolate_icon(target_time: dt.datetime,
                     icon_data: dict,
                     target_lat: np.ndarray,
                     target_lon: np.ndarray) -> np.ndarray:
    """
    ICON showers → dBZ поле на target grid за целевия момент.
    Времева интерполация между часовите стъпки, после
    пространствена (linear) към 1-km мрежата.
    """
    valid_times = icon_data["valid_times"]
    precip = icon_data.get("showers_mm", icon_data.get("precipitation_mm"))
    icon_lat = icon_data.get("lat")
    icon_lon = icon_data.get("lon")

    # Времеви скоби
    t0_idx, t1_idx = None, None
    for i, vt in enumerate(valid_times[:-1]):
        if vt <= target_time <= valid_times[i + 1]:
            t0_idx, t1_idx = i, i + 1
            break
    if t0_idx is None:
        t0_idx = t1_idx = len(valid_times) - 1

    if t0_idx == t1_idx:
        w = 0.0
    else:
        span = (valid_times[t1_idx] - valid_times[t0_idx]).total_seconds()
        off = (target_time - valid_times[t0_idx]).total_seconds()
        w = max(0.0, min(1.0, off / span)) if span > 0 else 0.0

    p_interp = precip[t0_idx] * (1 - w) + precip[t1_idx] * w

    if icon_lat is not None and icon_lon is not None and p_interp.ndim == 2:
        interp = RegularGridInterpolator(
            (icon_lat, icon_lon), p_interp,
            method="linear", bounds_error=False, fill_value=0.0)
        lon2d, lat2d = np.meshgrid(target_lon, target_lat)
        pts = np.column_stack([lat2d.ravel(), lon2d.ravel()])
        p_on_grid = interp(pts).reshape(len(target_lat), len(target_lon))
    else:
        p_on_grid = np.full((len(target_lat), len(target_lon)),
                            float(np.mean(p_interp)))

    # Конвективен пиков фактор: часовата акумулация подценява
    # моментния интензитет ~3x за конвективни клетки
    p_on_grid = p_on_grid * 3.0
    return precip_to_dbz(p_on_grid)


# ────────────────────────────────────────────────────────────
# Главен blend
# ────────────────────────────────────────────────────────────
def blend_nowcast_icon(forecast_dbz: np.ndarray,
                       forecast_times: list[dt.datetime],
                       icon_data: dict,
                       target_lat: np.ndarray,
                       target_lon: np.ndarray,
                       timestep_min: int = None) -> tuple:
    """
    Radar nowcast + ICON върху фиксирана 5-минутна blend ос (360 мин).

    Радарният кадър за всяка целева минута се избира ПО ВРЕМЕ
    (най-близък timestamp), не по индекс — така радарният timestep
    може да е произволен (5, 10, 14 мин...).

    Parameters
    ----------
    forecast_dbz   : (n_radar, ny, nx) S-PROG кадри
    forecast_times : list[datetime] — реалните валидни времена на кадрите
    timestep_min   : игнорира се (за съвместимост); blend оста е 5 мин.
    """
    blend_step = 5                      # фиксирана blend мрежа
    n_total = BLEND["n_steps"]          # 72 → 360 мин

    ny, nx = len(target_lat), len(target_lon)
    n_radar = forecast_dbz.shape[0]

    # Референтно време: старт на прогнозата (OBS момент)
    if n_radar >= 1 and len(forecast_times) >= 2:
        radar_step_min = (forecast_times[1] - forecast_times[0]
                          ).total_seconds() / 60.0
    else:
        radar_step_min = 5.0
    ref_time = forecast_times[0] - dt.timedelta(minutes=radar_step_min)

    # Толеранс за времево съответствие радар↔цел:
    # половин радарна стъпка + 1 мин
    match_tol_min = radar_step_min / 2.0 + 1.0

    fc_times_arr = np.array([t.timestamp() for t in forecast_times])

    blended = np.full((n_total, ny, nx), np.nan, dtype=np.float32)
    blend_times = []

    logger.info(f"Blend: радар {n_radar} кадъра на {radar_step_min:.0f} мин "
                f"(хоризонт {n_radar*radar_step_min:.0f} мин), "
                f"blend ос {n_total}×{blend_step} мин")

    for step in range(n_total):
        minutes = (step + 1) * blend_step
        target_time = ref_time + dt.timedelta(minutes=minutes)
        blend_times.append(target_time)

        rw, iw = blend_weights(minutes)

        # ── Радар: избор ПО ВРЕМЕ ────────────────────────
        z_radar = None
        if rw > 0 and n_radar > 0:
            diffs = np.abs(fc_times_arr - target_time.timestamp()) / 60.0
            best = int(np.argmin(diffs))
            if diffs[best] <= match_tol_min:
                z_radar = dbz_to_z(forecast_dbz[best])
        if z_radar is None:
            z_radar = np.zeros((ny, nx), dtype=np.float32)
            rw = 0.0
            # преразпредели тежестта към ICON
            if icon_data is not None:
                iw = 1.0 if minutes > 60 else iw

        # ── ICON ─────────────────────────────────────────
        if iw > 0 and icon_data is not None:
            icon_dbz = interpolate_icon(target_time, icon_data,
                                        target_lat, target_lon)
            z_icon = dbz_to_z(icon_dbz)
        else:
            z_icon = np.zeros((ny, nx), dtype=np.float32)
            iw = 0.0

        # ── Z-space смесване ─────────────────────────────
        z_blend = rw * z_radar + iw * z_icon
        blended[step] = z_to_dbz(z_blend)

        if step % 12 == 0 or step == n_total - 1:
            logger.info(f"  +{minutes:3d} мин {target_time.strftime('%H:%M')}: "
                        f"R:{rw:.0%} I:{iw:.0%} "
                        f"max={np.nanmax(blended[step]) if np.any(~np.isnan(blended[step])) else 0:.1f} dBZ")

    return blended, blend_times
