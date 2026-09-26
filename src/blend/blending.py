"""
Blending: Radar nowcast + ICON-EU
==================================
Blend мрежата е ФИКСИРАНА: 72 стъпки × 5 мин = 360 мин (6 часа).

За всяка целева минута T:
  - Радар: S-PROG кадърът, най-близък по време до T (в толеранс).
  - ICON: часовият прозорец (h-1, h], който съдържа T — без интерполация
    между часовете, за да се пази силата на ядрата.
  - Смесване в Z-space: Z = rw*Z_radar + iw*Z_icon.

ICON → dBZ (v5):
  showers (конвективен валеж за часа) → фон, чист Z-R
  LPI → ядра, само върху фон ≥ core_min_showers; dBZ = max(фон, ядро)
  LPI → dBZ по опорни точки от settings.ICON_DWD
"""

import os, sys, logging
import datetime as dt
import numpy as np
from scipy.interpolate import RegularGridInterpolator

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from config.settings import BLEND, ICON, ICON_DWD

logger = logging.getLogger("blend")

DBZ_HARD_CAP = 65.0


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
    """mm/h → dBZ (Marshall-Palmer Z-R). Под 0.1 mm/h → NaN."""
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
def blend_weights(minutes: float, radar_horizon: float = 60.0) -> tuple[float, float]:
    """
    Тегла (radar, icon) спрямо реалния радарен хоризонт.
    100% радар до 1/3 от хоризонта, после линейно до 0% точно в края му.
    Така няма скок, когато S-PROG кадрите свършат.
    """
    if radar_horizon <= 0:
        return 0.0, 1.0
    frac = minutes / radar_horizon
    if frac <= 1.0 / 3.0:
        return 1.0, 0.0
    if frac < 1.0:
        w = (frac - 1.0 / 3.0) / (2.0 / 3.0)
        return 1.0 - w, w
    return 0.0, 1.0


# ────────────────────────────────────────────────────────────
# ICON → dBZ
# ────────────────────────────────────────────────────────────
def _regrid(field2d, icon_data, target_lat, target_lon):
    """Линейна пространствена интерполация към target grid."""
    interp = RegularGridInterpolator(
        (icon_data["lat"], icon_data["lon"]), field2d,
        method="linear", bounds_error=False, fill_value=0.0)
    lon2d, lat2d = np.meshgrid(target_lon, target_lat)
    pts = np.column_stack([lat2d.ravel(), lon2d.ravel()])
    return interp(pts).reshape(len(target_lat), len(target_lon))


def interpolate_icon(target_time: dt.datetime,
                     icon_data: dict,
                     target_lat: np.ndarray,
                     target_lon: np.ndarray) -> np.ndarray:
    """ICON → dBZ поле на target grid за целевия момент."""
    times = icon_data["valid_times"]
    # Часът, чийто прозорец (vt-1ч, vt] съдържа target_time
    idx = next((i for i, vt in enumerate(times) if target_time <= vt),
               len(times) - 1)

    # Кеш по час — за 72 blend стъпки има само ~7 различни часа
    cache = icon_data.setdefault("_dbz_cache", {})
    if idx in cache:
        return cache[idx]

    showers = _regrid(icon_data["showers_mm"][idx], icon_data, target_lat, target_lon)
    dbz = precip_to_dbz(showers)                       # фон

    lpi_all = icon_data.get("lpi")
    if lpi_all is not None:
        lpi = _regrid(lpi_all[idx], icon_data, target_lat, target_lon)
        lpi_pts = ICON_DWD["lpi_anchors"]
        core_dbz = np.interp(lpi, lpi_pts, ICON_DWD["dbz_anchors"])
        core = (lpi >= lpi_pts[0]) & (showers >= ICON_DWD["core_min_showers"])
        dbz[core] = np.fmax(dbz[core], core_dbz[core])

    dbz = np.clip(dbz, None, DBZ_HARD_CAP).astype(np.float32)
    cache[idx] = dbz
    return dbz


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
    Радарният кадър за всяка целева минута се избира ПО ВРЕМЕ.
    timestep_min се игнорира (за съвместимост).
    """
    blend_step = 5
    n_total = BLEND["n_steps"]          # 72 → 360 мин

    ny, nx = len(target_lat), len(target_lon)
    n_radar = forecast_dbz.shape[0]

    if n_radar >= 1 and len(forecast_times) >= 2:
        radar_step_min = (forecast_times[1] - forecast_times[0]
                          ).total_seconds() / 60.0
    else:
        radar_step_min = 5.0
    ref_time = forecast_times[0] - dt.timedelta(minutes=radar_step_min)
    radar_horizon = n_radar * radar_step_min

    match_tol_min = radar_step_min / 2.0 + 1.0
    fc_times_arr = np.array([t.timestamp() for t in forecast_times])

    blended = np.full((n_total, ny, nx), np.nan, dtype=np.float32)
    blend_times = []
    actual_weights = []

    run = icon_data.get("run") if icon_data else None
    logger.info(f"Blend: радар {n_radar} кадъра на {radar_step_min:.0f} мин "
                f"(хоризонт {radar_horizon:.0f} мин), blend ос {n_total}×{blend_step} мин"
                + (f", ICON run {run:%Y-%m-%d %H} UTC" if run else ""))

    for step in range(n_total):
        minutes = (step + 1) * blend_step
        target_time = ref_time + dt.timedelta(minutes=minutes)
        blend_times.append(target_time)

        rw, iw = blend_weights(minutes, radar_horizon=radar_horizon)

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
            if icon_data is not None:
                iw = 1.0

        # ── ICON ─────────────────────────────────────────
        if iw > 0 and icon_data is not None:
            z_icon = dbz_to_z(interpolate_icon(target_time, icon_data,
                                               target_lat, target_lon))
        else:
            z_icon = np.zeros((ny, nx), dtype=np.float32)
            iw = 0.0

        # ── Z-space смесване ─────────────────────────────
        z_blend = rw * z_radar + iw * z_icon
        actual_weights.append((rw, iw))
        blended[step] = z_to_dbz(z_blend)

        if step % 12 == 0 or step == n_total - 1:
            mx = np.nanmax(blended[step]) if np.any(~np.isnan(blended[step])) else 0
            logger.info(f"  +{minutes:3d} мин {target_time.strftime('%H:%M')}: "
                        f"R:{rw:.0%} I:{iw:.0%} max={mx:.1f} dBZ")

    return blended, blend_times, actual_weights