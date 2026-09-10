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
def blend_weights(minutes: float, radar_horizon: float = 60.0) -> tuple[float, float]:
    """Тегла (radar, icon) като функция на minutes/radar_horizon.

    frac = minutes / radar_horizon (0=сега, 1=край на реалния радарен хоризонт).
    Преходът следва относителния хоризонт, не абсолютни минути — така при
    ts=13 мин (хоризонт 156) и ts=4 мин (хоризонт 60) схемата се държи еднакво.
    """
    if radar_horizon <= 0:
        return 0.0, 1.0
    frac = minutes / radar_horizon
    if frac <= 0.5:
        return 1.0, 0.0
    elif frac <= 1.0:
        w = (frac - 0.5) / 0.5
        return 1.0 - w * 0.3, w * 0.3          # 100%→70% радар до края на радара
    elif frac <= 2.0:
        w = frac - 1.0
        return 0.7 * (1.0 - w), 0.3 + 0.7 * w  # 70%→0% радар, ICON поема
    else:
        return 0.0, 1.0


# ────────────────────────────────────────────────────────────
# ICON интерполация (по време + пространство)
# ────────────────────────────────────────────────────────────
def _get_icon_field_raw(icon_data, field, t0_idx, t1_idx, w,
                         icon_lat, icon_lon, target_lat, target_lon):
    """Интерполира произволно ICON поле към target grid."""
    data = icon_data.get(field)
    if data is None:
        return np.zeros((len(target_lat), len(target_lon)))
    p = data[t0_idx] * (1 - w) + data[t1_idx] * w
    if icon_lat is not None and p.ndim == 2:
        interp = RegularGridInterpolator(
            (icon_lat, icon_lon), p,
            method="linear", bounds_error=False, fill_value=0.0)
        lon2d, lat2d = np.meshgrid(target_lon, target_lat)
        pts = np.column_stack([lat2d.ravel(), lon2d.ravel()])
        return interp(pts).reshape(len(target_lat), len(target_lon))
    return np.full((len(target_lat), len(target_lon)), float(np.mean(p)))

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

    # Конвективен enhancement (v3 — развързани роли):
    #   showers    → база (къде и колко вали, чист Z-R)
    #   LPI        → аддитивен бонус в dBZ (подчертава бурите, структура)
    #   cloud_top  → дълбочинен бонус (overshooting → град) + мек таван
    lpi_field = _get_icon_field_raw(icon_data, "lpi", t0_idx, t1_idx, w,
                                     icon_lat, icon_lon, target_lat, target_lon)
    ct_field = _get_icon_field_raw(icon_data, "cloud_top", t0_idx, t1_idx, w,
                                    icon_lat, icon_lon, target_lat, target_lon)

    LPI_REF = 25.0        # LPI за пълен структурен бонус
    BONUS_MAX = 15.0      # макс dBZ добавка от LPI
    DEPTH_BONUS = 8.0     # макс dBZ добавка за дълбока конвекция
    DBZ_HARD_CAP = 65.0

    # База от showers (NaN където няма валеж → няма конвективна област)
    dbz_base = precip_to_dbz(p_on_grid)

    # LPI бонус (аддитивен в dBZ), само където има база
    lpi_norm = np.clip(lpi_field / LPI_REF, 0.0, 1.0)
    dbz_bonus = lpi_norm * BONUS_MAX

    # Дълбочинен бонус: 10km→0, 14km→пълен
    dbz_depth = np.clip((ct_field - 10000.0) / 4000.0, 0.0, 1.0) * DEPTH_BONUS

    # Мек таван: 6km→50, 14km→65
    dbz_cap = np.clip(50.0 + (ct_field - 6000.0) / 8000.0 * 15.0, 45.0, 65.0)

    # Комбинация: бонусите само където базата е валидна (има валеж)
    base_valid = ~np.isnan(dbz_base)
    dbz_out = dbz_base.copy()
    dbz_out[base_valid] = np.minimum(
        dbz_base[base_valid] + dbz_bonus[base_valid] + dbz_depth[base_valid],
        dbz_cap[base_valid]
    )
    dbz_out = np.clip(dbz_out, None, DBZ_HARD_CAP)

    return dbz_out.astype(np.float32)

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
    actual_weights = []

    logger.info(f"Blend: радар {n_radar} кадъра на {radar_step_min:.0f} мин "
                f"(хоризонт {n_radar*radar_step_min:.0f} мин), "
                f"blend ос {n_total}×{blend_step} мин")

    for step in range(n_total):
        minutes = (step + 1) * blend_step
        target_time = ref_time + dt.timedelta(minutes=minutes)
        blend_times.append(target_time)

        rw, iw = blend_weights(minutes, radar_horizon=n_radar * radar_step_min)

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
        actual_weights.append((rw, iw))
        blended[step] = z_to_dbz(z_blend)

        if step % 12 == 0 or step == n_total - 1:
            logger.info(f"  +{minutes:3d} мин {target_time.strftime('%H:%M')}: "
                        f"R:{rw:.0%} I:{iw:.0%} "
                        f"max={np.nanmax(blended[step]) if np.any(~np.isnan(blended[step])) else 0:.1f} dBZ")

    return blended, blend_times, actual_weights
