"""
Composite — обединяване на радарните източници (v2)
====================================================
Правила (план 2026-09-25):
  - Времева ос: 10-мин марки на румънския композит.
  - Север (>=43°N): само RO. Юг (<43°N): GCD западно от 26°E, STS източно.
  - RO трябва да е точно на марката; ИАБГ скан се прикачва в ±3 мин.
  - Кадър, в който липсва активен източник, се изхвърля (без кърпене).
    Серията е само последната непрекъсната поредица от пълни кадри.
  - Източник, по-стар от 15 мин спрямо оста, е застоял за целия пуск
    → регионът му остава без покритие.
  - Сухо в покритието = 0 dBZ; извън покритието = NaN.
"""

import os, sys, logging
import datetime as dt
import numpy as np
from scipy.interpolate import RegularGridInterpolator

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from config.settings import DOMAIN, COVERAGE_MASKS, IABG_RADARS

logger = logging.getLogger("composite")

GRID_STEP_MIN = 10                       # каденция на RO
TOL_SEC = {"romania_composite": 60}      # RO точно на марката
TOL_SEC_DEFAULT = 180                    # ИАБГ ±3 мин
STALE_MIN = 15                           # източник по-стар от това → изключен
DBZ_MIN = 15.0                           # под това → сухо (0 dBZ)
DBZ_MAX = 70.0                           # физически таван


def make_target_grid(resolution_km=None):
    if resolution_km is None:
        resolution_km = DOMAIN["resolution_km"]
    dlat = resolution_km / 111.0
    dlon = resolution_km / (111.0 * np.cos(np.radians(42.75)))
    lat = np.arange(DOMAIN["lat_min"], DOMAIN["lat_max"] + dlat / 2, dlat)
    lon = np.arange(DOMAIN["lon_min"], DOMAIN["lon_max"] + dlon / 2, dlon)
    return lat, lon


def reproject_frame(frame, target_lat, target_lon):
    """Интерполира frame към целевия grid (nearest — пази пиковете)."""
    src_lat, src_lon, src_dbz = frame["lat"], frame["lon"], frame["dbz"]

    if src_lat[0] > src_lat[-1]:
        src_lat = src_lat[::-1]
        src_dbz = src_dbz[::-1, :]

    filled = np.nan_to_num(src_dbz, nan=-999.0)

    try:
        interp = RegularGridInterpolator(
            (src_lat, src_lon), filled,
            method="nearest", bounds_error=False, fill_value=-999.0)
    except Exception as e:
        logger.error(f"Интерполация: {e}")
        return np.full((len(target_lat), len(target_lon)), np.nan)

    lon2d, lat2d = np.meshgrid(target_lon, target_lat)
    pts = np.column_stack([lat2d.ravel(), lon2d.ravel()])
    result = interp(pts).reshape(len(target_lat), len(target_lon))
    result[result <= -998] = np.nan
    return result.astype(np.float32)


def coverage_mask(source, lat2d, lon2d):
    """Зона, за която източникът отговаря: регион + обхват на радара."""
    m = np.ones(lat2d.shape, dtype=bool)
    rules = COVERAGE_MASKS.get(source)
    if rules is None:
        return np.zeros(lat2d.shape, dtype=bool)   # непознат източник → нищо
    if "lat_min" in rules:
        m &= lat2d >= rules["lat_min"]
    if "lat_max" in rules:
        m &= lat2d < rules["lat_max"]
    if "lon_min" in rules:
        m &= lon2d >= rules["lon_min"]
    if "lon_max" in rules:
        m &= lon2d < rules["lon_max"]
    if source.startswith("iabg_"):
        r = IABG_RADARS[source[5:]]
        dy = (lat2d - r["lat"]) * 111.0
        dx = (lon2d - r["lon"]) * 111.0 * np.cos(np.radians(r["lat"]))
        m &= np.hypot(dx, dy) <= r["range_km"]
    return m


def create_composite(frames, ref_time, target_lat, target_lon):
    """Композит за една марка. frames — по един кадър на активен източник."""
    lon2d, lat2d = np.meshgrid(target_lon, target_lat)
    comp = np.full(lat2d.shape, np.nan, dtype=np.float32)
    covered = np.zeros(lat2d.shape, dtype=bool)
    sources = []

    for f in frames:
        src = f["source"]
        cov = coverage_mask(src, lat2d, lon2d)
        reproj = reproject_frame(f, target_lat, target_lon)
        # В покритието: NaN = сухо → 0; извън покритието → NaN
        layer = np.where(cov, np.nan_to_num(reproj, nan=0.0), np.nan)
        comp = np.fmax(comp, layer)
        covered |= cov
        sources.append(src)
        d = (f["timestamp"] - ref_time).total_seconds()
        logger.info(f"    {src}: {f['timestamp']:%H:%M:%S} Δ={d:+.0f}s")

    comp[covered & (comp < DBZ_MIN)] = 0.0
    n_clip = np.count_nonzero(comp > DBZ_MAX)
    if n_clip > 0:
        logger.warning(f"    Clip: {n_clip} px над {DBZ_MAX:.0f} dBZ → {DBZ_MAX:.0f}")
        comp[comp > DBZ_MAX] = DBZ_MAX

    echo = np.count_nonzero(comp >= DBZ_MIN)
    logger.info(f"    → {len(sources)} слоя, покритие {covered.mean():.0%}, ехо {echo} px")

    return {
        "timestamp": ref_time,
        "dbz": comp,
        "coverage": covered,
        "lat": target_lat,
        "lon": target_lon,
        "sources": sources,
    }


def _latest(frames):
    return max(f["timestamp"] for f in frames)


def create_composite_series(all_frames_by_source: dict,
                            n_composites: int = 5) -> list[dict]:
    """
    Хомогенна серия composites на 10-мин ос.
    all_frames_by_source: {"romania": [frames], "iabg_GCD": [frames], ...}
    """
    # Групирай по source (ключовете на входа не са същите като f["source"])
    by_src = {}
    for frames in all_frames_by_source.values():
        for f in frames:
            by_src.setdefault(f["source"], []).append(f)
    by_src = {s: v for s, v in by_src.items() if s in COVERAGE_MASKS}
    if not by_src:
        logger.error("Няма източници с дефинирано покритие")
        return []

    # Котва на оста: последният RO кадър
    if "romania_composite" in by_src:
        anchor = _latest(by_src["romania_composite"])
    else:
        t = max(_latest(v) for v in by_src.values())
        anchor = t.replace(minute=t.minute - t.minute % GRID_STEP_MIN,
                           second=0, microsecond=0)
        logger.warning("Няма RO — оста се котви на ИАБГ, северът е без покритие")

    # Застояли източници → изключени за целия пуск
    active = {}
    for src, frames in by_src.items():
        last = _latest(frames)
        age = (anchor - last).total_seconds() / 60
        if age > STALE_MIN:
            logger.warning(f"  {src}: застоял (последен {last:%H:%M} UTC, "
                           f"{age:.0f} мин преди оста) — изключен за пуска")
            continue
        active[src] = frames
    logger.info(f"Ос: {anchor:%H:%M} UTC, активни: {', '.join(active)}")

    # Назад по марките; спираме при първия непълен кадър след началото на серията
    target_lat, target_lon = make_target_grid()
    composites = []
    for k in range(n_composites + 3):
        T = anchor - dt.timedelta(minutes=GRID_STEP_MIN * k)
        chosen, missing = [], []
        for src, frames in active.items():
            best = min(frames, key=lambda f: abs((f["timestamp"] - T).total_seconds()))
            tol = TOL_SEC.get(src, TOL_SEC_DEFAULT)
            if abs((best["timestamp"] - T).total_seconds()) <= tol:
                chosen.append(best)
            else:
                missing.append(src)

        if missing:
            logger.info(f"  Кадър {T:%H:%M}: липсва {', '.join(missing)} — изхвърлен")
            if composites:
                break          # серията трябва да е непрекъсната
            continue

        logger.info(f"  Кадър {T:%H:%M}:")
        composites.append(create_composite(chosen, T, target_lat, target_lon))
        if len(composites) == n_composites:
            break

    composites.reverse()   # хронологичен ред
    if composites:
        logger.info(f"Серия: {len(composites)} composites, "
                    f"{composites[0]['timestamp']:%H:%M}–{composites[-1]['timestamp']:%H:%M} UTC, "
                    f"стъпка {GRID_STEP_MIN} мин")
    if len(composites) < 3:
        logger.warning("По-малко от 3 пълни кадъра — движението ще е ненадеждно")
    return composites