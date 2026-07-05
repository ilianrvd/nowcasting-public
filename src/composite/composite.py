"""
Composite — обединяване на всички радарни източници
====================================================
MAX-Z composite от Румъния + ИАБГ на общ lat/lon grid.
"""

import os, sys, logging
import numpy as np
from scipy.interpolate import RegularGridInterpolator

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from config.settings import DOMAIN

logger = logging.getLogger("composite")


def make_target_grid(resolution_km=None):
    if resolution_km is None:
        resolution_km = DOMAIN["resolution_km"]
    dlat = resolution_km / 111.0
    dlon = resolution_km / (111.0 * np.cos(np.radians(42.75)))
    lat = np.arange(DOMAIN["lat_min"], DOMAIN["lat_max"] + dlat / 2, dlat)
    lon = np.arange(DOMAIN["lon_min"], DOMAIN["lon_max"] + dlon / 2, dlon)
    return lat, lon


def reproject_frame(frame, target_lat, target_lon):
    """Интерполира frame към целевия grid."""
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


def create_composite(all_frames: list[dict],
                     match_sec: int = 600) -> dict | None:
    """
    MAX-Z composite от всички frames (Румъния + ИАБГ).
    """
    if not all_frames:
        return None

    target_lat, target_lon = make_target_grid()
    ref_time = max(f["timestamp"] for f in all_frames)

    layers, sources = [], []
    for f in all_frames:
        dt_sec = abs((f["timestamp"] - ref_time).total_seconds())
        if dt_sec > match_sec:
            logger.warning(f"  {f['source']}: Δ={dt_sec:.0f}s > {match_sec}s, пропуснат")
            continue
        reproj = reproject_frame(f, target_lat, target_lon)
        # Приложи маска за покритие
        from config.settings import COVERAGE_MASKS
        mask = COVERAGE_MASKS.get(f.get("source", ""), {})
        if mask:
            lon2d, lat2d = np.meshgrid(target_lon, target_lat)
            if "lat_min" in mask:
                reproj[lat2d < mask["lat_min"]] = np.nan
            if "lat_max" in mask:
                reproj[lat2d > mask["lat_max"]] = np.nan
            if "lon_min" in mask:
                reproj[lon2d < mask["lon_min"]] = np.nan
            if "lon_max" in mask:
                reproj[lon2d > mask["lon_max"]] = np.nan
            if "dbz_min" in mask:
                reproj[reproj < mask["dbz_min"]] = np.nan
        layers.append(reproj)
        sources.append(f["source"])
        logger.info(f"  {f['source']}: {f['timestamp']} Δ={dt_sec:.0f}s")

    if not layers:
        return None

    composite = np.nanmax(np.stack(layers), axis=0)
    composite[composite < 10.0] = np.nan
    valid = np.count_nonzero(~np.isnan(composite))
    logger.info(f"Composite: {len(sources)} слоя, {valid} valid px")

    return {
        "timestamp": ref_time,
        "dbz": composite,
        "lat": target_lat,
        "lon": target_lon,
        "sources": sources,
    }


def create_composite_series(all_frames_by_source: dict,
                            n_composites: int = 5) -> list[dict]:
    """
    Създава поредица composites за optical flow.
    all_frames_by_source: {"romania": [frames], "iabg_GCD": [frames], ...}
    """
    # Събери всички timestamps
    all_ts = set()
    for frames in all_frames_by_source.values():
        for f in frames:
            all_ts.add(f["timestamp"])

    if not all_ts:
        return []

    sorted_ts = sorted(all_ts)

    # Групирай по близки timestamps (±3 мин)
    groups = [[sorted_ts[0]]]
    for ts in sorted_ts[1:]:
        if (ts - groups[-1][-1]).total_seconds() < 180:
            groups[-1].append(ts)
        else:
            groups.append([ts])

    composites = []
    for group in groups[-n_composites:]:
        ref = max(group)
        frames_for_comp = []
        for frames in all_frames_by_source.values():
            best = min(frames, key=lambda f: abs((f["timestamp"] - ref).total_seconds()),
                       default=None)
            if best and abs((best["timestamp"] - ref).total_seconds()) < 600:
                frames_for_comp.append(best)

        comp = create_composite(frames_for_comp)
        if comp is not None and np.sum(~np.isnan(comp["dbz"])) > 500:
            composites.append(comp)

    logger.info(f"Серия: {len(composites)} composites")
    return composites
