"""
Ingest: ICON-EU от Open-Meteo API
==================================
Сваля precipitation forecast от ICON-EU и конвертира в dBZ grid
за blending с радарния nowcast.
Адаптирано от fog модела (historical-forecast-api работи от GitHub Actions).
"""

import os, sys, json, logging
import datetime as dt
import urllib.request, urllib.parse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from config.settings import ICON, DOMAIN

logger = logging.getLogger("ingest.icon")


def fetch_icon_precipitation(ref_time: dt.datetime = None,
                             forecast_hours: int = None) -> dict | None:
    """
    Сваля ICON-EU precipitation grid за домейна.

    Returns: dict с precipitation_mmh (T, ny, nx), lat, lon, valid_times
    """
    if ref_time is None:
        ref_time = dt.datetime.now(dt.timezone.utc)
    if forecast_hours is None:
        forecast_hours = ICON["forecast_hours"]

    today = ref_time.strftime("%Y-%m-%d")
    tomorrow = (ref_time + dt.timedelta(days=1)).strftime("%Y-%m-%d")

    params = {
        "latitude": f"{DOMAIN['lat_min']},{DOMAIN['lat_max']}",
        "longitude": f"{DOMAIN['lon_min']},{DOMAIN['lon_max']}",
        "hourly": "precipitation",
        "models": ICON["model"],
        "start_date": today,
        "end_date": tomorrow,
        "timeformat": "unixtime",
    }
    url = ICON["api_url"] + "?" + urllib.parse.urlencode(params)

    try:
        req = urllib.request.Request(url)
        req.add_header("User-Agent", "nowcasting-public/1.0")
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode())
    except Exception as e:
        logger.error(f"ICON-EU grid грешка: {e}")
        return None

    hourly = data.get("hourly", {})
    times_unix = hourly.get("time", [])
    precip = hourly.get("precipitation", [])

    if not times_unix or not precip:
        logger.error("ICON-EU: няма данни!")
        return None

    # Конвертирай times
    valid_times = [dt.datetime.fromtimestamp(t, tz=dt.timezone.utc)
                   for t in times_unix]

    # Филтрирай само бъдещите часове
    now = ref_time
    future_idx = [i for i, t in enumerate(valid_times)
                  if t >= now and i < len(precip)]
    future_idx = future_idx[:forecast_hours * 12]  # max

    if not future_idx:
        logger.warning("ICON-EU: няма бъдещи данни")
        return None

    precip_arr = np.array([precip[i] if precip[i] is not None else 0.0
                           for i in future_idx], dtype=np.float32)
    valid_arr = [valid_times[i] for i in future_idx]

    logger.info(f"  ICON: {len(future_idx)} часови стъпки, "
                f"max precip = {precip_arr.max():.1f} mm")

    return {
        "precipitation_mm": precip_arr,   # (T,) — единична точка
        "valid_times": valid_arr,
    }


def fetch_icon_grid(ref_time: dt.datetime = None,
                    grid_step_deg: float = 0.1,
                    cache_hours: float = 2.5) -> dict | None:
    """
    Сваля ICON-EU precipitation за ~0.2° grid (~22 km).
    Rate-limit стратегия за Open-Meteo free tier:
      - weight на заявка ≈ брой точки → chunks по ~260 точки
      - 1 chunk / 35 сек (< 600 calls/минута)
      - кеш 2.5 часа (ICON-EU се обновява на 3-6 ч)
    """
    import time as time_mod
    from config.settings import ICON_DIR

    if ref_time is None:
        ref_time = dt.datetime.now(dt.timezone.utc)

    # ── 1. КЕШ ───────────────────────────────────────────
    cache_path = os.path.join(ICON_DIR, "icon_cache.npz")
    if os.path.exists(cache_path):
        try:
            cached = np.load(cache_path, allow_pickle=True)
            cache_age_h = (ref_time.timestamp() - float(cached["fetched_ts"])) / 3600
            if cache_age_h < cache_hours:
                logger.info(f"ICON от кеш (възраст {cache_age_h:.1f} ч)")
                return {
                    "precipitation_mm": cached["precip"],
                    "cape": cached["cape"],
                    "showers_mm": cached["showers"],
                    "lat": cached["lat"],
                    "lon": cached["lon"],
                    "valid_times": [dt.datetime.fromtimestamp(t, tz=dt.timezone.utc)
                                    for t in cached["times_unix"]],
                }
            logger.info(f"ICON кеш е стар ({cache_age_h:.1f} ч) — ново теглене")
        except Exception as e:
            logger.warning(f"ICON кеш грешка: {e}")

    # ── 2. GRID ──────────────────────────────────────────
    today = ref_time.strftime("%Y-%m-%d")

    lats = np.arange(DOMAIN["lat_min"], DOMAIN["lat_max"] + 0.01, grid_step_deg)
    lons = np.arange(DOMAIN["lon_min"], DOMAIN["lon_max"] + 0.01, grid_step_deg)
    nlat, nlon = len(lats), len(lons)

    # Chunks по longitude: ~260 точки на chunk (< 600/мин при 35 сек паузи)
    max_pts_per_chunk = 260
    lon_cols_per_chunk = max(1, max_pts_per_chunk // nlat)
    lon_chunks = [lons[i:i + lon_cols_per_chunk]
                  for i in range(0, nlon, lon_cols_per_chunk)]

    logger.info(f"ICON-EU grid: {nlat}x{nlon} = {nlat*nlon} точки "
                f"(~{grid_step_deg} deg), {len(lon_chunks)} chunks")

    all_times = None
    precip_3d = None
    nt = 0
    lon_offset = 0
    ok_chunks = 0

    for chunk_idx, lon_chunk in enumerate(lon_chunks):
        if chunk_idx > 0:
            time_mod.sleep(35)   # < 600 calls/минута

        lat_pairs, lon_pairs = [], []
        for la in lats:
            for lo in lon_chunk:
                lat_pairs.append(f"{la:.3f}")
                lon_pairs.append(f"{lo:.3f}")

        params = {
            "latitude": ",".join(lat_pairs),
            "longitude": ",".join(lon_pairs),
            "hourly": "precipitation,showers,cape",
            "models": "icon_eu",
            "start_date": today,
            "end_date": today,
            "timeformat": "unixtime",
        }

        url = ICON["api_url"] + "?" + urllib.parse.urlencode(params)
        logger.info(f"  Chunk {chunk_idx+1}/{len(lon_chunks)}: "
                    f"{len(lat_pairs)} точки, URL={len(url)}")

        try:
            req = urllib.request.Request(url)
            req.add_header("User-Agent", "nowcasting-public/1.0")
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read().decode())
        except Exception as e:
            logger.error(f"  Chunk {chunk_idx+1} грешка: {e}")
            lon_offset += len(lon_chunk)
            continue

        results = data if isinstance(data, list) else [data]

        if all_times is None:
            times_unix = results[0].get("hourly", {}).get("time", [])
            if not times_unix:
                lon_offset += len(lon_chunk)
                continue
            all_times = [dt.datetime.fromtimestamp(t, tz=dt.timezone.utc)
                         for t in times_unix]
            nt = len(all_times)
            precip_3d = np.zeros((nt, nlat, nlon), dtype=np.float32)
            cape_3d = np.zeros((nt, nlat, nlon), dtype=np.float32)
            showers_3d = np.zeros((nt, nlat, nlon), dtype=np.float32)

        nlon_chunk = len(lon_chunk)
        for i, result in enumerate(results):
            precip = result.get("hourly", {}).get("precipitation", [])
            lat_idx = i // nlon_chunk
            lon_idx = lon_offset + (i % nlon_chunk)
            if lat_idx >= nlat or lon_idx >= nlon:
                continue
            n = min(len(precip), nt)
            for t in range(n):
                val = precip[t]
                precip_3d[t, lat_idx, lon_idx] = val if val is not None else 0.0
            cape = result.get("hourly", {}).get("cape", [])
            n_c = min(len(cape), nt)
            for t in range(n_c):
                val = cape[t]
                cape_3d[t, lat_idx, lon_idx] = val if val is not None else 0.0
            showers = result.get("hourly", {}).get("showers", [])
            n_s = min(len(showers), nt)
            for t in range(n_s):
                val = showers[t]
                showers_3d[t, lat_idx, lon_idx] = val if val is not None else 0.0

        ok_chunks += 1
        lon_offset += len(lon_chunk)

    if precip_3d is None or ok_chunks == 0:
        logger.error("ICON: нито един chunk не успя!")
        return None

    logger.info(f"  ICON grid: {precip_3d.shape}, {ok_chunks}/{len(lon_chunks)} "
                f"chunks OK, max = {precip_3d.max():.1f} mm")

    # ── 3. ЗАПАЗИ КЕШ ────────────────────────────────────
    if ok_chunks == len(lon_chunks):
        try:
            np.savez_compressed(
                cache_path,
                precip=precip_3d,
                cape=cape_3d,
                showers=showers_3d,
                lat=lats, lon=lons,
                times_unix=np.array([t.timestamp() for t in all_times]),
                fetched_ts=ref_time.timestamp(),
            )
            logger.info(f"  ICON кеш запазен: {cache_path}")
        except Exception as e:
            logger.warning(f"  Кеш запис грешка: {e}")
    else:
        logger.warning(f"  Непълни данни ({ok_chunks}/{len(lon_chunks)}) — кешът НЕ е запазен")

    return {
        "precipitation_mm": precip_3d,
        "cape": cape_3d,
        "showers_mm": showers_3d,
        "lat": lats,
        "lon": lons,
        "valid_times": all_times,
    }

def precip_to_dbz(precip_mmh: np.ndarray,
                  a: float = None, b: float = None) -> np.ndarray:
    """
    Конвертира precipitation rate (mm/h) → dBZ (Marshall-Palmer Z-R).
    Z = a * R^b  →  dBZ = 10 * log10(Z)
    """
    if a is None:
        a = ICON["zr_a"]
    if b is None:
        b = ICON["zr_b"]

    R = np.clip(precip_mmh, 0.01, None)
    Z = a * R ** b
    dbz = 10.0 * np.log10(Z)
    dbz[precip_mmh < 0.1] = np.nan
    return dbz.astype(np.float32)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(name)s] %(message)s")
    result = fetch_icon_precipitation()
    if result:
        print(f"ICON: {len(result['valid_times'])} стъпки")
        print(f"Max precip: {result['precipitation_mm'].max():.1f} mm")
