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
    logger.info(f"ICON-EU заявка: {today} +{forecast_hours}h")

    try:
        req = urllib.request.Request(url)
        req.add_header("User-Agent", "nowcasting-public/1.0")
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode())
    except Exception as e:
        logger.error(f"ICON-EU грешка: {e}")
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


def fetch_icon_grid(ref_time: dt.datetime = None) -> dict | None:
    """
    Сваля ICON-EU precipitation за grid от точки над България.
    Open-Meteo приема двойки lat,lon (не grid), до ~50 точки.
    """
    if ref_time is None:
        ref_time = dt.datetime.now(dt.timezone.utc)

    today = ref_time.strftime("%Y-%m-%d")
    tomorrow = (ref_time + dt.timedelta(days=1)).strftime("%Y-%m-%d")

    # Sparse grid ~1° стъпка → 4×9 = 36 точки
    lats = np.arange(DOMAIN["lat_min"], DOMAIN["lat_max"] + 0.1, 1.0)
    lons = np.arange(DOMAIN["lon_min"], DOMAIN["lon_max"] + 0.1, 1.0)

    # Генерирай двойки (всяка grid точка е отделна "локация")
    lat_pairs = []
    lon_pairs = []
    for la in lats:
        for lo in lons:
            lat_pairs.append(f"{la:.2f}")
            lon_pairs.append(f"{lo:.2f}")

    logger.info(f"ICON-EU grid: {len(lats)}×{len(lons)} = {len(lat_pairs)} точки")

    params = {
        "latitude": ",".join(lat_pairs),
        "longitude": ",".join(lon_pairs),
        "hourly": "precipitation",
        "models": "icon_eu",
        "start_date": today,
        "end_date": tomorrow,
        "timeformat": "unixtime",
    }

    url = ICON["api_url"] + "?" + urllib.parse.urlencode(params)

    try:
        req = urllib.request.Request(url)
        req.add_header("User-Agent", "nowcasting-public/1.0")
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode())
    except Exception as e:
        logger.error(f"ICON-EU grid грешка: {e}")
        return None

    # Open-Meteo връща масив от резултати за multi-point
    if isinstance(data, list):
        results = data
    elif isinstance(data, dict) and "hourly" in data:
        results = [data]
    else:
        logger.error("Неочакван формат от ICON API")
        return None

    # Събери timestamps от първия резултат
    first_hourly = results[0].get("hourly", {})
    times_unix = first_hourly.get("time", [])
    if not times_unix:
        logger.error("ICON: няма timestamps")
        return None

    all_times = [dt.datetime.fromtimestamp(t, tz=dt.timezone.utc)
                 for t in times_unix]
    nt = len(all_times)
    nlat, nlon = len(lats), len(lons)

    # Сглоби 3D масив (T, nlat, nlon)
    precip_3d = np.zeros((nt, nlat, nlon), dtype=np.float32)

    for i, result in enumerate(results):
        hourly = result.get("hourly", {})
        precip = hourly.get("precipitation", [])
        lat_idx = i // nlon
        lon_idx = i % nlon
        n = min(len(precip), nt)
        for t in range(n):
            val = precip[t]
            precip_3d[t, lat_idx, lon_idx] = val if val is not None else 0.0

    logger.info(f"  ICON grid: {precip_3d.shape}, "
                f"max = {precip_3d.max():.1f} mm")

    return {
        "precipitation_mm": precip_3d,
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
