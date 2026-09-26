"""
Ingest: ICON-EU директно от DWD Open Data (GRIB2)
=================================================
Полета от ЕДИН и същ run:
  rain_con, snow_con — конвективен валеж, акумулиран от началото на run-а (mm)
  lpi_con_max        — Lightning Potential Index, максимум за изминалия час (J/kg)
Runs на всеки 3 ч (00..21 UTC), почасови стъпки до +48 ч,
публикуване ~2 ч 40 мин след началото на run-а.

Изход (за blending.py):
  showers_mm  (T, ny, nx) — конвективен валеж за часа (h-1, h]
  lpi         (T, ny, nx) — LPI максимум за часа (h-1, h]
  lat, lon    — възходящи 1D оси, изрязани до DOMAIN
  valid_times — краят на всеки часов прозорец
"""

import os, sys, re, bz2, math, shutil, logging
import datetime as dt
import numpy as np
import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from config.settings import DOMAIN, ICON_DIR, ICON_DWD

logger = logging.getLogger("ingest.icon_dwd")

FIELDS = ("rain_con", "snow_con", "lpi_con_max")
FORECAST_HOURS = 6
MARGIN_DEG = 0.3          # поле около домейна при изрязване
_session = requests.Session()


def _fname(run: dt.datetime, step: int, field: str) -> str:
    return (f"icon-eu_europe_regular-lat-lon_single-level_"
            f"{run:%Y%m%d%H}_{step:03d}_{field.upper()}.grib2.bz2")


def _step_range(run: dt.datetime, now: dt.datetime) -> tuple[int, int]:
    """Стъпки, чиито часови прозорци покриват [now-1ч, now+6ч]."""
    h_start = (now - dt.timedelta(hours=1) - run).total_seconds() / 3600
    h_end = (now + dt.timedelta(hours=FORECAST_HOURS) - run).total_seconds() / 3600
    return max(0, math.ceil(h_start)), math.ceil(h_end) + 1


def _list_files(run_hh: int, field: str) -> tuple[str, set]:
    url = f"{ICON_DWD['base_url']}/{run_hh:02d}/{field}/"
    r = _session.get(url, timeout=30)
    r.raise_for_status()
    files = set(re.findall(
        r'icon-eu_europe_regular-lat-lon_single-level_\d{10}_\d{3}_[A-Z_]+\.grib2\.bz2',
        r.text))
    return url, files


def _find_run(now: dt.datetime):
    """Най-новият run, за който всички нужни стъпки и полета са публикувани."""
    base = now.replace(minute=0, second=0, microsecond=0)
    base -= dt.timedelta(hours=base.hour % 3)
    for k in range(5):
        run = base - dt.timedelta(hours=3 * k)
        s0, s1 = _step_range(run, now)
        if s1 > 48:
            continue
        urls, complete = {}, True
        for field in FIELDS:
            try:
                url, files = _list_files(run.hour, field)
            except Exception as e:
                logger.warning(f"  {field}: листинг грешка {e}")
                complete = False
                break
            need = [_fname(run, s, field) for s in range(max(0, s0 - 1), s1 + 1)]
            if not all(n in files for n in need):
                complete = False
                break
            urls[field] = url
        if complete:
            return run, s0, s1, urls
        logger.info(f"  Run {run:%Y-%m-%d %H} UTC не е пълен — пробвам предишния")
    return None


def _read_grib(path: str):
    import xarray as xr
    ds = xr.open_dataset(path, engine="cfgrib", backend_kwargs={"indexpath": ""})
    da = ds[list(ds.data_vars)[0]]
    lat = da["latitude"].values.astype(np.float64)
    lon = da["longitude"].values.astype(np.float64)
    data = da.values.astype(np.float32)
    ds.close()
    if lon.max() > 180:
        lon = np.where(lon > 180, lon - 360, lon)
    return lat, lon, data


def _load(run, step, field, url, cache_dir):
    fname = _fname(run, step, field)
    path = os.path.join(cache_dir, fname[:-4])        # без .bz2
    if not os.path.exists(path):
        r = _session.get(url + fname, timeout=60)
        r.raise_for_status()
        with open(path, "wb") as f:
            f.write(bz2.decompress(r.content))
    return _read_grib(path)


def _cleanup(keep_hours: float):
    root = os.path.join(ICON_DIR, "dwd")
    if not os.path.isdir(root):
        return
    cutoff = dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=keep_hours)
    for d in os.listdir(root):
        try:
            t = dt.datetime.strptime(d, "%Y%m%d%H").replace(tzinfo=dt.timezone.utc)
            if t < cutoff:
                shutil.rmtree(os.path.join(root, d), ignore_errors=True)
        except ValueError:
            pass


def fetch_icon_dwd(now: dt.datetime = None) -> dict | None:
    if now is None:
        now = dt.datetime.now(dt.timezone.utc)

    found = _find_run(now)
    if found is None:
        logger.error("ICON DWD: няма пълен run в последните 12 ч")
        return None
    run, s0, s1, urls = found
    logger.info(f"ICON DWD: run {run:%Y-%m-%d %H} UTC, стъпки +{s0}..+{s1} ч")

    cache_dir = os.path.join(ICON_DIR, "dwd", f"{run:%Y%m%d%H}")
    os.makedirs(cache_dir, exist_ok=True)

    lat_c = lon_c = None
    iy = ix = None
    flip = False

    def crop(lat, lon, data):
        nonlocal lat_c, lon_c, iy, ix, flip
        if iy is None:
            iy = np.where((lat >= DOMAIN["lat_min"] - MARGIN_DEG) &
                          (lat <= DOMAIN["lat_max"] + MARGIN_DEG))[0]
            ix = np.where((lon >= DOMAIN["lon_min"] - MARGIN_DEG) &
                          (lon <= DOMAIN["lon_max"] + MARGIN_DEG))[0]
            lat_c, lon_c = lat[iy], lon[ix]
            flip = lat_c[0] > lat_c[-1]
            if flip:
                lat_c = lat_c[::-1]
        out = data[np.ix_(iy, ix)]
        return out[::-1, :] if flip else out

    # Акумулиран конвективен валеж → часови суми
    acc = {}
    for s in range(max(0, s0 - 1), s1 + 1):
        r = crop(*_load(run, s, "rain_con", urls["rain_con"], cache_dir))
        sn = crop(*_load(run, s, "snow_con", urls["snow_con"], cache_dir))
        acc[s] = np.nan_to_num(r) + np.nan_to_num(sn)

    showers, lpi, times = [], [], []
    for s in range(s0, s1 + 1):
        prev = acc.get(s - 1, np.zeros_like(acc[s]))
        showers.append(np.clip(acc[s] - prev, 0.0, None))
        lpi.append(np.nan_to_num(crop(*_load(run, s, "lpi_con_max",
                                             urls["lpi_con_max"], cache_dir))))
        times.append(run + dt.timedelta(hours=s))

    showers = np.stack(showers).astype(np.float32)
    lpi = np.stack(lpi).astype(np.float32)

    # Статистика за калибровка на праговете (само във валежните зони)
    rain = showers >= ICON_DWD["core_min_showers"]
    if rain.any():
        l = lpi[rain]
        logger.info(f"  Showers max {showers.max():.1f} mm/h | LPI във валеж: "
                    f"p90={np.percentile(l, 90):.1f} p99={np.percentile(l, 99):.1f} "
                    f"max={l.max():.1f} | px с LPI≥20: {int((l >= 20).sum())}")
    else:
        logger.info(f"  Showers max {showers.max():.1f} mm/h — няма зони ≥ "
                    f"{ICON_DWD['core_min_showers']} mm/h")

    _cleanup(ICON_DWD["keep_runs_hours"])

    return {
        "showers_mm": showers,
        "lpi": lpi,
        "lat": lat_c,
        "lon": lon_c,
        "valid_times": times,
        "run": run,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(name)s] %(message)s")
    d = fetch_icon_dwd()
    if d:
        print(f"Run: {d['run']}")
        print(f"showers: {d['showers_mm'].shape}  lpi: {d['lpi'].shape}")
        print(f"lat {d['lat'][0]:.3f}..{d['lat'][-1]:.3f}  "
              f"lon {d['lon'][0]:.3f}..{d['lon'][-1]:.3f}")
        print("valid:", d["valid_times"][0], "→", d["valid_times"][-1])