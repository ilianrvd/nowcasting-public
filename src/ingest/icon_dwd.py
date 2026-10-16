"""
Ingest: ICON-EU директно от DWD Open Data (GRIB2)
=================================================
Полета от ЕДИН и същ run:
  tot_prec           — общ валеж (слоест + конвективен), акумулиран (mm) → фон
  rain_con, snow_con — конвективен валеж, акумулиран от началото на run-а (mm)
  lpi_con_max        — Lightning Potential Index, максимум за изминалия час (J/kg)
  u, v на 700 hPa    — вятър за проверка на движението от радара (една стъпка)
Runs на всеки 3 ч (00..21 UTC), почасови стъпки до +48 ч,
публикуване ~2 ч 40 мин след началото на run-а.

Изход (за blending.py):
  total_mm    (T, ny, nx) — общ валеж за часа (h-1, h]
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

FIELDS = ("tot_prec", "rain_con", "snow_con", "lpi_con_max")
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



def _fetch_wind700(run: dt.datetime, t_valid: dt.datetime, cache_dir: str, crop):
    """U/V на 700 hPa за стъпката, най-близка до t_valid. None при неуспех."""
    step = int(round((t_valid - run).total_seconds() / 3600))
    out = {}
    for comp in ("u", "v"):
        fname = (f"icon-eu_europe_regular-lat-lon_pressure-level_"
                 f"{run:%Y%m%d%H}_{step:03d}_700_{comp.upper()}.grib2.bz2")
        path = os.path.join(cache_dir, fname[:-4])
        if not os.path.exists(path):
            url = f"{ICON_DWD['base_url']}/{run.hour:02d}/{comp}/{fname}"
            r = _session.get(url, timeout=60)
            r.raise_for_status()
            with open(path, "wb") as f:
                f.write(bz2.decompress(r.content))
        lat, lon, data = _read_grib(path)
        out[comp] = np.nan_to_num(crop(lat, lon, data))
    return {"u": out["u"], "v": out["v"], "time": run + dt.timedelta(hours=step)}


def _get_grib(url_dir: str, fname: str, cache_dir: str, crop):
    path = os.path.join(cache_dir, fname[:-4])
    if not os.path.exists(path):
        r = _session.get(url_dir + fname, timeout=60)
        r.raise_for_status()
        with open(path, "wb") as f:
            f.write(bz2.decompress(r.content))
    lat, lon, data = _read_grib(path)
    return np.nan_to_num(crop(lat, lon, data))


def _fetch_site_winds(run: dt.datetime, s0: int, s1: int, cache_dir: str, crop):
    """
    Вятър за таблицата със стартовете, по часове s0..s1 (валидни моменти):
      u10/v10   — 10 m (m/s), моментни в часа
      gust      — VMAX_10M: максимален порив за изминалия час (m/s)
      u850/v850 — 850 hPa (m/s)
    Липсващо поле → None за него (страницата го показва като „—“).
    """
    base = ICON_DWD["base_url"]
    sl = lambda f, s: f"icon-eu_europe_regular-lat-lon_single-level_{run:%Y%m%d%H}_{s:03d}_{f.upper()}.grib2.bz2"
    pl = lambda c, s: f"icon-eu_europe_regular-lat-lon_pressure-level_{run:%Y%m%d%H}_{s:03d}_850_{c.upper()}.grib2.bz2"
    jobs = {"u10": lambda s: (f"{base}/{run.hour:02d}/u_10m/", sl("u_10m", s)),
            "v10": lambda s: (f"{base}/{run.hour:02d}/v_10m/", sl("v_10m", s)),
            "gust": lambda s: (f"{base}/{run.hour:02d}/vmax_10m/", sl("vmax_10m", s)),
            "u850": lambda s: (f"{base}/{run.hour:02d}/u/", pl("u", s)),
            "v850": lambda s: (f"{base}/{run.hour:02d}/v/", pl("v", s))}
    steps = list(range(max(1, s0), s1 + 1))           # VMAX за стъпка 0 няма смисъл
    out = {"times": [run + dt.timedelta(hours=s) for s in steps]}
    for key, job in jobs.items():
        try:
            out[key] = np.stack([_get_grib(*job(s), cache_dir, crop) for s in steps]).astype(np.float32)
        except Exception as e:
            logger.warning(f"  Вятър {key}: недостъпен ({str(e)[:70]})")
            out[key] = None
    return out


def fetch_icon_dwd(now: dt.datetime = None, t_wind: dt.datetime = None) -> dict | None:
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

    # Акумулиран валеж (общ и конвективен) → часови суми
    acc, acc_tot = {}, {}
    for s in range(max(0, s0 - 1), s1 + 1):
        r = crop(*_load(run, s, "rain_con", urls["rain_con"], cache_dir))
        sn = crop(*_load(run, s, "snow_con", urls["snow_con"], cache_dir))
        acc[s] = np.nan_to_num(r) + np.nan_to_num(sn)
        acc_tot[s] = np.nan_to_num(crop(*_load(run, s, "tot_prec", urls["tot_prec"], cache_dir)))

    showers, total, lpi, times = [], [], [], []
    for s in range(s0, s1 + 1):
        prev = acc.get(s - 1, np.zeros_like(acc[s]))
        showers.append(np.clip(acc[s] - prev, 0.0, None))
        prev_t = acc_tot.get(s - 1, np.zeros_like(acc_tot[s]))
        total.append(np.clip(acc_tot[s] - prev_t, 0.0, None))
        lpi.append(np.nan_to_num(crop(*_load(run, s, "lpi_con_max",
                                             urls["lpi_con_max"], cache_dir))))
        times.append(run + dt.timedelta(hours=s))

    showers = np.stack(showers).astype(np.float32)
    total = np.maximum(np.stack(total).astype(np.float32), showers)   # общият ≥ конвективния
    lpi = np.stack(lpi).astype(np.float32)

    logger.info(f"  Общ валеж max {total.max():.1f} mm/h, ≥0.1 mm/h в {(total >= 0.1).mean():.0%} "
                f"от полето | конвективен ≥0.1 mm/h в {(showers >= 0.1).mean():.0%}")

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

    # Вятър на 700 hPa за проверка на движението (не е задължителен)
    wind700 = None
    try:
        wind700 = _fetch_wind700(run, t_wind or now, cache_dir, crop)
        wind700["lat"], wind700["lon"] = lat_c, lon_c
        spd = np.hypot(wind700["u"], wind700["v"]) * 1.944
        logger.info(f"  Вятър 700 hPa ({wind700['time']:%H} UTC): "
                    f"средно {spd.mean():.0f} KT, max {spd.max():.0f} KT")
    except Exception as e:
        logger.warning(f"  Вятър 700 hPa недостъпен: {str(e)[:80]} — без проверка на движението")
        wind700 = None

    # Вятър за стартовете (10 m, пориви, 850 hPa) — по часове, не е задължителен
    winds = None
    try:
        winds = _fetch_site_winds(run, s0, s1, cache_dir, crop)
        if winds.get("u10") is not None:
            sp = np.hypot(winds["u10"], winds["v10"]) * 3.6
            g = winds["gust"] * 3.6 if winds.get("gust") is not None else None
            logger.info(f"  Вятър 10 m: {len(winds['times'])} часа, средно {sp.mean():.0f} km/h, "
                        f"max {sp.max():.0f} km/h" + (f", пориви до {g.max():.0f} km/h" if g is not None else ""))
    except Exception as e:
        logger.warning(f"  Вятър за стартовете недостъпен: {str(e)[:80]}")
        winds = None

    _cleanup(ICON_DWD["keep_runs_hours"])

    return {
        "total_mm": total,
        "showers_mm": showers,
        "lpi": lpi,
        "lat": lat_c,
        "lon": lon_c,
        "valid_times": times,
        "run": run,
        "wind700": wind700,
        "winds": winds,
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