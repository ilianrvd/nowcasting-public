"""
Сравнение ALADIN CE (ЧХМИ, 2.3 km) ↔ ICON-EU (DWD, 6.5 km) като фон за бленда
=============================================================================
Върви успоредно на основния пуск (.github/workflows/model_compare.yml), веднъж на час,
и не пипа картата. Въпрос: кой модел е по-добър за +60…+150 мин, където блендът
преминава от радара към модела.

Проверка за часа V (прозорец (V-1ч, V]):
  наблюдение — средна интензивност (mm/h) от всички румънски композити в прозореца
               (на 10 мин), Marshall–Palmer, без единични точки (speck);
  модели     — часов валеж за същия прозорец, от run-а, който основният пуск би имал
               в момента на издаване I = V − 30 мин − lead (lead = 60, 90, 120, 150):
               ICON-EU: run на 3 ч, достъпен ~2 ч 40 мин след началото
               ALADIN:  run на 6 ч, достъпен ~3 ч 30 мин след началото (истинското
                        закъснение се записва от Last-Modified на файла)
  кандидати  — icon, aladin, aladin_s (ALADIN, загладен с Гаус σ = 7 km — колко помага
               загладането срещу „двойното наказание“ при линейния бленд).
Мрежа: 43.0–48.0°N, 21.0–29.5°E през ~2.2 km (северна България + Румъния — зоната на
румънския композит), без морето и без точките извън обхвата на радарите.

Метрика: FSS (Roberts & Lean 2008) при прагове 0.5/2/5 mm/h и мащаби 10/30/60 km.
В CSV се пазят числителят и знаменателят, за да се сумират правилно за период
(tools/model_compare_report.py), плюс дяловете с валеж (честотно отклонение).

Употреба:
  python tools/model_compare.py                      # последният завършен час
  python tools/model_compare.py --time 2026-11-05T14 # конкретен час (ако данните още са на сървърите)
"""

import os, sys, re, bz2, csv, shutil, logging, argparse
import datetime as dt
import numpy as np
import requests

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from config.settings import ROMANIA, ICON_DWD          # noqa: E402

logger = logging.getLogger("model_compare")
UTC = dt.timezone.utc

CFG = {
    "aladin_url": ("https://opendata.chmi.cz/meteorology/weather/nwp_aladin/Lambert_2.3km/"
                   "{run:%H}/ALADLAMB4opendata_{run:%Y%m%d%H}_SURFPREC_TOTAL.grb.bz2"),
    "aladin_every_h": 6, "aladin_delay": dt.timedelta(hours=3, minutes=30),
    "icon_every_h": 3,   "icon_delay":   dt.timedelta(hours=2, minutes=40),
    "leads_min": (60, 90, 120, 150),
    "thresholds": (0.5, 2.0, 5.0),               # mm/h
    "scales_km": (10, 30, 60),
    "lat": (43.0, 48.0, 0.02), "lon": (21.0, 29.5, 0.027),
    "smooth_km": 7.0,
    "speck_px": 4,                               # по-малки петна (в px на композита) → махат се
    "min_obs_files": 3,                          # от 6 композита в часа
    "mp": (200.0, 1.6),                          # Z = a R^b
    "cache_dir": os.path.join(ROOT, "data", "compare"),
    "keep_h": 36,
}
# Брегът на Черно море (lat → lon); по-източното е море и не се проверява
_COAST = ((43.0, 27.95), (43.5, 28.62), (44.0, 28.68), (44.5, 28.78), (44.9, 29.0),
          (45.0, 29.65), (45.5, 29.75), (48.0, 29.75))

_session = requests.Session()


# ----------------------------------------------------------------------------- мрежа
def grid():
    la0, la1, dla = CFG["lat"]
    lo0, lo1, dlo = CFG["lon"]
    lat = np.arange(la0, la1 + dla / 2, dla)
    lon = np.arange(lo0, lo1 + dlo / 2, dlo)
    return lat, lon


def dx_km():
    lat, _ = grid()
    dlo = CFG["lon"][2] * 111.2 * np.cos(np.radians(lat.mean()))
    return 0.5 * (CFG["lat"][2] * 111.2 + dlo)


def sea_mask(lat, lon):
    lon2, lat2 = np.meshgrid(lon, lat)
    cl, co = zip(*_COAST)
    return lon2 > np.interp(lat2, cl, co)


# ----------------------------------------------------------------------------- време
def run_for(issue: dt.datetime, every_h: int, delay: dt.timedelta) -> dt.datetime:
    """Най-новият run (кратен на every_h часа), достъпен в момента issue."""
    t = issue - delay
    t = t.replace(minute=0, second=0, microsecond=0)
    return t - dt.timedelta(hours=t.hour % every_h)


def last_hour(now: dt.datetime) -> dt.datetime:
    """Последният завършен час, за който композитите вече са публикувани (~15 мин)."""
    return (now - dt.timedelta(minutes=15)).replace(minute=0, second=0, microsecond=0)


# ----------------------------------------------------------------------------- HTTP
def _download(url: str, path: str, timeout=120) -> dict:
    """Сваля url → path (поточно). Връща заглавките (Last-Modified, размер)."""
    tmp = path + ".part"
    with _session.get(url, timeout=timeout, stream=True) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
        hdr = {"last_modified": r.headers.get("Last-Modified", ""),
               "bytes": os.path.getsize(tmp)}
    os.replace(tmp, path)
    return hdr


def _get_text(url: str, timeout=30) -> str:
    r = _session.get(url, timeout=timeout)
    r.raise_for_status()
    return r.text


# ----------------------------------------------------------------------------- наблюдение
def dbz_to_rate(dbz):
    a, b = CFG["mp"]
    return (10.0 ** (dbz / 10.0) / a) ** (1.0 / b)


def read_ro(path: str):
    """
    Румънски композит (ODIM HDF5, AEQD) → (rate mm/h, lat1d↑, lon1d↑, време).
    За разлика от основния четец: undetect = 0 mm/h (няма ехо), nodata = NaN (извън
    обхвата) — за проверката двете са различни неща.
    """
    import h5py
    from pyproj import Transformer
    with h5py.File(path, "r") as f:
        what, where, ds = f["what"].attrs, f["where"].attrs, f["dataset1"]
        s = lambda v: v.decode() if isinstance(v, bytes) else str(v)
        ts = dt.datetime.strptime(s(what["date"]) + s(what["time"]), "%Y%m%d%H%M%S").replace(tzinfo=UTC)
        projdef = s(where["projdef"])
        nx, ny = int(where["xsize"]), int(where["ysize"])
        ll = (float(where["LL_lon"]), float(where["LL_lat"]))
        ur = (float(where["UR_lon"]), float(where["UR_lat"]))
        dw = ds["what"].attrs
        raw = ds["data1/data"][:].astype(np.float64)
        dbz = raw * float(dw["gain"]) + float(dw["offset"])
        rate = dbz_to_rate(dbz)
        rate[raw == float(dw["undetect"])] = 0.0
        rate[(raw == float(dw["nodata"])) | (dbz > 75.0)] = np.nan
    to_xy = Transformer.from_crs("EPSG:4326", projdef, always_xy=True)
    to_ll = Transformer.from_crs(projdef, "EPSG:4326", always_xy=True)
    (x0, y0), (x1, y1) = to_xy.transform(*ll), to_xy.transform(*ur)
    x = np.linspace(x0, x1, nx)
    y = np.linspace(y0, y1, ny)
    _, lat = to_ll.transform(np.full_like(y, (x0 + x1) / 2), y)
    lon, _ = to_ll.transform(x, np.full_like(x, (y0 + y1) / 2))
    rate = rate[::-1, :]                          # ODIM: ред 0 = север → обръщаме към юг-север
    return rate.astype(np.float32), lat, lon, ts


def despeck(rate, min_px: int, thr=0.5):
    """Маха изолирани петна с валеж < min_px точки (шум, единични отражения)."""
    from scipy import ndimage
    wet = np.nan_to_num(rate) >= thr
    lab, n = ndimage.label(wet)
    if n == 0:
        return rate
    sizes = ndimage.sum(wet, lab, index=np.arange(1, n + 1))
    small = np.isin(lab, np.where(sizes < min_px)[0] + 1)
    out = rate.copy()
    out[small] = 0.0
    return out


def to_grid_regular(field, src_lat, src_lon, lat, lon, fill=np.nan):
    """Линейна интерполация от правилна lat/lon мрежа (възходящи оси)."""
    from scipy.interpolate import RegularGridInterpolator
    if src_lat[0] > src_lat[-1]:
        src_lat, field = src_lat[::-1], field[::-1]
    f = RegularGridInterpolator((src_lat, src_lon), field, method="linear",
                                bounds_error=False, fill_value=fill)
    lon2, lat2 = np.meshgrid(lon, lat)
    return f(np.column_stack([lat2.ravel(), lon2.ravel()])).reshape(lat2.shape).astype(np.float32)


def ro_listing() -> list[tuple[dt.datetime, str]]:
    txt = _get_text(ROMANIA["base_url"])
    out = []
    for digits in sorted(set(re.findall(r"COMPOSITE_(\d+)dBZ\.hdf", txt))):
        if len(digits) < 12:                     # имената са YYYYMMDDHHMM + „0000“ (16 цифри)
            continue
        out.append((dt.datetime.strptime(digits[:12], "%Y%m%d%H%M").replace(tzinfo=UTC),
                    f"COMPOSITE_{digits}dBZ.hdf"))
    if out:
        logger.info(f"  RO листинг: {len(out)} композита, {out[0][0]:%d.%m %H:%M} … {out[-1][0]:%d.%m %H:%M} UTC")
    else:
        sample = re.findall(r'COMPOSITE_[^"<>]*', txt)[:3]
        logger.warning(f"  RO листинг: няма разпознати имена; пример: {sample}")
    return out


def observed_hour(V: dt.datetime, lat, lon, listing=None):
    """Средна интензивност за (V-1ч, V] върху мрежата за проверка. (rate, брой файлове)."""
    listing = ro_listing() if listing is None else listing
    pick = [(t, n) for t, n in listing if V - dt.timedelta(hours=1) < t <= V]
    d = os.path.join(CFG["cache_dir"], "ro")
    os.makedirs(d, exist_ok=True)
    fields = []
    for t, name in pick:
        p = os.path.join(d, name)
        try:
            if not os.path.exists(p):
                _download(ROMANIA["base_url"].rstrip("/") + "/" + name, p, timeout=60)
            rate, la, lo, _ = read_ro(p)
            rate = despeck(rate, CFG["speck_px"])
            fields.append(to_grid_regular(rate, la, lo, lat, lon))
        except Exception as e:
            logger.warning(f"  RO {name}: {type(e).__name__} {str(e)[:80]}")
    if len(fields) < CFG["min_obs_files"]:
        return None, len(fields)
    stack = np.stack(fields)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)      # точки без данни през целия час
        obs = np.nanmean(stack, axis=0)
    obs[np.isnan(stack).any(axis=0)] = np.nan      # само точки, покрити през целия час
    return obs.astype(np.float32), len(fields)


# ----------------------------------------------------------------------------- ALADIN
def _lcc_index(gid, lat, lon):
    """Индекси (ред, колона) в масива на съобщението за точките на мрежата за проверка."""
    import eccodes as ec
    from pyproj import Proj
    g = lambda k: ec.codes_get(gid, k)
    nx, ny = g("Nx"), g("Ny")
    lat1, lon1 = g("latitudeOfFirstGridPointInDegrees"), g("longitudeOfFirstGridPointInDegrees")
    lov, l1, l2 = g("LoVInDegrees"), g("Latin1InDegrees"), g("Latin2InDegrees")
    dxm, dym = g("DxInMetres"), g("DyInMetres")
    jpos, ineg = g("jScansPositively"), g("iScansNegatively")
    # ALADIN смята върху сфера с R = 6371229 m (независимо от формата на Земята в GRIB1)
    p = Proj(proj="lcc", lat_1=l1, lat_2=l2, lat_0=l1, lon_0=lov, R=6371229)
    x0, y0 = p(lon1, lat1)
    lon2, lat2 = np.meshgrid(lon, lat)
    x, y = p(lon2, lat2)
    col = (x0 - x) / dxm if ineg else (x - x0) / dxm
    row = (y - y0) / dym if jpos else (y0 - y) / dym
    # Проверка спрямо координатите, които смята eccodes (последната точка)
    try:
        la_all = ec.codes_get_double_array(gid, "latitudes")
        lo_all = ec.codes_get_double_array(gid, "longitudes")
        xl, yl = p(lo_all[-1], la_all[-1])
        err = np.hypot(abs((xl - x0) / dxm) - (nx - 1), abs((yl - y0) / dym) - (ny - 1))
        # GRIB1 не пази радиуса: eccodes смята с 6367470 m → ~1 px разлика в далечния ъгъл
        (logger.warning if err > 3 else logger.info)(
            f"  ALADIN: проекция спрямо eccodes в далечния ъгъл: {err:.2f} px")
    except Exception:
        pass
    return row, col, nx, ny


def aladin_hourly(run: dt.datetime, lat, lon, url_fmt=None) -> dict | None:
    """
    Часов валеж (mm/h) от ALADIN за всички стъпки на run-а, върху мрежата за проверка.
    {"steps": {s: поле за (s-1, s]}, "last_modified": ..., "bytes": ...}.
    Кешира се на run (npz) — целият файл (~всички 73 ч на цяла Европа) се тегли веднъж.
    """
    import eccodes as ec
    from scipy.ndimage import map_coordinates
    d = os.path.join(CFG["cache_dir"], "aladin")
    os.makedirs(d, exist_ok=True)
    npz = os.path.join(d, f"aladin_{run:%Y%m%d%H}.npz")
    if os.path.exists(npz):
        z = np.load(npz, allow_pickle=False)
        steps = {int(s): z[f"s{int(s)}"] for s in z["steps"]}
        return {"steps": steps, "last_modified": str(z["last_modified"]), "bytes": int(z["bytes"])}

    url = (url_fmt or CFG["aladin_url"]).format(run=run)
    bz = os.path.join(d, f"aladin_{run:%Y%m%d%H}.grb.bz2")
    grb = bz[:-4]
    try:
        hdr = _download(url, bz, timeout=300)
    except Exception as e:
        logger.warning(f"  ALADIN {run:%Y-%m-%d %H} UTC: {type(e).__name__} {str(e)[:100]}")
        return None
    with bz2.open(bz, "rb") as fi, open(grb, "wb") as fo:
        shutil.copyfileobj(fi, fo, 1 << 22)
    os.remove(bz)

    acc, idx = {}, None
    with open(grb, "rb") as f:
        while True:
            gid = ec.codes_grib_new_from_file(f)
            if gid is None:
                break
            try:
                vd, vt = ec.codes_get(gid, "validityDate"), ec.codes_get(gid, "validityTime")
                t = dt.datetime.strptime(f"{vd}{vt:04d}", "%Y%m%d%H%M").replace(tzinfo=UTC)
                s = int(round((t - run).total_seconds() / 3600))
                if idx is None:
                    idx = _lcc_index(gid, lat, lon)
                row, col, nx, ny = idx
                vals = ec.codes_get_values(gid).reshape(ny, nx)
                acc[s] = map_coordinates(vals, [row, col], order=1, mode="constant",
                                         cval=np.nan).astype(np.float32)
            finally:
                ec.codes_release(gid)
    os.remove(grb)
    if not acc:
        logger.warning(f"  ALADIN {run:%Y-%m-%d %H} UTC: празен файл")
        return None
    steps = {}
    for s in sorted(acc):
        if s == 0:
            continue
        prev = acc.get(s - 1)
        if prev is None:
            continue
        steps[s] = np.clip(acc[s] - prev, 0.0, None)
    np.savez_compressed(npz, steps=np.array(sorted(steps)),
                        last_modified=hdr["last_modified"], bytes=hdr["bytes"],
                        **{f"s{s}": v for s, v in steps.items()})
    logger.info(f"  ALADIN {run:%Y-%m-%d %H} UTC: {hdr['bytes'] / 1e6:.1f} MB, стъпки "
                f"{min(steps)}..{max(steps)}, публикуван {hdr['last_modified'] or '?'}")
    return {"steps": steps, **hdr}


# ----------------------------------------------------------------------------- ICON
def _icon_name(run, step):
    return (f"icon-eu_europe_regular-lat-lon_single-level_"
            f"{run:%Y%m%d%H}_{step:03d}_TOT_PREC.grib2.bz2")


def _read_regular(path):
    import xarray as xr
    ds = xr.open_dataset(path, engine="cfgrib", backend_kwargs={"indexpath": ""})
    da = ds[list(ds.data_vars)[0]]
    la = da["latitude"].values.astype(np.float64)
    lo = da["longitude"].values.astype(np.float64)
    v = np.nan_to_num(da.values.astype(np.float32))
    ds.close()
    lo = np.where(lo > 180, lo - 360, lo)
    return la, lo, v


def icon_hourly(run: dt.datetime, step: int, lat, lon, base_url=None) -> np.ndarray | None:
    """Часов валеж ICON-EU за (step-1, step] върху мрежата за проверка."""
    d = os.path.join(CFG["cache_dir"], "icon", f"{run:%Y%m%d%H}")
    os.makedirs(d, exist_ok=True)
    base = (base_url or ICON_DWD["base_url"]).rstrip("/") + f"/{run:%H}/tot_prec/"

    def acc(s):
        if s <= 0:
            return 0.0
        name = _icon_name(run, s)
        p = os.path.join(d, name[:-4])
        if not os.path.exists(p):
            tmp = p + ".bz2"
            _download(base + name, tmp, timeout=90)
            with bz2.open(tmp, "rb") as fi, open(p, "wb") as fo:
                shutil.copyfileobj(fi, fo)
            os.remove(tmp)
        la, lo, v = _read_regular(p)
        iy = np.where((la >= lat[0] - 1) & (la <= lat[-1] + 1))[0]
        ix = np.where((lo >= lon[0] - 1) & (lo <= lon[-1] + 1))[0]
        return to_grid_regular(v[np.ix_(iy, ix)], la[iy], lo[ix], lat, lon)

    try:
        return np.clip(acc(step) - acc(step - 1), 0.0, None).astype(np.float32)
    except Exception as e:
        logger.warning(f"  ICON {run:%Y-%m-%d %H} +{step}: {type(e).__name__} {str(e)[:100]}")
        return None


# ----------------------------------------------------------------------------- метрики
def smooth(field, sigma_km):
    from scipy.ndimage import gaussian_filter
    return gaussian_filter(np.nan_to_num(field), sigma_km / dx_km()).astype(np.float32)


def fss_parts(obs, fc, valid, thr, n):
    """
    Числител и знаменател на FSS в прозорец n×n (само валидните точки, дялове спрямо
    валидните в прозореца — краят на обхвата не се брои за „сухо“).
    FSS = 1 − num/den.
    """
    from scipy.ndimage import uniform_filter
    v = valid.astype(np.float64)
    bo = ((np.nan_to_num(obs) >= thr) & valid).astype(np.float64)
    bf = ((np.nan_to_num(fc) >= thr) & valid).astype(np.float64)
    w = uniform_filter(v, n, mode="constant")
    ok = valid & (w > 0.25)
    with np.errstate(invalid="ignore", divide="ignore"):
        fo = uniform_filter(bo, n, mode="constant")[ok] / w[ok]
        ff = uniform_filter(bf, n, mode="constant")[ok] / w[ok]
    num = float(np.mean((fo - ff) ** 2)) if fo.size else 0.0
    den = float(np.mean(fo ** 2) + np.mean(ff ** 2)) if fo.size else 0.0
    return num, den


def scale_px(km):
    return 2 * int(round(km / 2 / dx_km())) + 1


FIELDS = ["valid", "lead_min", "model", "run", "step", "age_h", "thr", "scale_km",
          "num", "den", "fss", "obs_frac", "fc_frac", "obs_mean", "fc_mean", "n_obs"]


def compare(V: dt.datetime, obs, n_obs, models: dict, valid) -> list[dict]:
    """models: {lead: {name: (field, run, step)}} → редове за CSV."""
    rows = []
    for lead, mm in models.items():
        for name, (fc, run, step) in mm.items():
            if fc is None:
                continue
            ok = valid & np.isfinite(fc)
            for thr in CFG["thresholds"]:
                of = float((obs[ok] >= thr).mean()) if ok.any() else 0.0
                ff = float((fc[ok] >= thr).mean()) if ok.any() else 0.0
                for km in CFG["scales_km"]:
                    num, den = fss_parts(obs, fc, ok, thr, scale_px(km))
                    rows.append({
                        "valid": f"{V:%Y-%m-%dT%H:%MZ}", "lead_min": lead, "model": name,
                        "run": f"{run:%Y-%m-%dT%HZ}", "step": step,
                        "age_h": round((V - run).total_seconds() / 3600, 1),
                        "thr": thr, "scale_km": km,
                        "num": f"{num:.6g}", "den": f"{den:.6g}",
                        "fss": f"{1 - num / den:.4f}" if den > 0 else "",
                        "obs_frac": f"{of:.5f}", "fc_frac": f"{ff:.5f}",
                        "obs_mean": f"{float(np.mean(obs[ok])):.4f}" if ok.any() else "",
                        "fc_mean": f"{float(np.mean(fc[ok])):.4f}" if ok.any() else "",
                        "n_obs": n_obs})
    return rows


def cleanup():
    cut = dt.datetime.now(UTC) - dt.timedelta(hours=CFG["keep_h"])
    root = CFG["cache_dir"]
    for sub in ("ro", "aladin", "icon"):
        p = os.path.join(root, sub)
        if not os.path.isdir(p):
            continue
        for name in os.listdir(p):
            q = os.path.join(p, name)
            if dt.datetime.fromtimestamp(os.path.getmtime(q), UTC) < cut:
                shutil.rmtree(q, ignore_errors=True) if os.path.isdir(q) else os.remove(q)


# ----------------------------------------------------------------------------- main
def run_hour(V: dt.datetime, out_dir: str, listing=None, aladin_url=None, icon_url=None) -> int:
    lat, lon = grid()
    logger.info(f"Час за проверка: (V-1ч, V], V = {V:%Y-%m-%d %H:%M} UTC")
    obs, n_obs = observed_hour(V, lat, lon, listing)
    if obs is None:
        logger.warning(f"  Наблюдение: само {n_obs} композита в часа — пропускам")
        return 0
    valid = np.isfinite(obs) & ~sea_mask(lat, lon)
    wet = float((obs[valid] >= CFG["thresholds"][0]).mean())
    logger.info(f"  Наблюдение: {n_obs} композита, покрити {valid.mean():.0%} от мрежата, "
                f"валеж ≥{CFG['thresholds'][0]} mm/h в {wet:.1%}, max {np.nanmax(obs):.1f} mm/h")

    al_cache, ic_cache, models = {}, {}, {}
    for lead in CFG["leads_min"]:
        issue = V - dt.timedelta(minutes=30 + lead)
        ra = run_for(issue, CFG["aladin_every_h"], CFG["aladin_delay"])
        ri = run_for(issue, CFG["icon_every_h"], CFG["icon_delay"])
        sa = int((V - ra).total_seconds() // 3600)
        si = int((V - ri).total_seconds() // 3600)
        if ra not in al_cache:
            al_cache[ra] = aladin_hourly(ra, lat, lon, aladin_url)
        if (ri, si) not in ic_cache:
            ic_cache[(ri, si)] = icon_hourly(ri, si, lat, lon, icon_url)
        a = al_cache[ra]
        fa = a["steps"].get(sa) if a else None
        models[lead] = {"icon": (ic_cache[(ri, si)], ri, si),
                        "aladin": (fa, ra, sa),
                        "aladin_s": (smooth(fa, CFG["smooth_km"]) if fa is not None else None, ra, sa)}
        logger.info(f"  +{lead} мин (издаване {issue:%H:%M}): ICON {ri:%d.%m %H} +{si} ч "
                    f"{'ok' if ic_cache[(ri, si)] is not None else 'липсва'} | "
                    f"ALADIN {ra:%d.%m %H} +{sa} ч {'ok' if fa is not None else 'липсва'}")

    # Сух час и сухи модели — не носи информация, не се записва
    any_wet = wet > 5e-4 or any(
        f is not None and float((np.nan_to_num(f)[valid] >= CFG["thresholds"][0]).mean()) > 5e-4
        for mm in models.values() for f, _, _ in mm.values())
    if not any_wet:
        logger.info("  Сухо навсякъде — без запис")
        return 0

    rows = compare(V, obs, n_obs, models, valid)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{V:%Y-%m}.csv")
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerows(rows)

    # Кратко резюме в лога: FSS при 0.5 mm/h / 30 km
    for lead in CFG["leads_min"]:
        r = {x["model"]: x["fss"] for x in rows
             if x["lead_min"] == lead and x["thr"] == 0.5 and x["scale_km"] == 30}
        logger.info(f"  FSS 0.5 mm/h, 30 km, +{lead}: " +
                    " | ".join(f"{k} {v or '—'}" for k, v in r.items()))
    logger.info(f"  {len(rows)} реда → {path}")
    return len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--time", help="V (UTC), напр. 2026-11-05T14; по подразбиране последният час")
    ap.add_argument("--out", default=os.path.join(ROOT, "out", "compare"))
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
                        datefmt="%H:%M:%S")
    if a.time:
        V = dt.datetime.strptime(a.time[:13], "%Y-%m-%dT%H").replace(tzinfo=UTC)
    else:
        V = last_hour(dt.datetime.now(UTC))
    os.makedirs(CFG["cache_dir"], exist_ok=True)
    cleanup()
    run_hour(V, a.out)


if __name__ == "__main__":
    main()
