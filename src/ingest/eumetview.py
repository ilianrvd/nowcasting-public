"""
Ingest: EUMETSAT EUMETView (WCS, без регистрация)
=================================================
  H40B   (mtg_fd__h40b)   — валеж от MTG FCI + LEO MW, mm/h, на 10 мин, ~20–30 мин закъснение
  LI AFA (li_afa)         — мълнии от MTG Lightning Imager, на 5 мин, ~10–14 мин закъснение.
                            От 10.2026 слоят е махнат от WCS и е само в WMS (оцветена
                            картинка). Броят светкавици за 5 мин се връща от цвета по
                            легендата на EUMETView (1 → 10 → 20+), таблица _AFA_LUT.
Сървър: https://view.eumetsat.int/geoserver/  (Fees: none, AccessConstraints: none)
"""

import io, os, re, logging
import datetime as dt
import numpy as np
import requests

logger = logging.getLogger("ingest.eumetview")

WCS = "https://view.eumetsat.int/geoserver/wcs"
WMS_LI = "https://view.eumetsat.int/geoserver/mtg_fd/wms"
LI_RES_DEG = 0.02            # ~2 km — решетката на FCI, на която е AFA
BBOX_LAT = (40.0, 45.0)      # малко по-широко от домейна — за адвекция отвън
BBOX_LON = (20.5, 30.0)


def _latest_time(cov: str, timeout: float = 90) -> dt.datetime:
    r = requests.get(WCS, params=dict(service="WCS", version="2.0.1",
                     request="DescribeCoverage", coverageId=cov), timeout=timeout)
    r.raise_for_status()
    end = re.search(r"<gml:endPosition>([^<]+)</gml:endPosition>", r.text).group(1)
    return dt.datetime.fromisoformat(end.replace("Z", "+00:00"))


def _get_coverage(cov: str, t: dt.datetime, timeout: float = 120):
    """GetCoverage → (bands, rows, cols), lat_1d (възходящо), lon_1d."""
    import tifffile
    ts = t.strftime("%Y-%m-%dT%H:%M:%S.000Z")
    params = [("service", "WCS"), ("version", "2.0.1"), ("request", "GetCoverage"),
              ("coverageId", cov), ("format", "image/tiff"),
              ("subset", f"Lat({BBOX_LAT[0]},{BBOX_LAT[1]})"),
              ("subset", f"Long({BBOX_LON[0]},{BBOX_LON[1]})"),
              ("subset", f'time("{ts}")')]
    r = requests.get(WCS, params=params, timeout=timeout)   # без общата сесия — безопасно в нишки
    r.raise_for_status()
    if "xml" in r.headers.get("Content-Type", ""):
        raise RuntimeError(f"{cov}: WCS грешка: {r.text[:300]}")

    logging.getLogger("tifffile").setLevel(logging.ERROR)   # заглушава GDAL_NODATA предупреждението
    with tifffile.TiffFile(io.BytesIO(r.content)) as tif:
        arr = tif.asarray().astype(np.float32)
        tags = {tg.code: tg.value for tg in tif.pages[0].tags.values()}
    if arr.ndim == 2:
        arr = arr[None]
    elif arr.shape[-1] <= 4 and arr.shape[0] > 4:
        arr = np.moveaxis(arr, -1, 0)

    rows, cols = arr.shape[1:]
    if 33922 in tags and 33550 in tags:                 # ModelTiepoint + ModelPixelScale
        tp, sc = tags[33922], tags[33550]
        x0, y0, sx, sy = tp[3], tp[4], sc[0], sc[1]
    elif 34264 in tags:                                  # ModelTransformation (4x4)
        m = tags[34264]
        x0, y0, sx, sy = m[3], m[7], m[0], -m[5]
    else:                                                # резервно: поисканият правоъгълник
        x0, y0 = BBOX_LON[0], BBOX_LAT[1]
        sx = (BBOX_LON[1] - BBOX_LON[0]) / cols
        sy = (BBOX_LAT[1] - BBOX_LAT[0]) / rows
        logger.debug(f"{cov}: няма GeoTIFF тагове — ползвам bbox")
    lon = x0 + (np.arange(cols) + 0.5) * sx
    lat = y0 - (np.arange(rows) + 0.5) * sy          # отгоре надолу
    arr, lat = arr[:, ::-1, :], lat[::-1]            # → възходящо по lat
    return arr, lat.astype(np.float64), lon.astype(np.float64)


CACHE_FILE = os.path.join(os.path.dirname(__file__), "..", "..", "data", "cache", "h40b_last.npz")


def _h40b_save_cache(d: dict):
    try:
        os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
        np.savez_compressed(CACHE_FILE, rain=d["rain"], lat=d["lat"], lon=d["lon"],
                            time=d["time"].strftime("%Y-%m-%dT%H:%M:%S"))
    except Exception as e:
        logger.debug(f"H40B кеш: не е записан ({e})")


def _h40b_load_cache(now: dt.datetime, max_age_min: float):
    """Последният успешен H40B от предишен пуск (GitHub Actions cache), ако е достатъчно пресен."""
    try:
        z = np.load(CACHE_FILE)
        t = dt.datetime.fromisoformat(str(z["time"])).replace(tzinfo=dt.timezone.utc)
    except Exception:
        return None
    age = (now - t).total_seconds() / 60
    if age > max_age_min:
        logger.warning(f"H40B кеш: {t:%H:%M} UTC е твърде стар ({age:.0f} мин)")
        return None
    logger.warning(f"H40B: ползвам кадъра от предишния пуск — {t:%H:%M} UTC ({age:.0f} мин)")
    return {"time": t, "rain": z["rain"], "quality": None, "lat": z["lat"], "lon": z["lon"],
            "source": "eumetsat_h40b_cache"}


def fetch_h40b(now: dt.datetime = None, max_age_min: float = 60,
               cache_max_age_min: float = 70) -> dict | None:
    """
    Последният кадър H40B: валеж mm/h (NaN = няма данни).
      1. кратко питане за последния кадър (DescribeCoverage, 20 s) — EUMETView често е бавен тук;
      2. ако не отговори: часът се изчислява (кадри на 10 мин, ~20–30 мин закъснение)
         и се пробват директно последните възможни кадри;
      3. ако и това не мине: последният успешен кадър от предишния пуск (кеш, ≤ cache_max_age_min).
    """
    import time as _time
    now = now or dt.datetime.now(dt.timezone.utc)
    t0 = _time.time()

    try:
        cands = [_latest_time("mtg_fd__h40b", timeout=20)]
        how = "DescribeCoverage"
    except Exception as e:
        base = (now - dt.timedelta(minutes=20)).replace(second=0, microsecond=0)
        base -= dt.timedelta(minutes=base.minute % 10)
        cands = [base - dt.timedelta(minutes=10 * k) for k in range(4)]
        how = f"изчислен час (DescribeCoverage: {str(e)[:50]})"

    last_err = None
    for t in cands:
        age = (now - t).total_seconds() / 60
        if age > max_age_min:
            break
        try:
            arr, lat, lon = _get_coverage("mtg_fd__h40b", t, timeout=45)
        except Exception as e:
            last_err = e
            continue
        rain = arr[0]
        rain[(rain < 0) | ~np.isfinite(rain)] = np.nan
        if not np.isfinite(rain).any():          # празен кадър (времето още го няма)
            last_err = "празен кадър"
            continue
        quality = arr[1] if arr.shape[0] > 1 else None
        logger.info(f"H40B: {t:%H:%M} UTC (закъснение {age:.0f} мин, {how}, "
                    f"{_time.time() - t0:.0f} s), {rain.shape}, "
                    f"валеж >0.1 mm/h в {int(np.nansum(rain > 0.1))} px, "
                    f"max {np.nanmax(rain):.1f} mm/h")
        d = {"time": t, "rain": rain, "quality": quality, "lat": lat, "lon": lon,
             "source": "eumetsat_h40b"}
        _h40b_save_cache(d)
        return d

    logger.warning(f"H40B недостъпен ({_time.time() - t0:.0f} s): {str(last_err)[:100]}")
    return _h40b_load_cache(now, cache_max_age_min)


# Легендата на li_afa (GetLegendGraphic, 640 px): цвят → светкавици/5 мин.
# Отметките 1 / 10 / 20+ са на x = 148 / 405.5 / 635; между тях — линейно.
_AFA_LUT = np.array([
    (255, 255, 233, 1.00),
    (255, 251, 193, 1.28),
    (255, 249, 189, 1.56),
    (255, 246, 182, 1.84),
    (255, 244, 178, 2.12),
    (255, 240, 170, 2.40),
    (255, 239, 166, 2.68),
    (255, 237, 161, 2.96),
    (255, 234, 155, 3.24),
    (255, 232, 151, 3.52),
    (255, 228, 143, 3.80),
    (254, 227, 139, 4.08),
    (254, 223, 132, 4.36),
    (254, 221, 129, 4.63),
    (254, 219, 124, 4.91),
    (254, 215, 117, 5.19),
    (254, 212, 113, 5.47),
    (254, 205, 105, 5.75),
    (254, 202, 102, 6.03),
    (254, 195, 94, 6.31),
    (254, 191, 91, 6.59),
    (254, 189, 87, 6.87),
    (254, 181, 79, 7.15),
    (254, 178, 76, 7.43),
    (254, 171, 72, 7.71),
    (254, 168, 71, 7.99),
    (253, 161, 68, 8.27),
    (253, 158, 67, 8.55),
    (253, 153, 65, 8.83),
    (253, 148, 63, 9.11),
    (253, 145, 61, 9.39),
    (253, 137, 58, 9.67),
    (253, 131, 57, 9.95),
    (253, 119, 54, 10.28),
    (253, 114, 52, 10.63),
    (252, 107, 51, 10.98),
    (252, 97, 47, 11.33),
    (252, 92, 46, 11.68),
    (252, 81, 42, 12.03),
    (251, 76, 41, 12.37),
    (246, 66, 39, 12.72),
    (244, 61, 37, 13.07),
    (242, 56, 36, 13.42),
    (237, 48, 33, 13.77),
    (235, 43, 32, 14.12),
    (230, 34, 29, 14.47),
    (228, 29, 28, 14.81),
    (222, 23, 29, 15.16),
    (219, 20, 29, 15.51),
    (213, 16, 31, 15.86),
    (208, 13, 32, 16.21),
    (205, 11, 33, 16.56),
    (198, 6, 35, 16.91),
    (195, 3, 36, 17.25),
    (188, 0, 37, 17.60),
    (183, 0, 38, 17.95),
    (176, 0, 38, 18.30),
    (167, 0, 38, 18.65),
    (161, 0, 38, 19.00),
    (150, 0, 38, 19.35),
    (145, 0, 38, 19.69),
    (134, 0, 29, 20.00),
], dtype=np.float32)


def _afa_count(rgba: np.ndarray) -> np.ndarray:
    """RGBA картинка от WMS (rows, cols, 4) → брой светкавици за 5 мин (0 = няма)."""
    alpha = rgba[..., 3] > 0
    out = np.zeros(rgba.shape[:2], dtype=np.float32)
    if alpha.any():
        px = rgba[alpha][:, :3].astype(np.float32)
        d = ((px[:, None, :] - _AFA_LUT[None, :, :3]) ** 2).sum(-1)
        out[alpha] = _AFA_LUT[d.argmin(1), 3]
    return out


def _li_latest_time(timeout: float = 60) -> dt.datetime:
    """Последният кадър на li_afa от WMS GetCapabilities (Dimension time, default=...)."""
    r = requests.get(WMS_LI, params=dict(service="WMS", version="1.3.0", request="GetCapabilities"),
                     timeout=timeout)
    r.raise_for_status()
    i = r.text.find("<Name>li_afa</Name>")
    if i < 0:
        raise RuntimeError("li_afa липсва в WMS GetCapabilities")
    m = re.search(r'<Dimension name="time" default="([^"]+)"', r.text[i:i + 6000])
    return dt.datetime.fromisoformat(m.group(1).replace("Z", "+00:00"))


def _li_getmap(t: dt.datetime, timeout: float = 30):
    """WMS GetMap (PNG) за bbox-а → (count, lat_1d възходящо, lon_1d)."""
    from PIL import Image
    w = int(round((BBOX_LON[1] - BBOX_LON[0]) / LI_RES_DEG))
    h = int(round((BBOX_LAT[1] - BBOX_LAT[0]) / LI_RES_DEG))
    params = dict(service="WMS", version="1.3.0", request="GetMap", layers="li_afa", styles="",
                  crs="EPSG:4326",                      # WMS 1.3.0 + EPSG:4326 → lat,lon
                  bbox=f"{BBOX_LAT[0]},{BBOX_LON[0]},{BBOX_LAT[1]},{BBOX_LON[1]}",
                  width=w, height=h, format="image/png", transparent="true",
                  time=t.strftime("%Y-%m-%dT%H:%M:%SZ"))
    r = requests.get(WMS_LI, params=params, timeout=timeout)
    r.raise_for_status()
    if "image" not in r.headers.get("Content-Type", ""):
        raise RuntimeError(f"WMS грешка: {r.text[:200]}")
    rgba = np.array(Image.open(io.BytesIO(r.content)).convert("RGBA"))
    count = _afa_count(rgba)[::-1, :]                    # отгоре надолу → възходящо по lat
    lat = BBOX_LAT[0] + (np.arange(h) + 0.5) * (BBOX_LAT[1] - BBOX_LAT[0]) / h
    lon = BBOX_LON[0] + (np.arange(w) + 0.5) * (BBOX_LON[1] - BBOX_LON[0]) / w
    return count, lat, lon


def fetch_li_frames(now: dt.datetime = None, back_min: float = 40,
                    max_age_min: float = 30) -> list[dict]:
    """
    Отделните 5-мин кадри на MTG LI за последните back_min минути (без сливане),
    за да може всеки радарен кадър да вземе мълниите от своя прозорец.
    Кадър, който не мине и след повторен опит, се пропуска.
    """
    import time
    now = now or dt.datetime.now(dt.timezone.utc)
    try:
        t_last = _li_latest_time()
    except Exception as e:
        logger.warning(f"LI AFA (WMS) недостъпен: {e}")
        return []
    age = (now - t_last).total_seconds() / 60
    if age > max_age_min:
        logger.warning(f"LI AFA: последният кадър {t_last:%H:%M} UTC е стар ({age:.0f} мин)")
        return []

    from concurrent.futures import ThreadPoolExecutor

    def _one(t):
        for attempt in (1, 2):
            try:
                count, lat, lon = _li_getmap(t, timeout=30)
                return {"time": t, "count": count, "lat": lat, "lon": lon}
            except Exception as e:
                if attempt == 1:
                    time.sleep(2)
                else:
                    logger.warning(f"LI AFA {t:%H:%M}: пропуснат ({str(e)[:60]})")
        return None

    times = [t_last - dt.timedelta(minutes=5 * k) for k in range(int(back_min // 5) + 1)]
    t_start = time.time()
    with ThreadPoolExecutor(max_workers=4) as ex:      # 4 заявки наведнъж
        frames = [f for f in ex.map(_one, times) if f is not None]
    frames.sort(key=lambda f: f["time"])

    if frames:
        n_px = int(max((f["count"] > 0).sum() for f in frames))
        mx = max(float(f["count"].max()) for f in frames)
        logger.info(f"LI AFA: {len(frames)} кадъра {frames[0]['time']:%H:%M}–"
                    f"{frames[-1]['time']:%H:%M} UTC (закъснение {age:.0f} мин), "
                    f"до {n_px} px с мълнии в кадър, max {mx:.0f} светкавици/5 мин, "
                    f"за {time.time() - t_start:.0f} s")
    else:
        logger.warning("LI AFA: нито един кадър не мина")
    return frames


def regrid(field, lat, lon, target_lat, target_lon, method="nearest"):
    """Прехвърля поле от решетката на EUMETView към целевата решетка."""
    from scipy.interpolate import RegularGridInterpolator
    interp = RegularGridInterpolator((lat, lon), np.nan_to_num(field, nan=0.0),
                                     method=method, bounds_error=False, fill_value=0.0)
    lon2d, lat2d = np.meshgrid(target_lon, target_lat)
    pts = np.column_stack([lat2d.ravel(), lon2d.ravel()])
    return interp(pts).reshape(len(target_lat), len(target_lon)).astype(np.float32)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
    fetch_h40b()
    fetch_li_frames(back_min=15)
