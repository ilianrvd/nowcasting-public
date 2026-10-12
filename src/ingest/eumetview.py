"""
Ingest: EUMETSAT EUMETView (WCS, без регистрация)
=================================================
  H40B   (mtg_fd__h40b)   — валеж от MTG FCI + LEO MW, mm/h, на 10 мин, ~25 мин закъснение
  LI AFA (mtg_fd__li_afa) — мълнии от MTG Lightning Imager, на 5 мин, ~10 мин закъснение
                            (WCS дава оцветена картинка; броят светкавици за 5 мин
                             се извлича от зеления канал на палитрата 1 → 20+)
Сървър: https://view.eumetsat.int/geoserver/  (Fees: none, AccessConstraints: none)
"""

import io, re, logging
import datetime as dt
import numpy as np
import requests

logger = logging.getLogger("ingest.eumetview")

WCS = "https://view.eumetsat.int/geoserver/wcs"
BBOX_LAT = (40.0, 45.0)      # малко по-широко от домейна — за адвекция отвън
BBOX_LON = (20.5, 30.0)
_session = requests.Session()


def _latest_time(cov: str) -> dt.datetime:
    r = _session.get(WCS, params=dict(service="WCS", version="2.0.1",
                     request="DescribeCoverage", coverageId=cov), timeout=60)
    r.raise_for_status()
    end = re.search(r"<gml:endPosition>([^<]+)</gml:endPosition>", r.text).group(1)
    return dt.datetime.fromisoformat(end.replace("Z", "+00:00"))


def _get_coverage(cov: str, t: dt.datetime):
    """GetCoverage → (bands, rows, cols), lat_1d (възходящо), lon_1d."""
    import tifffile
    ts = t.strftime("%Y-%m-%dT%H:%M:%S.000Z")
    params = [("service", "WCS"), ("version", "2.0.1"), ("request", "GetCoverage"),
              ("coverageId", cov), ("format", "image/tiff"),
              ("subset", f"Lat({BBOX_LAT[0]},{BBOX_LAT[1]})"),
              ("subset", f"Long({BBOX_LON[0]},{BBOX_LON[1]})"),
              ("subset", f'time("{ts}")')]
    r = _session.get(WCS, params=params, timeout=120)
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


def fetch_h40b(now: dt.datetime = None, max_age_min: float = 45) -> dict | None:
    """Последният кадър H40B: валеж mm/h (NaN = няма данни), качество %."""
    now = now or dt.datetime.now(dt.timezone.utc)
    try:
        t = _latest_time("mtg_fd__h40b")
        age = (now - t).total_seconds() / 60
        if age > max_age_min:
            logger.warning(f"H40B: последният кадър {t:%H:%M} UTC е стар ({age:.0f} мин)")
            return None
        arr, lat, lon = _get_coverage("mtg_fd__h40b", t)
    except Exception as e:
        logger.warning(f"H40B недостъпен: {e}")
        return None

    rain = arr[0]
    rain[(rain < 0) | ~np.isfinite(rain)] = np.nan
    quality = arr[1] if arr.shape[0] > 1 else None
    logger.info(f"H40B: {t:%H:%M} UTC (закъснение {age:.0f} мин), {rain.shape}, "
                f"валеж >0.1 mm/h в {int(np.nansum(rain > 0.1))} px, "
                f"max {np.nanmax(rain) if np.any(np.isfinite(rain)) else 0:.1f} mm/h")
    return {"time": t, "rain": rain, "quality": quality, "lat": lat, "lon": lon,
            "source": "eumetsat_h40b"}


def _afa_count(rgb: np.ndarray) -> np.ndarray:
    """RGB от палитрата жълто→червено (1 → 20+) → брой светкавици за 5 мин."""
    colored = rgb.max(axis=0) > 0
    g = rgb[1]
    count = 1.0 + 19.0 * (1.0 - g / 255.0)
    return np.where(colored, np.clip(count, 1.0, 20.0), 0.0).astype(np.float32)


def fetch_li_afa(now: dt.datetime = None, n_frames: int = 3,
                 max_age_min: float = 30) -> dict | None:
    """
    Мълнии от MTG LI: максимумът на брой светкавици/5 мин от последните n_frames кадъра
    (3 кадъра = последните 15 мин). Кадър, който не мине и след повторен опит, се пропуска.
    """
    import time
    now = now or dt.datetime.now(dt.timezone.utc)
    try:
        t_last = _latest_time("mtg_fd__li_afa")
    except Exception as e:
        logger.warning(f"LI AFA недостъпен: {e}")
        return None
    age = (now - t_last).total_seconds() / 60
    if age > max_age_min:
        logger.warning(f"LI AFA: последният кадър {t_last:%H:%M} UTC е стар ({age:.0f} мин)")
        return None

    count, lat, lon, got = None, None, None, []
    for k in range(n_frames):
        t = t_last - dt.timedelta(minutes=5 * k)
        for attempt in (1, 2):
            try:
                rgb, lat, lon = _get_coverage("mtg_fd__li_afa", t)
                c = _afa_count(rgb)
                count = c if count is None else np.maximum(count, c)
                got.append(f"{t:%H:%M}")
                break
            except Exception as e:
                if attempt == 1:
                    time.sleep(2)
                else:
                    logger.warning(f"LI AFA {t:%H:%M}: пропуснат ({str(e)[:60]})")

    if count is None:
        logger.warning("LI AFA: нито един кадър не мина")
        return None

    n_px = int((count > 0).sum())
    logger.info(f"LI AFA: {t_last:%H:%M} UTC (закъснение {age:.0f} мин), "
                f"кадри {', '.join(got)}: мълнии в {n_px} px, "
                f"max {count.max():.0f} светкавици/5 мин")
    return {"time": t_last, "count": count, "lat": lat, "lon": lon,
            "source": "eumetsat_li_afa"}

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
    h = fetch_h40b()
    li = fetch_li_afa()
    # Проверка и на кадъра с мълнии от сутринта
    t_test = dt.datetime(2026, 10, 12, 3, 25, tzinfo=dt.timezone.utc)
    rgb, lat, lon = _get_coverage("mtg_fd__li_afa", t_test)
    c = _afa_count(rgb)
    print(f"Тест 03:25 UTC: мълнии в {int((c > 0).sum())} px, "
          f"max {c.max():.0f}, медиана {np.median(c[c > 0]):.1f} светкавици/5 мин")