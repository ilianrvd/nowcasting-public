"""
Ingest: ИАБГ радари (PNG от weathermod-bg.eu)
=============================================
Сваля PNG картинки от wr.weathermod-bg.eu, декодира colormap → dBZ,
геореферира върху lat/lon grid.
Радари: GCD (Голям Чардак), STS (Старо Село), BRD (Бърдарски геран).
"""

import os, sys, json, logging, re
import datetime as dt
from io import BytesIO

import numpy as np
import requests
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from config.settings import IABG_RADARS, COLORMAP_RGB_TO_DBZ, RADAR_DIR

logger = logging.getLogger("ingest.iabg")


# ============================================================
# JSON списък с файлове
# ============================================================
def fetch_file_list(radar_id: str) -> list[dict]:
    """
    Сваля JSON списъка от wr.weathermod-bg.eu.
    Формат: {"img_list": {"15": {"wh_img": "...url...", "timp": "04.07.2026  07:54"}, ...}}
    """
    radar = IABG_RADARS[radar_id]
    # Cache-buster параметър (unix timestamp)
    import time
    url = radar["json_list"] + f"?{int(time.time())}"
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.error(f"[{radar_id}] JSON грешка: {e}")
        return []

    files = []
    img_list = data.get("img_list", {})
    for key, item in img_list.items():
        img_url = item.get("wh_img", "")
        timp = item.get("timp", "")
        if not img_url:
            continue

        # Извлечи filename от URL
        fname = img_url.split("/")[-1]

        # Парсни timestamp от timp "04.07.2026  07:54" или от filename
        ts = _parse_timestamp(radar_id, fname)
        if ts is None and timp:
            ts = _parse_timp(timp)

        if ts:
            files.append({"filename": fname, "url": img_url, "timestamp": ts})

    files.sort(key=lambda x: x["timestamp"])
    logger.info(f"[{radar_id}] {len(files)} файла от JSON")
    return files





def _parse_timestamp(radar_id: str, filename: str) -> dt.datetime | None:
    """GCD260704193827.CAPVXE1.png → 2026-07-04 19:38:27 LOCAL → 16:38:27 UTC"""
    m = re.search(rf'{radar_id}(\d{{12}})', filename, re.IGNORECASE)
    if not m:
        return None
    try:
        ts = dt.datetime.strptime(m.group(1), "%y%m%d%H%M%S")
        # ИАБГ дава локално време (EEST = UTC+3 лятно)
        local_tz = dt.timezone(dt.timedelta(hours=3))
        ts = ts.replace(tzinfo=local_tz)
        return ts.astimezone(dt.timezone.utc)
    except ValueError:
        return None


def _parse_timp(timp: str) -> dt.datetime | None:
    """'04.07.2026  19:38' LOCAL → UTC"""
    try:
        ts = dt.datetime.strptime(timp.strip(), "%d.%m.%Y %H:%M")
        local_tz = dt.timezone(dt.timedelta(hours=3))
        ts = ts.replace(tzinfo=local_tz)
        return ts.astimezone(dt.timezone.utc)
    except ValueError:
        return None

# ============================================================
# Сваляне на PNG
# ============================================================
def download_image(radar_id: str, filename: str,
                   full_url: str = None) -> np.ndarray | None:
    """Сваля PNG и връща RGB масив."""
    url = full_url or (IABG_RADARS[radar_id]["url_base"] + filename)
    try:
        resp = requests.get(url, timeout=20)
        resp.raise_for_status()
        img = Image.open(BytesIO(resp.content)).convert("RGB")
        return np.array(img)
    except Exception as e:
        logger.error(f"[{radar_id}] Грешка: {filename}: {e}")
        return None


# ============================================================
# RGB → dBZ декодиране
# ============================================================
def rgb_to_dbz(img_rgb: np.ndarray, tolerance: int = 25) -> np.ndarray:
    """Конвертира RGB картинка → 2D dBZ масив. NaN за неразпознати пиксели."""
    colors = np.array([c[0] for c in COLORMAP_RGB_TO_DBZ], dtype=np.float32)
    values = np.array([c[2] for c in COLORMAP_RGB_TO_DBZ], dtype=np.float32)

    h, w, _ = img_rgb.shape
    dbz = np.full((h, w), np.nan, dtype=np.float32)
    pixels = img_rgb.reshape(-1, 3).astype(np.float32)

    chunk = 100_000
    for s in range(0, len(pixels), chunk):
        e = min(s + chunk, len(pixels))
        diff = pixels[s:e, None, :] - colors[None, :, :]
        dist = np.sqrt(np.sum(diff ** 2, axis=2))
        idx = np.argmin(dist, axis=1)
        mask = dist[np.arange(e - s), idx] < tolerance
        flat = np.arange(s, e)
        dbz[flat[mask] // w, flat[mask] % w] = values[idx[mask]]

    valid = np.count_nonzero(~np.isnan(dbz))
    if valid > 0:
        logger.info(f"  dBZ: {valid} px, "
                    f"min={np.nanmin(dbz):.0f} max={np.nanmax(dbz):.0f}")
    return dbz


# ============================================================
# Геореференциране
# ============================================================
def georeference_cappi(dbz_raw: np.ndarray, radar_id: str) -> tuple:
    """
    CAPPI PNG → lat/lon grid.
    Центрирано на радара с range_km обхват.
    """
    radar = IABG_RADARS[radar_id]
    h, w = dbz_raw.shape

    # Crop легенда (ако >80% от дясната лента е NaN)
    strip = dbz_raw[:, -int(w * 0.12):]
    if np.sum(np.isnan(strip)) / strip.size > 0.5:
        w_new = int(w * 0.88)
        dbz_raw = dbz_raw[:, :w_new]
        h, w = dbz_raw.shape

    r = radar["range_km"]
    dlat = r / 111.0
    dlon = r / (111.0 * np.cos(np.radians(radar["lat"])))

    lat_1d = np.linspace(radar["lat"] + dlat, radar["lat"] - dlat, h)
    lon_1d = np.linspace(radar["lon"] - dlon, radar["lon"] + dlon, w)

    return dbz_raw, lat_1d, lon_1d


# ============================================================
# Пълен pipeline
# ============================================================
def ingest_iabg(radar_id: str, n_frames: int = 5) -> list[dict]:
    """Сваля и обработва последните n_frames от даден ИАБГ радар."""
    logger.info(f"=== Ingest {radar_id} ({IABG_RADARS[radar_id]['name']}) ===")

    file_list = fetch_file_list(radar_id)
    if not file_list:
        return []

    latest = file_list[-n_frames:]
    frames = []

    for finfo in latest:
        img = download_image(radar_id, finfo["filename"],
                             full_url=finfo.get("url"))
        if img is None:
            continue
        dbz = rgb_to_dbz(img)
        dbz, lat, lon = georeference_cappi(dbz, radar_id)
        frames.append({
            "timestamp": finfo["timestamp"],
            "dbz": dbz,
            "lat": lat,
            "lon": lon,
            "source": f"iabg_{radar_id}",
        })

    logger.info(f"[{radar_id}] {len(frames)} frames готови")
    return frames


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(name)s] %(message)s")
    for rid in ["GCD", "STS"]:
        if IABG_RADARS[rid]["enabled"]:
            frames = ingest_iabg(rid, 3)
