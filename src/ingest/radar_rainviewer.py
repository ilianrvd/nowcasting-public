"""
Ingest: Истанбул (MGM, TR34) — изходни картинки през RainViewer
================================================================
Източник:  https://data.rainviewer.com/images/TR34/
Файлове:   TR34_YYYYMMDD_HHMMSS_0_source.jpeg  (UTC слот, на 6 мин;
           реалният скан е ~2 мин след слота: 07:10 → 07:12:01Z)
Продукт:   PPI 0.3°, обхват 370 km, JPEG 880×720 (карта 720×720 + легенда)

Геореференция (калибрирана 2026-10-12 по 20 града в картинката, RMS ≈ 2 km):
  радар 41.348°N, 28.367°E; център на картата (x=360, y=359) px;
  0.975 px/km (пръстените през 100 km са на 97.5 px); азимутална
  еквидистантна проекция (AEQD) с център радара.

Цветове: непрекъсната скала 2…72 dBZ (вдясно на картинката). Всеки пиксел
се сравнява с цветовете на скалата; фонът на картата (суша, море, сиво извън
обхвата, линии, надписи) е далеч от тях и отпада. Пикселите под линиите и
надписите се попълват от съседите, за да няма дупки в ехото.

Изход: същият формат като ИАБГ — регулярна lat/lon решетка (1D lat, lon),
NaN извън обхвата, 0 = сухо в обхвата.
Данни: © MGM (Turkish State Meteorological Service), чрез RainViewer.
"""

import os, sys, re, logging
import datetime as dt
from io import BytesIO

import numpy as np
import requests
from PIL import Image
from scipy.ndimage import maximum_filter, map_coordinates

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from config.settings import RAINVIEWER_RADARS

logger = logging.getLogger("ingest.tr34")

BASE_URL = "https://data.rainviewer.com/images/TR34/"
HEADERS = {"User-Agent": "nowcasting-public/1.0 (github.com/ilianrvd/nowcasting-public)"}

# ── Геометрия на картинката ────────────────────────────────
RADAR_LAT = RAINVIEWER_RADARS["TR34"]["lat"]
RADAR_LON = RAINVIEWER_RADARS["TR34"]["lon"]
CX, CY = 360.0, 359.0          # пиксел на радара
PX_PER_KM = 0.975
RANGE_KM = float(RAINVIEWER_RADARS["TR34"]["range_km"])
MAP_W = 720                    # колоните вдясно са легендата
SCAN_OFFSET_MIN = 2            # реален скан ≈ слот + 2 мин
EARTH_R = 6371.0

# ── Цветова скала: (ред в легендата на оригинала, dBZ в средата на лентата) ──
_LEGEND_ROWS = [382, 397, 412, 427, 442, 457, 472, 487, 503, 518, 533, 548, 563, 578, 593, 608]
_LEGEND_DBZ  = [69, 63, 57.5, 54, 51.5, 47, 41.5, 38, 35.5, 31, 25.5, 22, 19.5, 15, 9.5, 4.5]
_LEGEND_X = (786, 796)         # колони на цветната лента
_COLOR_TOL = 40.0              # макс. RGB разстояние до цвят от скалата

_lut_rgb, _lut_dbz = None, None


def _build_lut(img: np.ndarray):
    """Таблица цвят → dBZ от цветната лента на самата картинка."""
    ys = np.arange(_LEGEND_ROWS[0] - 4, _LEGEND_ROWS[-1] + 5)
    rgb = img[ys, _LEGEND_X[0]:_LEGEND_X[1]].astype(np.float32).mean(axis=1)
    dbz = np.interp(ys, _LEGEND_ROWS, _LEGEND_DBZ)
    return rgb, dbz.astype(np.float32)


# ── Списък с кадри ─────────────────────────────────────────
def list_scans() -> list[dict]:
    """Файловете в папката TR34, сортирани по време (UTC на скана)."""
    r = requests.get(BASE_URL, headers=HEADERS, timeout=20)
    r.raise_for_status()
    names = sorted(set(re.findall(r'TR34_(\d{8})_(\d{6})_0_source\.jpe?g', r.text)))
    scans = []
    for d, t in names:
        slot = dt.datetime.strptime(d + t, "%Y%m%d%H%M%S").replace(tzinfo=dt.timezone.utc)
        scans.append({"name": f"TR34_{d}_{t}_0_source.jpeg",
                      "timestamp": slot + dt.timedelta(minutes=SCAN_OFFSET_MIN)})
    return scans


# ── Декодиране ─────────────────────────────────────────────
def decode_image(img: np.ndarray) -> np.ndarray:
    """RGB картинка → dBZ в пикселите на картата (720×720). NaN извън обхвата."""
    global _lut_rgb, _lut_dbz
    if _lut_rgb is None:
        _lut_rgb, _lut_dbz = _build_lut(img)

    m = img[:, :MAP_W].astype(np.float32)
    h, w, _ = m.shape
    px = m.reshape(-1, 3)

    # Най-близкият цвят от скалата (на парчета, за да не пълним паметта)
    dbz = np.zeros(len(px), dtype=np.float32)
    dist = np.full(len(px), np.inf, dtype=np.float32)
    for s in range(0, len(px), 60000):
        d = np.sqrt(((px[s:s + 60000, None, :] - _lut_rgb[None]) ** 2).sum(-1))
        k = d.argmin(1)
        dist[s:s + 60000] = d[np.arange(len(k)), k]
        dbz[s:s + 60000] = _lut_dbz[k]
    echo = (dist < _COLOR_TOL).reshape(h, w)
    dbz = np.where(echo, dbz.reshape(h, w), 0.0).astype(np.float32)

    # Линии и надписи (тъмни/бели, ненаситени) → попълни от съседите
    sat = m.max(-1) - m.min(-1)
    bright = m.mean(-1)
    overlay = (sat < 40) & ((bright < 110) | (bright > 235))
    dbz = np.where(overlay, maximum_filter(dbz, size=5), dbz)

    # Извън обхвата → NaN
    yy, xx = np.mgrid[0:h, 0:w]
    r_km = np.hypot(xx - CX, yy - CY) / PX_PER_KM
    dbz[r_km > RANGE_KM] = np.nan
    return dbz


def _aeqd_forward(lat, lon):
    """lat/lon (градуси) → x, y (km) спрямо радара."""
    la, lo = np.radians(lat), np.radians(lon)
    la0, lo0 = np.radians(RADAR_LAT), np.radians(RADAR_LON)
    cosc = np.sin(la0) * np.sin(la) + np.cos(la0) * np.cos(la) * np.cos(lo - lo0)
    c = np.arccos(np.clip(cosc, -1.0, 1.0))
    k = np.where(c > 1e-9, c / np.sin(np.maximum(c, 1e-9)), 1.0)
    x = EARTH_R * k * np.cos(la) * np.sin(lo - lo0)
    y = EARTH_R * k * (np.cos(la0) * np.sin(la) - np.sin(la0) * np.cos(la) * np.cos(lo - lo0))
    return x, y


def to_latlon_grid(dbz_px: np.ndarray, step_deg: float = 0.01):
    """Пикселите на картинката → регулярна lat/lon решетка (nearest)."""
    lat = np.arange(RADAR_LAT - 3.4, RADAR_LAT + 3.4, step_deg)
    lon = np.arange(RADAR_LON - 4.6, RADAR_LON + 4.6, step_deg)
    lon2d, lat2d = np.meshgrid(lon, lat)
    x, y = _aeqd_forward(lat2d, lon2d)
    col = CX + x * PX_PER_KM
    row = CY - y * PX_PER_KM
    filled = np.nan_to_num(dbz_px, nan=-999.0)
    out = map_coordinates(filled, [row, col], order=0, mode="constant", cval=-999.0)
    out = out.astype(np.float32)
    out[out <= -998] = np.nan
    return out, lat, lon


# ── Пълен pipeline ─────────────────────────────────────────
def ingest_tr34(n_frames: int = 6) -> list[dict]:
    """Последните n_frames кадъра на Истанбул като frames за композита."""
    logger.info("=== Ingest TR34 (Истанбул, MGM през RainViewer) ===")
    try:
        scans = list_scans()
    except Exception as e:
        logger.warning(f"[TR34] списъкът е недостъпен: {e}")
        return []
    if not scans:
        logger.warning("[TR34] няма файлове")
        return []

    frames = []
    for s in scans[-n_frames:]:
        try:
            r = requests.get(BASE_URL + s["name"], headers=HEADERS, timeout=30)
            r.raise_for_status()
            img = np.array(Image.open(BytesIO(r.content)).convert("RGB"))
        except Exception as e:
            logger.warning(f"[TR34] {s['name']}: {e}")
            continue
        if img.shape[0] != 720 or img.shape[1] != 880:
            logger.warning(f"[TR34] {s['name']}: неочакван размер {img.shape} — пропуснат")
            continue
        dbz_px = decode_image(img)
        dbz, lat, lon = to_latlon_grid(dbz_px)
        n_echo = int(np.nansum(dbz >= 15))
        logger.info(f"  {s['name']}  скан {s['timestamp']:%H:%M} UTC, "
                    f"ехо ≥15 dBZ: {n_echo} px, max {np.nanmax(dbz):.0f} dBZ")
        frames.append({"timestamp": s["timestamp"], "dbz": dbz, "lat": lat, "lon": lon,
                       "source": "rainviewer_TR34"})

    logger.info(f"[TR34] {len(frames)} frames готови")
    return frames


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
    if len(sys.argv) > 1:                       # тест върху локален файл
        img = np.array(Image.open(sys.argv[1]).convert("RGB"))
        dbz, lat, lon = to_latlon_grid(decode_image(img))
        print(f"{sys.argv[1]}: ехо ≥15 dBZ в {int(np.nansum(dbz >= 15))} px, "
              f"max {np.nanmax(dbz):.0f} dBZ, решетка {dbz.shape}")
    else:
        ingest_tr34(3)
