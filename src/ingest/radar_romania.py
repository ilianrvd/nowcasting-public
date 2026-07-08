"""
Ingest: Румъния национален COMPOSITE (ODIM HDF5)
=================================================
Сваля и чете COMPOSITE файлове от meteoromania.ro/radar/
Формат: ODIM HDF5, CAPPI, AEQD проекция.
Адаптирано от nowcasting системата на ДП РВД.
"""

import os, sys, logging, re
import datetime as dt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from config.settings import ROMANIA, ROMANIA_DIR, DOMAIN

logger = logging.getLogger("ingest.romania")


def read_romania_composite(filepath: str) -> dict | None:
    """
    Чете румънски CAPPI composite в ODIM HDF5 формат.
    Проекция: AEQD (Azimuthal Equidistant).

    Returns: dict с dbz, lat_1d, lon_1d, timestamp
    """
    import h5py
    from pyproj import Transformer

    logger.info(f"Чета: {os.path.basename(filepath)}")

    with h5py.File(filepath, 'r') as f:

        what  = f['what'].attrs
        where = f['where'].attrs
        ds    = f['dataset1']

        date_str = what['date'].decode() if isinstance(what['date'], bytes) else str(what['date'])
        time_str = what['time'].decode() if isinstance(what['time'], bytes) else str(what['time'])
        timestamp = dt.datetime.strptime(date_str + time_str, "%Y%m%d%H%M%S")
        timestamp = timestamp.replace(tzinfo=dt.timezone.utc)

        # Проекция
        projdef = where['projdef'].decode() if isinstance(where['projdef'], bytes) else str(where['projdef'])
        xsize  = int(where['xsize'])
        ysize  = int(where['ysize'])
        xscale = float(where['xscale'])
        yscale = float(where['yscale'])

        ll_lat = float(where['LL_lat'])
        ll_lon = float(where['LL_lon'])
        ur_lat = float(where['UR_lat'])
        ur_lon = float(where['UR_lon'])

        # Данни — gain/offset са в dataset1/what (не dataset1)
        data_what = ds['what'].attrs
        gain     = float(data_what['gain'])
        offset   = float(data_what['offset'])
        nodata   = float(data_what['nodata'])
        undetect = float(data_what['undetect'])

        raw = ds['data1/data'][:].astype(float)
        dbz = raw * gain + offset
        dbz[raw == nodata]   = np.nan
        dbz[raw == undetect] = np.nan

    # AEQD → lat/lon с pyproj
    transformer = Transformer.from_crs(projdef, "EPSG:4326", always_xy=True)

    # Изчисли AEQD координати на ъглите
    x_ll, y_ll = Transformer.from_crs("EPSG:4326", projdef, always_xy=True).transform(ll_lon, ll_lat)
    x_ur, y_ur = Transformer.from_crs("EPSG:4326", projdef, always_xy=True).transform(ur_lon, ur_lat)

    x_1d = np.linspace(x_ll, x_ur, xsize)
    y_1d = np.linspace(y_ll, y_ur, ysize)

    # Приближение: lat/lon за всяка клетка
    # За по-голяма скорост ползваме само 1D profiles по центъра
    cx = (x_ll + x_ur) / 2
    cy = (y_ll + y_ur) / 2

    # lon varies with x, lat varies with y (добро приближение за AEQD)
    _, lat_1d = transformer.transform(np.full_like(y_1d, cx), y_1d)
    lon_1d, _ = transformer.transform(x_1d, np.full_like(x_1d, cy))

    # ODIM HDF5: row 0 = север, но lat_1d тръгва от юг → flip данните
    if lat_1d[0] < lat_1d[-1]:
        dbz = dbz[::-1, :]

    logger.info(f"  {timestamp}  {ysize}×{xsize}  "
                f"({ll_lat:.1f}-{ur_lat:.1f}°N, {ll_lon:.1f}-{ur_lon:.1f}°E)  "
                f"max dBZ={np.nanmax(dbz):.1f}")

    return {
        "timestamp": timestamp,
        "dbz": dbz.astype(np.float32),
        "lat": lat_1d.astype(np.float64),
        "lon": lon_1d.astype(np.float64),
        "source": "romania_composite",
    }


def download_romania_composite() -> str | None:
    """
    Опитва да свали последния COMPOSITE от meteoromania.ro/radar/.
    Връща локалния път или None.
    """
    import requests
    from bs4 import BeautifulSoup

    url = ROMANIA["base_url"]
    logger.info(f"Сваляне от {url}")

    try:
        resp = requests.get(url, timeout=20)
        resp.raise_for_status()
    except Exception as e:
        logger.error(f"Грешка при достъп до {url}: {e}")
        return None

    # Парсвай Index of /radar/COMPOSITE/ за dBZ файлове
    links = re.findall(r'(COMPOSITE_\d+dBZ\.hdf)', resp.text)

    if not links:
        logger.warning("Няма COMPOSITE файлове на сайта!")
        return None

    links.sort()
    # Свали последните 6 файла
    for fname in links[-6:]:
        file_url = url.rstrip('/') + '/' + fname
        local_path = os.path.join(ROMANIA_DIR, os.path.basename(fname))
        if os.path.exists(local_path):
            continue
        logger.info(f"  Сваляне: {fname}")
        try:
            resp = requests.get(file_url, timeout=60)
            resp.raise_for_status()
            with open(local_path, 'wb') as f:
                f.write(resp.content)
            logger.info(f"  Запазен: {local_path}")
        except Exception as e:
            logger.error(f"  Грешка: {e}")
    return ROMANIA_DIR


def find_local_composites(n_latest: int = 5) -> list[str]:
    """Намира последните N локални COMPOSITE файла."""
    files = sorted([
        os.path.join(ROMANIA_DIR, f)
        for f in os.listdir(ROMANIA_DIR)
        if f.startswith("COMPOSITE") and f.endswith(".hdf")
    ])
    return files[-n_latest:]


def ingest_romania(n_frames: int = 5) -> list[dict]:
    """
    Пълен pipeline: свали + прочети последните composites.
    """
    logger.info("=== Ingest Румъния ===")

    # Опит за сваляне на последния
    download_romania_composite()

    # Прочети локалните файлове
    files = find_local_composites(n_frames)
    if not files:
        logger.warning("Няма румънски composite файлове!")
        return []

    frames = []
    for fpath in files:
        frame = read_romania_composite(fpath)
        if frame is not None:
            frames.append(frame)

    logger.info(f"  {len(frames)} frames прочетени")
    return frames


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(name)s] %(message)s")
    frames = ingest_romania(n_frames=3)
    for f in frames:
        print(f"  {f['timestamp']}  {f['dbz'].shape}  "
              f"max={np.nanmax(f['dbz']):.1f}")
