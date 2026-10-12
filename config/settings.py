"""
nowcasting_public — Конфигурация
================================
Публична nowcasting система за България.
Данни: Румъния (ODIM HDF5), ИАБГ радари (PNG), Blitzortung, ICON-EU.
"""

import os
import numpy as np

# ============================================================
# ДОМЕЙН
# ============================================================
DOMAIN = {
    "lat_min": 41.0, "lat_max": 44.5,
    "lon_min": 21.0, "lon_max": 29.5,
    "resolution_km": 1.0,
}

# ============================================================
# РУМЪНИЯ — COMPOSITE (ODIM HDF5)
# ============================================================
ROMANIA = {
    "base_url": "https://opendata.meteoromania.ro/radar/COMPOSITE/",
    "file_pattern": "COMPOSITE_{ts}dBZ.hdf",
    # Проекция AEQD: +proj=aeqd +lon_0=24.9805 +lat_0=45.9471 +ellps=sphere
    "proj_center_lat": 45.9471,
    "proj_center_lon": 24.9805,
    "xsize": 850, "ysize": 800,
    "xscale": 1472.03, "yscale": 1389.94,
    # Bounding box
    "LL_lat": 39.97, "LL_lon": 17.46,
    "UR_lat": 50.04, "UR_lon": 33.37,
    "update_min": 10,
    "enabled": True,
}

# ============================================================
# ИАБГ РАДАРИ (PNG от weathermod-bg.eu)
# ============================================================
IABG_RADARS = {
    "GCD": {
        "name": "Голям Чардак",
        "lat": 42.0836, "lon": 24.7353, "alt_m": 1515,
        "range_km": 250,
        "url_base": "https://wr.weathermod-bg.eu/data_png_mp/",
        "json_list": "https://www.weathermod-bg.eu/wr/json/GCD.json",
        "product": "CAPVXE1",
        "enabled": True,
    },
    "STS": {
        "name": "Старо Село",
        "lat": 42.6614, "lon": 26.2092, "alt_m": 692,
        "range_km": 250,
        "url_base": "https://wr.weathermod-bg.eu/data_png_mp/",
        "json_list": "https://www.weathermod-bg.eu/wr/json/STS.json",
        "product": "CAPVXE1",
        "enabled": True,
    },
    "BRD": {
        "name": "Бърдарски геран",
        "lat": 43.3833, "lon": 24.1000, "alt_m": 180,
        "range_km": 250,
        "url_base": "https://wr.weathermod-bg.eu/data_png_mp/",
        "json_list": "https://www.weathermod-bg.eu/wr/json/BRD.json",
        "product": "CAPVXE1",
        "enabled": False,   # северът е само RO (план 2026-09-25)
    },
}

# ============================================================
# ПОКРИТИЕ — един източник на регион, без припокриване
# ============================================================
# Север (>=43°N): само RO. Юг (<43°N): GCD западно, STS източно от 26°E.
# За ИАБГ зоната се реже допълнително до range_km около радара (composite.py).
COVERAGE_MASKS = {
    "romania_composite": {
        "lat_min": 43.0,   # север: само RO
    },
    "iabg_GCD": {
        "lat_max": 43.0,   # юг, западно от 26°E
        "lon_max": 26.0,
    },
    "iabg_STS": {
        "lat_max": 43.0,   # юг, източно от 26°E
        "lon_min": 26.0,
    },
    "rainviewer_TR34": {
        "lat_max": 43.0,   # същата зона като STS — само като негов резерв
        "lon_min": 26.0,
    },
}

# ============================================================
# РЕЗЕРВНИ РАДАРИ (изходни картинки през RainViewer)
# ============================================================
# Ползват се само когато основният източник (backup_for) е застоял
# или липсва — изборът е за целия пуск.
RAINVIEWER_RADARS = {
    "TR34": {
        "name": "Истанбул (MGM)",
        "lat": 41.348, "lon": 28.367,      # калибрирано по 20 града, RMS ≈ 2 km
        "range_km": 370,
        "backup_for": "iabg_STS",
        "enabled": True,
    },
}

# RGB → dBZ lookup (стандартна метео NWS скала от weathermod-bg.eu)
COLORMAP_RGB_TO_DBZ = [
    ((166, 247, 255), 30,  5.0),   # много светло синьо
    (( 41, 215, 255), 30, 15.0),   # светло циан
    (( 69, 174, 250), 30, 20.0),   # синьо
    (( 36, 255,  36), 30, 25.0),   # ярко зелено
    ((  0, 192,  55), 30, 30.0),   # зелено
    ((  0, 130,  39), 30, 35.0),   # тъмно зелено
    ((255, 255,  30), 30, 40.0),   # жълто
    ((255, 174,   0), 30, 45.0),   # оранжево
    ((237,   0,   0), 30, 50.0),   # червено
]

# ============================================================
# УСИЛВАНЕ С МЪЛНИИ (MTG LI) — радар и H40B, преди прогнозата
# ============================================================
# Където има сигнал и мълнии наблизо: dBZ = max(dBZ, праг).
LIGHTNING_BOOST = {
    "window_min": 15,          # мълниите за кадър T: 5-мин LI кадрите в (T-15, T] — 3 кадъра
    "dilate_km": 5,            # разширение на маската (разминаване във времето/LI пиксел)
    "radar_signal_dbz": 20.0,  # радарът „е дал сигнал“
    "sat_signal_mmh": 0.3,     # H40B „е дал сигнал“ (≈15 dBZ)
    "single_dbz": 45.0,        # единични мълнии
    "intense_dbz": 55.0,       # интензивни мълнии
    "intense_count": 5,        # светкавици/5 мин в пиксел за „интензивни“
}

# ============================================================
# BLITZORTUNG (не се ползва — заменен от MTG LI)
# ============================================================
BLITZORTUNG = {
    "ws_url": "wss://ws1.blitzortung.org",
    "subscribe_msg": '{"a": 111}',
    "bbox": {
        "lat_min": 41.0, "lat_max": 44.5,
        "lon_min": 21.0, "lon_max": 29.5,
    },
    "collect_seconds": 60,
}

# ============================================================
# ICON-EU (за blending, от Open-Meteo)
# ============================================================
ICON = {
    "api_url": "https://api.open-meteo.com/v1/dwd-icon",
    "model": "icon_eu",
    "variables": ["precipitation"],
    "forecast_hours": 7,   # 6h + 1 буфер
    "center_lat": 42.75,
    "center_lon": 25.50,
    # Z-R: Z = 200 * R^1.6 (Marshall-Palmer)
    "zr_a": 200.0,
    "zr_b": 1.6,
}

# ============================================================
# ICON-EU директно от DWD Open Data (GRIB2)
# ============================================================
ICON_DWD = {
    "base_url": "https://opendata.dwd.de/weather/nwp/icon-eu/grib",
    "keep_runs_hours": 24,          # локален кеш на GRIB файловете
    # LPI → dBZ за ядрата (плавно между опорните точки)
    "lpi_anchors": [20.0, 30.0, 40.0, 45.0],
    "dbz_anchors": [35.0, 50.0, 55.0, 65.0],
    "core_min_showers": 0.5,        # mm/h — ядро само върху такъв фон
}

# ============================================================
# NOWCASTING
# ============================================================
NOWCAST = {
    "timestep_min": 5,
    "n_leadtimes": 12,       # 12 × 5 = 60 мин S-PROG
    "n_cascade_levels": 6,
    "min_dbz": 10.0,
    # Движение (Lucas-Kanade): само ехо ≥ този праг — слабите петна не се следят
    "motion_min_dbz": 20.0,
    # Проверка срещу вятъра на 700 hPa от ICON: при голямо отклонение V = вятърът
    "wind_check": {
        "speed_factor": 0.7,        # системите се движат с ~0.7 от водещия поток
        "max_dir_diff_deg": 45.0,   # разлика в посоката (само ако и двете са ≥ min_speed_kt)
        "max_speed_ratio": 2.0,     # LK/ICON извън [1/2, 2] …
        "min_speed_diff_kt": 10.0,  # … и разлика поне толкова (за да не реагира при слаб вятър)
        "min_speed_kt": 8.0,
    },
}

# ============================================================
# BLENDING
# ============================================================
BLEND = {
    "total_hours": 6,
    "timestep_min": 5,
    "n_steps": 72,           # 6h × 12 стъпки/час
}

# ============================================================
# ПЪТИЩА
# ============================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR     = os.path.join(BASE_DIR, "data")
RADAR_DIR    = os.path.join(DATA_DIR, "radar")
ROMANIA_DIR  = os.path.join(RADAR_DIR, "romania")
LIGHTNING_DIR = os.path.join(DATA_DIR, "lightning")
ICON_DIR     = os.path.join(DATA_DIR, "icon")
OUTPUT_DIR   = os.path.join(DATA_DIR, "output")
MAPS_DIR     = os.path.join(OUTPUT_DIR, "maps")
LOG_DIR      = os.path.join(BASE_DIR, "logs")
PAGES_DIR    = os.path.join(BASE_DIR, "docs")

for d in [RADAR_DIR, ROMANIA_DIR, LIGHTNING_DIR, ICON_DIR,
          OUTPUT_DIR, MAPS_DIR, LOG_DIR, PAGES_DIR]:
    os.makedirs(d, exist_ok=True)