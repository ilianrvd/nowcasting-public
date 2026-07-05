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
        "enabled": True,
    },
}

# ============================================================
# ПОКРИТИЕ — без припокриване
# ============================================================
COVERAGE_MASKS = {
    "romania_composite": {
        "lat_min": 42.5,
    },
    "iabg_GCD": {
        "lat_max": 43.0,   # на юг от Румъния
        "lon_max": 26.0,   # на запад от Старо Село
    },
    "iabg_STS": {
        "lat_max": 43.0,   # на юг от Румъния
        "lon_min": 26.0,   # на изток от Голям Чардак
    },
    "iabg_BRD": {
        "lat_max": 43.0,   # на юг от Румъния
        "lon_max": 26.0,   # на запад от Старо Село
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
# BLITZORTUNG
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
    "api_url": "https://historical-forecast-api.open-meteo.com/v1/forecast",
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
# NOWCASTING
# ============================================================
NOWCAST = {
    "timestep_min": 5,
    "n_leadtimes": 12,       # 12 × 5 = 60 мин S-PROG
    "n_cascade_levels": 6,
    "min_dbz": 10.0,
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
