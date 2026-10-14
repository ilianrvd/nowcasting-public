"""
nowcasting_public — Конфигурация
================================
Публична nowcasting система за България.
Данни: Румъния (ODIM HDF5), ИАБГ радари (PNG, в сезона), Истанбул TR34 (RainViewer),
EUMETSAT MTG LI (мълнии) и H SAF H40B (валеж), DWD ICON-EU.
"""

import os

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
# САТЕЛИТ H40B — за зоната без радар
# ============================================================
SATELLITE = {
    "min_rate_mmh": 0.3,   # под това H40B не се показва (ситнежът му е твърде широк)
    "ramp_px": 40,         # плавен преход радар↔сателит (px ≈ km) — без права граница
}

# ============================================================
# УСИЛВАНЕ С МЪЛНИИ (MTG LI) — радар и H40B, преди прогнозата
# ============================================================
# Където има сигнал и мълнии наблизо: dBZ = max(dBZ, праг).
LIGHTNING_BOOST = {
    "window_min": 15,          # мълниите за кадър T: 5-мин LI кадрите в (T-15, T] — 3 кадъра
    "dilate_km": 5,            # разширение на маската (разминаване във времето/LI пиксел)
    "radar_signal_dbz": 20.0,  # радарът „е дал сигнал“
    "sat_signal_mmh": 1.0,     # H40B „е дал сигнал“ (≈23 dBZ) — по-строго от радара:
                               # AFA покрива и наковалнята, която в H40B е слаб валеж
    "sat_intense_min_mmh": 2.0,  # 55 dBZ върху H40B само при интензивни мълнии И ≥ 2 mm/h
    "single_dbz": 45.0,        # единични мълнии
    "intense_dbz": 55.0,       # интензивни мълнии
    "intense_count": 5,        # светкавици/5 мин в пиксел за „интензивни“
}

# ============================================================
# ICON-EU от Open-Meteo — резерв, ако DWD е недостъпен; Z-R за целия blend
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
    # Таван: ядрото ≤ фон + Δ (13.10.2026, брегът Бургас–Варна: LPI≥45 при 0.5 mm/h
    # даваше 65 dBZ; с Δ=18 → 36). LPI = къде, валежът = колко силно.
    "core_max_above_bg_db": 18.0,
}

# ============================================================
# NOWCASTING
# ============================================================
NOWCAST = {
    # Стъпката идва от серията (10 мин, RO); хоризонтът е HORIZON_MIN в nowcasting.py
    "n_cascade_levels": 6,
    # Метод: "anvil" (пази малките ядра; 14.10.2026 — при KIRK 45 dBZ до +60, S-PROG
    # ги губи до +30) или "sprog". ANVIL иска ar_order+2 = 4 кадъра, иначе S-PROG.
    "method": "anvil",
    "anvil": {"ar_order": 2, "ar_window_radius": 25, "num_workers": 1},
    # Движение (Lucas-Kanade): само ехо ≥ този праг — слабите петна не се следят
    "motion_min_dbz": 20.0,
    # Проверка срещу вятъра на 700 hPa от ICON: при голямо отклонение V = вятърът
    "wind_check": {
        # Сравнение ПО РЕГИОНИ (север/ЮИ/ЮЗ). 12.10.2026: клетките в Тракия ~40 km/h,
        # ICON 700 hPa там 40–50 km/h → движението ≈ 1.0 × вятъра (не 0.7).
        "speed_factor": 1.0,
        "max_dir_diff_deg": 45.0,   # разлика в посоката (ако и двете са ≥ min_speed_kt)
        "max_speed_ratio": 1.5,     # LK/ICON извън [1/1.5, 1.5] …
        "min_speed_diff_kt": 8.0,   # … и разлика поне толкова
        "min_speed_kt": 8.0,
        "min_echo_px": 50,          # регион с по-малко ехо → движението е от ICON
        "blend_sigma_px": 25,       # плавен преход LK↔ICON между регионите (px ≈ km)
    },
}

# ============================================================
# BLENDING
# ============================================================
BLEND = {
    "n_steps": 72,           # 72 × 5 мин = 6 часа
}

# ============================================================
# ПЪТИЩА
# ============================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR     = os.path.join(BASE_DIR, "data")
RADAR_DIR    = os.path.join(DATA_DIR, "radar")
ROMANIA_DIR  = os.path.join(RADAR_DIR, "romania")
ICON_DIR     = os.path.join(DATA_DIR, "icon")
OUTPUT_DIR   = os.path.join(DATA_DIR, "output")
MAPS_DIR     = os.path.join(OUTPUT_DIR, "maps")
LOG_DIR      = os.path.join(BASE_DIR, "logs")
PAGES_DIR    = os.path.join(BASE_DIR, "docs")

for d in [RADAR_DIR, ROMANIA_DIR, ICON_DIR,
          OUTPUT_DIR, MAPS_DIR, LOG_DIR, PAGES_DIR]:
    os.makedirs(d, exist_ok=True)