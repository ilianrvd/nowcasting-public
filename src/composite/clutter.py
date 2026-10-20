"""
Филтър за земни отражения (clutter) в композита
================================================
Пример: тясна линия ехо по билото на Западна Стара планина (Зайчар – Връшка чука –
Ком – Петрохан) в румънския композит — стои на едно място с дни, 15–40 dBZ,
почти без промяна от кадър на кадър.

Как се разпознава (самообучаващо се):
  • за всеки пиксел се пази колко често има ехо ≥ echo_dbz — плъзгаща се средна по
    кадрите (alpha ≈ 1/72 ≈ половин ден); пази се в data/cache/clutter.npz
    (в GitHub Actions — чрез actions/cache, като кеша на H40B);
  • clutter = ехото е там почти винаги (≥ freq_min), а наоколо (прозорец box_px)
    е поне contrast пъти по-рядко. Земното отражение е тясно и постоянно;
    при продължителен дъжд в широк район околността също е мокра → нищо не се маркира;
  • в маркираните пиксели се маха само ехо под max_dbz и не там, където има мълнии
    → истинска силна клетка над билото минава.
Първоначално знание: config/clutter_seed.npz (от кадрите на 20.10.2026), за да не се
чака денонощие, докато статистиката се натрупа.
"""

import os, sys, logging
import datetime as dt
import numpy as np
from scipy.ndimage import uniform_filter, binary_dilation

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from config.settings import CLUTTER

logger = logging.getLogger("composite.clutter")

_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
STATE_FILE = os.path.join(_ROOT, "data", "cache", "clutter.npz")
SEED_FILE = os.path.join(_ROOT, "config", "clutter_seed.npz")


def clutter_mask(freq: np.ndarray, cfg: dict = None) -> np.ndarray:
    cfg = cfg or CLUTTER
    local = uniform_filter(freq.astype(np.float32), size=cfg["box_px"], mode="constant")
    m = (freq >= cfg["freq_min"]) & (freq >= cfg["contrast"] * local)
    if cfg.get("dilate_px", 0) > 0 and m.any():
        m = binary_dilation(m, iterations=cfg["dilate_px"])
    return m


def _same_grid(z, lat, lon) -> bool:
    return (z["lat"].shape == lat.shape and z["lon"].shape == lon.shape
            and np.allclose(z["lat"], lat, atol=1e-6) and np.allclose(z["lon"], lon, atol=1e-6))


def _load_state(lat, lon):
    """(freq, последен час) — от кеша; ако го няма — от началното знание; иначе нули."""
    try:
        z = np.load(STATE_FILE)
        if _same_grid(z, lat, lon):
            last = str(z["last_time"])
            return z["freq"].astype(np.float32), (dt.datetime.fromisoformat(last) if last else None), "кеш"
    except Exception:
        pass
    try:
        z = np.load(SEED_FILE)
        if _same_grid(z, lat, lon):
            return z["seed"].astype(np.float32) * CLUTTER["seed_freq"], None, "начално знание"
    except Exception:
        pass
    return np.zeros((len(lat), len(lon)), np.float32), None, "нула"


def _save_state(freq, last_time, lat, lon):
    try:
        os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
        np.savez_compressed(STATE_FILE, freq=freq.astype(np.float16), lat=lat, lon=lon,
                            last_time=last_time.isoformat() if last_time else "")
    except Exception as e:
        logger.debug(f"clutter: състоянието не е записано ({e})")


def filter_clutter(composites: list[dict], li_frames: list[dict] = None) -> None:
    """Обновява статистиката с новите кадри и маха clutter от всички composites (на място)."""
    if not composites:
        return
    cfg = CLUTTER
    lat, lon = composites[-1]["lat"], composites[-1]["lon"]
    freq, last, src = _load_state(lat, lon)

    n_new = 0
    for comp in sorted(composites, key=lambda c: c["timestamp"]):
        t = comp["timestamp"]
        if last is not None and t <= last:
            continue
        d = comp["dbz"]
        cov = np.isfinite(d)
        echo = np.nan_to_num(d, nan=-99.0) >= cfg["echo_dbz"]
        freq[cov] = (1.0 - cfg["alpha"]) * freq[cov] + cfg["alpha"] * echo[cov]
        last, n_new = t, n_new + 1
    _save_state(freq, last, lat, lon)

    mask = clutter_mask(freq)
    if not mask.any():
        logger.info(f"  Clutter: няма маркирани пиксели (статистика: {src}, +{n_new} кадъра)")
        return

    ltg_fn = None
    if li_frames:
        from src.nowcast.lightning_boost import li_window
        ltg_fn = lambda c: li_window(li_frames, c["timestamp"], lat, lon)[0]

    removed = []
    for comp in composites:
        d = comp["dbz"]
        m = mask & np.isfinite(d) & (d > 0) & (d < cfg["max_dbz"])
        if ltg_fn is not None:
            cnt = ltg_fn(comp)
            if cnt is not None:
                m &= ~(cnt >= 1)
        d[m] = 0.0                                       # в покритието, но сухо
        removed.append(int(m.sum()))
    logger.info(f"  Clutter: маркирани {int(mask.sum())} px (статистика: {src}, +{n_new} кадъра); "
                f"махнато ехо по кадри: {removed}")
