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

Второ правило — по височината на върха на ехото (ETOP, файловете ..._Height.hdf на
румънския композит), за по-широки петна (21.10.2026 сутрин — инверсия, лъчът хваща
целия склон, а не само билото):
  • пикселът има ехо в ≥ static_min_frames от кадрите на серията, почти без промяна
    (std < static_max_std dB) — неподвижно като земята;
  • и в румънското покритие на ETOP няма връх в нито един кадър (земното отражение
    почти никога няма ETOP, истинският валеж почти винаги има) — или върхът е под
    static_low_top_km, но пикселът вече е „подозрителен“ по статистиката (freq ≥ static_freq).
  Само на север от 43° N (зоната на румънския радар). Мерено на 21.10 02:20–05:10 UTC:
  махнати 79% от фалшивото ехо по склона на Стара планина.
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


def _etop_on_grid(ts: dt.datetime, lat: np.ndarray, lon: np.ndarray):
    """ETOP (km) от румънския ..._Height.hdf за час ts върху нашата решетка.
    Връща (етоп, покритие): етоп = NaN където няма връх; покритие = вътре в продукта.
    None, ако файлът липсва."""
    import h5py
    from pyproj import Proj
    from src.ingest.radar_romania import HEIGHT_DIR
    fn = os.path.join(HEIGHT_DIR, f"COMPOSITE_{ts:%Y%m%d%H%M%S}00Height.hdf")
    if not os.path.exists(fn):
        return None
    with h5py.File(fn, "r") as h:
        w = h["where"].attrs
        dw = h["dataset1/what"].attrs
        raw = h["dataset1/data1/data"][:]
        pd = w["projdef"]; pd = pd.decode() if isinstance(pd, bytes) else str(pd)
        nod, und = float(dw["nodata"]), float(dw["undetect"])
        val = np.where((raw != nod) & (raw != und), raw * float(dw["gain"]) + float(dw["offset"]), np.nan)
        cov = raw != nod
        p = Proj(pd)
        x0, y0 = p(float(w["UL_lon"]), float(w["UL_lat"]))
        dx, dy = float(w["xscale"]), float(w["yscale"])
    LO, LA = np.meshgrid(lon, lat)
    x, y = p(LO, LA)
    j = np.round((x - x0) / dx - 0.5).astype(int)
    i = np.round((y0 - y) / dy - 0.5).astype(int)
    ok = (i >= 0) & (i < val.shape[0]) & (j >= 0) & (j < val.shape[1])
    et = np.full(LO.shape, np.nan, np.float32)
    cv = np.zeros(LO.shape, bool)
    et[ok], cv[ok] = val[i[ok], j[ok]], cov[i[ok], j[ok]]
    return et, cv


def static_mask(composites: list[dict], freq: np.ndarray, cfg: dict = None):
    """Неподвижно ехо без връх (или с нисък връх на „подозрителен“ пиксел) — по серията."""
    cfg = cfg or CLUTTER
    lat, lon = composites[-1]["lat"], composites[-1]["lon"]
    tops, covs, dbzs = [], [], []
    for c in composites:
        r = _etop_on_grid(c["timestamp"], lat, lon)
        if r is None:
            continue
        tops.append(r[0]); covs.append(r[1]); dbzs.append(c["dbz"])
    if len(dbzs) < cfg["static_min_frames"]:
        return None, (len(dbzs), float("nan"))
    Z = np.array(dbzs, dtype=np.float32)
    e = np.nan_to_num(Z, nan=-99.0) >= cfg["echo_dbz"]
    cnt = e.sum(0)
    import warnings
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        sd = np.nanstd(np.where(e, Z, np.nan), axis=0)
        T = np.where(e, np.array(tops), np.nan)
        anytop = np.isfinite(T).any(0)
        hmx = np.nanmax(np.where(np.isfinite(T), T, -1.0), axis=0)
    in_ro = np.array(covs).all(0) & (lat[:, None] >= cfg["static_lat_min"])
    low = ~anytop | ((hmx < cfg["static_low_top_km"]) & (freq >= cfg["static_freq"]))
    m = (cnt >= cfg["static_min_frames"]) & (np.nan_to_num(sd, nan=99.0) < cfg["static_max_std"]) & low & in_ro
    if m.any():
        m = binary_dilation(m, iterations=1) & in_ro
    # за проверка при истински валеж: какъв дял от ехото в последния кадър има връх
    last = e[-1] & in_ro
    share = float(np.isfinite(T[-1][last]).mean()) if last.any() else float("nan")
    return m, (len(dbzs), share)


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


def sea_filter(composites: list[dict]) -> None:
    """Черно море (ROMANIA_SEA): под min_dbz се маха само ехо без връх (ETOP).
    Без ETOP файл за кадъра — твърдият праг както преди."""
    from config.settings import ROMANIA_SEA as SEA
    if SEA.get("mode") != "no_top" or not composites:
        return
    lat, lon = composites[-1]["lat"], composites[-1]["lon"]
    box = ((lat[:, None] >= SEA["lat"][0]) & (lat[:, None] <= SEA["lat"][1]) &
           (lon[None, :] >= SEA["lon"][0]) & (lon[None, :] <= SEA["lon"][1]))
    kept, cut, hard = [], [], 0
    for c in composites:
        d = c["dbz"]
        weak = box & np.isfinite(d) & (d > 0) & (d < SEA["min_dbz"])
        r = _etop_on_grid(c["timestamp"], lat, lon)
        if r is None:
            m, hard = weak, hard + 1
        else:
            m = weak & ~np.isfinite(r[0])
        kept.append(int((weak & ~m).sum())); cut.append(int(m.sum()))
        d[m] = 0.0
    logger.info(f"  Море: под {SEA['min_dbz']:.0f} dBZ без връх махнато {cut}, с връх оставено {kept}"
                + (f" (без ETOP — твърд праг в {hard} кадъра)" if hard else ""))


def filter_clutter(composites: list[dict], li_frames: list[dict] = None) -> None:
    """Обновява статистиката с новите кадри и маха clutter от всички composites (на място)."""
    if not composites:
        return
    try:
        sea_filter(composites)                # първо морето — да не влиза в статистиката
    except Exception as e:
        logger.warning(f"  Море: {e}")
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
    n_learn = int(mask.sum())
    try:
        st, (n_et, top_share) = static_mask(composites, freq)
    except Exception as e:
        st, n_et, top_share = None, 0, float("nan")
        logger.warning(f"  Clutter: ETOP правилото не мина ({e})")
    n_static = int(st.sum()) if st is not None else 0
    st_note = ""
    rain_mode = bool(np.isfinite(top_share) and top_share > cfg["static_max_top_share"])
    if st is not None and rain_mode:
        st_note = (f" — ИЗКЛЮЧЕНО: {top_share:.0%} от ехото има връх (валеж); "
                   f"маската пази ехото с връх")
        st = None
    elif st is not None and not cfg.get("static_enabled", True):
        st_note = " — само за лога (static_enabled=False)"
        st = None
    if st is not None:
        mask = mask | st
    if not mask.any():
        logger.info(f"  Clutter: няма маркирани пиксели (статистика: {src}, +{n_new} кадъра; "
                    f"ETOP кадри: {n_et}){st_note}")
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
        if rain_mode:                                    # при валеж: маха се само ехо без връх
            r = _etop_on_grid(comp["timestamp"], lat, lon)
            if r is not None:
                m &= ~np.isfinite(r[0])
        d[m] = 0.0                                       # в покритието, но сухо
        removed.append(int(m.sum()))
    logger.info(f"  Clutter: маркирани {int(mask.sum())} px (по статистика {n_learn}, неподвижно ехо "
                f"без връх {n_static}; статистика: {src}, +{n_new} кадъра; ETOP кадри: {n_et}, "
                f"ехо с връх: {top_share:.0%}{st_note}); "
                f"махнато ехо по кадри: {removed}")
