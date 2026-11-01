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

Трето — шипове и единични точки (spike_filter, 30.10.2026), по всеки кадър и целия домейн:
  • шип = ядро ≥ spike_dbz от най-много spike_max_px пиксела, поне spike_db над медианата
    на околността spike_box×spike_box → става медианата. Освен ако е растяща клетка: в
    предишния кадър (10 мин) наблизо (spike_prev_box) е имало ехо най-много spike_growth_db
    по-слабо. Истински Cb не изскача от нищо до 60 dBZ за 10 мин и с ядро ≤ 4 km²;
    по-голямо ядро, мълнии или растеж отпреди → не се пипа. Истинското ядро расте постепенно (~5–10 dB/km), не с 20+ dB в един
    пиксел. 30.10 06:00: 62 dBZ в 2 px на вр. Ком сред сухо → ANVIL го размаза до 🔴 на +60;
    също 51 dBZ в морето при Констанца, 56 dBZ сред 20–25 dBZ дъжд в Северозапада;
  • петно ехо (8-свързано) под speck_min_px пиксела → сухо (картата и без това не ги рисува,
    но прогнозата ги местеше и размазваше);
  • където има мълнии — не се пипа.
"""

import os, sys, logging
import datetime as dt
import numpy as np
from scipy.ndimage import (uniform_filter, binary_dilation, median_filter, maximum_filter,
                           label, generate_binary_structure)

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


def clutter_mask(freq: np.ndarray, cfg: dict = None, cov: np.ndarray = None) -> np.ndarray:
    """cov — пикселите с радар: околността се осреднява само по тях. Иначе на ръба на
    покритието (43° N без ИАБГ, краищата на домейна) половината околност е „суха“ и
    продължителен валеж там минава за земно отражение (01.11.2026: в тест 585 px по
    ивицата 43.00–43.15° N); с ИАБГ на юг (април) ефектът изчезва — маската щеше да
    сменя поведението си между сезоните."""
    cfg = cfg or CLUTTER
    f = freq.astype(np.float32)
    if cov is None:
        local = uniform_filter(f, size=cfg["box_px"], mode="constant")
    else:
        num = uniform_filter(np.where(cov, f, 0.0).astype(np.float32), size=cfg["box_px"], mode="constant")
        den = uniform_filter(cov.astype(np.float32), size=cfg["box_px"], mode="constant")
        local = np.where(den > 0.05, num / np.maximum(den, 1e-6), 0.0)
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
    Без ETOP файл за кадъра — твърдият праг както преди.
    Само в зоната на румънския радар: ETOP е негов продукт, а морските отражения са
    негови. Южно от 43° N морето е на Старо село (от април) — там не се пипа, иначе
    всичко под 30 dBZ източно от 28° E (дъждът към Бургас) изчезваше."""
    from config.settings import ROMANIA_SEA as SEA
    from src.composite.composite import coverage_mask
    if SEA.get("mode") != "no_top" or not composites:
        return
    lat, lon = composites[-1]["lat"], composites[-1]["lon"]
    lon2d, lat2d = np.meshgrid(lon, lat)
    box = ((lat2d >= SEA["lat"][0]) & (lat2d <= SEA["lat"][1]) &
           (lon2d >= SEA["lon"][0]) & (lon2d <= SEA["lon"][1]) &
           coverage_mask("romania_composite", lat2d, lon2d))
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
        w = c.get("dbz_weak")                    # слабото ехо 7–15 (за картата): същото правило
        if w is not None:
            ww = box & np.isfinite(w)
            w[ww if r is None else (ww & ~np.isfinite(r[0]))] = np.nan
    logger.info(f"  Море: под {SEA['min_dbz']:.0f} dBZ без връх махнато {cut}, с връх оставено {kept}"
                + (f" (без ETOP — твърд праг в {hard} кадъра)" if hard else ""))


def spike_filter(composites: list[dict], ltg_fn=None, cfg: dict = None) -> None:
    """Шипове → медианата на околността; петна под speck_min_px → сухо. На място."""
    cfg = cfg or CLUTTER
    if not composites or not cfg.get("spike_enabled", True):
        return
    s8 = generate_binary_structure(2, 2)
    n_spk, n_spe, n_grow, mx = [], [], 0, 0.0
    prev = None                                   # предишният кадър, преди филтъра
    order = sorted(range(len(composites)), key=lambda i: composites[i]["timestamp"] or 0)
    for c in (composites[i] for i in order):
        d = c["dbz"]
        cov = np.isfinite(d)
        z = np.where(cov, d, 0.0)
        z_orig = z.copy()
        keep = np.zeros(d.shape, bool)
        if ltg_fn is not None:
            cnt = ltg_fn(c)
            if cnt is not None:
                keep = cnt >= 1
        med = median_filter(z, size=cfg["spike_box"])
        hot = cov & (z >= cfg["spike_dbz"])
        lab_h, n_h = label(hot, structure=s8)
        small = np.zeros(d.shape, bool)
        if n_h:
            small = (lab_h > 0) & (np.bincount(lab_h.ravel())[lab_h] <= cfg["spike_max_px"])
        spk = small & (z - med >= cfg["spike_db"]) & ~keep
        if prev is not None and spk.any():
            # растяща клетка: 10 мин по-рано наблизо е имало ехо до spike_growth_db по-слабо
            grow = spk & (maximum_filter(prev, size=cfg["spike_prev_box"]) >= z - cfg["spike_growth_db"])
            n_grow += int(grow.sum())
            spk &= ~grow
        prev = z_orig
        if spk.any():
            mx = max(mx, float(z[spk].max()))
            d[spk] = med[spk]
        echo = cov & (np.where(cov, d, 0.0) >= cfg["echo_dbz"])
        lab, n = label(echo, structure=s8)
        spe = np.zeros(d.shape, bool)
        if n:
            size = np.bincount(lab.ravel())
            spe = (lab > 0) & (size[lab] < cfg["speck_min_px"]) & ~keep
            d[spe] = 0.0
        n_spk.append(int(spk.sum())); n_spe.append(int(spe.sum()))
    logger.info(f"  Шипове: {n_spk} px" + (f" (max {mx:.0f} dBZ)" if mx else "")
                + (f", оставени като растящи клетки: {n_grow} px" if n_grow else "")
                + f"; петна под {cfg['speck_min_px']} px: {n_spe} px")


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

    ltg_fn = None
    if li_frames:
        from src.nowcast.lightning_boost import li_window
        ltg_fn = lambda c: li_window(li_frames, c["timestamp"], lat, lon)[0]
    try:
        spike_filter(composites, ltg_fn)      # шипове и единични точки — също преди статистиката
    except Exception as e:
        logger.warning(f"  Шипове: {e}")
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

    mask = clutter_mask(freq, cov=np.isfinite(composites[-1]["dbz"]))
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
        if comp.get("dbz_weak") is not None:             # слабото ехо (за картата) — също
            comp["dbz_weak"][mask] = np.nan
    logger.info(f"  Clutter: маркирани {int(mask.sum())} px (по статистика {n_learn}, неподвижно ехо "
                f"без връх {n_static}; статистика: {src}, +{n_new} кадъра; ETOP кадри: {n_et}, "
                f"ехо с връх: {top_share:.0%}{st_note}); "
                f"махнато ехо по кадри: {removed}")
