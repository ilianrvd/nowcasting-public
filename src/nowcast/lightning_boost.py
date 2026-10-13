"""
Усилване на наблюдението там, където има мълнии
===============================================
Просто правило, еднакво за радара и за сателита (H40B):

  където има валеж (сигнал) и мълнии наблизо:
      dBZ = max(dBZ, 45)   при единични мълнии
      dBZ = max(dBZ, 55)   при интензивни мълнии (≥ intense_count светкавици/5 мин)

Никъде не се намалява и не се добавя ехо, където няма сигнал.

Тайминг: всеки кадър взима мълниите от своите последни window_min минути
(максимумът на 5-мин кадрите на MTG LI в (T − window_min, T], т.е. 3 кадъра).
Разминаването е до ~15 мин → маската на мълниите се разширява с dilate_km,
вместо да се мести отделно.

Усилването се прави ПРЕДИ прогнозата, върху наблюдението — после ядрата
се движат и развиват заедно с радара (S-PROG) или със сателитния слой.
Прагове: settings.LIGHTNING_BOOST.
"""

import os, sys, logging
import datetime as dt
import numpy as np
from scipy.ndimage import maximum_filter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from config.settings import LIGHTNING_BOOST as CFG

logger = logging.getLogger("nowcast.ltg")


def li_window(li_frames: list[dict], t_end: dt.datetime,
              target_lat: np.ndarray, target_lon: np.ndarray, dilate: bool = True):
    """
    Мълниите за кадър с час t_end върху целевата решетка:
    максимум на 5-мин кадрите в (t_end − window_min, t_end], разширен с dilate_km.
    Връща (count, брой ползвани LI кадри); count е None, ако няма кадри в прозореца.
    """
    from src.ingest.eumetview import regrid

    t0 = t_end - dt.timedelta(minutes=CFG["window_min"])
    sel = [f for f in li_frames if t0 < f["time"] <= t_end]
    if not sel:
        return None, 0

    count = None
    for f in sel:
        c = regrid(f["count"], f["lat"], f["lon"], target_lat, target_lon, method="nearest")
        count = c if count is None else np.maximum(count, c)

    km_per_px = abs(float(target_lat[1] - target_lat[0])) * 111.0
    r = int(round(CFG["dilate_km"] / km_per_px))
    if r > 0 and dilate:
        count = maximum_filter(count, size=2 * r + 1)
    return count, len(sel)


def boost_dbz(dbz: np.ndarray, count: np.ndarray, signal_dbz: float,
              intense_signal_dbz: float = None):
    """
    dBZ = max(dBZ, 45/55) там, където dBZ ≥ signal_dbz и има мълнии.
    intense_signal_dbz: по-висок праг за 55 dBZ (ползва се при H40B).
    NaN (без покритие) остава NaN. Връща (ново поле, px до 45, px до 55).
    """
    out = dbz.copy()
    sig = np.isfinite(out) & (out >= signal_dbz)
    sig_int = sig if intense_signal_dbz is None else sig & (out >= intense_signal_dbz)
    single = sig & (count >= 1)
    intense = sig_int & (count >= CFG["intense_count"])
    n1 = int((single & (out < CFG["single_dbz"])).sum())
    n2 = int((intense & (out < CFG["intense_dbz"])).sum())
    out[single] = np.maximum(out[single], CFG["single_dbz"])
    out[intense] = np.maximum(out[intense], CFG["intense_dbz"])
    return out, n1, n2


def _breakdown(dbz: np.ndarray, count: np.ndarray, signal_dbz: float,
               nan_label: str = "без радар") -> str:
    """Къде попадат мълниите (след разширението) — за диагностика в лога."""
    ltg = count >= 1
    n = int(ltg.sum())
    if n == 0:
        return "мълнии в домейна: 0 px"
    nocov = int((ltg & ~np.isfinite(dbz)).sum())
    d = np.nan_to_num(dbz, nan=-99.0)
    weak = int((ltg & (d > -99) & (d < signal_dbz)).sum())
    strong = int((ltg & (d >= CFG["single_dbz"])).sum())
    return (f"мълнии {n} px: {nan_label} {nocov}, сухо/под {signal_dbz:.0f} dBZ {weak}, "
            f"вече ≥{CFG['single_dbz']:.0f} dBZ {strong}")


def boost_composites(composites: list[dict], li_frames: list[dict], n_last: int = 3):
    """
    Усилва последните n_last композита (тези, които S-PROG ползва),
    всеки с мълниите от своя прозорец. Оригиналът остава в comp["dbz_radar"]
    (за OBS панела). Връща мълниите на последния кадър (за сателита) или None.
    """
    last_count = None
    for comp in composites[-n_last:]:
        count, n_li = li_window(li_frames, comp["timestamp"], comp["lat"], comp["lon"])
        if comp is composites[-1]:
            last_count = count
            # за контура на OBS панела: мълниите без разширение
            comp["li_raw"], _ = li_window(li_frames, comp["timestamp"],
                                          comp["lat"], comp["lon"], dilate=False)
        if count is None:
            logger.info(f"  {comp['timestamp']:%H:%M}: няма LI кадри в прозореца — без усилване")
            continue
        comp.setdefault("dbz_radar", comp["dbz"].copy())
        comp["dbz"], n1, n2 = boost_dbz(comp["dbz_radar"], count, CFG["radar_signal_dbz"])
        logger.info(f"  {comp['timestamp']:%H:%M}: радар усилен от {n_li} LI кадъра — "
                    f"до {CFG['single_dbz']:.0f} dBZ: {n1} px, до {CFG['intense_dbz']:.0f} dBZ: {n2} px"
                    f"  [{_breakdown(comp['dbz_radar'], count, CFG['radar_signal_dbz'])}]")
    return last_count
