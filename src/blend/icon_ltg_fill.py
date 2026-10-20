"""
Резерв №3 за зоната без радар: ICON при +0, коригиран по мълниите (MTG LI)
=========================================================================
Включва се, когато H40B липсва или е твърде стар (и кешът също). Веригата за
зоната без радар: H40B → кеш на H40B → ICON + LI → (без LI) само ICON.

Корекцията (прагове в settings.ICON_LTG):
  1. Основа: ICON dBZ за часа на наблюдението (същото като в blend-а: фон от
     общия валеж + ядра от LPI).
  2. Мълниите от последните 15 мин (разширени с dilate_km) → групи.
  3. За всяка група: конвективна клетка на ICON (≥ conv_dbz) в радиус radius_km
     → клетката се премества с центъра си върху мълниите; на старото ѝ място
     остава фонът около нея (≤ strat_dbz).
  4. Върху мълниите: ≥ 45 dBZ, при интензивни ≥ 55 (както при радара); група
     без клетка на ICON наблизо → ново ядро на мястото на мълниите.
  5. Конвективно ядро на ICON без мълнии в radius_km → най-много cap_no_ltg_dbz
     (жълто, не червено): моделът често разминава конвекцията.
  6. Слоестият валеж без мълнии остава какъвто е в модела.

Интерфейсът е като на SatelliteLayer (time, z_at, label), за да го ползват
blend-ът и картата без промени.
"""

import os, sys, logging
import datetime as dt
import numpy as np
from scipy.ndimage import label, center_of_mass, distance_transform_edt, map_coordinates, binary_dilation

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from config.settings import ICON_LTG, LIGHTNING_BOOST

logger = logging.getLogger("blend.icon_ltg")


def correct_icon_with_lightning(dbz_icon: np.ndarray, count: np.ndarray | None,
                                zone: np.ndarray, km_per_px: float, cfg: dict = None):
    """
    dbz_icon: ICON dBZ (NaN = без валеж); count: мълнии в прозореца (разширени), или None;
    zone: True там, където няма радар (само там се търсят групи мълнии).
    Връща (ново dBZ поле, статистика).
    """
    cfg = cfg or ICON_LTG
    out = np.array(dbz_icon, dtype=np.float32, copy=True)
    stats = {"moved": 0, "shift_km": [], "new": 0, "weakened": 0, "groups": 0}
    if count is None:
        return out, stats

    d0 = np.nan_to_num(out, nan=-99.0)
    ny, nx = out.shape
    R = cfg["radius_km"] / km_per_px

    ltg = (count >= 1) & zone
    lab_l, n_l = label(ltg)
    # клетка = ядро ≥ conv_dbz + „раменете“ ѝ (до 8 dB по-слабо, до ~6 px навън) — местят се заедно
    core = d0 >= cfg["conv_dbz"]
    conv = binary_dilation(core, iterations=6, mask=d0 >= cfg["conv_dbz"] - 8.0) | core
    lab_c, n_c = label(conv)
    stats["groups"] = n_l
    moved = set()

    for j in range(1, n_l + 1):
        mj = lab_l == j
        if mj.sum() < cfg["min_group_px"]:
            continue
        cy, cx = center_of_mass(mj)
        # прозорец около групата ± R — разстоянието се смята само там
        ys, xs = np.nonzero(mj)
        y0, y1 = max(0, int(ys.min() - R - 1)), min(ny, int(ys.max() + R + 2))
        x0, x1 = max(0, int(xs.min() - R - 1)), min(nx, int(xs.max() + R + 2))
        dist = distance_transform_edt(~mj[y0:y1, x0:x1])
        cand = lab_c[y0:y1, x0:x1]
        best, best_d = 0, None
        for k in np.unique(cand[(cand > 0) & (dist <= R)]):
            if k in moved:
                continue
            dk = float(dist[cand == k].min())
            if best_d is None or dk < best_d:
                best, best_d = int(k), dk
        if best:
            obj = lab_c == best
            oy, ox = center_of_mass(obj)
            dy, dx = int(round(cy - oy)), int(round(cx - ox))
            py, px = np.nonzero(obj)
            vals = d0[py, px]
            # старото място → фонът около клетката (медиана на пръстен от 3 px), не по-силен от strat_dbz
            ring = binary_dilation(obj, iterations=3) & ~obj
            rv = d0[ring]; rv = rv[rv > -99]
            fill = min(float(np.median(rv)), cfg["strat_dbz"]) if rv.size else np.nan
            out[obj] = fill
            ty, tx = py + dy, px + dx
            ok = (ty >= 0) & (ty < ny) & (tx >= 0) & (tx < nx)
            cur = np.nan_to_num(out[ty[ok], tx[ok]], nan=-99.0)
            out[ty[ok], tx[ok]] = np.maximum(cur, vals[ok])
            moved.add(best)
            stats["moved"] += 1
            stats["shift_km"].append(float(np.hypot(dy, dx)) * km_per_px)
        else:
            stats["new"] += 1

    # усилване върху мълниите (единични / интензивни), както при радара
    single = (count >= 1) & zone
    intense = (count >= LIGHTNING_BOOST["intense_count"]) & zone
    cur = np.nan_to_num(out, nan=-99.0)
    cur[single] = np.maximum(cur[single], LIGHTNING_BOOST["single_dbz"])
    cur[intense] = np.maximum(cur[intense], LIGHTNING_BOOST["intense_dbz"])

    # ядра на ICON без мълнии наблизо → таван (само в зоната без радар)
    if n_c:
        dist_ltg = distance_transform_edt(~(count >= 1)) if (count >= 1).any() else np.full(out.shape, np.inf)
        for k in range(1, n_c + 1):
            if k in moved:
                continue
            obj = (lab_c == k) & zone
            if not obj.any() or float(dist_ltg[obj].min()) <= R:
                continue
            hot = obj & (cur > cfg["cap_no_ltg_dbz"])
            if hot.any():
                cur[hot] = cfg["cap_no_ltg_dbz"]
                stats["weakened"] += 1

    out = np.where(cur > -99, cur, np.nan).astype(np.float32)
    return out, stats


class IconLtgLayer:
    """ICON (+ LI корекция) като Z поле, което се мести по V до всеки целеви момент."""

    def __init__(self, icon_data: dict, li_count, t0: dt.datetime, V: np.ndarray, ts_min: float,
                 target_lat: np.ndarray, target_lon: np.ndarray, cov_w: np.ndarray):
        from src.blend.blending import interpolate_icon

        km_per_px = abs(float(target_lat[1] - target_lat[0])) * 111.0
        dbz0 = interpolate_icon(t0, icon_data, target_lat, target_lon)   # кеширано — не се пипа
        zone = cov_w < 0.5
        dbz, st = correct_icon_with_lightning(dbz0, li_count, zone, km_per_px)

        self.z = np.where(np.isfinite(dbz), 10.0 ** (dbz / 10.0), 0.0).astype(np.float32)
        self.time = t0
        self.with_ltg = li_count is not None
        self.label = "модел ICON, коригиран по мълнии" if self.with_ltg else "модел ICON"
        self.stats = st
        self._li, self._zone, self._km = li_count, zone, km_per_px
        self._corr_cache = {}

        self.vx = (V[0] / ts_min).astype(np.float32)
        self.vy = (V[1] / ts_min).astype(np.float32)
        ny, nx = self.z.shape
        self.yy, self.xx = np.mgrid[0:ny, 0:nx].astype(np.float32)

        run = icon_data.get("run")
        if self.with_ltg:
            sh = st["shift_km"]
            logger.info(f"Резерв ICON+LI ({t0:%H:%M} UTC"
                        + (f", ICON run {run:%d.%m %H} UTC" if run else "") + f"): групи мълнии {st['groups']}, "
                        f"клетки преместени {st['moved']}"
                        + (f" (средно {np.mean(sh):.0f} km, max {max(sh):.0f} km)" if sh else "")
                        + f", нови ядра {st['new']}, отслабени ядра без мълнии {st['weakened']}")
        else:
            logger.info(f"Резерв ICON ({t0:%H:%M} UTC): без мълнии — само моделът")

    def correct_icon(self, dbz_icon: np.ndarray, target_time: dt.datetime) -> np.ndarray:
        """
        Същата корекция върху ICON за по-късен момент (за blend-а): мълниите се местят
        с V до target_time и ICON се съпоставя с тях. Без LI — ICON без промяна.
        Кеш по (час на ICON, 30-мин интервал).
        """
        if self._li is None:
            return dbz_icon
        k = (target_time - self.time).total_seconds() / 60.0
        key = (id(dbz_icon), int(k // 30))
        if key not in self._corr_cache:
            kk = (key[1] + 0.5) * 30.0
            coords = [self.yy - self.vy * kk, self.xx - self.vx * kk]
            li = map_coordinates(self._li, coords, order=0, mode="constant", cval=0.0)
            self._corr_cache[key], _ = correct_icon_with_lightning(dbz_icon, li, self._zone, self._km)
        return self._corr_cache[key]

    def z_at(self, target_time: dt.datetime) -> np.ndarray:
        """Z полето, преместено от t0 до target_time (обратна траектория)."""
        k = (target_time - self.time).total_seconds() / 60.0
        coords = [self.yy - self.vy * k, self.xx - self.vx * k]
        return map_coordinates(self.z, coords, order=1, mode="constant", cval=0.0)
