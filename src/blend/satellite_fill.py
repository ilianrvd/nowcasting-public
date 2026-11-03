"""
Сателитен слой за зоните без радарно покритие
=============================================
H40B (EUMETSAT, MTG FCI + LEO MW) → Z, преместен по вектора на движението
от S-PROG от СВОЯ час (≈25 мин закъснение) до целевия момент.
Само адвекция, без еволюция.

coverage_weight: 1 навътре в радарното покритие, линейно до 0 в последните
ramp_px пиксела към ръба му, 0 извън него. Така радарът и сателитът се
сшиват плавно, без шев.

Невидим ръб (SATELLITE["pad_km"], 03.11.2026): слоевете за попълване (H40B,
ICON+LI) се пазят на решетка, по-широка от домейна; в прогнозата на наветрения
край влиза това, което реално идва отвън. known_at(t) казва кои пиксели имат
източник вътре в разширената решетка — за останалите blend-ът взима ICON.
"""

import logging
import datetime as dt
import numpy as np
from scipy.ndimage import distance_transform_edt, map_coordinates

logger = logging.getLogger("blend.sat")


def coverage_weight(last_dbz: np.ndarray, ramp_px: int = 15) -> np.ndarray:
    """Тегло на радара по пиксели от маската на последния composite (NaN = няма радар)."""
    cov = np.isfinite(last_dbz)
    if cov.all():
        return np.ones(cov.shape, dtype=np.float32)
    if not cov.any():
        return np.zeros(cov.shape, dtype=np.float32)
    dist = distance_transform_edt(cov)          # разстояние до най-близкия пиксел без радар
    return np.clip(dist / float(ramp_px), 0.0, 1.0).astype(np.float32)


def advect_weight(layer, w0: np.ndarray, t: dt.datetime, t0: dt.datetime) -> np.ndarray:
    """
    Теглото на радара w0 (coverage_weight при t0), преместено с движението на слоя до t.
    Така всеки облак носи теглото от мястото, откъдето е тръгнал: H40B облак от зоната
    без радар остава с теглото на сателита и след като влезе в радарната зона (иначе
    угасваше на 43° N — 03.11.2026, ЮИ → N, север → W); радарна клетка, излязла от
    покритието, остава радарна. w0 е за часа на радара t0 (не на H40B); при t0 — същото
    като w0. Извън домейна → 0 (без радар).
    """
    k = (t - t0).total_seconds() / 60.0
    if abs(k) < 0.5:
        return w0
    coords = [layer.yy - layer.vy * k, layer.xx - layer.vx * k]
    return np.clip(map_coordinates(w0, coords, order=1, mode="constant", cval=0.0), 0.0, 1.0)


def padded_grid(lat: np.ndarray, lon: np.ndarray, pad: int):
    """Решетката на домейна, разширена с pad пиксела от всяка страна (същата стъпка)."""
    dlat, dlon = float(lat[1] - lat[0]), float(lon[1] - lon[0])
    k = np.arange(1, pad + 1)
    plat = np.concatenate([lat[0] - dlat * k[::-1], lat, lat[-1] + dlat * k])
    plon = np.concatenate([lon[0] - dlon * k[::-1], lon, lon[-1] + dlon * k])
    return plat, plon


def pad_px(lat: np.ndarray) -> int:
    from config.settings import SATELLITE
    km = abs(float(lat[1] - lat[0])) * 111.0
    return int(round(SATELLITE.get("pad_km", 0) / km))


class PaddedAdvection:
    """
    Поле Z на разширената решетка (self.zp, pad пиксела извън домейна), което се мести
    по V до целевия момент. V извън домейна = V на най-близкия ръб.
    Атрибутите vx, vy, yy, xx са за домейна (ползват се и от advect_weight).
    """

    def _init_motion(self, V: np.ndarray, ts_min: float, pad: int):
        self.pad = pad
        self.vx = (V[0] / ts_min).astype(np.float32)          # px/мин, домейн
        self.vy = (V[1] / ts_min).astype(np.float32)
        ny, nx = self.vx.shape
        self.yy, self.xx = np.mgrid[0:ny, 0:nx].astype(np.float32)
        self._vxp = np.pad(self.vx, pad, mode="edge")
        self._vyp = np.pad(self.vy, pad, mode="edge")
        self._yyp, self._xxp = np.mgrid[0:ny + 2 * pad, 0:nx + 2 * pad].astype(np.float32)

    @property
    def z(self):
        """Полето в домейна (без невидимия ръб) — за обратна съвместимост."""
        p = self.pad
        return self.zp[p:self.zp.shape[0] - p, p:self.zp.shape[1] - p] if p else self.zp

    def _minutes(self, t):
        return (t - self.time).total_seconds() / 60.0

    def z_at(self, target_time: dt.datetime) -> np.ndarray:
        """Z в домейна, преместено от self.time до target_time (обратна траектория)."""
        k, p = self._minutes(target_time), self.pad
        coords = [self.yy + p - self.vy * k, self.xx + p - self.vx * k]
        return map_coordinates(self.zp, coords, order=1, mode="constant", cval=0.0)

    def zp_at(self, target_time: dt.datetime) -> np.ndarray:
        """Цялата разширена решетка, преместена до target_time."""
        k = self._minutes(target_time)
        coords = [self._yyp - self._vyp * k, self._xxp - self._vxp * k]
        return map_coordinates(self.zp, coords, order=1, mode="constant", cval=0.0)

    def known_at(self, target_time: dt.datetime) -> np.ndarray:
        """0…1: източникът на пиксела е вътре в разширената решетка (1) или извън нея (0)."""
        k, p = self._minutes(target_time), self.pad
        cy = self.yy + p - self.vy * k
        cx = self.xx + p - self.vx * k
        hy, hx = self.zp.shape
        inside = (cy >= 0) & (cy <= hy - 1) & (cx >= 0) & (cx <= hx - 1)
        return inside.astype(np.float32)


class SatelliteLayer(PaddedAdvection):
    """Последният кадър H40B като Z поле, което се мести по V до всеки целеви момент."""

    def __init__(self, h40b: dict, V: np.ndarray, ts_min: float,
                 target_lat: np.ndarray, target_lon: np.ndarray,
                 zr_a: float = 200.0, zr_b: float = 1.6, min_rate: float = 0.1, pad: int = None):
        from src.ingest.eumetview import regrid

        pad = pad_px(target_lat) if pad is None else pad
        plat, plon = padded_grid(target_lat, target_lon, pad)
        rain = regrid(h40b["rain"], h40b["lat"], h40b["lon"], plat, plon, method="linear")
        rain = np.clip(np.nan_to_num(rain, nan=0.0), 0.0, None)
        self.zp = np.where(rain >= min_rate,
                           zr_a * np.maximum(rain, 0.01) ** zr_b, 0.0).astype(np.float32)
        self.time = h40b["time"]
        # V е в пиксели за една стъпка на S-PROG (ts_min) → пиксели за минута
        self._init_motion(V, ts_min, pad)

        inner = rain[pad:rain.shape[0] - pad, pad:rain.shape[1] - pad] if pad else rain
        n_rain = int((inner >= min_rate).sum())
        n_out = int((rain >= min_rate).sum()) - n_rain
        logger.info(f"Сателит H40B: {self.time:%H:%M} UTC, валеж ≥{min_rate} mm/h "
                    f"в {n_rain} px, max {inner.max():.1f} mm/h; невидим ръб {pad} px: "
                    f"още {n_out} px с валеж")

    def apply_lightning(self, count: np.ndarray, at_time: dt.datetime,
                        signal_mmh: float, intense_min_mmh: float = None,
                        zr_a: float = 200.0, zr_b: float = 1.6):
        """
        Мести H40B до at_time (часа на радара), усилва там с мълниите от същия
        прозорец (същото правило като за радара) и оттук нататък слоят се движи
        от at_time с вече усиленото поле.
        """
        from src.nowcast.lightning_boost import boost_dbz, _breakdown
        zp = self.zp_at(at_time)                       # цялата решетка до часа на радара
        p = self.pad
        z0 = zp[p:zp.shape[0] - p, p:zp.shape[1] - p] if p else zp
        dbz = np.where(z0 > 0, 10.0 * np.log10(np.maximum(z0, 1e-6)), np.nan)
        to_dbz = lambda r: 10.0 * np.log10(zr_a * r ** zr_b)
        signal_dbz = to_dbz(signal_mmh)
        int_dbz = to_dbz(intense_min_mmh) if intense_min_mmh else None
        brk = _breakdown(dbz, count, signal_dbz, nan_label="без валеж в H40B")
        dbz, n1, n2 = boost_dbz(dbz, count, signal_dbz, int_dbz)
        z0[...] = np.where(np.isfinite(dbz), 10.0 ** (dbz / 10.0), 0.0)   # мълниите — само в домейна
        self.zp = zp.astype(np.float32)
        logger.info(f"Сателит H40B: преместен {self.time:%H:%M}→{at_time:%H:%M} UTC и усилен "
                    f"с мълнии (сигнал ≥{signal_mmh} mm/h = {signal_dbz:.0f} dBZ) — "
                    f"до 45 dBZ: {n1} px, до 55 dBZ: {n2} px  [{brk}]")
        self.time = at_time



def fill_nowcast_for_map(forecast: dict, cov_w: np.ndarray, sat_layer) -> None:
    """
    За nowcast картата: в зоната без радар показва H40B (преместен до всеки час),
    смесен с радара по същото тегло на покритието като в blend-а:
        Z = cov·Z_радар + (1−cov)·Z_сателит
    Записва forecast["forecast_dbz_plot"] и last_composite["dbz_obs_plot"].
    Самата прогноза (forecast_dbz) не се променя.
    """
    def to_z(d):
        d = np.nan_to_num(d, nan=-999.0)
        return np.where(d > -900, 10.0 ** (d / 10.0), 0.0)

    def to_dbz(z):
        out = np.full(z.shape, np.nan, dtype=np.float32)
        ok = z > 1.0                      # > 0 dBZ
        out[ok] = 10.0 * np.log10(z[ok])
        return out

    comp = forecast["last_composite"]
    obs = comp.get("dbz_radar", comp["dbz"])
    z_obs = cov_w * to_z(obs) + (1.0 - cov_w) * sat_layer.z_at(comp["timestamp"])
    comp["dbz_obs_plot"] = to_dbz(z_obs)

    fc = forecast["forecast_dbz"]
    out = np.empty_like(fc)
    for i, t in enumerate(forecast["timestamps"]):
        w = advect_weight(sat_layer, cov_w, t, comp["timestamp"])   # теглото пътува с облаците
        out[i] = to_dbz(w * to_z(fc[i]) + (1.0 - w) * sat_layer.z_at(t))
    forecast["forecast_dbz_plot"] = out
    comp["sat_on_map"] = True


class WeakRadarLayer:
    """
    Слабото ехо на румънския радар (7–15 dBZ, comp["dbz_weak"]) — само за картата,
    само където H40B потвърждава валеж в радиус gate_km (settings.WEAK_RADAR).
    at(t) → dBZ поле, преместено по V до момента t (NaN = няма); след max_lead_min → None.
    """

    def __init__(self, comp: dict, sat_layer, V: np.ndarray, ts_min: float, cfg: dict = None):
        from scipy.ndimage import distance_transform_edt, label
        from config.settings import WEAK_RADAR, SATELLITE
        cfg = cfg or WEAK_RADAR
        self.cfg, self.t0 = cfg, comp["timestamp"]
        w = comp.get("dbz_weak")
        lat = comp["lat"]
        km_px = abs(float(lat[1] - lat[0])) * 111.0
        z_min = 200.0 * SATELLITE["min_rate_mmh"] ** 1.6           # H40B ≥ min_rate
        rain = sat_layer.z_at(self.t0) >= z_min
        gate = distance_transform_edt(~rain) * km_px <= cfg["gate_km"] if rain.any() else rain
        ok = np.isfinite(w) & gate if w is not None else np.zeros(lat.shape + comp["lon"].shape, bool)
        # единични точки (под 4 px) — шум, както на картата
        lab, n = label(ok)
        if n:
            size = np.bincount(lab.ravel())
            ok &= size[lab] >= 4
        self.dbz = np.where(ok, w, np.nan).astype(np.float32) if w is not None else None
        n_all = int(np.isfinite(w).sum()) if w is not None else 0
        logger.info(f"Слабо ехо на радара (7–15 dBZ): {n_all} px, потвърдени от H40B {int(ok.sum())} px "
                    f"(в {cfg['gate_km']:.0f} km) → на картата до +{cfg['max_lead_min']} мин")
        self.vx = (V[0] / ts_min).astype(np.float32)
        self.vy = (V[1] / ts_min).astype(np.float32)
        ny, nx = ok.shape
        self.yy, self.xx = np.mgrid[0:ny, 0:nx].astype(np.float32)

    def at(self, t: dt.datetime):
        if self.dbz is None:
            return None
        k = (t - self.t0).total_seconds() / 60.0
        if k > self.cfg["max_lead_min"] + 1:
            return None
        if abs(k) < 0.5:
            return self.dbz
        from scipy.ndimage import map_coordinates
        coords = [self.yy - self.vy * k, self.xx - self.vx * k]
        return map_coordinates(self.dbz, coords, order=0, mode="constant", cval=np.nan)
