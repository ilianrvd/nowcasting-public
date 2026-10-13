"""
Сателитен слой за зоните без радарно покритие
=============================================
H40B (EUMETSAT, MTG FCI + LEO MW) → Z, преместен по вектора на движението
от S-PROG от СВОЯ час (≈25 мин закъснение) до целевия момент.
Само адвекция, без еволюция.

coverage_weight: 1 навътре в радарното покритие, линейно до 0 в последните
ramp_px пиксела към ръба му, 0 извън него. Така радарът и сателитът се
сшиват плавно, без шев.
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


class SatelliteLayer:
    """Последният кадър H40B като Z поле, което се мести по V до всеки целеви момент."""

    def __init__(self, h40b: dict, V: np.ndarray, ts_min: float,
                 target_lat: np.ndarray, target_lon: np.ndarray,
                 zr_a: float = 200.0, zr_b: float = 1.6, min_rate: float = 0.1):
        from src.ingest.eumetview import regrid

        rain = regrid(h40b["rain"], h40b["lat"], h40b["lon"],
                      target_lat, target_lon, method="linear")
        rain = np.clip(np.nan_to_num(rain, nan=0.0), 0.0, None)
        self.z = np.where(rain >= min_rate,
                          zr_a * np.maximum(rain, 0.01) ** zr_b, 0.0).astype(np.float32)
        self.time = h40b["time"]

        # V е в пиксели за една стъпка на S-PROG (ts_min) → пиксели за минута
        self.vx = (V[0] / ts_min).astype(np.float32)
        self.vy = (V[1] / ts_min).astype(np.float32)
        ny, nx = self.z.shape
        self.yy, self.xx = np.mgrid[0:ny, 0:nx].astype(np.float32)

        n_rain = int((rain >= min_rate).sum())
        logger.info(f"Сателит H40B: {self.time:%H:%M} UTC, валеж ≥{min_rate} mm/h "
                    f"в {n_rain} px, max {rain.max():.1f} mm/h")

    def apply_lightning(self, count: np.ndarray, at_time: dt.datetime,
                        signal_mmh: float, intense_min_mmh: float = None,
                        zr_a: float = 200.0, zr_b: float = 1.6):
        """
        Мести H40B до at_time (часа на радара), усилва там с мълниите от същия
        прозорец (същото правило като за радара) и оттук нататък слоят се движи
        от at_time с вече усиленото поле.
        """
        from src.nowcast.lightning_boost import boost_dbz, _breakdown
        z0 = self.z_at(at_time)
        dbz = np.where(z0 > 0, 10.0 * np.log10(np.maximum(z0, 1e-6)), np.nan)
        to_dbz = lambda r: 10.0 * np.log10(zr_a * r ** zr_b)
        signal_dbz = to_dbz(signal_mmh)
        int_dbz = to_dbz(intense_min_mmh) if intense_min_mmh else None
        brk = _breakdown(dbz, count, signal_dbz, nan_label="без валеж в H40B")
        dbz, n1, n2 = boost_dbz(dbz, count, signal_dbz, int_dbz)
        self.z = np.where(np.isfinite(dbz), 10.0 ** (dbz / 10.0), 0.0).astype(np.float32)
        logger.info(f"Сателит H40B: преместен {self.time:%H:%M}→{at_time:%H:%M} UTC и усилен "
                    f"с мълнии (сигнал ≥{signal_mmh} mm/h = {signal_dbz:.0f} dBZ) — "
                    f"до 45 dBZ: {n1} px, до 55 dBZ: {n2} px  [{brk}]")
        self.time = at_time

    def z_at(self, target_time: dt.datetime) -> np.ndarray:
        """Z полето, преместено от часа на H40B до target_time (обратна траектория)."""
        k = (target_time - self.time).total_seconds() / 60.0     # минути
        coords = [self.yy - self.vy * k, self.xx - self.vx * k]
        return map_coordinates(self.z, coords, order=1, mode="constant", cval=0.0)


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
        out[i] = to_dbz(cov_w * to_z(fc[i]) + (1.0 - cov_w) * sat_layer.z_at(t))
    forecast["forecast_dbz_plot"] = out
    comp["sat_on_map"] = True
