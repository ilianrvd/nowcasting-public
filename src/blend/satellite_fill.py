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

    def z_at(self, target_time: dt.datetime) -> np.ndarray:
        """Z полето, преместено от часа на H40B до target_time (обратна траектория)."""
        k = (target_time - self.time).total_seconds() / 60.0     # минути
        coords = [self.yy - self.vy * k, self.xx - self.vx * k]
        return map_coordinates(self.z, coords, order=1, mode="constant", cval=0.0)
