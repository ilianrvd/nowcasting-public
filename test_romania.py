"""Диагностика на Румъния reprojection."""
import numpy as np
from src.ingest.radar_romania import read_romania_composite, find_local_composites

files = find_local_composites(1)
if not files:
    print("Няма файлове!")
    exit()

frame = read_romania_composite(files[-1])
dbz = frame["dbz"]
lat = frame["lat"]
lon = frame["lon"]

print(f"lat: {lat[0]:.2f} → {lat[-1]:.2f}  (len={len(lat)})")
print(f"lon: {lon[0]:.2f} → {lon[-1]:.2f}  (len={len(lon)})")
print(f"dbz shape: {dbz.shape}")
print(f"dbz max: {np.nanmax(dbz):.1f}")

# Къде е максимумът?
idx = np.unravel_index(np.nanargmax(dbz), dbz.shape)
print(f"Max at row={idx[0]}, col={idx[1]}")
print(f"Max at lat={lat[idx[0]]:.2f}, lon={lon[idx[1]]:.2f}")

# Проверка: колко пиксели > 40 dBZ в зоната 42-44°N
mask_bg = (lat >= 42.0) & (lat <= 44.0)
bg_rows = np.where(mask_bg)[0]
if len(bg_rows) > 0:
    bg_dbz = dbz[bg_rows[0]:bg_rows[-1]+1, :]
    print(f"\nВ зоната 42-44°N:")
    print(f"  >40 dBZ: {np.sum(bg_dbz > 40)} px")
    print(f"  >30 dBZ: {np.sum(bg_dbz > 30)} px")
    print(f"  >20 dBZ: {np.sum(bg_dbz > 20)} px")
    print(f"  max: {np.nanmax(bg_dbz):.1f}")
else:
    print("Няма данни в 42-44°N!")