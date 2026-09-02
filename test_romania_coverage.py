import numpy as np
from src.ingest.radar_romania import read_romania_composite, find_local_composites

files = find_local_composites(1)
frame = read_romania_composite(files[-1])
dbz = frame["dbz"]
lat = frame["lat"]
lon = frame["lon"]

# Къде има валидни данни (не NaN)
valid = ~np.isnan(dbz)

# Най-южната ширина с данни
rows_with_data = np.where(valid.any(axis=1))[0]
if len(rows_with_data) > 0:
    south_idx = rows_with_data[0] if lat[0] < lat[-1] else rows_with_data[-1]
    south_lat = lat[south_idx]
    print(f"Най-южна ширина с румънски данни: {south_lat:.2f}°N")

# Данни в Черно море зона (lat 42-44, lon 28-30)
sea_mask = (lat[:, None] >= 42) & (lat[:, None] <= 44) & \
           (lon[None, :] >= 28) & (lon[None, :] <= 30)
sea_data = dbz[sea_mask & valid]
print(f"\nЧерно море зона (42-44°N, 28-30°E):")
print(f"  Валидни пиксели: {len(sea_data)}")
if len(sea_data) > 0:
    print(f"  dBZ диапазон: {sea_data.min():.1f} - {sea_data.max():.1f}")
    print(f"  Медиана: {np.median(sea_data):.1f}")

# Разпределение по ширина в морската зона
print(f"\nПокритие по ширина (само lon 28-30°E):")
for target_lat in [41, 42, 43, 44, 45]:
    idx = np.argmin(np.abs(lat - target_lat))
    row = dbz[idx]
    lon_mask = (lon >= 28) & (lon <= 30)
    n = np.count_nonzero(~np.isnan(row[lon_mask]))
    print(f"  {target_lat}°N: {n} валидни пиксела в 28-30°E")