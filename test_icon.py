"""Диагностика на ICON-EU данни."""
import numpy as np
import datetime as dt
from src.ingest.icon_eu import fetch_icon_grid, precip_to_dbz

data = fetch_icon_grid()
if not data:
    print("Няма данни!")
    exit()

precip = data["precipitation_mm"]
lats = data["lat"]
lons = data["lon"]
times = data["valid_times"]

print(f"Shape: {precip.shape}")
print(f"Lats: {lats}")
print(f"Lons: {lons}")
print(f"Times: {len(times)} стъпки, {times[0]} → {times[-1]}")
print(f"Max precip: {precip.max():.2f} mm")
print(f"Non-zero: {np.count_nonzero(precip > 0)} от {precip.size}")

# Покажи първите 12 часови стъпки
print("\nЧасови стъпки (средно/max по целия grid):")
for t in range(min(24, len(times))):
    p = precip[t]
    if p.max() > 0:
        print(f"  {times[t].strftime('%H:%M')} UTC  "
              f"mean={p.mean():.3f}  max={p.max():.2f} mm  "
              f"non-zero={np.count_nonzero(p > 0)}/{p.size}")

# Конвертирай в dBZ и покажи
print("\nDBZ конверсия (за max timestep):")
t_max = np.argmax(precip.max(axis=(1,2)))
p_max = precip[t_max]
dbz = precip_to_dbz(p_max)
print(f"  Time: {times[t_max].strftime('%H:%M')} UTC")
print(f"  Precip: max={p_max.max():.2f} mm")
print(f"  dBZ: max={np.nanmax(dbz):.1f}, non-NaN={np.count_nonzero(~np.isnan(dbz))}/{dbz.size}")

# Покажи grid стойности за max timestep
print(f"\nGrid (mm) за {times[t_max].strftime('%H:%M')} UTC:")
print(f"       ", end="")
for lo in lons:
    print(f"{lo:6.1f}", end="")
print()
for i, la in enumerate(lats):
    print(f"  {la:5.1f} ", end="")
    for j in range(len(lons)):
        print(f"{p_max[i,j]:6.2f}", end="")
    print() 