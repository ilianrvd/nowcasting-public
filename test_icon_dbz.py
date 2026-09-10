"""Тест матрица ICON→dBZ — верига v3 (роли: showers=база, LPI=структура, cloud_top=дълбочина+таван)."""
import numpy as np

ZR_A, ZR_B = 200.0, 1.6      # потвърди спрямо ICON["zr_a"/"zr_b"]
LPI_REF = 25.0               # LPI за пълен бонус
BONUS_MAX = 15.0             # макс dBZ добавка от LPI (структура/интензивност)
DEPTH_BONUS = 8.0            # макс dBZ добавка за дълбока конвекция
DBZ_HARD_CAP = 65.0

def precip_to_dbz(p):
    R = np.clip(p, 0.01, None)
    Z = ZR_A * R ** ZR_B
    return 10.0 * np.log10(Z)

def cloud_top_cap(ct_m):
    """cloud_top (m) → мек таван dBZ. 6km→50, 14km→65."""
    return np.clip(50.0 + (ct_m - 6000) / 8000.0 * 15.0, 45.0, 65.0)

def depth_bonus(ct_m):
    """Дълбочинен бонус: overshooting топове → повече dBZ. 10km→0, 14km→пълен."""
    return np.clip((ct_m - 10000) / 4000.0, 0.0, 1.0) * DEPTH_BONUS

def chain(showers, lpi, cloud_top):
    if showers < 0.1 and lpi <= 0:
        return np.nan, 0, 0, 0, 0
    dbz_base = precip_to_dbz(showers) if showers >= 0.1 else 0.0
    lpi_norm = np.clip(lpi / LPI_REF, 0.0, 1.0)
    dbz_bonus = lpi_norm * BONUS_MAX
    dbz_dep = depth_bonus(cloud_top)
    dbz_cap = cloud_top_cap(cloud_top)
    dbz = min(dbz_base + dbz_bonus + dbz_dep, dbz_cap)
    dbz = np.clip(dbz, None, DBZ_HARD_CAP)
    return dbz, dbz_base, dbz_bonus, dbz_dep, dbz_cap

rows = [
    (0.5,  0,  3000,  "~20 (моръсене)"),
    (2.0,  0,  5000,  "~30 (умерен)"),
    (2.0,  10, 10000, "~35 (умерена буря)"),
    (8.0,  25, 12000, "~50 (силна буря)"),
    (15.0, 30, 14000, "~55-65 (град)"),
]

print(f"{'showers':>8} {'lpi':>5} {'ct(km)':>7} {'база':>6} {'LPI+':>6} {'дълб+':>6} {'таван':>6} {'dBZ':>6}  очаквано")
print("-" * 86)
for showers, lpi, ct, exp in rows:
    dbz, base, bonus, dep, cap = chain(showers, lpi, ct)
    dbz_s = f"{dbz:.1f}" if not np.isnan(dbz) else "NaN"
    print(f"{showers:>8.1f} {lpi:>5.0f} {ct/1000:>7.1f} {base:>6.1f} {bonus:>6.1f} {dep:>6.1f} {cap:>6.1f} {dbz_s:>6}  {exp}")

print("\nLPI чувствителност (showers=2, ct=10km):")
for lpi in [0, 5, 10, 15, 25, 40]:
    dbz, base, bonus, dep, cap = chain(2.0, lpi, 10000)
    print(f"  LPI={lpi:>3}: база={base:.1f} LPI+={bonus:.1f} → dBZ={dbz:.1f}")

print("\ncloud_top (showers=15, LPI=30) — дълбочина + таван:")
for ct in [6000, 8000, 10000, 12000, 14000]:
    dbz, base, bonus, dep, cap = chain(15.0, 30, ct)
    print(f"  ct={ct/1000:.0f}km: база={base:.1f} LPI+={bonus:.1f} дълб+={dep:.1f} таван={cap:.1f} → dBZ={dbz:.1f}")