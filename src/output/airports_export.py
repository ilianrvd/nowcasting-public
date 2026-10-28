"""
Изход за летищата: docs/embed/data/airports.json
=================================================
Малък JSON за отделния проект airport-nowcast (таблица 0–12 ч). Чете се от Pages.
Нищо от съществуващия изход не се променя — само се добавя този файл след export_web.

За всяко летище (settings.AIRPORTS_EXPORT):
  source     — откъде идва ехото в пиксела на летището в t0:
               radar_ro | radar_iabg | tr34 | h40b | icon_ltg | icon | none
  motion     — движение на ехото около летището (V, средно в радиус motion_radius_km),
               dir_deg = НАКЪДЕТО се движи (както MOV в SIGMET и „Движение“ на картата)
  lightning  — MTG LI за последните 15 мин: брой LI пиксели (~2 km) с мълнии
               в пръстените 0–8, 8–16, 16–30 km
  leads      — 0, 30 … 360 мин: radar_weight, максимум dBZ в 3 / 10 km и в пръстена 10–20 km,
               convective, средна интензивност в 3 km (Z-R 200·R^1.6)
lead 0 = последният композит (наблюдението, с попълването в зоната без радар), не прогноза.
null = няма стойност (извън покритие / липсващ кадър). dBZ < 5 → 0.
"""

import os, sys, json, logging
import datetime as dt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from config.settings import AIRPORTS_EXPORT, COVERAGE_MASKS, LIGHTNING_BOOST

logger = logging.getLogger("output.airports")

OUT_FILE = "airports.json"


def _iso(t):
    return t.astimezone(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ") if t else None


def _dist_km(lat, lon, la0, lo0):
    """Разстояние (km) от точката до всеки пиксел на решетката lat×lon (равнинно приближение)."""
    dy = (np.asarray(lat)[:, None] - la0) * 111.2
    dx = (np.asarray(lon)[None, :] - lo0) * 111.2 * np.cos(np.radians(la0))
    return np.hypot(dy, dx)


def _in_mask(src, la, lo):
    m = COVERAGE_MASKS.get(src, {})
    return (la >= m.get("lat_min", -90) and la <= m.get("lat_max", 90)
            and lo >= m.get("lon_min", -180) and lo <= m.get("lon_max", 180))


def _source(comp, iy, ix, la, lo):
    """Откъде е ехото в пиксела на летището в t0."""
    cov = comp.get("coverage")
    covered = bool(cov[iy, ix]) if cov is not None else bool(np.isfinite(comp["dbz"][iy, ix]))
    if covered:
        for src in comp.get("sources", []):
            if _in_mask(src, la, lo):
                if src == "romania_composite":
                    return "radar_ro"
                if src.startswith("iabg_"):
                    return "radar_iabg"
                if src.startswith("rainviewer_"):
                    return "tr34"
        return "radar_ro" if _in_mask("romania_composite", la, lo) else "none"
    if comp.get("sat_on_map"):
        return "h40b"
    return {"icon_ltg": "icon_ltg", "icon": "icon"}.get(comp.get("fill_src"), "none")


def _stats(field, d, valid, cfg):
    """dBZ максимуми и интензивност за едно поле; None където няма стойност."""
    if field is None or not valid:
        return None
    from src.nowcast.nowcasting import dbz_to_r
    f = np.nan_to_num(np.asarray(field, dtype=np.float32), nan=0.0)   # NaN тук = сухо
    f = np.where(f < 5.0, 0.0, f)
    core, near = d <= cfg["r_core_km"], d <= cfg["r_near_km"]
    ring = (d > cfg["r_near_km"]) & (d <= cfg["r_ring_km"])
    rate = np.where(f > 0, dbz_to_r(f), 0.0)
    r1 = lambda x: round(float(x), 1)
    return {"dbz_max_3km": r1(f[core].max()), "dbz_max_10km": r1(f[near].max()),
            "dbz_max_10_20km": r1(f[ring].max()) if ring.any() else None,
            "rate_mmh_3km": round(float(rate[core].mean()), 2)}


def _lightning(li_frames, t0, la0, lo0, rings):
    """Брой LI пиксели с мълнии в пръстените (последните window_min минути до t0)."""
    if not li_frames:
        return None
    t_from = t0 - dt.timedelta(minutes=LIGHTNING_BOOST["window_min"])
    sel = [f for f in li_frames if t_from < f["time"] <= t0]
    if not sel:
        return None
    cnt = None
    for f in sel:
        cnt = f["count"] if cnt is None else np.maximum(cnt, f["count"])
    d = _dist_km(sel[0]["lat"], sel[0]["lon"], la0, lo0)
    hit = cnt >= 1
    out, r_in = {}, 0.0
    for r in rings:
        out[f"r{int(r)}"] = int((hit & (d > r_in) & (d <= r)).sum())
        r_in = r
    return out


def export_airports(composites, forecast, blended=None, blend_times=None,
                    li_frames=None, V=None, icon_data=None, out_dir=None):
    from src.blend.blending import blend_weights
    from src.nowcast.nowcasting import motion_summary
    from src.output.web_export import OUT_DIR

    cfg = AIRPORTS_EXPORT
    comp = forecast["last_composite"] if forecast else composites[-1]
    t0 = comp["timestamp"]
    lat, lon = comp["lat"], comp["lon"]
    fc_times = forecast.get("timestamps", []) if forecast else []
    horizon = (fc_times[-1] - t0).total_seconds() / 60 if fc_times else 0.0
    ts = (fc_times[1] - fc_times[0]).total_seconds() / 60 if len(fc_times) >= 2 else 10.0
    V = V if V is not None else (forecast.get("motion") if forecast else None)
    obs = comp.get("dbz_obs_plot", comp.get("dbz_radar", comp["dbz"]))
    fc = forecast.get("forecast_dbz_plot", forecast.get("forecast_dbz")) if forecast else None

    def field_at(m):
        """Полето за водещо време m: 0 = наблюдението; иначе blend-ът (без него — nowcast-ът)."""
        if m == 0:
            return obs
        if blended is not None and blend_times:
            k = int(np.argmin([abs((bt - t0).total_seconds() / 60 - m) for bt in blend_times]))
            if abs((blend_times[k] - t0).total_seconds() / 60 - m) <= 2.5:
                return blended[k]
            return None
        if fc is not None and fc_times:
            k = int(np.argmin([abs((ft - t0).total_seconds() / 60 - m) for ft in fc_times]))
            if abs((fc_times[k] - t0).total_seconds() / 60 - m) <= 2.5:
                return fc[k]
        return None

    method = (forecast.get("method", "") if forecast else "").lower().replace("-", "")
    out = {"created": _iso(dt.datetime.now(dt.timezone.utc)), "radar_time": _iso(t0),
           "method": method if method in ("anvil", "sprog") else (method or None),
           "radar_horizon_min": int(round(horizon)), "airports": {}}

    for icao, (la0, lo0) in cfg["airports"].items():
        iy, ix = int(np.argmin(np.abs(lat - la0))), int(np.argmin(np.abs(lon - lo0)))
        d = _dist_km(lat, lon, la0, lo0)
        src = _source(comp, iy, ix, la0, lo0)

        motion = None
        if V is not None:
            disk = d <= cfg["motion_radius_km"]
            ms = motion_summary(np.where(disk[None], V, 0.0),
                                {"dbz": np.where(disk, comp.get("dbz_radar", comp["dbz"]), np.nan)}, ts)
            if ms["n_px"] > 0:
                motion = {"dir_deg": int(round(ms["dir_deg"])) % 360, "speed_kmh": int(round(ms["kmh"]))}

        ltg = _lightning(li_frames, t0, la0, lo0, cfg["ltg_rings_km"])

        leads = []
        for m in cfg["leads_min"]:
            rw = 1.0 if m == 0 else (blend_weights(m, radar_horizon=horizon)[0] if horizon else 0.0)
            iw = 0.0 if m == 0 else 1.0 - rw
            # без радар и без попълване: стойност има само там, където ICON вече води (както
            # „няма данни“ на картата); без ICON изобщо — null
            if src == "none":
                valid = icon_data is not None and m > 0 and iw >= 0.5
            else:
                valid = True
            st = _stats(field_at(m), d, valid, cfg)
            conv = None
            if st is not None:
                conv = bool(st["dbz_max_10km"] >= cfg["conv_dbz"])
                if m == 0 and ltg is not None:
                    near = sum(v for r, v in zip(cfg["ltg_rings_km"], ltg.values()) if r <= cfg["conv_ltg_km"])
                    conv = conv or near > 0
            leads.append({"lead_min": m, "valid": _iso(t0 + dt.timedelta(minutes=m)),
                          "radar_weight": round(float(rw), 2),
                          "dbz_max_3km": st and st["dbz_max_3km"],
                          "dbz_max_10km": st and st["dbz_max_10km"],
                          "dbz_max_10_20km": st and st["dbz_max_10_20km"],
                          "convective": conv,
                          "rate_mmh_3km": st and st["rate_mmh_3km"]})

        out["airports"][icao] = {"source": src, "motion": motion, "lightning": ltg, "leads": leads}

    path = os.path.join(out_dir or OUT_DIR, OUT_FILE)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fo:
        json.dump(out, fo, ensure_ascii=False, indent=1)
    logger.info("  Летища: " + " · ".join(
        f"{k} {v['source']} {v['leads'][0]['dbz_max_10km']}/{max((x['dbz_max_10km'] or 0) for x in v['leads'])} dBZ"
        for k, v in out["airports"].items()) + f" → {OUT_FILE}")
    return path
