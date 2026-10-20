#!/usr/bin/env python3
"""
nowcasting_public — Главен pipeline
====================================
1. Радари: Румъния (ODIM HDF5) + ИАБГ (PNG, в сезона) → резерв Истанбул (TR34)
2. Мълнии: MTG Lightning Imager (EUMETView); H40B се тегли паралелно
3. Композит на 10-мин ос + усилване с мълнии (45/55 dBZ)
4. ICON-EU (DWD): валеж, LPI, вятър 700 hPa
5. S-PROG; движение от LK, проверено с ICON по региони
6. Сателит H40B за зоната без радар; blending с ICON (0–6 часа)
7. PNG карти + GitHub Pages данни

Стартирай:
  python run_nowcast.py
  python run_nowcast.py --no-lightning --no-icon --no-satellite
"""

import os, sys, time, logging, argparse
import datetime as dt

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.settings import IABG_RADARS, MAPS_DIR, LOG_DIR, BASE_DIR, DOMAIN


def setup_logging(verbose=False):
    os.makedirs(LOG_DIR, exist_ok=True)
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(LOG_DIR, "nowcast.log"),
                                encoding="utf-8"),
        ])


def main():
    parser = argparse.ArgumentParser(description="Nowcasting Public")
    parser.add_argument("--iabg", default="GCD,STS",
                        help="ИАБГ радари: GCD,STS (BRD е изключен в settings)")
    parser.add_argument("--no-romania", action="store_true")
    parser.add_argument("--no-lightning", action="store_true")
    parser.add_argument("--no-icon", action="store_true")
    parser.add_argument("--no-backup", action="store_true",
                        help="без резервни радари (Истанбул за STS)")
    parser.add_argument("--force-icon-ltg", action="store_true",
                        help="тест: резервът ICON + мълнии вместо H40B в зоната без радар")
    parser.add_argument("--no-satellite", action="store_true",
                        help="без H40B за зоните без радар (там ICON от +0)")
    parser.add_argument("--n-frames", type=int, default=5)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    setup_logging(args.verbose)
    log = logging.getLogger("main")
    t0 = time.time()

    log.info("=" * 60)
    log.info("NOWCASTING PUBLIC — START")
    log.info("=" * 60)

    # Изтрий стари входни файлове (>2 часа)
    import glob
    import re
    from config.settings import ROMANIA_DIR
    cutoff = dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=2)
    for pattern in [os.path.join(ROMANIA_DIR, "*.hdf")]:
        for f in glob.glob(pattern):
            try:
                m = re.search(r'(\d{14})', os.path.basename(f))
                if m:
                    ft = dt.datetime.strptime(m.group(1), "%Y%m%d%H%M%S")
                    ft = ft.replace(tzinfo=dt.timezone.utc)
                    if ft < cutoff:
                        os.remove(f)
                        log.info(f"  Изтрит: {os.path.basename(f)}")
            except Exception:
                pass

    # H40B: във фон от самото начало — EUMETView е бавен, а трябва чак във фаза 5a
    from concurrent.futures import ThreadPoolExecutor
    pool = ThreadPoolExecutor(max_workers=1)
    h40b_future = None
    if not args.no_satellite:
        from src.ingest.eumetview import fetch_h40b
        h40b_future = pool.submit(fetch_h40b)

    # ── 1. INGEST РУМЪНИЯ ─────────────────────────────────
    all_frames = {}

    if not args.no_romania:
        log.info("── Фаза 1a: Румъния composite ──")
        from src.ingest.radar_romania import ingest_romania
        ro_frames = ingest_romania(n_frames=args.n_frames + 2)   # резерв за изпуснати марки
        if ro_frames:
            all_frames["romania"] = ro_frames

    # ── 2. INGEST ИАБГ ────────────────────────────────────
    log.info("── Фаза 1b: ИАБГ радари ──")
    from src.ingest.radar_iabg import ingest_iabg
    iabg_ids = [r.strip().upper() for r in args.iabg.split(",")]
    for rid in iabg_ids:
        if rid in IABG_RADARS and IABG_RADARS[rid]["enabled"]:
            frames = ingest_iabg(rid, n_frames=15)   # целият час от JSON (4-мин каденция)
            if frames:
                all_frames[f"iabg_{rid}"] = frames

    # ── 2b. РЕЗЕРВНИ РАДАРИ (Истанбул за STS) ─────────────
    # Теглят се само ако основният липсва или е застоял спрямо RO.
    from config.settings import RAINVIEWER_RADARS
    ro_last = max((f["timestamp"] for f in all_frames.get("romania", [])), default=None)
    for rid, cfg in RAINVIEWER_RADARS.items():
        if not cfg.get("enabled") or args.no_backup:
            continue
        primary = cfg["backup_for"]                       # напр. "iabg_STS"
        prim_frames = all_frames.get(primary, [])
        prim_last = max((f["timestamp"] for f in prim_frames), default=None)
        stale = (prim_last is None or ro_last is None or
                 (ro_last - prim_last).total_seconds() / 60 > 15)
        if not stale:
            continue
        log.info(f"── Фаза 1c: резерв {rid} ({cfg['name']}) за {primary} ──")
        try:
            from src.ingest.radar_rainviewer import ingest_tr34
            frames = ingest_tr34(n_frames=12)   # ~70 мин: да покрие серията на RO
            if frames:
                all_frames[f"rainviewer_{rid}"] = frames
        except Exception as e:
            log.warning(f"  {rid}: {e}")

    if not all_frames:
        log.error("ФАТАЛНО: Няма данни от нито един радар!")
        sys.exit(1)

    total_frames = sum(len(v) for v in all_frames.values())
    log.info(f"  Общо: {total_frames} frames от {len(all_frames)} източника")

    # ── 3. МЪЛНИИ (H40B вече се тегли във фон от началото на пуска) ──

    # Отделните 5-мин кадри за ~40 мин назад: всеки от последните 3 композита
    # (на 10 мин) взима мълниите от своите 15 мин.
    li_frames = []

    if not args.no_lightning:
        log.info("── Фаза 2: Мълнии (MTG Lightning Imager) ──")
        try:
            from src.ingest.eumetview import fetch_li_frames
            li_frames = fetch_li_frames(back_min=40)
            if li_frames:
                rows = (li_frames[0]["lat"] >= DOMAIN["lat_min"]) & (li_frames[0]["lat"] <= DOMAIN["lat_max"])
                cols = (li_frames[0]["lon"] >= DOMAIN["lon_min"]) & (li_frames[0]["lon"] <= DOMAIN["lon_max"])
                n_dom = int(max((f["count"][np.ix_(rows, cols)] > 0).sum() for f in li_frames))
                log.info(f"  От тях в домейна: до {n_dom} px в кадър")
        except Exception as e:
            log.warning(f"  LI AFA: {e}")

    # ── 4. COMPOSITE ──────────────────────────────────────
    log.info("── Фаза 3: Composite ──")
    from src.composite.composite import create_composite_series
    composites = create_composite_series(all_frames, n_composites=args.n_frames)

    if not composites:
        log.error("ФАТАЛНО: Няма composites!")
        sys.exit(1)

    # ── 4b. УСИЛВАНЕ С МЪЛНИИ (преди прогнозата) ─────────
    li_last = None
    if li_frames:
        log.info("── Фаза 3b: Усилване с мълнии (45/55 dBZ където има сигнал) ──")
        from src.nowcast.lightning_boost import boost_composites
        li_last = boost_composites(composites, li_frames, n_last=3)

    # ── 5. ICON (преди S-PROG: вятърът на 700 hPa проверява движението) ──
    icon_data = None
    blended, blend_times, actual_weights = None, None, None

    if not args.no_icon:
        log.info("── Фаза 4a: ICON-EU ──")
        try:
            from src.ingest.icon_dwd import fetch_icon_dwd
            icon_data = fetch_icon_dwd(t_wind=composites[-1]["timestamp"])
        except Exception as e:
            log.warning(f"  ICON DWD: {e}")
        if icon_data is None:
            log.warning("  ICON DWD недостъпен — резерв Open-Meteo (без LPI ядра и без вятър)")
            try:
                from src.ingest.icon_eu import fetch_icon_grid
                icon_data = fetch_icon_grid()
                if icon_data is not None:
                    composites[-1]["icon_src"] = "open-meteo"   # за посочването на източника
            except Exception as e:
                log.warning(f"  ICON Open-Meteo: {e}")

    # ── 6. S-PROG ─────────────────────────────────────────
    log.info("── Фаза 4: Nowcast (ANVIL / S-PROG) ──")
    from src.nowcast.nowcasting import run_sprog
    forecast = run_sprog(composites,
                         wind700=icon_data.get("wind700") if icon_data else None)

    if forecast is None:
        log.error("ФАТАЛНО: Nowcast неуспешен!")
        sys.exit(1)

    comp = forecast["last_composite"]
    fc_times = forecast["timestamps"]
    real_ts = round((fc_times[1] - fc_times[0]).total_seconds() / 60) if len(fc_times) >= 2 else 10

    # ── 7. САТЕЛИТ H40B за зоната без радар (независимо от ICON) ──
    from src.blend.satellite_fill import coverage_weight, SatelliteLayer, fill_nowcast_for_map
    from config.settings import SATELLITE
    cov_w = coverage_weight(comp["dbz"], ramp_px=SATELLITE["ramp_px"])
    sat_layer = None
    if h40b_future is not None and float((cov_w < 1.0).mean()) > 0.01:
        log.info("── Фаза 5a: Сателит H40B ──")
        try:
            h40b = h40b_future.result(timeout=240)
            if args.force_icon_ltg:
                log.info("  --force-icon-ltg: H40B се пропуска (тест на резерва)")
                h40b = None
            if h40b is not None:
                sat_layer = SatelliteLayer(h40b, forecast["motion"], real_ts,
                                           comp["lat"], comp["lon"],
                                           min_rate=SATELLITE["min_rate_mmh"])
                if li_last is not None:
                    from config.settings import LIGHTNING_BOOST
                    sat_layer.apply_lightning(li_last, comp["timestamp"],
                                              LIGHTNING_BOOST["sat_signal_mmh"],
                                              LIGHTNING_BOOST.get("sat_intense_min_mmh"))
        except Exception as e:
            log.warning(f"  H40B: {e}")
        if sat_layer is None and icon_data is not None:
            # Резерв №3: ICON при +0, коригиран по мълниите (без LI — само ICON)
            try:
                from src.blend.icon_ltg_fill import IconLtgLayer
                sat_layer = IconLtgLayer(icon_data, li_last, comp["timestamp"], forecast["motion"],
                                         real_ts, comp["lat"], comp["lon"], cov_w)
                comp["fill_src"] = "icon_ltg" if sat_layer.with_ltg else "icon"
                comp["sat_missing"] = True
                log.warning(f"  H40B липсва — в зоната без радар: {sat_layer.label}")
            except Exception as e:
                log.warning(f"  Резерв ICON+LI: {e}")
                sat_layer = None
        if sat_layer is not None:
            fill_nowcast_for_map(forecast, cov_w, sat_layer)
            if comp.get("fill_src"):
                comp["sat_on_map"] = False          # не е сателит — картата го отбелязва отделно
        else:
            comp["sat_missing"] = True
            log.warning("  H40B липсва — зоната без радар е празна на nowcast картата, ICON от +0 в blend-а")
    pool.shutdown(wait=False)

    # ── 8. BLENDING ───────────────────────────────────────
    if icon_data is not None:
        log.info("── Фаза 5b: Blending ──")
        from src.blend.blending import blend_nowcast_icon
        blended, blend_times, actual_weights = blend_nowcast_icon(
            forecast["forecast_dbz"], fc_times, icon_data, comp["lat"], comp["lon"],
            coverage_w=cov_w, sat_layer=sat_layer)

    # ── 9. КАРТИ ─────────────────────────────────────────
    log.info("── Фаза 6: Карти ──")
    ts_str = dt.datetime.utcnow().strftime("%Y%m%d_%H%M")
    os.makedirs(MAPS_DIR, exist_ok=True)

    from src.output.visualize import plot_nowcast, plot_blended, generate_pages_data

    plot_nowcast(forecast, os.path.join(MAPS_DIR, f"nowcast_{ts_str}.png"))
    plot_nowcast(forecast, os.path.join(MAPS_DIR, "nowcast_latest.png"))   # стабилно име за Pages

    if blended is not None:
        plot_blended(blended, blend_times, forecast["last_composite"],
                     output_path=os.path.join(MAPS_DIR, f"blended_{ts_str}.png"),
                     actual_weights=actual_weights)
        plot_blended(blended, blend_times, forecast["last_composite"],
                     output_path=os.path.join(MAPS_DIR, "blended_latest.png"),
                     actual_weights=actual_weights)

    generate_pages_data(forecast["last_composite"], forecast,
                        blended, blend_times)

    # Публичната страница (docs/embed/): кадри за Leaflet + таблица за стартовете
    try:
        from src.output.web_export import export_web
        idat = icon_data or {}
        export_web(forecast, blended, blend_times, winds=idat.get("winds"),
                   wind_grid=(idat["lat"], idat["lon"]) if idat.get("winds") else None)
    except Exception as e:
        log.warning(f"  Web експорт: {e}")

    # Копирай картите в docs/ за GitHub Pages
    import shutil
    pages_maps = os.path.join(BASE_DIR, "docs", "maps")
    os.makedirs(pages_maps, exist_ok=True)
    for fname in ["nowcast_latest.png", "blended_latest.png", "meta.json"]:
        src = os.path.join(MAPS_DIR, fname)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(pages_maps, fname))
    log.info("Pages обновени: docs/maps/")
    elapsed = time.time() - t0
    log.info("=" * 60)
    log.info(f"ГОТОВО за {elapsed:.1f} сек")
    log.info("=" * 60)


if __name__ == "__main__":
    # Изход без финализация на интерпретатора: фоновата нишка за H40B може още да
    # е вътре в C код (requests/tifffile), ако EUMETView отговаря бавно — тогава
    # нормалното затваряне гърми с "double free or corruption" (exit 134), стъпката
    # в Actions пада и Pages не се публикува, въпреки че всичко е записано.
    import traceback
    code = 0
    try:
        main()
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
    except Exception:
        traceback.print_exc()
        code = 1
    finally:
        logging.shutdown()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(code)
