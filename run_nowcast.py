#!/usr/bin/env python3
"""
nowcasting_public — Главен pipeline
====================================
1. Румъния composite (ODIM HDF5)
2. ИАБГ радари (PNG scraping)
3. Blitzortung мълнии
4. ICON-EU precipitation
5. S-PROG nowcasting (0–60 мин)
6. Blending с ICON (0–6 часа)
7. PNG карти + GitHub Pages данни

Стартирай:
  python run_nowcast.py
  python run_nowcast.py --no-lightning --no-icon
  python run_nowcast.py --iabg GCD,STS,BRD
"""

import os, sys, time, logging, argparse
import datetime as dt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.settings import (IABG_RADARS, NOWCAST, MAPS_DIR,
                              LOG_DIR, BASE_DIR)


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
    parser.add_argument("--iabg", default="GCD,STS,BRD",
                        help="ИАБГ радари: GCD,STS,BRD")
    parser.add_argument("--no-romania", action="store_true")
    parser.add_argument("--no-lightning", action="store_true")
    parser.add_argument("--no-icon", action="store_true")
    parser.add_argument("--lightning-sec", type=int, default=30)
    parser.add_argument("--n-frames", type=int, default=5)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    setup_logging(args.verbose)
    log = logging.getLogger("main")
    t0 = time.time()

    log.info("=" * 60)
    log.info("NOWCASTING PUBLIC — START")
    log.info("=" * 60)

    # ── 1. INGEST РУМЪНИЯ ─────────────────────────────────
    all_frames = {}

    if not args.no_romania:
        log.info("── Фаза 1a: Румъния composite ──")
        from src.ingest.radar_romania import ingest_romania
        ro_frames = ingest_romania(n_frames=args.n_frames)
        if ro_frames:
            all_frames["romania"] = ro_frames

    # ── 2. INGEST ИАБГ ────────────────────────────────────
    log.info("── Фаза 1b: ИАБГ радари ──")
    from src.ingest.radar_iabg import ingest_iabg
    iabg_ids = [r.strip().upper() for r in args.iabg.split(",")]
    for rid in iabg_ids:
        if rid in IABG_RADARS and IABG_RADARS[rid]["enabled"]:
            frames = ingest_iabg(rid, n_frames=args.n_frames)
            if frames:
                all_frames[f"iabg_{rid}"] = frames

    if not all_frames:
        log.error("ФАТАЛНО: Няма данни от нито един радар!")
        sys.exit(1)

    total_frames = sum(len(v) for v in all_frames.values())
    log.info(f"  Общо: {total_frames} frames от {len(all_frames)} източника")

    # ── 3. МЪЛНИИ ─────────────────────────────────────────
    lightning, ltg_density, ltg_lat, ltg_lon = [], None, None, None

    if not args.no_lightning:
        log.info("── Фаза 2: Blitzortung ──")
        try:
            from src.lightning.blitzortung import LightningCollector, strikes_to_grid
            coll = LightningCollector()
            lightning = coll.collect_sync(seconds=args.lightning_sec)
            if lightning:
                ltg_density, ltg_lat, ltg_lon = strikes_to_grid(lightning)
        except Exception as e:
            log.warning(f"  Blitzortung: {e}")

    # ── 4. COMPOSITE ──────────────────────────────────────
    log.info("── Фаза 3: Composite ──")
    from src.composite.composite import create_composite_series
    composites = create_composite_series(all_frames, n_composites=args.n_frames)

    if not composites:
        log.error("ФАТАЛНО: Няма composites!")
        sys.exit(1)

    # ── 5. S-PROG ─────────────────────────────────────────
    log.info("── Фаза 4: S-PROG Nowcast ──")
    from src.nowcast.nowcasting import run_sprog, enhance_with_lightning
    forecast = run_sprog(composites)

    if forecast is None:
        log.error("ФАТАЛНО: Nowcast неуспешен!")
        sys.exit(1)

    if ltg_density is not None:
        forecast = enhance_with_lightning(forecast, ltg_density, ltg_lat, ltg_lon)

    # ── 6. ICON + BLENDING ────────────────────────────────
    icon_data = None
    blended, blend_times = None, None

    if not args.no_icon:
        log.info("── Фаза 5: ICON-EU + Blending ──")
        try:
            from src.ingest.icon_eu import fetch_icon_grid
            icon_data = fetch_icon_grid()
        except Exception as e:
            log.warning(f"  ICON: {e}")

    if icon_data is not None:
        from src.blend.blending import blend_nowcast_icon
        comp = forecast["last_composite"]
        blended, blend_times = blend_nowcast_icon(
            forecast["forecast_dbz"], forecast["timestamps"],
            icon_data, comp["lat"], comp["lon"])

    # ── 7. ВИЗУАЛИЗАЦИЯ ──────────────────────────────────
    log.info("── Фаза 6: Карти ──")
    ts_str = dt.datetime.utcnow().strftime("%Y%m%d_%H%M")
    os.makedirs(MAPS_DIR, exist_ok=True)

    from src.output.visualize import plot_nowcast, plot_blended, generate_pages_data

    nc_path = plot_nowcast(forecast,
                           os.path.join(MAPS_DIR, f"nowcast_{ts_str}.png"),
                           lightning)

    # Стабилно име за GitHub Pages
    plot_nowcast(forecast, os.path.join(MAPS_DIR, "nowcast_latest.png"), lightning)

    if blended is not None:
        bl_path = plot_blended(blended, blend_times, forecast["last_composite"],
                               os.path.join(MAPS_DIR, f"blended_{ts_str}.png"))
        plot_blended(blended, blend_times, forecast["last_composite"],
                     os.path.join(MAPS_DIR, "blended_latest.png"))

    generate_pages_data(forecast["last_composite"], forecast,
                        blended, blend_times)

    # Копирай картите в docs/ за GitHub Pages
    import shutil
    pages_maps = os.path.join(BASE_DIR, "docs", "maps")
    os.makedirs(pages_maps, exist_ok=True)
    for fname in ["nowcast_latest.png", "blended_latest.png", "meta.json"]:
        src = os.path.join(MAPS_DIR, fname)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(pages_maps, fname))
    log.info(f"Pages обновени: docs/maps/")
    elapsed = time.time() - t0
    log.info("=" * 60)
    log.info(f"ГОТОВО за {elapsed:.1f} сек")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
