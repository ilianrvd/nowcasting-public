"""
Output: PNG карти и GitHub Pages данни
=======================================
Генерира карти от nowcast и blended прогноза.
"""

import os, sys, logging
import datetime as dt
import numpy as np
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from config.settings import DOMAIN, MAPS_DIR

logger = logging.getLogger("output")

# Радарна цветова скала
RADAR_COLORS = [
    '#00ECEC', '#00A0F0', '#0000F0',
    '#00FF00', '#00C800', '#009000',
    '#FFFF00', '#E7C000', '#FF9000',
    '#FF0000', '#D60000', '#C00000',
    '#FF00FF', '#9955C9',
]
RADAR_LEVELS = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70]

# Посочване на източниците (RainViewer го изисква изрично)
CREDITS = {
    "romania_composite": "радар: NMA România",
    "iabg_GCD": "радар: ИАБГ",
    "iabg_STS": "радар: ИАБГ",
    "rainviewer_TR34": "радар Истанбул: MGM, Weather data by RainViewer (rainviewer.com)",
}
# Същото за GitHub Pages — с линкове
CREDIT_LINKS = {
    "romania_composite": ("NMA România", "https://www.meteoromania.ro/"),
    "iabg_GCD": ("ИАБГ", "https://www.weathermod-bg.eu/"),
    "iabg_STS": ("ИАБГ", "https://www.weathermod-bg.eu/"),
    "rainviewer_TR34": ("MGM (Истанбул) — Weather data by RainViewer", "https://www.rainviewer.com/"),
}
FIXED_LINKS = [
    ("EUMETSAT (MTG Lightning Imager, H SAF H40B)", "https://view.eumetsat.int/"),
    ("DWD ICON-EU", "https://opendata.dwd.de/"),
]


def _add_credits(fig, comp, extra=()):
    """Ред с източниците най-долу на картата."""
    parts = []
    for s in comp.get("sources", []):
        c = CREDITS.get(s)
        if c and c not in parts:
            parts.append(c)
    parts += list(extra)
    fig.text(0.5, 0.005, "Данни: " + " · ".join(parts),
             ha="center", va="bottom", fontsize=7, color="#555555")


def _setup_cmap():
    import matplotlib.colors as mcolors
    cmap = mcolors.ListedColormap(RADAR_COLORS)
    norm = mcolors.BoundaryNorm(RADAR_LEVELS, cmap.N)
    cmap.set_bad(alpha=0.0)
    return cmap, norm

def _mov_text(comp):
    """Движението в заглавието: по региони (север/ЮИ/ЮЗ) и откъде е всяко."""
    mi = comp.get("motion_info")
    if not mi:
        return ""
    regs = mi.get("regions")
    if regs:
        lab = {"север": "С", "ЮИ": "ЮИ", "ЮЗ": "ЮЗ"}
        parts = []
        for r in regs:
            mv = "STNR" if r["dir_txt"] == "STNR" else f"{r['dir_txt']} {r['kt']:.0f}KT"
            parts.append(f"{lab.get(r['name'], r['name'])}: {mv}"
                         + (" (ICON)" if r["src"] == "ICON" else ""))
        return "   |   MOV  " + "  ·  ".join(parts)
    if mi["dir_txt"] == "STNR":
        return "   |   STNR"
    return f"   |   MOV {mi['dir_txt']} {mi['kt']:.0f}KT ({mi['kmh']:.0f} km/h)"


def plot_nowcast(forecast, output_path=None):
    """Карта: наблюдение + S-PROG прогнози (до 60 мин)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        HAS_CARTOPY = True
    except ImportError:
        HAS_CARTOPY = False

    fc_dbz = forecast.get("forecast_dbz_plot", forecast["forecast_dbz"])   # с H40B без радар
    fc_times = forecast["timestamps"]
    comp = forecast["last_composite"]
    cmap, norm = _setup_cmap()

    # Панели около +5, +15, +30, +45, +60 мин (реални минути, по стъпката на S-PROG)
    target_mins = [5, 15, 30, 45, 60]
    if len(fc_times) >= 2:
        ts_min = (fc_times[1] - fc_times[0]).total_seconds() / 60
    else:
        ts_min = 5
    steps = [max(0, round(m / ts_min) - 1) for m in target_mins]
    steps = list(dict.fromkeys(s for s in steps if s < fc_dbz.shape[0]))   # без дубликати
    n_panels = 1 + len(steps)
    ncols = min(4, n_panels)
    nrows = (n_panels + ncols - 1) // ncols

    kw = {"projection": ccrs.PlateCarree()} if HAS_CARTOPY else {}
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows),
                             subplot_kw=kw)
    axes = np.array(axes).flatten()

    def _setup_ax(ax):
        if HAS_CARTOPY:
            ax.set_extent([DOMAIN["lon_min"], DOMAIN["lon_max"],
                           DOMAIN["lat_min"], DOMAIN["lat_max"]])
            ax.add_feature(cfeature.BORDERS, linewidth=0.8)
            ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
            ax.add_feature(cfeature.LAND, facecolor='#f0f0f0', alpha=0.3)

    # Panel 0: OBS
    ax = axes[0]
    _setup_ax(ax)
    # OBS = радарът без усилването от LI (+ H40B в зоната без радар, ако го има)
    raw = comp.get("dbz_obs_plot", comp.get("dbz_radar", comp["dbz"]))
    obs = np.ma.masked_where(np.isnan(raw) | (raw < 5), raw)
    tr = ccrs.PlateCarree() if HAS_CARTOPY else None
    ax.pcolormesh(comp["lon"], comp["lat"], obs, cmap=cmap, norm=norm, transform=tr)
    li = comp.get("li_raw")
    if li is not None and (li >= 1).any():           # MTG LI за последните 15 мин
        ax.contour(comp["lon"], comp["lat"], (li >= 1).astype(float), levels=[0.5],
                   colors="black", linewidths=0.6, transform=tr, zorder=11)
        if (li >= 5).any():
            ax.contour(comp["lon"], comp["lat"], (li >= 5).astype(float), levels=[0.5],
                       colors="magenta", linewidths=0.9, transform=tr, zorder=12)
    sources = comp.get("sources", [])
    ax.set_title(f"OBS {comp['timestamp'].strftime('%H:%M')} UTC"
                 + ("  + H40B" if comp.get("sat_on_map") else "")
                 + ("  (H40B липсва)" if comp.get("sat_missing") else "")
                 + ("  (контур: мълнии 15 мин)" if li is not None and (li >= 1).any() else "") + "\n"
                 f"{', '.join(sources)}", fontsize=9)

    # Forecast panels
    for i, step in enumerate(steps):
        ax = axes[i + 1]
        _setup_ax(ax)
        fc = np.ma.masked_where(np.isnan(fc_dbz[step]) | (fc_dbz[step] < 5),
                                fc_dbz[step])
        ax.pcolormesh(comp["lon"], comp["lat"], fc, cmap=cmap, norm=norm, transform=tr)
        real_mins = round((fc_times[step] - comp["timestamp"]).total_seconds() / 60)
        ax.set_title(f"+{real_mins} min  {fc_times[step].strftime('%H:%M')} UTC",
                     fontsize=9)

    for j in range(n_panels, len(axes)):
        axes[j].set_visible(False)

    plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                 ax=axes[:n_panels], label='dBZ', shrink=0.6, pad=0.02)
    plt.suptitle(f'NOWCASTING PUBLIC — {forecast.get("method", "S-PROG")}{_mov_text(comp)}',
                 fontsize=13, fontweight='bold')

    _add_credits(fig, comp, (["сателит: EUMETSAT / H SAF H40B"] if comp.get("sat_on_map") else [])
                 + ["мълнии: EUMETSAT MTG LI"])

    if output_path is None:
        output_path = os.path.join(MAPS_DIR, "nowcast.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    logger.info(f"Nowcast карта: {output_path}")
    return output_path


def plot_blended(blended_dbz, blend_times, comp, output_path=None, actual_weights=None):
    """Карта: blended прогноза до 6 часа."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        HAS_CARTOPY = True
    except ImportError:
        HAS_CARTOPY = False

    cmap, norm = _setup_cmap()

    # Панели: +30, +60, +120, +180, +240, +300 мин
    show_minutes = [30, 60, 120, 180, 240, 300]
    ts_min = 5
    show_steps = [m // ts_min - 1 for m in show_minutes]
    show_steps = [s for s in show_steps if s < blended_dbz.shape[0]]

    ncols = min(3, len(show_steps))
    nrows = (len(show_steps) + ncols - 1) // ncols

    kw = {"projection": ccrs.PlateCarree()} if HAS_CARTOPY else {}
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows),
                             subplot_kw=kw)
    axes = np.array(axes).flatten()

    for i, step in enumerate(show_steps):
        ax = axes[i]
        if HAS_CARTOPY:
            ax.set_extent([DOMAIN["lon_min"], DOMAIN["lon_max"],
                           DOMAIN["lat_min"], DOMAIN["lat_max"]])
            ax.add_feature(cfeature.BORDERS, linewidth=0.8)
            ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
            ax.add_feature(cfeature.LAND, facecolor='#f0f0f0', alpha=0.3)

        tr = ccrs.PlateCarree() if HAS_CARTOPY else None
        fc = np.ma.masked_where(np.isnan(blended_dbz[step]) | (blended_dbz[step] < 5),
                                blended_dbz[step])
        ax.pcolormesh(comp["lon"], comp["lat"], fc, cmap=cmap, norm=norm, transform=tr)

        mins = (step + 1) * ts_min
        if actual_weights is not None and step < len(actual_weights):
            rw, iw = actual_weights[step]
        else:
            rw, iw = 0.0, 1.0
        ax.set_title(f"+{mins} min  {blend_times[step].strftime('%H:%M')}\n"
                     f"R:{rw:.0%} I:{iw:.0%}", fontsize=9)

    for j in range(len(show_steps), len(axes)):
        axes[j].set_visible(False)

    plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                 ax=axes[:len(show_steps)], label='dBZ', shrink=0.6, pad=0.02)
    plt.suptitle(f'BLENDED FORECAST — Radar + ICON-EU (0–6h){_mov_text(comp)}',
                 fontsize=13, fontweight='bold')

    _add_credits(fig, comp, (["сателит: EUMETSAT / H SAF H40B"] if comp.get("sat_on_map") else [])
                 + ["модел: DWD ICON-EU"])

    if output_path is None:
        output_path = os.path.join(MAPS_DIR, "blended.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    logger.info(f"Blended карта: {output_path}")
    return output_path


def _credit_links(comp):
    """Списък {name, url} за страницата в GitHub Pages."""
    out, seen = [], set()
    for s in comp.get("sources", []):
        if s in CREDIT_LINKS and CREDIT_LINKS[s][0] not in seen:
            name, url = CREDIT_LINKS[s]
            out.append({"name": name, "url": url})
            seen.add(name)
    out += [{"name": n, "url": u} for n, u in FIXED_LINKS]
    return out


def generate_pages_data(comp, forecast, blended_dbz, blend_times,
                        output_dir=None):
    """Генерира JSON метаданни за GitHub Pages dashboard."""
    if output_dir is None:
        output_dir = MAPS_DIR

    meta = {
        "generated_utc": dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "observation_time": comp["timestamp"].strftime("%Y-%m-%dT%H:%M:%SZ"),
        "sources": comp.get("sources", []),
        "nowcast_minutes": 60,          # nowcast картата показва до +60 мин
        "blend_hours": 6,
        "domain": DOMAIN,
        "credits": _credit_links(comp),
    }

    meta_path = os.path.join(output_dir, "meta.json")
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    logger.info(f"Pages meta: {meta_path}")
    return meta_path
