"""
Lightning: Blitzortung WebSocket
================================
LZW decode, парсване, density grid.
"""

import os, sys, json, time, logging, threading
import datetime as dt
from collections import deque
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from config.settings import BLITZORTUNG, DOMAIN

logger = logging.getLogger("lightning")


def lzw_decode(data: bytes) -> str:
    d = list(data.decode("latin-1"))
    if not d:
        return ""
    e, c, f, g, h, o = {}, d[0], d[0], [d[0]], 256, 256
    for i in range(1, len(d)):
        a = ord(d[i])
        a = d[i] if h > a else (e[a] if a in e else f + c)
        g.append(a); c = a[0]; e[o] = f + c; o += 1; f = a
    return "".join(g)


def parse_strike(raw: str) -> dict | None:
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None
    if "lat" not in data or "lon" not in data:
        return None
    lat, lon = data["lat"], data["lon"]
    bb = BLITZORTUNG["bbox"]
    if not (bb["lat_min"] <= lat <= bb["lat_max"] and
            bb["lon_min"] <= lon <= bb["lon_max"]):
        return None
    ts = dt.datetime.fromtimestamp(data.get("time", 0) / 1e9, tz=dt.timezone.utc)
    return {"timestamp": ts, "lat": lat, "lon": lon,
            "polarity": data.get("pol", 0),
            "n_sensors": len(data.get("sig", []))}


class LightningCollector:
    def __init__(self, buffer_sec=None):
        self.buffer_sec = buffer_sec or BLITZORTUNG["collect_seconds"]
        self.strikes = deque(maxlen=10000)
        self._ws = None

    def _on_message(self, ws, msg):
        decoded = lzw_decode(msg) if isinstance(msg, bytes) else msg
        s = parse_strike(decoded)
        if s:
            self.strikes.append(s)

    def _on_open(self, ws):
        ws.send(BLITZORTUNG["subscribe_msg"])
        logger.info("Blitzortung WebSocket отворен")

    def _on_error(self, ws, err):
        logger.debug(f"WS грешка: {err}")

    def collect_sync(self, seconds=None) -> list[dict]:
        sec = seconds or self.buffer_sec
        logger.info(f"Blitzortung: събиране {sec}s...")
        try:
            import websocket
            self._ws = websocket.WebSocketApp(
                BLITZORTUNG["ws_url"],
                on_message=self._on_message,
                on_open=self._on_open, on_error=self._on_error)
            t = threading.Thread(target=self._ws.run_forever, daemon=True)
            t.start()
            time.sleep(sec)
            self._ws.close()
        except ImportError:
            logger.warning("websocket-client не е инсталиран")
        except Exception as e:
            logger.warning(f"Blitzortung грешка: {e}")

        strikes = list(self.strikes)
        logger.info(f"  {len(strikes)} мълнии за България")
        return strikes


def strikes_to_grid(strikes, resolution_km=2.0, sigma_km=5.0):
    dlat = resolution_km / 111.0
    dlon = resolution_km / (111.0 * np.cos(np.radians(42.75)))
    lat_1d = np.arange(DOMAIN["lat_min"], DOMAIN["lat_max"], dlat)
    lon_1d = np.arange(DOMAIN["lon_min"], DOMAIN["lon_max"], dlon)
    density = np.zeros((len(lat_1d), len(lon_1d)), dtype=np.float32)

    for s in strikes:
        iy = int((s["lat"] - DOMAIN["lat_min"]) / dlat)
        ix = int((s["lon"] - DOMAIN["lon_min"]) / dlon)
        if 0 <= iy < len(lat_1d) and 0 <= ix < len(lon_1d):
            density[iy, ix] += 1

    if sigma_km > 0 and density.max() > 0:
        from scipy.ndimage import gaussian_filter
        density = gaussian_filter(density, sigma=sigma_km / resolution_km)

    return density, lat_1d, lon_1d
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(name)s] %(message)s")
    collector = LightningCollector()
    strikes = collector.collect_sync(seconds=30)
    print(f"\nСъбрани {len(strikes)} мълнии:")
    for s in strikes[:10]:
        print(f"  {s['timestamp']}  ({s['lat']:.4f}, {s['lon']:.4f})")
    if strikes:
        density, lat, lon = strikes_to_grid(strikes)
        print(f"Density grid: {density.shape}, max={density.max():.3f}")