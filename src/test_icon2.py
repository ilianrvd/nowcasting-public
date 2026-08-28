import urllib.request, json
import datetime as dt

r = urllib.request.urlopen(
    'https://api.open-meteo.com/v1/dwd-icon?latitude=42.7&longitude=25.5'
    '&hourly=showers,convective_cloud_top,lightning_potential'
    '&models=icon_eu&forecast_days=1&timeformat=unixtime'
)
d = json.loads(r.read())
h = d['hourly']

for i, t in enumerate(h['time']):
    s = h['showers'][i] or 0
    c = h['convective_cloud_top'][i] or 0
    l = h['lightning_potential'][i] or 0
    if s > 0 or c > 0 or l > 0:
        ts = dt.datetime.fromtimestamp(t, tz=dt.timezone.utc)
        print(f'{ts.strftime("%H:%M")}  showers={s:.1f}mm  '
              f'cloud_top={c:.0f}m  LPI={l:.2f}')