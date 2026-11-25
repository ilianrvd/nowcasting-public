# METAR: `obs/metar_latest.json`

Публикува се в Pages: https://ilianrvd.github.io/nowcasting-public/obs/metar_latest.json

Источник: Aviation Weather Center (AWC, `aviationweather.gov/api/data/metar`). Летища: LBSF, LBBG, LBWN, LBPD, LBGO. METAR и SPECI за последните 3 часа, най-старото първо. Файлът се обновява с всеки пуск (на ~20 мин); Pages връща `Cache-Control: max-age=600`.

Личен проект, не е официален продукт на сертифициран доставчик. Контакт: Илиан Манафов, ilianm@mail.bg.

```json
{
  "generated_at": "2026-11-25T07:05:12Z",
  "fetched_at": "2026-11-25T07:05:10Z",
  "source": "awc",
  "reports": [
    {"id": "LBSF", "time": "2026-11-25T06:00:00Z", "t": -1, "td": -1,
     "wind_kt": 0, "wind_dir": null, "vis_m": 300, "rvr_m": 600, "wx": ["FG"],
     "cloud_base_ft": null, "vv_ft": 100,
     "raw": "METAR LBSF 250600Z 00000KT 0300 R27/0600N FG VV001 M01/M01 Q1030 NOSIG="}
  ]
}
```

| Поле | Описание |
|---|---|
| `generated_at` | Кога е сглобен файлът (UTC, ISO 8601, `Z`). |
| `fetched_at` | Кога последно е успяло изтеглянето от AWC. При неуспех файлът се публикува пак, със съобщенията от кеша и с оригиналните им `time`; `fetched_at` остава старо. Възрастта на данните се гледа по `time` и `fetched_at`. |
| `source` | `awc`. |
| `id` | ICAO код (виж `static/points.json`). |
| `time` | Часът на наблюдението от съобщението, UTC. |
| `t`, `td` | °C, цели градуси (както в METAR). `null` ако липсват. |
| `wind_kt` | Средна скорост в kt (MPS/KMH, ако се появят, се превръщат). |
| `wind_dir` | Градуси, откъдето духа; `null` при VRB и при щил (00000KT). |
| `vis_m` | Преобладаваща видимост в метри; CAVOK и 9999 са 10000. |
| `rvr_m` | Най-ниската RVR в метри (`P`/`M` префиксите се игнорират), `null` ако няма. |
| `wx` | Групите на явленията от основната част, както са в съобщението (`FG`, `BR`, `BCFG`, `-RA`, `+TSRA`, `VCFG`…). Без тренд (TEMPO/BECMG) и без последни явления (`RE…`). |
| `cloud_base_ft` | Основата на най-ниския слой BKN/OVC (таван) във фута; `null` ако няма. |
| `vv_ft` | Вертикална видимост във фута; `null` ако няма. |
| `raw` | Оригиналното съобщение, винаги се пази. |

Всички полета са от самото съобщение (`raw`), а не от AWC JSON: той дава видимост в мили.

Вътрешното състояние „метар → цвят на летището“ (`embed/data/sites.json`) не се променя.
