# Nowcasting Public — България 🇧🇬⛈

Публична nowcasting система за конвективни явления над България.
Изцяло базирана на свободно достъпни данни.

## Източници на данни

| Източник | Формат | Покритие | Обновяване |
|----------|--------|----------|------------|
| 🇷🇴 Румъния (ANM) | ODIM HDF5 COMPOSITE | Цяла Румъния + Северна България | ~10 мин |
| 🇧🇬 ИАБГ Голям Чардак | PNG (CAPPI 2.5 km) | Тракийска низина, Родопи | ~4 мин |
| 🇧🇬 ИАБГ Старо Село | PNG (CAPPI 2.5 km) | Югоизточна България | ~4 мин |
| 🇧🇬 ИАБГ Бърдарски геран | PNG (CAPPI 2.5 km) | Северозападна България | ~4 мин |
| ⚡ Blitzortung | WebSocket JSON | Пълно покритие | Real-time |
| 🌤 ICON-EU (DWD) | Open-Meteo API | Цяла Европа | 1 час |

## Архитектура

```
Входни данни
├── Румъния COMPOSITE (ODIM HDF5)
├── ИАБГ GCD/STS/BRD (PNG → dBZ)
├── Blitzortung мълнии (WebSocket)
└── ICON-EU precipitation (Open-Meteo)
         ↓
    MAX-Z Composite
         ↓
    S-PROG Nowcast (0–60 мин)
         ↓
    Blending с ICON-EU (0–6 часа)
         ↓
    PNG карти + GitHub Pages
```

## Blending схема

```
0–30 мин   → 100% радар
30–60 мин  → 90% радар + 10% ICON
60–90 мин  → 70% радар + 30% ICON
90–120 мин → 50% радар + 50% ICON
120–180 мин → 20–50% радар + 50–80% ICON
180–360 мин → 0–20% радар + 80–100% ICON
```

## Стартиране

```bash
# Пълен pipeline
python run_nowcast.py

# Без мълнии и ICON (само радарна екстраполация)
python run_nowcast.py --no-lightning --no-icon

# Само определени ИАБГ радари
python run_nowcast.py --iabg GCD,STS

# Без Румъния
python run_nowcast.py --no-romania
```

## GitHub Pages

Dashboard-ът се обновява автоматично чрез GitHub Actions на всеки 10 минути.
Настрой GitHub Pages от `docs/` директорията.

## Зависимости

```bash
pip install -r requirements.txt
```
