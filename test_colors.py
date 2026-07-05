"""Анализ на цветовете в ИАБГ радарна картинка за калибриране на colormap."""

import requests
import numpy as np
from PIL import Image
from io import BytesIO
from collections import Counter

# Свали JSON с файлове
resp = requests.get("https://www.weathermod-bg.eu/wr/json/GCD.json")
data = resp.json()
urls = [v["wh_img"] for v in data["img_list"].values()]
print(f"Налични: {len(urls)} картинки")

# Свали първата
print(f"Сваляне: {urls[0].split('/')[-1]}")
resp = requests.get(urls[0], timeout=20)
img = np.array(Image.open(BytesIO(resp.content)).convert("RGB"))
print(f"Размер: {img.shape}")

# Намери цветните пиксели (без сиво/черно/бяло)
pixels = img.reshape(-1, 3)
mask = ~((pixels[:, 0] == pixels[:, 1]) & (pixels[:, 1] == pixels[:, 2]))
colored = [tuple(p) for p in pixels[mask]]

print(f"\nЦветни пиксели: {len(colored)} от {len(pixels)}")
print(f"\nТоп 30 цвята:")
for rgb, cnt in Counter(colored).most_common(30):
    print(f"  RGB({rgb[0]:3d}, {rgb[1]:3d}, {rgb[2]:3d})  count={cnt:6d}")

# Запази картинката локално за визуална проверка
Image.open(BytesIO(requests.get(urls[0]).content)).save("data/radar/gcd/test_image.png")
print(f"\nКартинка запазена: data/radar/gcd/test_image.png")
