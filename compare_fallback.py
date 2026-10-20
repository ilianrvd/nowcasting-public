"""
Сравнение на резерва ICON + мълнии с H40B за един и същ момент.
Пуска се от C:\\nowcasting_public, когато H40B работи и има мълнии в южна България:
    python compare_fallback.py
Пуска nowcast два пъти (с H40B и с --force-icon-ltg) и прави compare\\fallback_<час>.png
с двете nowcast карти една до друга. Прати ми PNG-то и реда „Резерв ICON+LI“ от изхода.
"""
import os, shutil, subprocess, sys, datetime as dt
from PIL import Image

MAP = os.path.join("data", "output", "maps", "nowcast_latest.png")
os.makedirs("compare", exist_ok=True)
stamp = dt.datetime.utcnow().strftime("%Y%m%d_%H%M")
out = []
for tag, extra in (("h40b", []), ("icon_ltg", ["--force-icon-ltg"])):
    print(f"\n===== пуск: {tag} =====", flush=True)
    r = subprocess.run([sys.executable, "run_nowcast.py", *extra], capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    for line in (r.stdout + r.stderr).splitlines():
        if any(k in line for k in ("H40B", "Резерв", "Без радар", "Попълване")):
            print("  " + line)
    dst = os.path.join("compare", f"{tag}_{stamp}.png")
    shutil.copy(MAP, dst)
    out.append(Image.open(dst).convert("RGB"))
w = sum(i.width for i in out); h = max(i.height for i in out)
both = Image.new("RGB", (w, h), "white")
x = 0
for i in out:
    both.paste(i, (x, 0)); x += i.width
path = os.path.join("compare", f"fallback_{stamp}.png")
both.save(path)
print(f"\nГотово: {path}  (вляво H40B, вдясно ICON + мълнии)")
