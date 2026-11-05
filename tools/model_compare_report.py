"""
Обобщение на сравнението ALADIN ↔ ICON (tools/model_compare.py)
================================================================
Чете CSV файловете от клона model-compare и дава FSS за целия период
(FSS = 1 − Σnum / Σden — правилното сумиране, не средно от часовите FSS)
и честотното отклонение (Σ дял с валеж в модела / Σ дял в наблюдението).

  git fetch origin model-compare && git worktree add ../mc origin/model-compare
  python tools/model_compare_report.py ../mc/results/*.csv
  python tools/model_compare_report.py ../mc/results/*.csv --from 2026-12-01 --wet 0.02

--wet: само часове, в които наблюдението има валеж ≥0.5 mm/h в поне този дял от мрежата
(отделя валежните ситуации от „модел вали, а реално е сухо“).
"""

import csv, glob, argparse
from collections import defaultdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--from", dest="t0", default="")
    ap.add_argument("--to", dest="t1", default="9999")
    ap.add_argument("--wet", type=float, default=0.0)
    a = ap.parse_args()

    files = sorted({f for p in a.files for f in glob.glob(p)})
    num, den, ofr, ffr, n = (defaultdict(float) for _ in range(5))
    hours = set()
    wet_hours = {}
    rows = []
    for fn in files:
        with open(fn, newline="") as f:
            rows += list(csv.DictReader(f))
    for r in rows:                                   # дял с валеж на часа при 0.5 mm/h
        if r["thr"] in ("0.5", "0.50") and r["model"] == "icon" and r["scale_km"] == "10":
            wet_hours[r["valid"]] = float(r["obs_frac"])
    for r in rows:
        if not (a.t0 <= r["valid"][:10] <= a.t1):
            continue
        if wet_hours.get(r["valid"], 0.0) < a.wet:
            continue
        k = (r["model"], int(r["lead_min"]), float(r["thr"]), int(r["scale_km"]))
        num[k] += float(r["num"])
        den[k] += float(r["den"])
        ofr[k] += float(r["obs_frac"])
        ffr[k] += float(r["fc_frac"])
        n[k] += 1
        hours.add(r["valid"])

    if not n:
        print("Няма редове за избрания период.")
        return
    models = sorted({k[0] for k in n}, key=lambda m: ["icon", "aladin", "aladin_s"].index(m)
                    if m in ("icon", "aladin", "aladin_s") else 9)
    leads = sorted({k[1] for k in n})
    thrs = sorted({k[2] for k in n})
    scales = sorted({k[3] for k in n})
    print(f"Часове: {len(hours)}  ({min(hours)} … {max(hours)})"
          + (f", само с валеж ≥ {a.wet:.0%}" if a.wet else ""))

    for thr in thrs:
        print(f"\n=== Праг {thr} mm/h ===")
        hdr = f"{'lead':>5} {'km':>4} " + " ".join(f"{m:>9}" for m in models)
        print("FSS (по-високо = по-добре)\n" + hdr)
        for lead in leads:
            for km in scales:
                cells = []
                for m in models:
                    k = (m, lead, thr, km)
                    cells.append(f"{1 - num[k] / den[k]:9.3f}" if den[k] > 0 else f"{'—':>9}")
                print(f"{lead:>5} {km:>4} " + " ".join(cells))
        print("Честотно отклонение (1 = точно, >1 = прекалено много валеж)")
        for lead in leads:
            cells = []
            for m in models:
                k = (m, lead, thr, scales[0])
                cells.append(f"{ffr[k] / ofr[k]:9.2f}" if ofr[k] > 0 else f"{'—':>9}")
            print(f"{lead:>5} {'':>4} " + " ".join(cells))


if __name__ == "__main__":
    main()
