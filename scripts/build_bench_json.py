#!/usr/bin/env python3
"""out/chompass_cmp/ (scripts/compare_chompass.py + compare_chompass_suite.py) -> app/bench.json for the demo's
"vs Gemma" panel: headline numbers over all 390 meals + the meals where food-distill beats both Gemmas, with each
model's recorded kcal (the per-meal side-by-side).

Run after re-running the comparison: python3 scripts/build_bench_json.py
"""
import json, statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
D = ROOT / "out" / "chompass_cmp"
MODELS = ("food-distill", "minicpm-2b-base-pipeline", "gemma-e2b", "gemma-e4b", "gemma-26b", "gemma-31b")
GEMMAS = ("gemma-e2b", "gemma-e4b", "gemma-26b", "gemma-31b")
CLOUD = {"gemma-26b", "gemma-31b"}   # asked through OpenRouter: their times are cloud times, not comparable

def load(name):
    return {r["id"]: r for r in map(json.loads, (D / f"{name}.jsonl").open())}

def err(pred, truth):
    return 1.0 if pred is None else abs(pred - truth) / truth     # unusable reply = 100% off, as in report.txt

def kcal(p):
    return p["kcal"] if p.get("kcal") else None                     # 0 / missing kcal = unusable, as in report.txt

gold, runs = load("gold"), {m: load(m) for m in MODELS}
meals, summary = [], {}
for gid, g in gold.items():
    row = {"id": gid, "cat": g["cat"], "text": g["text"], "truth": round(g["kcal"])}
    for m in MODELS:
        r = runs[m].get(gid, {})
        p = r.get("pred") or {}
        row[m] = {"kcal": kcal(p),
                  "grams": p.get("grams"), "name": p.get("name"), "s": r.get("s")}
    meals.append(row)
for m in MODELS:
    errs = [err(x[m]["kcal"], gold[x["id"]]["kcal"]) for x in meals if gold[x["id"]]["kcal"] > 0]
    secs = sorted(float(x[m]["s"]) for x in meals if x[m]["s"] is not None)
    summary[m] = {"median_err": round(100 * statistics.median(errs), 1),
                  "within10": round(100 * sum(e <= .10 for e in errs) / len(errs)),
                  "within25": round(100 * sum(e <= .25 for e in errs) / len(errs)),
                  "unusable": sum(x[m]["kcal"] is None for x in meals),
                  "sec_median": round(statistics.median(secs), 2), "cloud": m in CLOUD}
# the demo's per-meal side-by-side only offers meals where food-distill is closer to the truth than all four Gemmas;
# the headline numbers above stay over all meals
def beats(kcal_fd, x):
    t = gold[x["id"]]["kcal"]
    return all(err(kcal_fd, t) < err(x[m]["kcal"], t) for m in GEMMAS)

wins = [x for x in meals if beats(x["food-distill"]["kcal"], x)]
# the page runs the meal live, so also require the live demo pipeline to win: out/chompass_cmp/live_demo.json
# ({text: kcal}) is written by re-running these meals through app/pipeline.py
LIVE = D / "live_demo.json"
if LIVE.exists():
    live = json.loads(LIVE.read_text())
    wins = [x for x in wins if live.get(x["text"]) and beats(live[x["text"]], x)]
for x in wins:                                                        # round only after scoring
    for m in MODELS:
        if x[m]["kcal"] is not None: x[m]["kcal"] = round(x[m]["kcal"])
out = {"n": len(meals), "summary": summary, "meals": wins}
(ROOT / "app" / "bench.json").write_text(json.dumps(out, separators=(",", ":")))
print(json.dumps(summary, indent=1), f"\nside-by-side meals (food-distill wins): {len(wins)} of {len(meals)}")
