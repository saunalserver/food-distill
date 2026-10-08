#!/usr/bin/env python3
"""Run the Chompass in-repo benchmark manifests (docs/benchmarks/food_accuracy) through our
three setups, scored with Chompass's own score.py definitions so the numbers line up with
their published baselines: parse_ok_rate, wmape = Σ|err|/Σ|gt| over kcal+protein+carbs+fat
combined (parsed rows), within_20pct_calories_rate, mae_calories.

A parsed reply with a missing kcal counts as a parse failure (their ParsedPrediction asserts
non-null calories); a missing protein/carbs/fat is skipped from the wmape sums like their
null-filtered means.

  python3 scripts/compare_chompass_suite.py run text|realistic NAME URL MODEL
  python3 scripts/compare_chompass_suite.py report text|realistic NAME [NAME ...]
  python3 scripts/compare_chompass_suite.py check          # scorer self-check
"""
import json, os, sys, time, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from compare_chompass import PROMPT, parse_reply  # noqa: E402

CHOMPASS = Path(os.environ.get("CHOMPASS_DIR", Path.home() / "chompass"))   # a clone of codeberg.org/fitguy/Chompass
MANIFESTS = {
    "text": CHOMPASS / "docs/benchmarks/food_accuracy/manifest/eval_text.jsonl",
    "realistic": CHOMPASS / "docs/benchmarks/food_accuracy/manifest/eval_grounded_realistic_text.jsonl",
}
OUT = ROOT / "out" / "chompass_cmp"
NUTR = (("kcal", "calories"), ("protein", "protein_g"), ("carbs", "carbs_g"), ("fat", "fat_g"))


def load(set_):
    rows = [json.loads(l) for l in open(MANIFESTS[set_])]
    return [{"id": r["id"], "text": r["text"], "slice": r.get("slice", ""),
             **{k: r[g] for k, g in NUTR}} for r in rows]


def run(set_, name, url, model):
    OUT.mkdir(parents=True, exist_ok=True)
    rows = load(set_)
    path = OUT / f"suite_{set_}_{name}.jsonl"
    done = {json.loads(l)["id"] for l in open(path)} if path.exists() else set()

    def ask(text):
        body = {"model": model, "messages": [{"role": "user", "content": PROMPT % text}],
                "temperature": 0.1, "top_k": 40, "top_p": 0.95, "max_tokens": 2048, "seed": 1,
                "chat_template_kwargs": {"enable_thinking": False}}
        headers = {"Content-Type": "application/json"}
        if os.environ.get("FD_API_KEY"):   # hosted endpoint (OpenRouter) for the bigger Gemma models
            headers["Authorization"] = f"Bearer {os.environ['FD_API_KEY']}"
            body["reasoning"] = {"enabled": False}
        req = urllib.request.Request(url, json.dumps(body).encode(), headers)
        t0 = time.time()
        with urllib.request.urlopen(req, timeout=300) as r: resp = json.load(r)
        return resp, time.time() - t0

    ask("a banana")  # warm-up, not timed
    with open(path, "a") as f:
        for r in rows:
            if r["id"] in done: continue
            try:
                resp, dt = ask(r["text"])
                reply = resp["choices"][0]["message"].get("content") or ""
                out = {"id": r["id"], "s": round(dt, 2), "reply": reply, "pred": parse_reply(reply)}
            except Exception as e:  # noqa: BLE001 — a failed call is a result too
                out = {"id": r["id"], "s": None, "reply": "", "pred": None, "error": str(e)[:300]}
            f.write(json.dumps(out, ensure_ascii=False) + "\n"); f.flush()
    print(f"{set_}/{name}: done -> {path}")


def score_rows(gold_rows, res_rows):
    """Chompass score.py aggregates: returns dict per row-group."""
    g = {r["id"]: r for r in gold_rows}
    n = len(res_rows)
    parsed = [r for r in res_rows if r.get("pred") and r["pred"].get("kcal") is not None]
    abs_sum = gt_sum = 0.0
    kcal_mae, within, secs = [], [], []
    for r in parsed:
        p, t = r["pred"], g[r["id"]]
        for k, _ in NUTR:
            if p.get(k) is None or t[k] is None: continue  # null nutrient skipped like their _mean null-filter
            abs_sum += abs(p[k] - t[k]); gt_sum += abs(t[k])
        kcal_mae.append(abs(p["kcal"] - t["kcal"]))
        within.append(abs(p["kcal"] - t["kcal"]) / t["kcal"] <= 0.20 if t["kcal"] else None)
        if r.get("s"): secs.append(r["s"])
    w = [x for x in within if x is not None]
    return {"n": n, "parse_ok": len(parsed) / n if n else 0.0,
            "wmape": abs_sum / gt_sum if gt_sum else None,
            "kcal_mae": sum(kcal_mae) / len(kcal_mae) if kcal_mae else None,
            "within20": sum(w) / len(w) if w else None,
            "med_s": sorted(secs)[len(secs) // 2] if secs else None}


def report(set_, names):
    rows = load(set_)
    slices = sorted({r["slice"] for r in rows} - {""})
    for name in names:
        res = [json.loads(l) for l in open(OUT / f"suite_{set_}_{name}.jsonl")]
        a = score_rows(rows, res)
        print(f"## {name}: n={a['n']} parse_ok={a['parse_ok']:.0%} wmape={a['wmape']:.1%} "
              f"kcal_mae={a['kcal_mae']:.1f} within20={a['within20']:.0%} med={a['med_s']:.2f}s")
        for sl in slices:
            s = score_rows([r for r in rows if r["slice"] == sl], [r for r in res if g_id(r, rows) == sl])
            print(f"   {sl:10s} n={s['n']:2d} wmape={s['wmape']:.1%} within20={s['within20']:.0%}")
    print()


def g_id(r, rows):
    m = {x["id"]: x["slice"] for x in rows}
    return m.get(r["id"], "")


def check():
    gold = [{"id": "a", "text": "x", "slice": "", "kcal": 100.0, "protein": 10.0, "carbs": 20.0, "fat": 5.0},
            {"id": "b", "text": "y", "slice": "", "kcal": 200.0, "protein": None, "carbs": 0.0, "fat": 10.0}]
    res = [{"id": "a", "s": 1.0, "pred": {"kcal": 120.0, "protein": 12.0, "carbs": 16.0, "fat": 5.0}},
           {"id": "b", "s": 2.0, "pred": {"kcal": 100.0, "protein": None, "carbs": 0.0, "fat": 10.0}}]
    a = score_rows(gold, res)
    # a: |20|+|2|+|4|+|0| = 26 over 100+10+20+5=135; b: |100|+|0|+|0|=100 over 200+0+10=210 -> 126/345
    assert abs(a["wmape"] - 126 / 345) < 1e-9, a["wmape"]
    assert a["within20"] == 0.5 and a["parse_ok"] == 1.0 and abs(a["kcal_mae"] - 60.0) < 1e-9
    assert score_rows(gold, res[:1] + [{"id": "b", "pred": None}])["parse_ok"] == 0.5
    print("suite scorer self-checks pass")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "run": run(*sys.argv[2:6])
    elif cmd == "report": report(sys.argv[2], sys.argv[3:])
    elif cmd == "check": check()
    else: print(__doc__)
