#!/usr/bin/env python3
"""Pick-task training labels: for parsed items, run the real search (app/fooddb.candidates) and have an
OpenRouter teacher choose the matching candidate — the same decision the student makes at runtime
(app/fooddb.PICK_SYSTEM / pick_prompt), so the student learns it from a stronger model.

Several items per request (amortizes instructions), resumable, hard budget cap.

  python3 scripts/or_pick.py LABELS.jsonl OUT.jsonl [--model ...] [--k 10] [--budget 1] [--limit N]
LABELS rows: {"id", "items": [...]} (from scripts/or_label.py). OUT rows:
  {"id", "i", "item", "cands": [describe lines], "cand_ids": [...], "pick": n (1-based, 0 = none)}
"""
import argparse, json, os, random, sys, threading, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "app")); sys.path.insert(0, str(ROOT / "scripts"))
import fooddb  # noqa: E402
from or_label import call  # noqa: E402

SYSTEM = """You are a nutritionist matching logged foods to a nutrition database (values per 100 g).
For each numbered case, choose the candidate that is the same food as logged:
- the same product when a brand or restaurant is given (else the closest generic equivalent);
- the same raw/cooked state as stated (raw grains = dry grains); "" means the food as normally eaten;
- generic reference rows are preferred for generic foods; avoid rows with implausible numbers;
- 0 if no candidate is the same food or a close equivalent.
Output ONLY a JSON array: [{"case": 1, "pick": <number>}, ...]"""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("labels"); ap.add_argument("out")
    ap.add_argument("--model", default="anthropic/claude-haiku-5.5")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--per", type=int, default=12, help="cases per request")
    ap.add_argument("--budget", type=float, default=1.0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="max items")
    ap.add_argument("--reasoning", default="")
    a = ap.parse_args()
    key = os.environ["OPENROUTER_API_KEY"]
    c = fooddb.connect()
    done = {(r["id"], r["i"]) for r in map(json.loads, open(a.out))} if Path(a.out).exists() else set()
    cases = []
    for r in map(json.loads, open(a.labels)):
        for i, it in enumerate(r["items"]):
            if (r["id"], i) not in done: cases.append((r["id"], i, it))
    random.Random(3).shuffle(cases)
    cases = cases[: a.limit or None]
    print(f"{len(cases)} items to pick, k={a.k}", flush=True)
    lock = threading.Lock(); spent = [0.0]; kept = [0]; out = open(a.out, "a"); t0 = time.time()
    batches = [cases[j:j + a.per] for j in range(0, len(cases), a.per)]

    def work(bi):
        if spent[0] + a.workers * spent[0] / max(1, bi) >= a.budget: return
        rows = []
        for rid, i, it in batches[bi]:
            with lock: cands = fooddb.candidates(c, it, a.k)
            if cands: rows.append((rid, i, it, cands))
        if not rows: return
        user = "\n\n".join(f"Case {n}. {fooddb.pick_prompt(it, cands)}" for n, (_, _, it, cands) in enumerate(rows, 1))
        txt, cost = call(a.model, [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
                         key, 6000, a.reasoning)
        s, e = txt.find("["), txt.rfind("]")
        try: arr = json.loads(txt[s:e + 1]) if 0 <= s < e else []
        except json.JSONDecodeError: arr = []
        got = {}
        for o in arr:
            try: got[int(o["case"])] = int(o["pick"])
            except (KeyError, TypeError, ValueError): pass
        with lock:
            spent[0] += cost
            for n, (rid, i, it, cands) in enumerate(rows, 1):
                p = got.get(n)
                if p is None or not 0 <= p <= len(cands): continue
                out.write(json.dumps({"id": rid, "i": i, "item": it, "cands": [fooddb.describe(x) for x in cands],
                                      "cand_ids": [x["id"] for x in cands], "pick": p}, ensure_ascii=False) + "\n")
                kept[0] += 1
            out.flush()
            if (bi + 1) % 25 == 0:
                print(f"req {bi + 1}/{len(batches)} kept={kept[0]} spent=${spent[0]:.3f} ({time.time() - t0:.0f}s)", flush=True)

    with ThreadPoolExecutor(a.workers) as ex:
        list(ex.map(work, range(len(batches))))
    out.close()
    print(f"DONE kept {kept[0]} picks, ${spent[0]:.3f} (${spent[0] / max(1, kept[0]) * 1000:.3f}/1k)")

if __name__ == "__main__":
    main()
