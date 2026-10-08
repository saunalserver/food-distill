#!/usr/bin/env python3
"""The untrained base model (MiniCPM5-2B, no fine-tune) inside the food-distill pipeline: same search, pick,
portion rules and output grammar; only the model differs. Without training it gets the full written contract and
the teacher's worked examples as its system prompt. Output rows match compare_chompass.py, so its `report` works.

  python3 scripts/compare_untrained.py URL [meals|text|realistic]     # URL = a llama-server running the base GGUF
  python3 scripts/compare_chompass.py report food-distill minicpm-2b-base-pipeline
"""
import json, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "app")); sys.path.insert(0, str(ROOT / "scripts"))
import fooddb, pipeline  # noqa: E402
from compare_chompass import OUT  # noqa: E402

NAME = "minicpm-2b-base-pipeline"
contract = (ROOT / "prompts" / "parse_system_v2.txt").read_text()
examples = "\n\n".join("Message: " + b.replace("\n", "\nParse: ", 1)          # same layout as the teacher prompt
                       for b in (ROOT / "prompts" / "teacher_examples_v2.txt").read_text().strip().split("\n\n"))
pipeline.PARSE_SYS = (contract + "\n\nWorked examples:\n\n" + examples +
                      '\n\nReply with only the JSON object {"items":[...]} for the user\'s meal.')


def main(url, kind="meals"):
    if kind == "meals":
        path, rows = OUT / f"{NAME}.jsonl", [json.loads(l) for l in open(OUT / "gold.jsonl")]
    else:
        import compare_chompass_suite as cs
        path, rows = OUT / f"suite_{kind}_{NAME}.jsonl", cs.load(kind)
    done = {json.loads(l)["id"] for l in open(path)} if path.exists() else set()
    c = fooddb.connect()
    with open(path, "a") as f:
        for r in rows:
            if r["id"] in done: continue
            t0 = time.time()
            try:
                items = pipeline.parse(r["text"], url=url, timeout=120)
                _, tot = pipeline.analyze(items, c=c, url=url)
                pred = {k: tot.get(k) for k in ("kcal", "protein", "carbs", "fat", "grams")} if items else None
                out = {"id": r["id"], "s": round(time.time() - t0, 2), "reply": json.dumps({"items": items}), "pred": pred}
            except Exception as e:  # noqa: BLE001 — a failed call is a result too
                out = {"id": r["id"], "s": None, "reply": "", "pred": None, "error": str(e)[:300]}
            f.write(json.dumps(out, ensure_ascii=False) + "\n"); f.flush()
    print(f"{NAME} {kind}: done -> {path}")


if __name__ == "__main__":
    if len(sys.argv) < 2: print(__doc__)
    else: main(*sys.argv[1:3])
