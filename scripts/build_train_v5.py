#!/usr/bin/env python3
"""v5 training set: parse rows (teacher-labeled texts) + typo augmentation + pick rows -> data/train.jsonl,
data/val.jsonl in chat form ({"messages": [...]}) for scripts/train_student.py.

The student gets SHORT system prompts (prompts/student_*_v5.txt): it learns the contract from data,
and short prompts make training ~3x cheaper and runtime prefill fast.

Texts near any eval text (same normalized text or >= 0.8 token overlap) are dropped.

  python3 scripts/build_train_v5.py [--labels data/gen/v5_labels.jsonl] [--picks data/gen/v5_picks.jsonl]
"""
import argparse, json, random, re, sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PARSE_SYS = (ROOT / "prompts" / "student_parse_v5.txt").read_text().strip()
sys.path.insert(0, str(ROOT / "app"))
from fooddb import PICK_SYSTEM, pick_prompt_lines  # noqa: E402

UNIT_WORDS = set("g kg oz lb ml l tbsp tsp cup cups slice slices piece pieces bowl plate handful clove can bottle "
                 "glass pint serving scoop shot lbs ozs".split())

def toks(t): return re.findall(r"[a-z0-9]+", t.lower())

def near_eval_filter(eval_texts):
    sets = [set(toks(t)) for t in eval_texts]
    exact = {" ".join(toks(t)) for t in eval_texts}
    inv = defaultdict(list)
    for i, s in enumerate(sets):
        for w in s: inv[w].append(i)
    def bad(text):
        tt = toks(text)
        if " ".join(tt) in exact: return True
        s = set(tt)
        if len(s) < 3: return False
        cand = Counter(i for w in s for i in inv.get(w, ()))
        return any(n / len(s | sets[i]) >= 0.8 for i, n in cand.items() if n >= 0.8 * len(s))
    return bad

def typo(word, rng):
    i = rng.randrange(1, len(word) - 1)
    op = rng.choice("swap drop double vowel".split())
    if op == "swap": return word[:i] + word[i + 1] + word[i] + word[i + 2:]
    if op == "drop": return word[:i] + word[i + 1:]
    if op == "double": return word[:i] + word[i] + word[i:]
    v = "aeiou"; return word[:i] + (rng.choice(v) if word[i] in v else word[i]) + word[i + 1:]

def augment(text, rng):
    """1-2 typos in alphabetic words (not units/numbers); the label stays valid."""
    words = text.split(" ")
    idx = [i for i, w in enumerate(words) if w.isalpha() and len(w) > 4 and w.lower() not in UNIT_WORDS]
    if not idx: return None
    for i in rng.sample(idx, min(len(idx), rng.choice((1, 1, 2)))): words[i] = typo(words[i], rng)
    out = " ".join(words)
    return out if out != text else None

def parse_row(text, items):
    return {"messages": [{"role": "system", "content": PARSE_SYS}, {"role": "user", "content": text},
                         {"role": "assistant", "content": json.dumps({"items": items}, ensure_ascii=False,
                                                                     separators=(",", ":"))}], "task": "parse"}

def pick_row(p):
    return {"messages": [{"role": "system", "content": PICK_SYSTEM},
                         {"role": "user", "content": pick_prompt_lines(p["item"], p["cands"])},
                         {"role": "assistant", "content": str(p["pick"])}], "task": "pick"}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default=str(ROOT / "data" / "gen" / "v5_labels.jsonl"))
    ap.add_argument("--texts", default=str(ROOT / "data" / "gen" / "v5_texts.jsonl"))
    ap.add_argument("--picks", default=str(ROOT / "data" / "gen" / "v5_picks.jsonl"))
    ap.add_argument("--aug", type=float, default=0.2, help="typo copies as a fraction of parse rows")
    ap.add_argument("--val", type=float, default=0.03)
    a = ap.parse_args()
    rng = random.Random(17)
    texts = {r["id"]: r["text"] for r in map(json.loads, open(a.texts))}
    bad = near_eval_filter([json.loads(l)["text"] for l in open(ROOT / "eval" / "v2" / "orig.jsonl")])
    labels = [r for r in map(json.loads, open(a.labels)) if r["id"] in texts]
    kept = [r for r in labels if not bad(texts[r["id"]])]
    print(f"parse: {len(labels)} labeled, {len(labels) - len(kept)} dropped as near-eval")
    ids = sorted({r["id"] for r in kept}); rng.shuffle(ids)
    val_ids = set(ids[: int(len(ids) * a.val)])
    train, val = [], []
    for r in kept:
        (val if r["id"] in val_ids else train).append(parse_row(texts[r["id"]], r["items"]))
    n_aug = 0
    for r in rng.sample([r for r in kept if r["id"] not in val_ids], int(len(kept) * a.aug)):
        t = augment(texts[r["id"]], rng)
        if t and not bad(t): train.append(parse_row(t, r["items"])); n_aug += 1
    picks = [json.loads(l) for l in open(a.picks)] if Path(a.picks).exists() else []
    for p in picks:
        (val if p["id"] in val_ids else train).append(pick_row(p))
    rng.shuffle(train)
    for name, rows in (("train", train), ("val", val)):
        with open(ROOT / "data" / f"{name}.jsonl", "w") as f:
            for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    stats = {"train": len(train), "val": len(val), "typo_aug": n_aug, "picks": len(picks),
             "train_tasks": dict(Counter(r["task"] for r in train)),
             "pick_none_frac": round(sum(p["pick"] == 0 for p in picks) / max(1, len(picks)), 3)}
    print(json.dumps(stats))
    with open(ROOT / "data" / "metrics.jsonl", "a") as f:
        f.write(json.dumps({"ts": __import__("datetime").datetime.now().isoformat(timespec="seconds"),
                            "phase": "v5", "event": "dataset_built", "data": stats}) + "\n")

if __name__ == "__main__":
    main()
