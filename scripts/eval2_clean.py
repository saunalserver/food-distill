#!/usr/bin/env python3
"""Merge eval/v2/raw/*.jsonl -> eval/v2/orig.jsonl with the agreed cross-writer conventions applied.

Deterministic and re-runnable (raw files are never edited). Each change is logged to stdout.
  - grains (rice, pasta, noodles, oats, quinoa, ...) are raw unless cooking is said (project rule)
  - plain water is never an item; unbranded sparkling water neither
  - pints are their own unit
  - hand fixes listed in FIX (by id)
"""
import json, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = sorted((ROOT / "eval" / "v2" / "raw").glob("*.jsonl"))
OUT = ROOT / "eval" / "v2" / "orig.jsonl"   # then scripts/eval2_merge.py -> all.jsonl

GRAIN = {"rice", "pasta", "spaghetti", "penne", "macaroni", "noodle", "noodles", "oats", "quinoa", "couscous",
         "bulgur", "barley", "farro", "fusilli", "linguine", "fettuccine", "rigatoni", "orzo", "udon", "soba",
         "vermicelli", "millet", "buckwheat", "grits", "polenta", "rotini", "farfalle", "lasagna noodles"}
GRAIN_MOD = {"white", "brown", "jasmine", "basmati", "sticky", "glutinous", "wild", "long", "short", "grain",
             "sushi", "arborio", "whole", "wheat", "egg", "plain", "steamed", "black", "red", "rolled", "steel",
             "cut", "quick", "dry", "japanese", "thai", "rice", "parboiled", "converted", "extra", "side", "of"}
COOKED_HINT = re.compile(r"cook|fried|fry|boil|steam|bake|roast|grill|leftover|reheat|microwav|poach|saut|toast|"
                         r"stir|\bhot\b|warm|porridge|overnight")

DROP = {"world-0141"}                       # duplicate text of typos-0264
FIX = {  # id -> list of (item index, field, value); index None = whole items list
    "fastfood-0049": [(0, "qty", 4), (1, "size", "medium"), (4, "size", "medium")],   # Cane's Box Combo = 4 fingers
    "fastfood-0192": [(1, "size", "medium"), (5, "size", "medium")],
    "combos-0100": [(1, "qty", 1)], "combos-0101": [(1, "qty", 1)], "combos-0102": [(1, "qty", 1)],  # "eggs" unspecified -> 1
    "homecook-0073": [(1, "state", "raw")],   # "80g dry egg noodles"
    "packaged-0151": [(0, "unit", "pint")], "packaged-0155": [(0, "unit", "pint")],   # ice cream by the pint
}

def plain_grain(food):
    toks = food.split()
    return any(t in GRAIN for t in toks) and all(t in GRAIN or t in GRAIN_MOD for t in toks)

def cooked_said(text, food, prep):
    """A cooking word in prep, or within two words of the grain in the text ("leftover rice", "rice (cooked)")."""
    if COOKED_HINT.search(prep.lower()): return True
    words = [t for t in food.split() if t in GRAIN] or food.split()[-1:]
    for w in words:
        for m in re.finditer(rf"{w[:5]}\w*", text):
            near = " ".join(text[:m.start()].split()[-2:]) + " " + " ".join(text[m.end():].split()[:2])
            if COOKED_HINT.search(near): return True
    return False

def main():
    rows = [json.loads(l) for p in RAW for l in open(p) if l.strip()]
    out, n = [], {"grain": 0, "water": 0, "pint": 0, "fix": 0}
    for r in rows:
        if r["id"] in DROP: print("drop", r["id"]); continue
        text = r["text"].lower()
        items = r["expected"]["items"]
        for i, f, v in FIX.get(r["id"], []):
            print(f"fix {r['id']} item {i} {f}: {items[i][f]!r} -> {v!r}"); items[i][f] = v; n["fix"] += 1
        keep = []
        for it in items:
            food = it["food"]
            if re.fullmatch(r"(plain |tap |ice |bottled |still )?water", food) or (food == "sparkling water" and not it["brand"]):
                print(f"water {r['id']}: drop {food!r}"); n["water"] += 1; continue
            if plain_grain(food) and it["state"] != "raw" and not cooked_said(text, food, it["prep"]):
                print(f"grain {r['id']}: {food!r} {it['state']!r} -> 'raw'   | {r['text'][:70]}")
                it["state"] = "raw"; it["prep"] = re.sub(r"\b(steamed|cooked)\b,?\s*", "", it["prep"]).strip()
                n["grain"] += 1
            if re.search(r"\bpints?\b", text) and it["unit"] in ("glass", "serving") and \
                    re.search(r"beer|lager|ale\b|stout|ipa|guinness|cider|bitter|pilsner|porter", food):
                print(f"pint {r['id']}: {food!r} {it['qty']} {it['unit']} -> pint"); it["unit"] = "pint"; n["pint"] += 1
            keep.append(it)
        r["expected"]["items"] = keep
        out.append(r)
    with open(OUT, "w") as f:
        for r in out: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{len(out)} rows -> {OUT}  changes: {n}", file=sys.stderr)

if __name__ == "__main__":
    main()
