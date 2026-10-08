#!/usr/bin/env python3
"""Merge the two blind relabels (eval/v2/relabel/A-*, B-*) into eval/v2/all.jsonl.

Rows where A and B list the same foods: items from A; any field where B differs becomes an accepted
alternative (item["alt"][field] = [values]); food synonyms from B and the original writer go into accept.
Rows where they list different foods: taken from eval/v2/adjudicated.jsonl if present, else written
to eval/v2/relabel/disputed.jsonl for adjudication (and left out of all.jsonl until adjudicated).

Reads eval/v2/orig.jsonl (the original writers' labels after scripts/eval2_clean.py).
"""
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from score_parse import score_row  # noqa: E402

V2 = ROOT / "eval" / "v2"
FIELDS = ("qty", "unit", "size", "state", "brand", "prep")

def load(pattern):
    out = {}
    for p in sorted((V2 / "relabel").glob(pattern)):
        out.update({r["id"]: r["items"] for r in map(json.loads, open(p))})
    return out

def normalize(it):
    it = dict(it)
    it.setdefault("accept", []); it.setdefault("alt", {}); it.setdefault("db_hint", "")
    if not isinstance(it["accept"], list): it["accept"] = [it["accept"]] if it["accept"] else []
    return it

def expand_coffee(it, text):
    """Frozen rule 2: Tim Hortons "double double" = coffee + 2 tbsp cream + 2 tsp sugar (In-N-Out's is a burger)."""
    m = {"double double": 2, "triple triple": 3, "four by four": 4}.get(it["food"])
    if not m or "in-n-out" in (it["brand"] + text.lower()) or "animal" in text.lower(): return [it]
    base = {"brand": "", "size": "", "state": "", "prep": "", "accept": [], "alt": {}, "db_hint": ""}
    return [{**it, "food": "coffee", "brand": "tim hortons", "unit": "serving", "accept": ["brewed coffee"]},
            {**base, "food": "cream", "qty": m * it["qty"], "unit": "tbsp", "accept": ["coffee cream", "18% cream"]},
            {**base, "food": "sugar", "qty": m * it["qty"], "unit": "tsp", "accept": ["white sugar"]}]

def main():
    orig = {r["id"]: r for r in map(json.loads, open(V2 / "orig.jsonl"))}
    A, B = load("A-*.jsonl"), load("B-*.jsonl")
    adj_path = V2 / "adjudicated.jsonl"
    adj = {r["id"]: r for r in map(json.loads, open(adj_path))} if adj_path.exists() else {}
    out, disputed, n = [], [], {"agreed": 0, "adjudicated": 0, "disputed": 0}
    for rid, o in orig.items():
        a, b = A[rid], B[rid]
        ga = [{**i, "accept": []} for i in a]          # gold-shaped copies of A's items
        s = score_row(b, ga)
        if rid in adj:
            if adj[rid].get("drop"): n["dropped"] = n.get("dropped", 0) + 1; continue
            items = adj[rid]["items"]; n["adjudicated"] += 1
        elif s["row_food"]:
            pairs = {id(g): p for p, g in s["pairs"]}
            om = score_row(o["expected"]["items"], ga)
            opairs = {id(g): p for p, g in om["pairs"]}
            items = []
            for it, g in zip(a, ga):
                bi, oi = pairs[id(g)], opairs.get(id(g))
                acc = {bi["food"]} | ({oi["food"], *oi.get("accept", [])} if oi else set())
                alt = {f: [bi[f]] for f in FIELDS if f != "prep" and bi[f] != it[f]}
                items.append({**it, "accept": sorted(x for x in acc if x != it["food"]), "alt": alt,
                              "db_hint": oi.get("db_hint", "") if oi else ""})
            n["agreed"] += 1
        else:
            disputed.append({"id": rid, "text": o["text"], "A": a, "B": b,
                             "orig": [{k: v for k, v in i.items() if k not in ("db_hint",)} for i in o["expected"]["items"]]})
            n["disputed"] += 1
            continue
        items = [normalize(it) for it in items]
        items = [x for it in items for x in expand_coffee(it, o["text"])]
        out.append({"id": rid, "cat": o["cat"], "text": o["text"], "expected": {"items": items},
                    "notes": o.get("notes", ""), "label": "adjudicated" if rid in adj else "agreed"})
    with open(V2 / "all.jsonl", "w") as f:
        for r in out: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(V2 / "relabel" / "disputed.jsonl", "w") as f:
        for r in disputed: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(n, f"-> {len(out)} rows in all.jsonl, {len(disputed)} disputed")

if __name__ == "__main__":
    main()
