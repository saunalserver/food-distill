#!/usr/bin/env python3
"""Retrieval recall on match-labeled eval items: is an acceptable row (gold `ok`) among the top-k candidates?
Compares app/fooddb.candidates (current) with candidates_v1 (2026-10-07 morning). Prints misses with --show."""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "app"))
import fooddb
c = fooddb.connect()
gold = {r["id"]: r for r in map(json.loads, open(ROOT / "eval" / "v2" / "all.jsonl"))}
labs = [r for p in sorted((ROOT / "eval" / "v2" / "match").glob("match-*.jsonl")) for r in map(json.loads, open(p))]
show = "--show" in sys.argv
for k in (1, 5, 10, 15):
    res = {}
    for name, fn in (("v1", fooddb.candidates_v1), ("new", fooddb.candidates)):
        hit = n = 0
        for m in labs:
            if not m["ok"] or m["id"] not in gold: continue
            items = gold[m["id"]]["expected"]["items"]
            if m["i"] >= len(items) or items[m["i"]]["food"] != m["food"]: continue
            it = items[m["i"]]; n += 1
            cands = fn(c, it, k); ok = set(m["ok"])
            got = any(x["id"] in ok for x in cands); hit += got
            if show and name == "new" and k == 10 and not got:
                best = c.execute("select name, brand, src from food where id = ?", (m["best"],)).fetchone()
                print(f"MISS {it['food']!r} b={it['brand']!r} st={it['state']!r} | best: {best[0]} [{best[1]}] ({best[2]}) | got: {[x['name'][:28] for x in cands[:3]]}")
        res[name] = f"{hit}/{n} = {hit / max(1, n):.3f}"
    print(f"recall@{k}: {res}")
