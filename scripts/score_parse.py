#!/usr/bin/env python3
"""Score v2 parses against eval/v2 labels. Library + CLI.

Items are matched by food name (normalized; the label's `accept` list counts), then fields are compared.
  row_exact : every item matched, none extra, and qty/unit/size/state all equal
  row_food  : same set of foods (names only)
  item P/R  : by food name
  field acc : among matched items

  python3 scripts/score_parse.py PRED.jsonl [GOLD.jsonl]     PRED rows: {"id", "items": [...]}
"""
import json, re, sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GOLD = ROOT / "eval" / "v2" / "all.jsonl"

def _sing(w):
    if len(w) > 4 and w.endswith("ies"): return w[:-3] + "y"
    if len(w) > 4 and re.search(r"(ches|shes|sses|xes|oes)$", w): return w[:-2]
    if len(w) > 3 and w.endswith("s") and not w.endswith(("ss", "us", "is")): return w[:-1]
    return w

def norm(s):
    s = str(s).lower().replace("&", " and ").replace("’", "'")
    toks = [_sing(t) for t in re.findall(r"[a-z0-9]+", s.replace("'", "")) if t not in ("a", "the", "of")]
    return " ".join(toks)

def names(it):
    return {norm(it["food"]), *(norm(a) for a in it.get("accept", []))} | {" ".join(sorted(norm(it["food"]).split()))}

# words that only restate the category ("parmesan cheese" = "parmesan", "french fries" = "fries")
FILLER = {"cheese", "bread", "lettuce", "sauce", "dressing", "coffee", "drink", "popcorn", "french", "plain",
          "fresh", "whole", "regular", "classic", "original", "homemade", "side", "ice", "cream"}

def _core(n):
    toks = [t for t in n.split() if t not in FILLER]
    return " ".join(sorted(toks)) if toks else n

def same_food(p, g):
    n = norm(p["food"])
    ns = names(g)
    return n in ns or " ".join(sorted(n.split())) in ns or _core(n) in {_core(x) for x in ns}

def qty_eq(a, b):
    try: return abs(float(a) - float(b)) <= 0.011 * max(1.0, abs(float(b)))
    except (TypeError, ValueError): return False

FIELDS = ("qty", "unit", "size", "state", "brand")

def field_eq(p, g, f):
    """Gold value or any accepted alternative (g["alt"][f], from labeler disagreement)."""
    vals = [g[f], *g.get("alt", {}).get(f, [])]
    if f == "qty": return any(qty_eq(p.get("qty"), v) for v in vals)
    return any(norm(p.get(f, "")) == norm(v) for v in vals)

def score_row(pred, gold):
    """pred, gold: item lists. Returns dict of counts for one row."""
    left, pairs, opt = list(pred or []), [], 0
    for g in sorted(gold, key=lambda g: bool(g.get("optional"))):     # required items claim matches first
        j = next((j for j, p in enumerate(left) if same_food(p, g)), None)
        if j is not None: pairs.append((left.pop(j), g)); opt += bool(g.get("optional"))
    req = sum(not g.get("optional") for g in gold)
    # optional items (plate components only some versions have) count neither as missed nor as extra
    out = {"n_gold": req + opt, "n_pred": len(pred or []), "n_match": len(pairs)}
    for f in FIELDS: out[f] = sum(field_eq(p, g, f) for p, g in pairs)
    full = all(field_eq(p, g, f) for p, g in pairs for f in ("qty", "unit", "size", "state"))
    out["row_food"] = not left and len(pairs) - opt == req
    out["row_exact"] = out["row_food"] and full
    out["pairs"] = pairs
    return out

def summarize(results, cats=None):
    t = Counter()
    by_cat = defaultdict(Counter)
    for rid, r in results.items():
        for k, v in r.items():
            if k != "pairs": t[k] += int(v)
        if cats:
            by_cat[cats[rid]]["n"] += 1; by_cat[cats[rid]]["exact"] += r["row_exact"]
    n = len(results)
    s = {"rows": n, "row_exact": t["row_exact"] / n, "row_food": t["row_food"] / n,
         "item_P": t["n_match"] / max(1, t["n_pred"]), "item_R": t["n_match"] / max(1, t["n_gold"])}
    for f in FIELDS: s[f] = t[f] / max(1, t["n_match"])
    if cats: s["by_cat"] = {c: round(v["exact"] / v["n"], 3) for c, v in sorted(by_cat.items())}
    return s

def load_gold(path=GOLD):
    return {r["id"]: r for r in map(json.loads, open(path))}

def main():
    gold = load_gold(sys.argv[2] if len(sys.argv) > 2 else GOLD)
    pred = {r["id"]: r.get("items") for r in map(json.loads, open(sys.argv[1]))}
    res = {rid: score_row(pred[rid], gold[rid]["expected"]["items"]) for rid in pred if rid in gold}
    print(json.dumps(summarize(res, {rid: gold[rid]["cat"] for rid in res}), indent=1))

if __name__ == "__main__":
    main()
