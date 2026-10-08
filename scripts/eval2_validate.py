#!/usr/bin/env python3
"""Schema check for eval v2 rows (see eval/v2/SPEC.md). Stdlib only.

Usage: python3 scripts/eval2_validate.py FILE.jsonl [FILE.jsonl ...]
Exit 1 if any row is invalid; prints one line per problem."""
import json, re, sys

UNITS = set("g kg oz lb ml l fl_oz tbsp tsp cup slice piece bowl plate handful clove can "
            "bottle glass pint serving scoop shot".split())
SIZES = {"", "small", "medium", "large", "extra large"}
STATES = {"", "raw", "cooked"}
CATS = {"fastfood", "packaged", "typos", "world", "homecook", "combos", "vague", "edge"}
KEYS = {"food", "brand", "qty", "unit", "size", "state", "prep", "accept", "db_hint"}

def check(row):
    errs = []
    if not re.fullmatch(r"[a-z]+-\d{4}", str(row.get("id", ""))): errs.append("bad id")
    if row.get("cat") not in CATS: errs.append(f"bad cat {row.get('cat')!r}")
    if not str(row.get("text", "")).strip(): errs.append("empty text")
    items = row.get("expected", {}).get("items")
    if not isinstance(items, list): return errs + ["expected.items not a list"]
    for i, it in enumerate(items):
        p = f"item {i}"
        if not (KEYS - {"db_hint"}) <= set(it) <= KEYS | {"alt", "optional"}: errs.append(f"{p}: keys {sorted(set(it) ^ KEYS)} wrong")
        food = it.get("food", "")
        if not food or food != food.lower() or food != food.strip(): errs.append(f"{p}: food must be lowercase, non-empty")
        if it.get("brand", "") != str(it.get("brand", "")).lower(): errs.append(f"{p}: brand must be lowercase")
        if not isinstance(it.get("qty"), (int, float)) or it.get("qty") <= 0: errs.append(f"{p}: qty must be a positive number")
        if it.get("unit") not in UNITS: errs.append(f"{p}: unit {it.get('unit')!r} not allowed")
        if it.get("size") not in SIZES: errs.append(f"{p}: size {it.get('size')!r} not allowed")
        if it.get("state") not in STATES: errs.append(f"{p}: state {it.get('state')!r} not allowed")
        if not isinstance(it.get("accept"), list): errs.append(f"{p}: accept must be a list")
    return errs

def main():
    bad = total = 0; ids = set(); texts = set()
    for path in sys.argv[1:]:
        for n, line in enumerate(open(path), 1):
            if not line.strip(): continue
            total += 1
            try: row = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"{path}:{n}: not JSON ({e})"); bad += 1; continue
            errs = check(row)
            if row.get("id") in ids: errs.append("duplicate id")
            t = re.sub(r"\W+", " ", str(row.get("text", "")).lower()).strip()
            if t in texts: errs.append("duplicate text")
            ids.add(row.get("id")); texts.add(t)
            for e in errs: print(f"{path}:{n} [{row.get('id')}]: {e}")
            bad += bool(errs)
    print(f"{total} rows, {bad} invalid")
    sys.exit(1 if bad else 0)

if __name__ == "__main__":
    main()
