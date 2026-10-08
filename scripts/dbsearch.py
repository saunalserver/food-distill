#!/usr/bin/env python3
"""Search data/fooddb.sqlite from the shell (for humans and labeling agents).

  python3 scripts/dbsearch.py "chicken breast raw"            all words, all sources
  python3 scripts/dbsearch.py "chicken breast" --any          any word
  python3 scripts/dbsearch.py "big mac" --src usda_sr,usda_fndds --limit 40
  python3 scripts/dbsearch.py --id 123456                     one row + its portions
Prints: id | src | kcal/100g | P/C/F | name [brand]
"""
import argparse, re, sqlite3
from pathlib import Path

DB = Path(__file__).resolve().parent.parent / "data" / "fooddb.sqlite"

def row_line(r):
    b = f" [{r['brand']}]" if r["brand"] else ""
    f = lambda v: "?" if v is None else f"{v:.0f}"
    return f"{r['id']:>7} | {r['src']:<12} | {f(r['kcal']):>4} kcal | {f(r['protein'])}/{f(r['carbs'])}/{f(r['fat'])} | {r['name']}{b}"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("query", nargs="?", default="")
    ap.add_argument("--any", action="store_true")
    ap.add_argument("--src", default="")
    ap.add_argument("--limit", type=int, default=25)
    ap.add_argument("--id", type=int)
    a = ap.parse_args()
    c = sqlite3.connect(f"file:{DB}?mode=ro", uri=True); c.row_factory = sqlite3.Row
    if a.id:
        r = c.execute("select * from food where id = ?", (a.id,)).fetchone()
        print(row_line(r)); print(f"   serving: {r['serving_g']} g {r['serving_label']}  category: {r['category']}")
        for p in c.execute("select label, grams from portion where food_id = ?", (a.id,)): print(f"   portion: {p[0]} = {p[1]} g")
        return
    toks = re.findall(r"[a-z0-9]+", a.query.lower())
    q = (" OR " if a.any else " ").join(f'"{t}"' for t in toks)
    srcs = [s for s in a.src.split(",") if s]
    where = f" and f.src in ({','.join('?' * len(srcs))})" if srcs else ""
    rows = c.execute(f"select f.* from food_fts join food f on f.id = food_fts.rowid where food_fts match ?{where} "
                     f"order by bm25(food_fts) limit ?", (q, *srcs, a.limit)).fetchall()
    for r in rows: print(row_line(r))
    if not rows: print("(no rows)")

if __name__ == "__main__":
    main()
