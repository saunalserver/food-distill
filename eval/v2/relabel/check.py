import json,sys
UNITS=set("g kg oz lb ml l fl_oz tbsp tsp cup slice piece bowl plate handful clove can bottle glass pint serving scoop shot".split())
inp=[json.loads(l)["id"] for l in open(sys.argv[1])]
out=[json.loads(l) for l in open(sys.argv[2]) if l.strip()]
ids=[r["id"] for r in out]; errs=[]
if sorted(ids)!=sorted(inp): errs.append(f"ids differ: missing {sorted(set(inp)-set(ids))[:10]} extra/dup {[i for i in ids if ids.count(i)>1 or i not in inp][:10]}")
for r in out:
  for it in r.get("items",[]):
    if not {"food","brand","qty","unit","size","state","prep"} <= set(it) <= {"food","brand","qty","unit","size","state","prep","accept","alt","optional"}: errs.append(f"{r['id']}: keys {sorted(it)}")
    if it.get("unit") not in UNITS: errs.append(f"{r['id']}: unit {it.get('unit')}")
    if it.get("size") not in ("","small","medium","large","extra large"): errs.append(f"{r['id']}: size {it.get('size')}")
    if it.get("state") not in ("","raw","cooked"): errs.append(f"{r['id']}: state {it.get('state')}")
    if not isinstance(it.get("qty"),(int,float)) or it["qty"]<=0: errs.append(f"{r['id']}: qty {it.get('qty')}")
print("\n".join(errs[:50]) or "OK", len(out), "rows")
