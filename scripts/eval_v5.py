#!/usr/bin/env python3
"""End-to-end eval of the v5 chain on eval/v2: parse -> search -> pick -> portion -> kcal.

Parse source:   --parse-url http://127.0.0.1:8139/v1/chat/completions (a llama-server running the student)
                or --parse-file PRED.jsonl ({"id","items"}; e.g. a teacher run) or --parse-gold
Pick:           --pick-url URL (student), or --pick first (rank 1), or --pick oracle (best gold-ok candidate)

Scores
  parse : scripts/score_parse.py on every row
  match : on rows with match labels (eval/v2/match/match-*.jsonl): retrieval recall@k (an ok row among the
          candidates) and pick accuracy (chosen row in ok) — on gold-parsed items, so search/pick are
          measured independently of parse errors
  kcal  : per meal, |predicted total - gold total| / gold total, where gold = gold parse x gold best row x
          the shared portion logic. Reported: median error, share of meals within 10% / 25%.

  python3 scripts/eval_v5.py --parse-url ... --pick-url ... [--k 10] [--limit 0] [--out out/eval_v5.jsonl]
"""
import argparse, json, statistics, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "app")); sys.path.insert(0, str(ROOT / "scripts"))
import fooddb  # noqa: E402
from score_parse import load_gold, score_row, summarize  # noqa: E402

PARSE_SYS = (ROOT / "prompts" / "student_parse_v5.txt").read_text().strip()
GRAMMAR = (ROOT / "prompts" / "parse_v5.gbnf").read_text()   # forces valid JSON + allowed enums
FIELDS = ("food", "brand", "qty", "unit", "size", "state", "prep")

def chat(url, system, user, max_tokens=700, grammar=None):
    body = {"temperature": 0, "max_tokens": max_tokens, "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
    if grammar: body["grammar"] = grammar
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)["choices"][0]["message"]["content"]

def parse_text(url, text):
    out = chat(url, PARSE_SYS, text, grammar=GRAMMAR)
    try:
        items = json.loads(out[out.find("{"): out.rfind("}") + 1]).get("items", [])
        return [{k: it.get(k, "" if k != "qty" else 1) for k in FIELDS} for it in items if isinstance(it, dict)]
    except (json.JSONDecodeError, AttributeError):
        return []

def load_match():
    m = {}
    for p in sorted((ROOT / "eval" / "v2" / "match").glob("match-*.jsonl")):
        for r in map(json.loads, open(p)):
            m[(r["id"], r["i"])] = r
    return m

def choose(c, item, cands, mode, url, ok=None):
    if not cands: return None
    if mode == "first": return cands[0]
    if mode == "oracle":
        return next((x for x in cands if ok and x["id"] in ok), None)
    i = fooddb.pick(item, cands, url=url)
    return cands[0] if i is None else cands[i]     # same as app resolve(): "none" -> top result, flagged

def kcal_of(c, item, row):
    if row is None: return None
    g, _, _ = fooddb.grams(c, item, row)
    return None if g is None or row["kcal"] is None else row["kcal"] * g / 100

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parse-url"); ap.add_argument("--parse-file"); ap.add_argument("--parse-gold", action="store_true")
    ap.add_argument("--pick", default="model", choices=("model", "first", "oracle"))
    ap.add_argument("--pick-url", default=fooddb.LLAMA)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--only-matched", action="store_true", help="only rows with match labels (faster)")
    ap.add_argument("--out", default=str(ROOT / "out" / "eval_v5.jsonl"))
    a = ap.parse_args()
    gold = load_gold(); match = load_match(); c = fooddb.connect()
    rows = list(gold.values())
    if a.only_matched: rows = [r for r in rows if any(k[0] == r["id"] for k in match)]
    rows = rows[: a.limit or None]
    pred_file = {r["id"]: r["items"] for r in map(json.loads, open(a.parse_file))} if a.parse_file else {}
    t0 = time.time()

    def parse_one(r):
        if a.parse_gold: return [{k: it[k] for k in FIELDS} for it in r["expected"]["items"]]
        if a.parse_file: return pred_file.get(r["id"])
        return parse_text(a.parse_url, r["text"])
    with ThreadPoolExecutor(a.workers) as ex:
        parsed = dict(zip([r["id"] for r in rows], ex.map(parse_one, rows)))
    t_parse = time.time() - t0

    res, recall, picks, kcal_err, out = {}, [], [], [], []
    for r in rows:
        rid = r["id"]; pred = parsed.get(rid)
        if pred is None: continue
        res[rid] = score_row(pred, r["expected"]["items"])
        labeled = [(i, it, match[(rid, i)]) for i, it in enumerate(r["expected"]["items"])
                   if (rid, i) in match and match[(rid, i)]["food"] == it["food"]]
        # search + pick on GOLD items (independent of parse)
        for i, it, m in labeled:
            ok = set(m["ok"])
            if not ok: continue
            cands = fooddb.candidates(c, it, a.k)
            recall.append(any(x["id"] in ok for x in cands))
            ch = choose(c, it, cands, a.pick, a.pick_url, ok)
            picks.append(ch is not None and ch["id"] in ok)
        # meal kcal: full chain on PREDICTED items vs gold items x gold best rows
        if labeled and len(labeled) == len(r["expected"]["items"]) and all(m["best"] for _, _, m in labeled):
            gk = [kcal_of(c, it, c.execute("select * from food where id = ?", (m["best"],)).fetchone())
                  for _, it, m in labeled]
            if any(x is None for x in gk) or sum(gk) <= 0: continue
            pk = 0.0
            if a.parse_gold:   # gold items: the oracle can use each item's ok-set
                for _, it, m in labeled:
                    ch = choose(c, it, fooddb.candidates(c, it, a.k), a.pick, a.pick_url, set(m["ok"]))
                    pk += kcal_of(c, it, ch) or 0.0
            else:
                for it in pred:
                    ch = choose(c, it, fooddb.candidates(c, it, a.k), "first" if a.pick == "oracle" else a.pick, a.pick_url)
                    pk += kcal_of(c, it, ch) or 0.0
            err = abs(pk - sum(gk)) / sum(gk)
            kcal_err.append(err)
            out.append({"id": rid, "text": r["text"], "gold_kcal": round(sum(gk)), "pred_kcal": round(pk),
                        "err": round(err, 3), "pred": pred})
    s = summarize(res, {rid: gold[rid]["cat"] for rid in res})
    s["parse_s_per_row"] = round(t_parse / max(1, len(rows)), 2)
    if recall: s["retrieval_recall@%d" % a.k] = round(sum(recall) / len(recall), 3); s["n_match_items"] = len(recall)
    if picks: s["pick_acc"] = round(sum(picks) / len(picks), 3)
    if kcal_err:
        s["meals_kcal"] = len(kcal_err)
        s["kcal_err_median"] = round(statistics.median(kcal_err), 3)
        s["kcal_within_10%"] = round(sum(e <= 0.10 for e in kcal_err) / len(kcal_err), 3)
        s["kcal_within_25%"] = round(sum(e <= 0.25 for e in kcal_err) / len(kcal_err), 3)
    with open(a.out, "w") as f:
        for o in out: f.write(json.dumps(o, ensure_ascii=False) + "\n")
    print(json.dumps(s, indent=1))

if __name__ == "__main__":
    main()
