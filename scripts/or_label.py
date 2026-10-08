#!/usr/bin/env python3
"""Label food-diary texts with an OpenRouter teacher under the v2 contract (the student's own task).

Batches several texts per request (amortizes the long system prompt), temperature 0, validates
items, hard budget cap from OpenRouter's usage.cost. Resumable: ids already in OUT are skipped.

  python3 scripts/or_label.py IN.jsonl OUT.jsonl --model anthropic/claude-haiku-5.5 [--budget 1] [--batch 15]
IN rows need "id" and "text". OUT rows: {"id", "items", "model"}.
"""
import argparse, json, os, re, sys, threading, time, urllib.error, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from or_v2 import valid_item  # noqa: E402

CONTRACT = (ROOT / "prompts" / "parse_system_v2.txt").read_text()
EXAMPLES = "\n\n".join("Message: " + b.replace("\n", "\nParse: ", 1)
                       for b in (ROOT / "prompts" / "teacher_examples_v2.txt").read_text().strip().split("\n\n"))
SYSTEM = f"""You are the reference labeler for a food-log parser. Apply this contract EXACTLY to each message:
---
{CONTRACT}
---
Conventions (they follow from the contract; apply them consistently):
- A cooking method goes in prep with state "cooked": "2 fried eggs" -> food "egg", qty 2, state "cooked", prep "fried".
  Foods that are cooked by nature when served (bacon at breakfast, toast, a burger patty in a burger meal, fried
  chicken) need no prep word: "toast" is food "toast", state "", prep "".
- Rice, pasta, noodles, oats, quinoa, couscous and other plain grains are state "raw" unless a cooking word is right
  next to them ("cooked rice", "steamed rice", "leftover pasta"): "curry w rice" -> rice, state "raw". Dishes named
  after a grain (fried rice, biryani, ramen, mac and cheese, oatmeal) are state "".
- food: plain everyday name, singular, lowercase; never add detail that is not in the text ("beef steak" stays
  "beef steak"). Keep product names and fill brand when stated or certain.
- Never split a single dish into ingredients; never merge separate foods.
- "1 cream" / "2 milk" at a coffee shop means creamer portions: cream or milk, unit "tbsp", one per portion
  ("double double" = coffee + 2 tbsp cream + 2 tsp sugar); "a splash" of milk/cream in coffee or tea = 1 tbsp;
  "2 sugars" = 2 tsp sugar.

Worked examples (follow these conventions exactly):
{EXAMPLES}

Input: numbered messages. Output ONLY a JSON array, one object per message, in order:
[{{"i": 1, "items": [{{"food":..., "brand":..., "qty":..., "unit":..., "size":..., "state":..., "prep":...}}]}}, ...]"""

def call(model, msgs, key, max_tokens, reasoning, temperature=0):
    body = {"model": model, "temperature": temperature, "max_tokens": max_tokens, "messages": msgs}
    if reasoning == "off": body["reasoning"] = {"enabled": False}
    elif reasoning: body["reasoning"] = {"effort": reasoning}
    req = urllib.request.Request("https://openrouter.ai/api/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json", "Authorization": f"Bearer {key}",
                                  "User-Agent": "food-distill/2.0"})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                d = json.load(r)
            return (d["choices"][0]["message"]["content"] or ""), float((d.get("usage") or {}).get("cost") or 0)
        except urllib.error.HTTPError as e:
            msg = e.read()[:300]
            if e.code in (400, 401, 402, 404): print("HTTP", e.code, msg, file=sys.stderr); return "", 0.0
            time.sleep(5 * (attempt + 1))
        except (urllib.error.URLError, TimeoutError, ConnectionError, KeyError, json.JSONDecodeError):
            time.sleep(5 * (attempt + 1))
    return "", 0.0

def salvage(txt):
    """Complete {"i": ..} objects from a truncated array (output cut off by max_tokens)."""
    dec, out, pos = json.JSONDecoder(), [], txt.find("{")
    while 0 <= pos < len(txt):
        try:
            o, end = dec.raw_decode(txt, pos); out.append(o); pos = txt.find("{", end)
        except json.JSONDecodeError:
            pos = txt.find('{"i"', pos + 1)
    return out

def parse_out(txt, n):
    a, b = txt.find("["), txt.rfind("]")
    try: arr = json.loads(txt[a:b + 1]) if 0 <= a < b else salvage(txt)
    except json.JSONDecodeError: arr = salvage(txt)
    out = {}
    for o in arr or []:
        if not isinstance(o, dict) or not isinstance(o.get("items"), list): continue
        try: i = int(o.get("i"))
        except (TypeError, ValueError): continue
        items = [valid_item(it) for it in o["items"]]
        if 1 <= i <= n and all(x is not None for x in items): out[i] = items
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inp"); ap.add_argument("out")
    ap.add_argument("--model", required=True)
    ap.add_argument("--budget", type=float, default=1.0)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--reasoning", default="", help="off|low|medium|high (default: the model's own)")
    ap.add_argument("--max-tokens", type=int, default=16000)
    a = ap.parse_args()
    key = os.environ["OPENROUTER_API_KEY"]
    rows = [json.loads(l) for l in open(a.inp) if l.strip()]
    done = {json.loads(l)["id"] for l in open(a.out)} if Path(a.out).exists() else set()
    todo = [r for r in rows if r["id"] not in done][: a.limit or None]
    batches = [todo[i:i + a.batch] for i in range(0, len(todo), a.batch)]
    lock = threading.Lock(); spent = [0.0]; kept = [0]; miss = [0]; out = open(a.out, "a"); t0 = time.time()
    print(f"{a.model}: {len(todo)} rows in {len(batches)} requests, budget ${a.budget}", flush=True)

    def work(bi):
        if spent[0] + a.workers * (spent[0] / max(1, bi)) >= a.budget: return   # leave room for in-flight requests
        b = batches[bi]
        user = "\n".join(f"{i}. {r['text']}" for i, r in enumerate(b, 1))
        txt, cost = call(a.model, [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
                         key, a.max_tokens, a.reasoning)
        got = parse_out(txt, len(b))
        if len(got) < len(b):
            with open(str(a.out) + ".debug", "a") as dbg:
                dbg.write(json.dumps({"batch": [r["id"] for r in b], "got": sorted(got), "txt": txt[-3000:]}) + "\n")
        with lock:
            spent[0] += cost
            for i, r in enumerate(b, 1):
                if i in got:
                    out.write(json.dumps({"id": r["id"], "items": got[i], "model": a.model}) + "\n"); kept[0] += 1
                else: miss[0] += 1
            out.flush()
            if (bi + 1) % 25 == 0:
                print(f"req {bi + 1}/{len(batches)} kept={kept[0]} missed={miss[0]} spent=${spent[0]:.4f} "
                      f"({time.time() - t0:.0f}s)", flush=True)

    with ThreadPoolExecutor(a.workers) as ex:
        list(ex.map(work, range(len(batches))))
    out.close()
    print(f"DONE {a.model} kept {kept[0]} missed {miss[0]} ${spent[0]:.4f} "
          f"(${spent[0] / max(1, kept[0]) * 1000:.3f}/1k rows) {time.time() - t0:.0f}s")

if __name__ == "__main__":
    main()
