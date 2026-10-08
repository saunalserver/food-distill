#!/usr/bin/env python3
"""v2 teacher data: OpenRouter model writes realistic food-diary entries + their
parse under the v2 contract (prompts/parse_system_v2.txt: free-text food, brand,
size, raw/cooked state). Diversity comes from rotating themes and from seed foods
sampled out of data/fooddb.sqlite (generic dishes + branded products), so the
long tail of the database shows up in training text.

Resumable (appends; request ids continue), hard budget cap from OpenRouter's own
usage.cost. Needs OPENROUTER_API_KEY.

  python3 scripts/or_v2.py N_REQUESTS [--model openai/gpt-6-luna] [--budget 3.0]
Out: data/gen/v2_parse.jsonl
"""
import argparse, json, os, random, re, sqlite3, sys, threading, time, urllib.error, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "gen" / "v2_parse.jsonl"
CONTRACT = (ROOT / "prompts" / "parse_system_v2.txt").read_text()
UNITS = set("g kg oz lb ml l fl_oz tbsp tsp cup slice piece bowl plate handful clove can bottle glass pint "
            "serving scoop shot".split())
SIZES = {"", "small", "medium", "large", "extra large"}
STATES = {"", "raw", "cooked"}

CUISINES = ["Indian", "Thai", "Vietnamese", "Chinese (Cantonese, Sichuan, takeout)", "Japanese", "Korean",
            "Filipino", "Mexican", "Italian", "French", "British/Irish", "Lebanese/Middle Eastern", "Greek",
            "Turkish", "Ethiopian", "Caribbean/Jamaican", "Peruvian", "Spanish", "German", "Polish/Ukrainian",
            "Canadian (poutine, butter tarts, nanaimo bars, tourtière)", "Persian", "Malaysian/Singaporean",
            "Indonesian", "Brazilian", "Hawaiian (poke, loco moco)", "Taiwanese (bubble tea, beef noodle soup)",
            "Pakistani", "Moroccan", "Southern US / soul food", "Jewish deli", "Russian"]
THEMES = [  # (weight, theme)
    (6, "fast-food orders at US and Canadian chains (McDonald's, Tim Hortons, A&W, Wendy's, Burger King, Taco Bell, "
        "Chipotle, Subway, KFC, Popeyes, Chick-fil-A, Five Guys, Harvey's, Mary Brown's, Domino's, Pizza Hut, "
        "Freshii, Panera …): exact menu item names, sizes, combos/meal deals, customizations"),
    (4, "coffee shops and drinks (Starbucks, Tim Hortons, Blenz, Second Cup, bubble tea): sizes, milk types, "
        "syrups, sugars, 'double double', smoothies, energy drinks"),
    (5, "branded packaged foods, snacks and drinks from grocery stores (US/Canada/UK brands, President's Choice, "
        "No Name, Kirkland, Trader Joe's), named exactly as on the package"),
    (4, "sit-down restaurant meals, incl. Vancouver/Canadian chains (Cactus Club, Earls, White Spot, Joey, "
        "Boston Pizza, The Keg, Olive Garden, Cheesecake Factory): dish names, sides, drinks"),
    (8, "dishes from {cuisine} cuisine, named the way people in Canada/US type them, with sides and drinks"),
    (6, "meal-prep / macro tracking: raw ingredients with exact grams, ml, oz, cups, tbsp (state raw by default, "
        "grains dry), some explicitly cooked ('200g cooked rice', 'grilled salmon 150g')"),
    (4, "home-cooked meals described as dishes or ingredient lists, mixing raw and explicitly cooked items"),
    (5, "named plates and meal deals that must be expanded into components (full english/irish breakfast, "
        "big breakfast, bento, thali, mezze, combo meals) and 'X and Y' plates"),
    (6, "typos, autocorrect and voice-dictation errors, slang and shorthand on any foods "
        "(e.g. 'bug mac', 'chiken', 'brocoli', 'sammich', 'a coupla', 'bfast', 'oj'); the label has the "
        "corrected food"),
    (5, "vague and fuzzy amounts (some, a bit of, a splash, a handful, half, 2-3, a few, ¾ cup, 1/3 of a pizza), "
        "size words, no amounts at all"),
    (3, "alcohol and bar food (pints, cans, shots, cocktails, wine by the glass, wings, nachos)"),
    (3, "desserts, bakery and sweets (named pastries, cakes, cookies, ice cream with toppings)"),
    (3, "breakfast and brunch in many styles"),
    (3, "whole-day logs with several meals in one message (breakfast: … lunch: … snacks: …), 5-12 items"),
    (2, "supplements and gym food (protein shakes with exact scoops/ml, bars, pre-workout, creatine)"),
    (2, "edge cases: messages with no food (items []), water, corrections ('2 eggs, actually 3'), "
        "negations ('no sugar'), leftovers, emojis, numbers as words"),
]

SYSTEM = f"""You create training data for a small food-log parser. Write meal-diary messages the way real people
type or dictate them into a phone calorie app, each paired with its exact correct parse.

The parse must follow this contract EXACTLY (it is the parser's own system prompt):
---
{CONTRACT}
---
Conventions for the parse:
- Put cooking methods in prep and set state "cooked": a fried egg is food "egg", state "cooked", prep "fried".
- Seed foods are given in database wording ("Chicken, broiler, breast, raw"); write them in the TEXT the way a
  person would say them ("chicken breast"), and in the parse use plain everyday names.
- Messages: mostly lowercase, short, abbreviations, varied length (30% one item, 40% two-three, 30% four+).
  No two messages alike. Never mention calories or macros in the text.

Output ONLY a JSON array of {{"text": str, "items": [ ...contract items... ]}} objects."""

def seeds(db, rng, n=12):
    out = []
    for src, k in (("usda_fndds", 4), ("cofid", 1), ("ciqual", 1), ("cnf", 1), ("usda_sr", 1), ("branded", 4)):
        if src == "branded":
            rows = db.execute("select name, brand from food where src in ('off','usda_branded') and brand != '' "
                              "and rowid >= abs(random()) % (select max(rowid) from food) limit ?", (k,)).fetchall()
            out += [f"{b} {nm}" for nm, b in rows]
        else:
            rows = db.execute("select name from food where src = ? order by random() limit ?", (src, k)).fetchall()
            out += [r[0] for r in rows]
    rng.shuffle(out)
    return out[:n]

def theme(rng):
    w, t = zip(*THEMES)
    th = rng.choices(t, weights=w)[0]
    return th.format(cuisine=rng.choice(CUISINES)) if "{cuisine}" in th else th

def valid_item(it):
    if not isinstance(it, dict): return None
    try:
        food = str(it["food"]).strip().lower()
        q = it["qty"]
        if isinstance(q, str): q = float(q)
        out = {"food": food, "brand": str(it.get("brand") or "").strip().lower(), "qty": q,
               "unit": str(it.get("unit", "")).strip().lower(), "size": str(it.get("size") or "").strip().lower(),
               "state": str(it.get("state") or "").strip().lower(), "prep": str(it.get("prep") or "").strip()}
    except (KeyError, ValueError, TypeError):
        return None
    if not food or len(food) > 80 or isinstance(q, bool) or not isinstance(q, (int, float)) or not (0 < q < 5000): return None
    if out["unit"] not in UNITS or out["size"] not in SIZES or out["state"] not in STATES: return None
    if re.search(r"\d+\s*(kcal|cal\b|calories)", food + " " + out["prep"]): return None
    return out

def call(model, prompt, key, max_tokens=6000):
    body = {"model": model, "temperature": 1.0, "max_tokens": max_tokens,
            "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]}
    req = urllib.request.Request("https://openrouter.ai/api/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json", "Authorization": f"Bearer {key}",
                                  "User-Agent": "food-distill/2.0"})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                d = json.load(r)
            cost = float((d.get("usage") or {}).get("cost") or 0)
            txt = d["choices"][0]["message"]["content"] or ""
            a, b = txt.find("["), txt.rfind("]")
            try: return (json.loads(txt[a:b + 1]) if 0 <= a < b else None), cost
            except json.JSONDecodeError: return None, cost
        except (urllib.error.URLError, TimeoutError, ConnectionError, KeyError):
            time.sleep(5 * (attempt + 1))
    return None, 0.0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("n", type=int)
    ap.add_argument("--model", default="openai/gpt-6-luna")
    ap.add_argument("--budget", type=float, default=3.0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--per", type=int, default=10, help="entries per request")
    a = ap.parse_args()
    key = os.environ["OPENROUTER_API_KEY"]
    db = sqlite3.connect(f"file:{ROOT / 'data' / 'fooddb.sqlite'}?mode=ro", uri=True, check_same_thread=False)
    start = 0
    if OUT.exists():
        for l in open(OUT):
            start = max(start, json.loads(l)["req"] + 1)
    lock = threading.Lock(); spent = [0.0]; kept = [0]; bad = [0]; out = open(OUT, "a"); t0 = time.time()
    print(f"{a.model}: {a.n} requests from req {start}, budget ${a.budget}", flush=True)

    def work(i):
        if spent[0] >= a.budget: return
        rng = random.Random(1000 + i)
        with lock: sd = seeds(db, rng)
        th = theme(rng)
        prompt = (f"Write {a.per} messages. Theme: {th}.\nSeed foods — use several of them (in everyday words), "
                  f"plus anything else that fits the theme: {'; '.join(sd)}")
        entries, cost = call(a.model, prompt, key)
        rows = []
        for e in entries or []:
            if not isinstance(e, dict) or not isinstance(e.get("text"), str) or not isinstance(e.get("items"), list): continue
            text = " ".join(e["text"].split())
            items = [valid_item(it) for it in e["items"]]
            if not (2 < len(text) < 500) or any(x is None for x in items): continue
            if not items and "edge" not in th: continue
            rows.append({"text": text, "expected": {"items": items}, "src": "or-v2", "model": a.model,
                         "theme": th.split(":")[0][:60], "req": i})
        with lock:
            spent[0] += cost
            for r in rows: out.write(json.dumps(r) + "\n")
            out.flush(); kept[0] += len(rows); bad[0] += not rows
            done = i - start + 1
            if done % 20 == 0:
                print(f"req ~{done}/{a.n} kept={kept[0]} empty_reqs={bad[0]} spent=${spent[0]:.3f} "
                      f"({time.time() - t0:.0f}s)", flush=True)

    with ThreadPoolExecutor(a.workers) as ex:
        list(ex.map(work, range(start, start + a.n)))
    out.close()
    print(f"DONE kept {kept[0]} rows, ${spent[0]:.3f} spent → {OUT}")

if __name__ == "__main__":
    main()
