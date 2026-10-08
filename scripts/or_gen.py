#!/usr/bin/env python3
"""v5 training TEXTS (no labels): an OpenRouter model writes realistic food-diary messages.
Labels come separately from the best teacher (scripts/or_label.py), so cheap generation and
accurate labeling are decoupled.

Diversity: weighted themes (scripts/or_v2.py THEMES + additions) x seed foods sampled from
data/fooddb.sqlite (generic dishes + popular branded products). Texts that duplicate an eval text
(normalized) or each other are dropped. Resumable, hard budget cap from OpenRouter usage.cost.

  python3 scripts/or_gen.py N_REQUESTS [--model anthropic/claude-haiku-5.5] [--per 25] [--budget 0.5]
Out: data/gen/v5_texts.jsonl  {"id": "g000123-04", "text", "theme", "model"}
"""
import argparse, json, os, random, re, sqlite3, sys, threading, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from or_v2 import CUISINES, THEMES as V2_THEMES  # noqa: E402
from or_label import call  # noqa: E402

OUT = ROOT / "data" / "gen" / "v5_texts.jsonl"
THEMES = V2_THEMES + [
    (5, "coffee and tea orders with additions spelled out the way people log them (2 milk 2 sugar, double double, "
        "a splash of cream, oat milk, pumps of syrup, honey), plus whatever was eaten with them"),
    (5, "foods with toppings, spreads, condiments, sauces and dressings named (toast w butter and jam, bagel w cream "
        "cheese, fries w gravy, salad w ranch on the side, pancakes w syrup, 'extra cheese', 'extra sauce')"),
    (3, "Vancouver / BC life: sushi and izakaya, dim sum, pho, ramen, Japadog, Cactus Club, White Spot, Tim Hortons, "
        "Save-On-Foods / Superstore / No Name / President's Choice products, Nanaimo bars, BC salmon, craft beer"),
    (3, "very short one-item logs (2-5 words), sometimes just a product name or a number and a food"),
    (2, "grains and starches logged by dry weight for meal prep (75g rice, 100g pasta, 40g oats) mixed with "
        "a few explicitly cooked ones ('1 cup cooked rice', 'leftover pasta')"),
]

# --focus: the student's weak spots in the first v5 eval (typos 19% exact, combos 25%, fastfood 32%; invented
# items from chatter like "mario kart night"; shared/partial amounts; corrections)
FOCUS = [
    (3, "heavy typos, autocorrect and voice-dictation errors on everyday and branded foods (hedhog, expresso, "
        "biskotti, pastrmi, bologines, chiken, brocoli, sammich, 'a coupla', 'bfast'), the way people really mistype"),
    (3, "named plates, combos and meal deals at chains and diners (big mac meal, tenders meal, 2 can dine, value meal, "
        "full english, grand slam, thali, bento, poutine combo), with swaps and sizes ('w a coke zero', 'large fries')"),
    (2, "shared and partial amounts: split with friends, ate half, 'had maybe 12 of the 20 nuggets', 'bottle of wine "
        "split 3 ways', '50/50 with the toddler', 'a few bites of his fries', leftovers of a share"),
    (2, "messages with context words that are NOT food (game night, mario kart, christmas movie night, at brunch, "
        "after the gym, at the bbq, at my desk) and emoji-only or emoji-heavy logs (🥯 with 🧈, 🍜, 🍕x2)"),
    (2, "exact fast-food and coffee-chain orders with sizes, customizations and brand product names (Tim Hortons, "
        "McDonald's, Wendy's, A&W, Subway, Starbucks, Chick-fil-A, Popeyes, Taco Bell, Five Guys, Harvey's)"),
    (1, "corrections and negations: '2 eggs, actually 3', 'oatmeal w brown sugar — skip the sugar', 'no cheese', "
        "'hold the mayo', 'scratch that, it was a large'"),
]

SYSTEM = """You write realistic messages that people type or dictate into a phone calorie-tracking app.
Rules:
- Mostly lowercase, casual, abbreviations (w/, n, 2x, tbsp, lg), missing punctuation, some typos.
- Vary length: ~30% one item, ~40% two or three, ~30% four or more (some whole-day logs).
- Vary how amounts are given: none, counts, grams/oz/ml, cups/tbsp, sizes, vague words.
- Seed foods come in database wording ("Chicken, broiler, breast, raw"); write them the way a person
  would ("chicken breast"). Use several seeds plus anything else that fits the theme.
- Never mention calories or macros. No two messages alike. No numbering.
Output ONLY a JSON array of strings."""

def norm_text(t):
    return re.sub(r"\W+", " ", t.lower()).strip()

def seeds(db, rng, n=12):
    out = []
    for src, k in (("usda_fndds", 4), ("cofid", 1), ("ciqual", 1), ("cnf", 1), ("usda_sr", 1)):
        out += [r[0] for r in db.execute("select name from food where src = ? order by random() limit ?", (src, k))]
    out += [f"{b} {nm}" for nm, b in db.execute(
        "select name, brand from food where src = 'off' and popularity >= 5 and brand != '' order by random() limit 3")]
    out += [f"{b} {nm}" for nm, b in db.execute(
        "select name, brand from food where src = 'usda_branded' and rowid >= abs(random()) % "
        "(select max(rowid) from food) and brand != '' limit 1")]
    rng.shuffle(out)
    return out[:n]

def theme(rng, themes=None):
    w, t = zip(*(themes or THEMES))
    th = rng.choices(t, weights=w)[0]
    return th.format(cuisine=rng.choice(CUISINES)) if "{cuisine}" in th else th

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("n", type=int)
    ap.add_argument("--model", default="anthropic/claude-haiku-5.5")
    ap.add_argument("--per", type=int, default=25)
    ap.add_argument("--budget", type=float, default=0.5)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--reasoning", default="off")
    ap.add_argument("--focus", action="store_true", help="only the FOCUS themes (weak spots)")
    a = ap.parse_args()
    key = os.environ["OPENROUTER_API_KEY"]
    db = sqlite3.connect(f"file:{ROOT / 'data' / 'fooddb.sqlite'}?mode=ro", uri=True, check_same_thread=False)
    seen = {norm_text(json.loads(l)["text"]) for l in open(ROOT / "eval" / "v2" / "orig.jsonl")}
    start = 0
    if OUT.exists():
        for l in open(OUT):
            r = json.loads(l); seen.add(norm_text(r["text"])); start = max(start, int(r["id"][1:7]) + 1)
    lock = threading.Lock(); spent = [0.0]; kept = [0]; dup = [0]; out = open(OUT, "a"); t0 = time.time()
    print(f"{a.model}: {a.n} requests from req {start}, budget ${a.budget}", flush=True)

    def work(i):
        if spent[0] + a.workers * spent[0] / max(1, i - start) >= a.budget: return
        rng = random.Random(5000 + i)
        with lock: sd = seeds(db, rng)
        th = theme(rng, FOCUS if a.focus else None)
        prompt = f"Write {a.per} messages. Theme: {th}.\nSeed foods: {'; '.join(sd)}"
        txt, cost = call(a.model, [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
                         key, 4000, a.reasoning, temperature=1.0)
        s, e = txt.find("["), txt.rfind("]")
        try: arr = json.loads(txt[s:e + 1]) if 0 <= s < e else []
        except json.JSONDecodeError: arr = []
        with lock:
            spent[0] += cost
            for j, t in enumerate(x for x in arr if isinstance(x, str)):
                t = " ".join(t.split()); k = norm_text(t)
                if not (2 < len(t) < 600) or k in seen: dup[0] += 1; continue
                seen.add(k)
                out.write(json.dumps({"id": f"g{i:06d}-{j:02d}", "text": t, "theme": th.split(":")[0][:60],
                                      "model": a.model}, ensure_ascii=False) + "\n"); kept[0] += 1
            out.flush()
            if (i - start + 1) % 50 == 0:
                print(f"req ~{i - start + 1}/{a.n} kept={kept[0]} dup={dup[0]} spent=${spent[0]:.3f} "
                      f"({time.time() - t0:.0f}s)", flush=True)

    with ThreadPoolExecutor(a.workers) as ex:
        list(ex.map(work, range(start, start + a.n)))
    out.close()
    print(f"DONE kept {kept[0]} texts (dropped {dup[0]} dup/eval), ${spent[0]:.3f} → {OUT}")

if __name__ == "__main__":
    main()
