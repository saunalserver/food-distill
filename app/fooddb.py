"""Search + pick + portion over data/fooddb.sqlite (built by scripts/build_fooddb.py).

    parsed item {"food","brand","qty","unit","size","state","prep"}
      -> candidates(item)   full-text search across all sources, ranked
      -> pick(item, cands)  the model chooses one (grammar-constrained), or rules if no model
      -> grams(item, food)  portion row / label serving / borrowed USDA portion / household measure
      -> nutrients scaled to grams

Stdlib only. The model is only ever asked to choose between real database rows;
every number comes from the database.
"""
import json, os, re, sqlite3, time, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "fooddb.sqlite"
# the student model runs in a llama.cpp llama-server (an RTX 2060 here); point FD_MODEL_URL at it
LLAMA = os.environ.get("FD_MODEL_URL", "http://127.0.0.1:8139/v1/chat/completions")

GENERIC = ("usda_fndds", "usda_sr", "usda_found", "cnf", "cofid", "ciqual")
BRANDED = ("usda_branded", "off")
# lower = preferred when relevance ties. FNDDS = "as eaten" foods with household portions.
SRC_RANK = {"usda_fndds": 0, "usda_sr": 1, "cnf": 2, "usda_found": 3, "cofid": 4, "ciqual": 5,
            "usda_branded": 6, "off": 7}
COOK_WORDS = ("cooked", "boiled", "fried", "grilled", "baked", "roasted", "steamed", "broiled",
              "braised", "poached", "scrambled", "toasted", "stewed", "sauteed", "sautéed", "microwaved",
              "pan-fried", "dry-fried", "reheated", "prepared")
RAW_WORDS = ("raw", "dry", "uncooked", "unprepared", "fresh")
# words that turn a basic food into a different product; penalized unless the user said them
HEAVY = ("breaded", "battered", "babyfood", "baby food", "infant", "glutinous", "pudding", "sauce",
         "salad", "burrito", "taquito", "quesadilla", "sandwich", "creamed", "deviled", "benedict",
         "juice", "canned", "frozen", "dehydrated", "powder", "mix", "imitation", "substitute",
         "flavored", "tenders", "nuggets", "patty", "split", "soup", "casserole", "with ", "dressing", "dip",
         "kit", "seasoning", "syrup", "flour", "extract", "concentrate")
STOP = {"of", "with", "and", "a", "the", "in", "on", "w"}

MASS = {"g": 1.0, "kg": 1000.0, "oz": 28.3495, "lb": 453.592}
VOL_ML = {"ml": 1.0, "l": 1000.0, "fl_oz": 29.5735, "tsp": 4.93, "tbsp": 14.79, "cup": 236.6,
          "shot": 44.4, "glass": 250.0, "pint": 500.0}   # pint: between US 473 and Canadian/UK 568 ml
PORTION_WORDS = {"tbsp": ["tablespoon", "tbsp"], "tsp": ["teaspoon", "tsp"], "cup": ["cup"],
                 "slice": ["slice"], "piece": ["piece", "each", "whole", "item"], "bowl": ["bowl"],
                 "plate": ["plate"], "handful": ["handful"], "clove": ["clove"], "can": ["can"],
                 "bottle": ["bottle"], "glass": ["glass"], "serving": ["serving"], "scoop": ["scoop"],
                 "shot": ["shot", "jigger"], "pint": ["pint"]}

def connect(path=DB_PATH):
    c = sqlite3.connect(f"file:{path}?mode=ro", uri=True, check_same_thread=False)
    c.row_factory = sqlite3.Row
    return c

def _tokens(s):
    return [t for t in re.findall(r"[a-z0-9]+", s.lower()) if t not in STOP]

def _fts(c, table, query, srcs, limit):
    q = (f"select f.*, bm25({table}) as score from {table} join food f on f.id = {table}.rowid "
         f"where {table} match ? and f.src in ({','.join('?' * len(srcs))}) order by score limit ?")
    try:
        return list(c.execute(q, (query, *srcs, limit)))
    except sqlite3.OperationalError:
        return []

# everyday names -> database wording; searched IN ADDITION to the user's words (small penalty)
ALIASES = {
    "burger": "hamburger", "burgers": "hamburger", "chips": "potato chips", "crisps": "potato chips",
    "pop": "soft drink", "soda": "soft drink", "soft drink": "soft drink nfs", "coke": "soft drink cola",
    "pepsi": "soft drink cola", "diet coke": "soft drink cola diet", "coke zero": "soft drink cola diet",
    "sprite": "soft drink lemon lime", "7up": "soft drink lemon lime", "ginger ale": "soft drink ginger ale",
    "salad": "lettuce salad assorted vegetables", "side salad": "lettuce salad assorted vegetables",
    "green salad": "lettuce salad assorted vegetables", "garden salad": "salad garden",
    "toast": "bread white toasted", "donut": "doughnut", "donuts": "doughnut", "timbits": "doughnut holes",
    "sticky rice": "glutinous rice", "wings": "chicken wing", "chicken wings": "chicken wing",
    "sub": "submarine sandwich", "hoagie": "submarine sandwich", "cheesesteak": "steak sandwich",
    "philly cheesesteak": "steak sandwich", "hash browns": "potato hash brown", "hash brown": "potato hash brown",
    "sugar": "sugars granulated", "cream": "cream fluid light coffee", "coffee cream": "cream fluid light coffee",
    "creamer": "coffee creamer",
    "jam": "jams and preserves", "coffee": "coffee brewed", "black coffee": "coffee brewed",
    "drip coffee": "coffee brewed", "filter coffee": "coffee brewed", "tea": "tea brewed", "black tea": "tea brewed",
    "oj": "orange juice", "fries": "french fries", "mayo": "mayonnaise", "pb": "peanut butter",
    "egg": "egg whole", "eggs": "egg whole", "milk": "milk nfs", "rice": "rice white", "pasta": "pasta",
    "noodles": "noodles", "steak": "beef steak", "ground beef": "beef ground", "chicken": "chicken breast",
    "bacon": "pork bacon", "sausage": "pork sausage", "ham": "pork ham", "wrap": "sandwich wrap",
    "bbq": "barbecue", "veggie": "vegetable", "veggies": "vegetables", "ranch": "ranch dressing",
}
# single words too vague to search alone in a backoff
BACKOFF_STOP = {"papa", "mama", "baby", "big", "little", "mini", "style", "classic", "original", "special",
                "super", "deluxe", "supreme", "house", "homemade", "fresh", "plain", "mixed", "spicy", "sweet",
                "hot", "cold", "iced", "red", "green", "white", "black", "brown", "golden", "crispy", "extra"}
# trailing words that only say what kind of dish it is; dropping them is a mild backoff
DISH_TAIL = {"sandwich", "burger", "bowl", "plate", "wrap", "dish", "meal", "combo", "platter"}

def _search_variants(toks):
    """(tokens, penalty) to search: the user's words, an alias, then word-dropping backoffs."""
    out, phrase = [(toks, 0.0)], " ".join(toks)
    if phrase in ALIASES:                                                       # whole-name alias = canonical meaning
        out.append((ALIASES[phrase].split(), -2.5))
    else:                                                                       # word-level aliases ("philly cheesesteak")
        sub = [w for t in toks for w in (ALIASES[t].split() if t in ALIASES and len(toks) > 1 else [t])]
        if sub != toks: out.append((sub, 0.5))
    if len(toks) >= 2:
        if toks[-1] in DISH_TAIL: out.append((toks[:-1], 3.5))                  # "cheesesteak sandwich" -> "cheesesteak"
        for i in range(len(toks) - 1):                                          # drop one modifier, keep the head noun
            out.append((toks[:i] + toks[i + 1:], 3.0))
        out.append((toks[-1:], 5.0))                                            # head noun alone
        if toks[-1] in ALIASES: out.append((ALIASES[toks[-1]].split(), 5.5))
    return out

def candidates(c, item, k=8, stats=None):
    """Up to k distinct database rows for one parsed item, best first.

    Searches the user's words, an everyday->database alias, and word-dropping backoffs (each later variant
    with a penalty), in generic sources first and branded ones when a brand is given or nothing generic fits;
    then re-ranks by state, missing words, dish words the user did not say, source and name length."""
    food, brand = item.get("food", ""), item.get("brand", "")
    toks = _tokens(food)
    if not toks: return []
    btoks = _tokens(brand)
    pool, via_alias = {}, set()
    searches = []                                                              # (query, hits) for the demo view
    def fts(table, query, srcs, limit):
        rows = _fts(c, table, query, srcs, limit)
        searches.append((query.replace('"', ""), len(rows)))
        return rows
    def add(rows, bonus):
        # bm25 is not comparable across queries: scale each result list to 0..-3 (best = -3), then add the
        # query's bonus/penalty, so "which search found it" decides more than raw term statistics
        best = min((r["score"] for r in rows), default=0) or -1
        for r in rows:
            sc = -3 * r["score"] / best + bonus
            if r["id"] not in pool or pool[r["id"]][1] > sc: pool[r["id"]] = (r, sc)
    quote = lambda ts: " ".join(f'"{t}"' for t in ts)
    state, prep = item.get("state", ""), (item.get("prep") or "").lower()
    method = [w for w in COOK_WORDS if w in prep]
    variants = _search_variants(toks)
    if brand:   # the product under its brand, in label databases and FNDDS/SR restaurant rows
        add(fts("food_fts", quote(toks + btoks), BRANDED + GENERIC, 40), -6)
        add(fts("food_fts", quote(toks), ("usda_fndds", "usda_sr"), 20), -3)
        add(fts("food_fts", quote(toks), BRANDED, 25), -1)
    full_hit = False
    for vt, pen in variants:
        if pen >= 2.0 and full_hit: break                                      # backoffs only when nothing fits
        if pen >= 2.0 and len(vt) == 1 and vt[0] in BACKOFF_STOP: continue    # never search "papa", "style" alone
        if method: add(fts("food_fts", quote(vt + method[:1]), GENERIC, 30), pen - 3)   # "fried egg" -> Egg, fried
        elif state == "raw": add(fts("food_fts", quote(vt + ["raw"]), GENERIC, 30), pen - 2)
        rows = fts("food_fts", quote(vt), GENERIC, 60)
        add(rows, pen)
        if pen < 0: via_alias.update(r["id"] for r in rows)                   # the canonical row for this name
        if rows and pen < 2.0: full_hit = True
    if not full_hit:   # nothing generic names this food: label databases (dishes like butter chicken, jollof)
        add(fts("food_fts", quote(toks), BRANDED, 40), 1.0)
    if len(pool) < k and not full_hit:  # typos: trigram substring
        tri = [t for t in toks if len(t) >= 3]
        if tri: add(fts("food_tri", " OR ".join(f'"{t[:max(3, len(t) - 1)]}"' for t in tri), GENERIC, 60), 8)
    if not pool:   # still nothing ("hedhog"): any shared 3-letter piece, all sources; rows sharing the most rank first
        grams3 = {t[i:i + 3] for t in toks if len(t) >= 4 for i in range(len(t) - 2)}
        if grams3: add(fts("food_tri", " OR ".join(f'"{g}"' for g in sorted(grams3)), GENERIC + BRANDED, 40), 10)

    want_cook = method or (["cooked"] if state == "cooked" else [])
    plain = len(toks) <= 2 and not brand
    alias_words = [w for vt, pen in variants if pen <= 0.5 and vt != toks for w in vt]
    said = food + " " + " ".join(alias_words)                                # words the user (or their alias) said
    n_said = max(len(toks), max((len(vt) for vt, pen in variants if pen <= 0.5), default=0))
    def rank(entry):
        r, s = entry; name = r["name"].lower()
        if state == "raw":
            s += -2 if any(w in name for w in RAW_WORDS) else (2 if any(w in name for w in COOK_WORDS) else 0)
        elif state == "cooked":
            s += -2 if any(w in name for w in want_cook or COOK_WORDS) else (2 if any(w in name for w in RAW_WORDS) else 0)
        missing = sum(1 for t in toks if t not in name and t.rstrip("s") not in name)
        if r["id"] not in via_alias:
            s += 1.5 * sum(1 for w in HEAVY if w in name and w not in said)
            s += 0.15 * max(0, len(_tokens(name)) - n_said - 2)
        s += 1.5 * missing + 0.3 * SRC_RANK[r["src"]]
        if plain and re.search(r"\bnfs\b|\bns as to\b", name): s -= 1.0      # FNDDS "X, NFS" / "NS as to" = generic default
        if r["src"] == "off": s -= min(1.5, (r["popularity"] or 0) / 50)
        if r["kcal"] is None or (r["src"] in BRANDED and not r["kcal"]): s += 5   # unusable / suspicious label row
        return s
    ranked = sorted(pool.values(), key=rank)
    if stats is not None: stats.update(pool=len(pool), searches=searches)
    out, seen = [], set()
    for r, _ in ranked:
        key = (r["name"].lower(), r["brand"].lower(), round(r["kcal"] or 0))
        if key in seen: continue
        seen.add(key); out.append(r)
        if len(out) == k: break
    return out

HEAVY_V1 = ("breaded", "battered", "babyfood", "baby food", "infant", "glutinous", "pudding", "sauce",
         "salad", "burrito", "taquito", "quesadilla", "sandwich", "creamed", "deviled", "benedict",
         "juice", "canned", "frozen", "dehydrated", "powder", "mix", "imitation", "substitute",
         "flavored", "tenders", "nuggets", "patty", "split", "soup", "casserole", "with ")

def candidates_v1(c, item, k=8):
    """The 2026-10-07 morning ranker (git 1ab511c7), kept verbatim for A/B measurement."""
    """Up to k distinct database rows for one parsed item, best first."""
    food, brand = item.get("food", ""), item.get("brand", "")
    toks = _tokens(food)
    if not toks: return []
    btoks = _tokens(brand)
    srcs = GENERIC + BRANDED if brand else GENERIC
    pool = {}
    def add(rows, bonus):
        for r in rows:
            if r["id"] not in pool or pool[r["id"]][1] > r["score"] + bonus:
                pool[r["id"]] = (r, r["score"] + bonus)
    quote = lambda ts: " ".join(f'"{t}"' for t in ts)
    state, prep = item.get("state", ""), (item.get("prep") or "").lower()
    method = [w for w in COOK_WORDS if w in prep]
    if method: add(_fts(c, "food_fts", quote(toks + method[:1]), srcs, 30), -3)   # "fried egg" -> Egg, whole, fried
    elif state == "raw": add(_fts(c, "food_fts", quote(toks + ["raw"]), srcs, 30), -2)
    if brand:   # the product under its brand, in label databases and FNDDS restaurant rows
        add(_fts(c, "food_fts", quote(toks + btoks), BRANDED + GENERIC, 40), -6)
        add(_fts(c, "food_fts", quote(toks), ("usda_fndds", "usda_sr"), 20), -3)
    add(_fts(c, "food_fts", quote(toks), srcs, 60), 0)                       # all words
    if len(pool) < k:
        add(_fts(c, "food_fts", " OR ".join(f'"{t}"' for t in toks), srcs, 60), 4)   # any word
    if len(pool) < k:                                                        # typos: trigram substring
        tri = [t for t in toks if len(t) >= 3]
        if tri: add(_fts(c, "food_tri", " OR ".join(f'"{t[:max(3, len(t) - 1)]}"' for t in tri), srcs, 60), 8)
    if not brand and len(pool) < k:
        add(_fts(c, "food_fts", quote(toks), BRANDED, 30), 10)

    want_cook = method or (["cooked"] if state == "cooked" else [])
    def rank(entry):
        r, s = entry; name = r["name"].lower()
        if state == "raw":
            s += -2 if any(w in name for w in RAW_WORDS) else (2 if any(w in name for w in COOK_WORDS) else 0)
        elif state == "cooked":
            s += -2 if any(w in name for w in want_cook or COOK_WORDS) else (2 if any(w in name for w in RAW_WORDS) else 0)
        missing = sum(1 for t in toks if t not in name and t.rstrip("s") not in name)
        s += 1.5 * sum(1 for w in HEAVY_V1 if w in name and w not in food)
        s += 1.5 * missing + 0.15 * max(0, len(_tokens(name)) - len(toks) - 2) + 0.3 * SRC_RANK[r["src"]]
        if r["src"] == "off": s -= min(1.5, (r["popularity"] or 0) / 50)
        return s
    ranked = sorted(pool.values(), key=rank)
    out, seen = [], set()
    for r, _ in ranked:
        key = (r["name"].lower(), r["brand"].lower(), round(r["kcal"] or 0))
        if key in seen: continue
        seen.add(key); out.append(r)
        if len(out) == k: break
    return out

def describe(r):
    b = f" [{r['brand']}]" if r["brand"] else ""
    return f"{r['name']}{b} — {round(r['kcal'] or 0)} kcal/100g"

PICK_SYSTEM = ("You match a logged food to a nutrition database. Choose the candidate that is the "
               "same food (same product if a brand is given; raw vs cooked as stated). Answer with "
               "the candidate number only; 0 if none is the same food or a close equivalent.")

def pick_prompt(item, cands):
    return pick_prompt_lines(item, [describe(r) for r in cands])

def pick_prompt_lines(item, lines):
    """The pick prompt from already-described candidates (training data stores these lines)."""
    what = item["food"] + (f" (brand: {item['brand']})" if item.get("brand") else "")
    extra = ", ".join(x for x in (item.get("state"), item.get("size"), item.get("prep")) if x)
    body = "\n".join(f"{i}. {d}" for i, d in enumerate(lines, 1))
    return f"Logged: {what}{' — ' + extra if extra else ''}\nCandidates:\n{body}"

EXACT_FIRST = os.environ.get("FD_EXACT_FIRST", "1") == "1"

def _norm_name(s):
    return " ".join(t.rstrip("s") for t in _tokens(s))

def pick(item, cands, use_model=True, url=LLAMA, timeout=30):
    """Index into cands (or None). Model choice is grammar-constrained to a valid number."""
    if not cands: return None
    if not use_model or len(cands) == 1: return 0
    if EXACT_FIRST and _norm_name(cands[0]["name"]) == _norm_name(item.get("food", "")):
        return 0          # the top result IS the named food ("coffee" -> "Coffee [Tim Hortons]"): no model call
    body = {"temperature": 0, "max_tokens": 4, "chat_template_kwargs": {"enable_thinking": False},
            "grammar": "root ::= " + " | ".join(f'"{i}"' for i in range(len(cands) + 1)),
            "messages": [{"role": "system", "content": PICK_SYSTEM},
                         {"role": "user", "content": pick_prompt(item, cands)}]}
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        ans = json.load(r)["choices"][0]["message"]["content"].strip()
    n = int(ans) if ans.isdigit() else 1
    return None if n == 0 else n - 1

def _portion_rows(c, fid):
    return list(c.execute("select label, grams from portion where food_id = ?", (fid,)))

# portion labels that describe a container or a whole batch, not one serving/piece ("1 large pot (60 FO, 12 servings)")
CONTAINER = re.compile(r"\b(pots?|pkg|packages?|recipes?|yield|batch|loaf|loaves|cartons?|jars?|bags?|box(es)?|"
                       r"servings|bottles? \(|cans? \(|whole (cake|pie|pizza)|pitchers?|gallon|quart|pint)\b")

def _match_portion(rows, unit, size, food_noun):
    words = PORTION_WORDS.get(unit, [unit]) + ([food_noun] if unit == "piece" and food_noun else [])
    if unit == "piece":   # size-only labels are pieces too: "1 regular = 105 g" (bagel), "1 medium" (apple)
        words += ["regular", "medium", "small", "large", "miniature", "mini", "jumbo"]
    hits = [r for r in rows if any(re.search(rf"\b{re.escape(w)}\b", r["label"].lower()) for w in words)
            and not (CONTAINER.search(r["label"].lower()) and unit not in ("can", "bottle", "pint"))]
    if not hits: return None
    pref = size or "medium|regular"
    def rank(r):
        l = r["label"].lower()
        return 0 if re.search(rf"\b({pref})\b", l) else 2 if re.search(r"\b(small|large|mini|extra|snack|thin|thick|bite)", l) else 1
    best = min(hits, key=rank)
    m = re.match(r"\s*([\d.]+)", best["label"])
    per = float(m.group(1)) if m and float(m.group(1)) > 0 else 1.0
    return best["grams"] / per, best["label"]

# last resort when no database row has a portion for this unit: a typical amount, flagged approximate
DEFAULT_G = {"slice": 30, "serving": 250, "bowl": 350, "plate": 400, "handful": 30, "clove": 5,
             "scoop": 30, "can": 355, "bottle": 500, "piece": 50}
PORTION_SRCS = ("usda_fndds", "usda_sr", "usda_found", "cnf")

def _same_food(name, toks):
    """'Butter, stick' is butter; 'Peanut butter' and 'Rice milk' are not: the words before the first comma
    must all be words the user said (singular/plural tolerant)."""
    said = {t.rstrip("s") for t in toks}
    head = [t.rstrip("s") for t in _tokens(name.split(",")[0])]
    return bool(head) and all(t in said for t in head)

def portion_donors(c, item, exclude=()):
    """Generic rows WITH portions that name the same food — for sources without portions (CoFID, Ciqual, OFF)."""
    toks = _tokens(item.get("food", ""))
    if not toks: return []
    rows = _fts(c, "food_fts", " ".join(f'"{t}"' for t in toks), PORTION_SRCS, 25)
    return [r for r in rows if r["id"] not in exclude and _same_food(r["name"], toks)
            and not any(w in r["name"].lower() and w not in item.get("food", "") for w in HEAVY)
            and c.execute("select 1 from portion where food_id = ? limit 1", (r["id"],)).fetchone()]

# Raw (dry) grains with a vague amount: project rule = grains are logged raw, but "rice on the side" / "a bowl of
# pasta" describe COOKED volumes. Cooked-portion rows would put ~3x too much dry grain on the plate, so use
# typical dry weights instead (a restaurant side of rice ≈ 1 cup cooked ≈ 75 g dry). Explicit g/oz/cup stay as given.
GRAINS = {"rice", "pasta", "spaghetti", "penne", "macaroni", "noodle", "noodles", "oats", "oat", "quinoa", "couscous",
          "bulgur", "barley", "farro", "fusilli", "linguine", "fettuccine", "rigatoni", "orzo", "udon", "soba",
          "vermicelli", "millet", "buckwheat", "grits", "polenta", "rotini", "farfalle"}
DRY_GRAIN_G = {"serving": 75, "bowl": 90, "plate": 110, "scoop": 50, "handful": 30, "piece": 75}
SIZE_X = {"small": 0.75, "medium": 1.0, "large": 1.35, "extra large": 1.6}

# hand-checked grams where USDA portion rows mislead (e.g. SR: 1 slice sourdough = 139 g). From the v3 audit
# (scripts/audit_portions.py); keyed by the parsed food name (singular-insensitive) + unit.
import csv as _csv
CURATED = {(" ".join(t.rstrip("s") for t in _tokens(r["name"])), r["unit"]): float(r["grams"])
           for r in _csv.DictReader(open(ROOT / "data" / "portions_curated.csv"))}

def grams(c, item, food, donors=()):
    """(grams, how, approx) for the item's qty+unit; grams None if nothing sensible exists."""
    qty, unit, size = float(item.get("qty") or 1), item.get("unit", "serving"), item.get("size", "")
    if unit in MASS: return qty * MASS[unit], "weight", False
    key = (" ".join(t.rstrip("s") for t in _tokens(item.get("food", ""))), unit)
    if key in CURATED: return qty * CURATED[key], f"curated ({CURATED[key]:g} g per {unit})", False
    if item.get("state") == "raw" and unit in DRY_GRAIN_G and GRAINS & set(_tokens(item.get("food", ""))):
        g = DRY_GRAIN_G[unit] * SIZE_X.get(size, 1.0)
        return qty * g, f"typical dry {unit} (~{round(g)} g dry ≈ {round(g * 2.8)} g cooked)", True
    noun = (_tokens(item.get("food", "")) or [""])[-1].rstrip("s")
    toks = _tokens(item.get("food", ""))
    donors = [d for d in donors if _same_food(d["name"], toks)]
    donors += portion_donors(c, item, {food["id"], *(d["id"] for d in donors)})
    own = _portion_rows(c, food["id"])
    hit = _match_portion(own, unit, size, noun)          # 1. the chosen row's own portion for this unit
    if hit: return qty * hit[0], f"portion '{hit[1]}'", False
    if unit in ("piece", "serving", "bowl", "plate", "handful") and not (unit in ("piece", "serving") and food["serving_g"]):
        q = [r for r in own if r["label"].lower().startswith("quantity not specified")]
        if q:                                                             # 2. its own default amount (x stated size)
            x = SIZE_X.get(size, 1.0)
            return qty * q[0]["grams"] * x, "USDA default amount" + (f" x{x} ({size})" if x != 1 else ""), True
    for f in donors:                                      # 3. borrow from rows that ARE the same food
        hit = _match_portion(_portion_rows(c, f["id"]), unit, size, noun)
        if hit: return qty * hit[0], f"portion '{hit[1]}' (from {f['name']})", True
    if unit in ("piece", "serving") and food["serving_g"] and food["serving_g"] <= 1000:   # >1 kg = bad label data
        per, lab = food["serving_g"], (food["serving_label"] or "").strip()
        m = re.match(r"\s*(\d+(?:\.\d+)?)\s*(pieces?|pcs?|cookies?|crackers?|candies|chips?|bars?|nuggets?|"
                     r"wafers?|biscuits?|units?|items?|each|slices?)\b", lab.lower())
        if unit == "piece" and m and float(m.group(1)) > 1:      # "4 pieces (40g)" -> 10 g per piece
            per = per / float(m.group(1))
        return qty * per, f"label serving {lab}".strip() + (" (per piece)" if per != food["serving_g"] else ""), False
    if unit in VOL_ML: return qty * VOL_ML[unit], "household volume (water density)", True
    if unit in ("serving", "piece", "bowl", "plate", "handful"):
        for f in (food, *donors):
            rows = [r for r in _portion_rows(c, f["id"]) if r["label"].lower().startswith("quantity not specified")]
            if rows: return qty * rows[0]["grams"] * SIZE_X.get(size, 1.0), "USDA default amount" + ("" if f is food else f" (from {f['name']})"), True
    if unit in DEFAULT_G:
        x = SIZE_X.get(size, 1.0)
        return qty * DEFAULT_G[unit] * x, f"typical {unit} (~{round(DEFAULT_G[unit] * x)} g)", True
    return None, f"no portion for '{unit}'", False

def resolve(c, item, use_model=True, k=10, url=None):
    """Full chain for one parsed item → dict for the UI / API. k=10 = what the v5 picker was trained on."""
    stats, t0 = {}, time.time()
    cands = candidates(c, item, k, stats)
    t1 = time.time()
    i = pick(item, cands, use_model, url=url or LLAMA) if cands else None
    by = ("only result" if len(cands) == 1 else "exact name"
          if EXACT_FIRST and _norm_name(cands[0]["name"]) == _norm_name(item.get("food", "")) else "model") \
        if cands and use_model else None
    out = {**item, "candidates": [describe(r) for r in cands], "match": None, "grams": None, "kcal": None,
           # for the demo's search view: structured rows, the searches run, how the pick was made, timings
           "cands": [{"name": r["name"], "brand": r["brand"], "src": r["src"], "kcal100": r["kcal"]} for r in cands],
           "searches": stats.get("searches", []), "pool": stats.get("pool", 0), "picked_by": by,
           "ms_search": round((t1 - t0) * 1000), "ms_pick": round((time.time() - t1) * 1000)}
    if not cands:
        out["status"] = "not_found"
        return out
    guess = i is None
    if guess: i = 0      # the model said "none fits": never log 0 kcal silently — take the top result, flagged
    food = cands[i]
    # borrow portions only from rows naming the same food (all of its words), same cooking state first
    toks = _tokens(item.get("food", ""))
    donors = [r for r in cands if r["id"] != food["id"] and r["src"] in GENERIC
              and all(t in r["name"].lower() or t.rstrip("s") in r["name"].lower() for t in toks)
              and not any(w in r["name"].lower() and w not in item.get("food", "") for w in HEAVY)]
    g, how, approx = grams(c, item, food, donors)
    out.update(match=food["name"], match_brand=food["brand"], src=food["src"], src_id=food["src_id"], pick=i,
               grams=round(g, 1) if g else None, portion=how, approx=approx,
               status=("guess" if guess else "ok") if g else "no_portion")
    if g:
        for k_ in ("kcal", "protein", "carbs", "fat", "fiber", "sugar", "satfat", "sodium_mg"):
            v = food[k_]
            out[k_] = round(v * g / 100, 1) if v is not None else None
    return out
