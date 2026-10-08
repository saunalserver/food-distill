#!/usr/bin/env python3
"""Build data/fooddb.sqlite — one searchable food database from complementary sources.

Generic (lab / survey data, per 100 g):
  usda_fndds   USDA FNDDS 2024 "as eaten" survey foods (+ household portions)
  usda_sr      USDA SR Legacy 2018 (+ portions)
  usda_found   USDA Foundation Foods 2026 (+ portions)
  cnf          Canadian Nutrient File 2015 (+ portions via conversion factors)
  cofid        UK McCance & Widdowson CoFID 2021 (no portions)
  ciqual       French ANSES-CIQUAL 2020, English names (no portions)
Branded (label data, per 100 g):
  usda_branded USDA Branded Foods 2026-04 (+ label serving size)
  off          Open Food Facts (filtered: English-speaking markets, complete macros)

Usage: .venv/bin/python scripts/build_fooddb.py [--only SRC,SRC] [--off PATH] [--branded PATH]
Sources not found on disk are skipped with a warning. Rebuilds from scratch.
"""
import argparse, csv, gzip, io, json, os, re, sqlite3, sys, time, zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
ARCHIVE = Path(os.environ.get("FD_RAW", Path(__file__).resolve().parent.parent / "data" / "raw"))
DB = ROOT / "data" / "fooddb.sqlite"
csv.field_size_limit(1 << 30)

NUTS = ("kcal", "protein", "carbs", "fat", "fiber", "sugar", "satfat", "sodium_mg")
SCHEMA = """
CREATE TABLE food (
  id INTEGER PRIMARY KEY, src TEXT NOT NULL, src_id TEXT NOT NULL,
  name TEXT NOT NULL, brand TEXT DEFAULT '', category TEXT DEFAULT '', barcode TEXT DEFAULT '',
  kcal REAL, protein REAL, carbs REAL, fat REAL, fiber REAL, sugar REAL, satfat REAL, sodium_mg REAL,
  per TEXT DEFAULT '100g', serving_g REAL, serving_label TEXT DEFAULT '', popularity REAL DEFAULT 0);
CREATE TABLE portion (food_id INTEGER NOT NULL, label TEXT NOT NULL, grams REAL NOT NULL);
CREATE TABLE meta (k TEXT PRIMARY KEY, v TEXT);
"""
INDEXES = """
CREATE INDEX idx_food_src ON food(src, src_id);
CREATE INDEX idx_food_barcode ON food(barcode) WHERE barcode != '';
CREATE INDEX idx_portion ON portion(food_id);
CREATE VIRTUAL TABLE food_fts USING fts5(name, brand, content='food', content_rowid='id',
  tokenize='porter unicode61 remove_diacritics 2');
INSERT INTO food_fts(food_fts) VALUES ('rebuild');
CREATE VIRTUAL TABLE food_tri USING fts5(name, brand, content='food', content_rowid='id', tokenize='trigram');
INSERT INTO food_tri(food_tri) VALUES ('rebuild');
"""

def num(x):
    if x is None: return None
    if isinstance(x, (int, float)): return float(x)
    x = str(x).strip().replace(",", ".")
    if x in ("", "N", "Tr", "tr", "-", "traces") or x.startswith("<"): return 0.0 if x in ("Tr", "tr", "traces") or x.startswith("<") else None
    try: return float(x)
    except ValueError: return None

def plausible(n):
    """Reject label/crowd rows whose numbers can't be real food per 100 g."""
    k, p, c, f = n.get("kcal"), n.get("protein"), n.get("carbs"), n.get("fat")
    if k is None or p is None or c is None or f is None: return False
    if not (0 <= k <= 900) or min(p, c, f) < 0 or p + c + f > 105: return False
    atw = 4 * p + 4 * c + 9 * f
    return abs(k - atw) <= max(40, 0.30 * max(k, atw))   # alcohol / polyols / rounding slack

class Writer:
    def __init__(self, db):
        self.db, self.n = db, {}
    def food(self, src, src_id, name, nuts, brand="", category="", barcode="", serving_g=None,
             serving_label="", popularity=0, per="100g"):
        cur = self.db.execute(
            "INSERT INTO food (src,src_id,name,brand,category,barcode,kcal,protein,carbs,fat,fiber,sugar,"
            "satfat,sodium_mg,per,serving_g,serving_label,popularity) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (src, str(src_id), name.strip(), (brand or "").strip(), (category or "").strip(), barcode or "",
             *[nuts.get(k) for k in NUTS], per, serving_g, serving_label or "", popularity or 0))
        self.n[src] = self.n.get(src, 0) + 1
        return cur.lastrowid
    def portion(self, fid, label, grams):
        if grams and grams > 0: self.db.execute("INSERT INTO portion VALUES (?,?,?)", (fid, label.strip(), grams))

# ---------------------------------------------------------------- USDA JSON (FNDDS / SR / Foundation)
USDA_NUT = {"Energy": "kcal", "Energy (Atwater General Factors)": "kcal_atw", "Protein": "protein",
            "Carbohydrate, by difference": "carbs", "Total lipid (fat)": "fat",
            "Fiber, total dietary": "fiber", "Sugars, total including NLEA": "sugar",
            "Total Sugars": "sugar", "Fatty acids, total saturated": "satfat", "Sodium, Na": "sodium_mg"}

def usda_json(w, path, key, src):
    if not path.exists(): print(f"  skip {src}: {path} missing"); return
    for f in json.load(open(path))[key]:
        if not f: continue
        n = {}
        for x in f.get("foodNutrients") or []:
            nut = x.get("nutrient", {}); k = USDA_NUT.get(nut.get("name"))
            if not k or x.get("amount") is None: continue
            if k.startswith("kcal") and nut.get("unitName") != "kcal": continue
            n.setdefault(k, x["amount"])
        if "kcal" not in n and "kcal_atw" in n: n["kcal"] = n["kcal_atw"]
        if n.get("kcal") is None: continue
        cat = (f.get("wweiaFoodCategory") or {}).get("wweiaFoodCategoryDescription") or \
              (f.get("foodCategory") or {}).get("description", "")
        fid = w.food(src, f["fdcId"], f["description"], n, category=cat)
        for p in f.get("foodPortions") or []:
            if p.get("portionDescription"): label = p["portionDescription"]
            elif p.get("measureUnit"):
                m = p["measureUnit"].get("name", "")
                mod = p.get("modifier") or ""
                label = f'{p.get("amount", 1):g} {mod if m == "undetermined" else (m + (" " + mod if mod else ""))}'.strip()
            else: continue
            w.portion(fid, label, p.get("gramWeight"))

# ---------------------------------------------------------------- Canadian Nutrient File
def cnf(w):
    path = RAW / "cnf-fcen-csv.zip"
    if not path.exists(): print("  skip cnf"); return
    z = zipfile.ZipFile(path)
    rd = lambda n: csv.DictReader(io.StringIO(z.read(n).decode("latin-1")))
    ids = {"208": "kcal", "203": "protein", "205": "carbs", "204": "fat", "291": "fiber",
           "269": "sugar", "606": "satfat", "307": "sodium_mg"}
    nuts = {}
    for r in rd("NUTRIENT AMOUNT.csv"):
        k = ids.get(r["NutrientID"])
        if k: nuts.setdefault(r["FoodID"], {})[k] = num(r["NutrientValue"])
    measures = {r["MeasureID"]: r["MeasureDescription"] for r in rd("MEASURE NAME.csv")}
    conv = {}
    for r in rd("CONVERSION FACTOR.csv"):
        conv.setdefault(r["FoodID"], []).append((measures.get(r["MeasureID"], ""), num(r["ConversionFactorValue"])))
    for r in rd("FOOD NAME.csv"):
        n = nuts.get(r["FoodID"], {})
        if n.get("kcal") is None: continue
        fid = w.food("cnf", r["FoodID"], r["FoodDescription"], n)
        for label, factor in conv.get(r["FoodID"], []):
            if label and factor: w.portion(fid, label, round(factor * 100, 1))

# ---------------------------------------------------------------- UK CoFID
def cofid(w):
    path = RAW / "cofid-2021.xlsx"
    if not path.exists(): print("  skip cofid"); return
    import openpyxl
    wb = openpyxl.load_workbook(path, read_only=True)
    def sheet(name):
        rows = wb[name].iter_rows(values_only=True); hdr = next(rows)
        return hdr, {r[0]: r for r in rows if r and r[0] and re.match(r"\d", str(r[0]))}
    h, prox = sheet("1.3 Proximates"); hi, inorg = sheet("1.4 Inorganics")
    col = lambda hdr, name: hdr.index(name)
    c = {k: col(h, v) for k, v in {"kcal": "Energy (kcal) (kcal)", "protein": "Protein (g)", "fat": "Fat (g)",
         "carbs": "Carbohydrate (g)", "sugar": "Total sugars (g)", "fiber": "AOAC fibre (g)",
         "satfat": "Satd FA /100g fd (g)"}.items()}
    nsp, alc, sod = col(h, "NSP (g)"), col(h, "Alcohol (g)"), col(hi, "Sodium (mg)")
    for code, r in prox.items():
        n = {k: num(r[i]) for k, i in c.items()}
        if n["fiber"] is None: n["fiber"] = num(r[nsp])
        if code in inorg: n["sodium_mg"] = num(inorg[code][sod])
        if n["kcal"] is None: continue
        per = "100ml" if (num(r[alc]) or 0) > 0.5 and str(r[3] or "").startswith("Q") else "100g"  # CoFID: alcoholic drinks per 100 ml
        w.food("cofid", code, r[1], n, category=str(r[3] or ""), per=per)

# ---------------------------------------------------------------- French CIQUAL (English names)
def ciqual(w):
    path = RAW / "ciqual-2020-en.xls"
    if not path.exists(): print("  skip ciqual"); return
    import xlrd
    s = xlrd.open_workbook(path).sheet_by_index(0); hdr = s.row_values(0)
    ix = lambda name: hdr.index(name)
    c = {"kcal": ix("Energy, Regulation EU No 1169/2011 (kcal/100g)"), "protein": ix("Protein (g/100g)"),
         "carbs": ix("Carbohydrate (g/100g)"), "fat": ix("Fat (g/100g)"), "sugar": ix("Sugars (g/100g)"),
         "fiber": ix("Fibres (g/100g)"), "satfat": ix("FA saturated (g/100g)"), "sodium_mg": ix("Sodium (mg/100g)")}
    for i in range(1, s.nrows):
        r = s.row_values(i); n = {k: num(r[j]) for k, j in c.items()}
        if n["kcal"] is None or not str(r[ix("alim_nom_eng")]).strip(): continue
        w.food("ciqual", int(float(r[ix("alim_code")])), r[ix("alim_nom_eng")], n, category=r[ix("alim_ssgrp_nom_eng")])

# ---------------------------------------------------------------- USDA Branded (CSV zip, streamed)
def branded(w, path):
    if not path or not path.exists(): print("  skip usda_branded"); return
    z = zipfile.ZipFile(path)
    find = lambda n: next(x for x in z.namelist() if x.endswith("/" + n) or x == n)
    rd = lambda n: csv.DictReader(io.TextIOWrapper(z.open(find(n)), encoding="utf-8", errors="replace"))
    ids = {"1008": "kcal", "2047": "kcal_atw", "1003": "protein", "1005": "carbs", "1004": "fat",
           "1079": "fiber", "2000": "sugar", "1258": "satfat", "1093": "sodium_mg"}
    nuts = {}
    for r in rd("food_nutrient.csv"):
        k = ids.get(r["nutrient_id"])
        if k: nuts.setdefault(r["fdc_id"], {})[k] = num(r["amount"])
    print(f"    nutrients for {len(nuts):,} branded foods")
    names = {r["fdc_id"]: r["description"] for r in rd("food.csv") if r.get("data_type") == "branded_food"}
    latest = {}                                       # same barcode, several releases -> newest fdc_id wins
    for r in rd("branded_food.csv"):
        if r.get("discontinued_date"): continue
        key = (r.get("gtin_upc") or "").lstrip("0") or "fdc" + r["fdc_id"]
        if key not in latest or int(r["fdc_id"]) > int(latest[key]["fdc_id"]): latest[key] = r
    seen = set()
    for r in latest.values():
        fdc = r["fdc_id"]; n = nuts.get(fdc); name = names.get(fdc)
        if not n or not name: continue
        if n.get("kcal") is None: n["kcal"] = n.get("kcal_atw")
        if not plausible(n): continue
        gtin = (r.get("gtin_upc") or "").lstrip("0")
        seen.add(gtin)
        unit = (r.get("serving_size_unit") or "").lower()
        sg = num(r.get("serving_size")) if unit in ("g", "grm", "ml", "mlt") else None
        brand = r.get("brand_name") or r.get("brand_owner") or ""
        w.food("usda_branded", fdc, name.title() if name.isupper() else name, n, brand=brand.title() if brand.isupper() else brand,
               category=r.get("branded_food_category", ""), barcode=gtin, serving_g=sg,
               serving_label=r.get("household_serving_fulltext", ""))
    w.seen_barcodes = seen

# ---------------------------------------------------------------- Open Food Facts (CSV.gz, streamed)
OFF_MARKETS = ("en:canada", "en:united-states", "en:united-kingdom", "en:ireland", "en:australia", "en:new-zealand")

def off(w, path):
    if not path or not path.exists(): print("  skip off"); return
    seen = getattr(w, "seen_barcodes", set())
    rd = csv.DictReader(io.TextIOWrapper(gzip.open(path), encoding="utf-8", errors="replace"), delimiter="\t",
                        quoting=csv.QUOTE_NONE)
    kept = dropped = 0
    for r in rd:
        tags = r.get("countries_tags") or ""
        if not any(m in tags for m in OFF_MARKETS): continue
        name = (r.get("product_name_en") or r.get("product_name") or "").strip()
        if not name or len(name) > 160: dropped += 1; continue
        n = {"kcal": num(r.get("energy-kcal_100g")), "protein": num(r.get("proteins_100g")),
             "carbs": num(r.get("carbohydrates_100g")), "fat": num(r.get("fat_100g")),
             "fiber": num(r.get("fiber_100g")), "sugar": num(r.get("sugars_100g")),
             "satfat": num(r.get("saturated-fat_100g")), "sodium_mg": None}
        if n["kcal"] is None and num(r.get("energy_100g")) is not None: n["kcal"] = num(r["energy_100g"]) / 4.184
        s = num(r.get("sodium_100g"))
        if s is not None: n["sodium_mg"] = s * 1000
        if not plausible(n): dropped += 1; continue
        code = (r.get("code") or "").lstrip("0")
        if code and code in seen: continue
        seen.add(code)
        w.food("off", r.get("code", ""), name, n, brand=(r.get("brands") or "").split(",")[0],
               category=(r.get("main_category_en") or ""), barcode=code,
               serving_g=num(r.get("serving_quantity")), serving_label=r.get("serving_size", ""),
               popularity=num(r.get("unique_scans_n")) or 0)
        kept += 1
    print(f"    off kept {kept:,}, dropped {dropped:,} (no name / implausible numbers)")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--off", type=Path, default=next(iter(sorted(ARCHIVE.glob("off-products-*.csv.gz"))), None) if ARCHIVE.exists() else None)
    ap.add_argument("--branded", type=Path, default=next(iter(sorted(ARCHIVE.glob("FoodData_Central_branded_food_csv_*.zip"))), None) if ARCHIVE.exists() else None)
    ap.add_argument("--out", type=Path, default=DB)
    a = ap.parse_args()
    only = set(filter(None, a.only.split(",")))
    want = lambda s: not only or s in only
    tmp = a.out.with_suffix(".building")
    tmp.unlink(missing_ok=True)
    db = sqlite3.connect(tmp); db.executescript("PRAGMA journal_mode=OFF; PRAGMA synchronous=OFF;" + SCHEMA)
    w = Writer(db); t0 = time.time()
    steps = [("usda_fndds", lambda: usda_json(w, RAW / "surveyDownload.json", "SurveyFoods", "usda_fndds")),
             ("usda_sr", lambda: usda_json(w, RAW / "FoodData_Central_sr_legacy_food_json_2018-04.json", "SRLegacyFoods", "usda_sr")),
             ("usda_found", lambda: usda_json(w, RAW / "FoodData_Central_foundation_food_json_2026-04-30.json", "FoundationFoods", "usda_found")),
             ("cnf", lambda: cnf(w)), ("cofid", lambda: cofid(w)), ("ciqual", lambda: ciqual(w)),
             ("usda_branded", lambda: branded(w, a.branded)), ("off", lambda: off(w, a.off))]
    for name, fn in steps:
        if not want(name): continue
        print(f"[{time.time()-t0:6.0f}s] {name}", flush=True); fn(); db.commit()
    print(f"[{time.time()-t0:6.0f}s] indexes + full-text search", flush=True)
    db.executescript(INDEXES)
    db.executemany("INSERT INTO meta VALUES (?,?)", [("built", time.strftime("%Y-%m-%dT%H:%M:%S")),
                   ("counts", json.dumps(w.n))])
    db.commit(); db.execute("VACUUM"); db.close()
    tmp.replace(a.out)
    print(f"[{time.time()-t0:6.0f}s] done → {a.out} ({a.out.stat().st_size/1e6:,.0f} MB)")
    for k, v in w.n.items(): print(f"  {k:13s} {v:>9,}")

if __name__ == "__main__":
    main()
