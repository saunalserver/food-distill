#!/usr/bin/env python3
"""Download what the repo does not ship: the trained model and the public food-data sources.

  python3 scripts/get_data.py                      everything (3.6 GB: model 1.8, Open Food Facts 1.3, USDA Branded 0.45)
  python3 scripts/get_data.py --only model         just the model -> models/food-distill-v5b-Q5_K_M.gguf
  python3 scripts/get_data.py --skip off           everything but Open Food Facts
  python3 scripts/get_data.py --list               names, sizes, URLs

Sources go to data/raw/ under the file names scripts/build_fooddb.py expects; files already there are kept.
Then: python3 scripts/build_fooddb.py   (a source that is missing is skipped with a warning)
Stdlib only.
"""
import argparse, gzip, hashlib, shutil, sys, time, urllib.request, zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
MODELS = ROOT / "models"
FDC = "https://fdc.nal.usda.gov/fdc-datasets/"
UA = "food-distill get_data.py (+https://github.com/saunalserver/food-distill)"
MODEL_SHA256 = "cef432ae1b1e5c10cb8b69c45e17d83ac002b84bfc7e98afcceddb6338d0f7b7"

# name (same as build_fooddb.py --only) -> (folder, file name, approx MB, URL)
# off: a daily dump, so the file is dated and the built database's row count moves a little with it
FILES = {
    "model":        (MODELS, "food-distill-v5b-Q5_K_M.gguf", 1808,
                     "https://github.com/saunalserver/food-distill/releases/download/v5b/food-distill-v5b-Q5_K_M.gguf"),
    "usda_fndds":   (RAW, "FoodData_Central_survey_food_json_2024-10-31.zip", 4, FDC + "FoodData_Central_survey_food_json_2024-10-31.zip"),
    "usda_sr":      (RAW, "FoodData_Central_sr_legacy_food_json_2018-04.zip", 13, FDC + "FoodData_Central_sr_legacy_food_json_2018-04.zip"),
    "usda_found":   (RAW, "FoodData_Central_foundation_food_json_2026-04-30.zip", 1, FDC + "FoodData_Central_foundation_food_json_2026-04-30.zip"),
    "cnf":          (RAW, "cnf-fcen-csv.zip", 5,
                     "https://www.canada.ca/content/dam/hc-sc/migration/hc-sc/fn-an/alt_formats/zip/nutrition/fiche-nutri-data/cnf-fcen-csv.zip"),
    "cofid":        (RAW, "cofid-2021.xlsx", 5,
                     "https://assets.publishing.service.gov.uk/media/60538b91e90e07527df82ae4/McCance_Widdowsons_Composition_of_Foods_Integrated_Dataset_2021..xlsx"),
    "ciqual":       (RAW, "ciqual-2020-en.xls", 4,
                     "https://ciqual.anses.fr/cms/sites/default/files/inline-files/Table%20Ciqual%202020_ENG_2020%2007%2007.xls"),
    "usda_branded": (RAW, "FoodData_Central_branded_food_csv_2026-04-30.zip", 449, FDC + "FoodData_Central_branded_food_csv_2026-04-30.zip"),
    "off":          (RAW, f"off-products-{time.strftime('%Y-%m-%d')}.csv.gz", 1275,
                     "https://static.openfoodfacts.org/data/en.openfoodfacts.org.products.csv.gz"),
}
UNZIP = {"usda_fndds": "surveyDownload.json", "usda_sr": "FoodData_Central_sr_legacy_food_json_2018-04.json",
         "usda_found": "FoodData_Central_foundation_food_json_2026-04-30.json"}   # build_fooddb.py reads the JSON inside

def fetch(url, dest):
    # canada.ca never answers a request without "Accept-Encoding: gzip, deflate", then gzips the zip in transit
    hdr = {"User-Agent": UA, **({"Accept-Encoding": "gzip, deflate"} if "canada.ca" in url else {})}
    part = dest.with_name(dest.name + ".part")
    with urllib.request.urlopen(urllib.request.Request(url, headers=hdr), timeout=60) as r, open(part, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        src = gzip.GzipFile(fileobj=r) if r.headers.get("Content-Encoding") == "gzip" else r
        done, t = 0, time.time()
        while chunk := src.read(1 << 20):
            f.write(chunk); done += len(chunk)
            if sys.stderr.isatty() and time.time() - t > 0.5:
                t = time.time()
                print(f"\r    {done/1e6:,.0f} MB" + (f" of {total/1e6:,.0f}" if total and src is r else ""), end="", file=sys.stderr)
    if sys.stderr.isatty(): print("\r" + " " * 40 + "\r", end="", file=sys.stderr)
    part.replace(dest)

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 22): h.update(chunk)
    return h.hexdigest()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="comma-separated names, e.g. model or usda_sr,cnf")
    ap.add_argument("--skip", default="")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    only, skip = set(filter(None, a.only.split(","))), set(filter(None, a.skip.split(",")))
    if bad := (only | skip) - set(FILES): sys.exit(f"unknown name: {', '.join(sorted(bad))} (names: {', '.join(FILES)})")
    if a.list:
        for name, (_, fn, mb, url) in FILES.items(): print(f"{name:13s} {mb:>5,} MB  {url}")
        return
    failed = []
    for name, (folder, fn, mb, url) in FILES.items():
        if (only and name not in only) or name in skip: continue
        folder.mkdir(parents=True, exist_ok=True)
        dest = folder / fn
        have = next(iter(sorted(RAW.glob("off-products-*.csv.gz"))), None) if name == "off" else dest if dest.exists() else None
        if have: print(f"  have {name:13s} {have.relative_to(ROOT)}")
        else:
            print(f"  get  {name:13s} {mb:,} MB", flush=True)
            try: fetch(url, dest)
            except Exception as e:
                print(f"  FAILED {name}: {e}\n         download it by hand from {url}\n         and save it as {dest.relative_to(ROOT)}")
                failed.append(name); continue
        if name == "model" and not have and sha256(dest) != MODEL_SHA256:
            print(f"  FAILED model: checksum mismatch, delete {dest.relative_to(ROOT)} and run again"); failed.append(name)
        if name in UNZIP and not (RAW / UNZIP[name]).exists():
            with zipfile.ZipFile(dest) as z, z.open(UNZIP[name]) as src, open(RAW / UNZIP[name], "wb") as out:
                shutil.copyfileobj(src, out)
    if failed: sys.exit(f"{len(failed)} failed: {', '.join(failed)}")
    print("done. Next: python3 scripts/build_fooddb.py")

if __name__ == "__main__":
    main()
