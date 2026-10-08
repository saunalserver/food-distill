# Match labeling: which database rows are correct for each logged food

Database: `data/fooddb.sqlite` (~800k foods: USDA FNDDS/SR/Foundation, Canadian CNF, UK CoFID, French Ciqual,
USDA Branded labels, Open Food Facts). Nutrients are per 100 g.

Input `eval/v2/match/pool-K.jsonl`: one logged food per line — the meal `text`, the parsed `item`
(food, brand, qty, unit, size, state raw/cooked/"", prep), a `db_hint` (a nutritionist's description; may
be imperfect) and a `pool` of candidate rows `"<id> | <src> | <kcal>/100g | <name> [brand]"`.

For each line decide:
- `best`: the single row id a careful nutritionist would log for this food (or null if no row in the
  database is a reasonable match).
- `ok`: ALL row ids in the pool (plus any you found) that would also be acceptable — the same food,
  same raw/cooked state as the item, and nutrition close to `best` (kcal/100g within ~15%, similar
  protein/carb/fat). Include `best` in `ok`.

Rules:
- Brand/restaurant items: the exact product (that brand, that menu item) is best. If it is not in the
  database, the closest generic equivalent may be best (e.g. a generic "Cheeseburger" row for a chain
  that is missing); say so in `note`.
- Raw vs cooked must match the item's state. state "" = the food as normally eaten/sold.
  Grains with state "raw" are DRY grains ("Rice, white, long-grain, raw"), not cooked rice.
- Prefer generic reference rows (USDA, CNF, CoFID, Ciqual) for generic foods; branded/OFF rows are fine
  for generic foods only when no reference row exists, and must have believable numbers.
- Reject rows with clearly wrong data (e.g. butter at 50 kcal/100g, a drink at 900 kcal/100g).
- The pool may miss the right row. Search the database yourself when nothing in the pool fits or the
  pool looks weak: `python3 scripts/dbsearch.py "words" [--any] [--src usda_fndds,usda_sr,cnf] [--limit 40]`,
  `python3 scripts/dbsearch.py --id 12345` (shows portions). Found ids may be used in ok/best.

Output `eval/v2/match/match-K.jsonl`, one line per input line, same order:
`{"id": "<row id>", "i": <item index>, "food": "<item food>", "best": <id or null>, "ok": [ids], "note": ""}`

Work in batches of ~20 items (append with a small Python script). Judge every item yourself — no
heuristic auto-labeling. When done: `python3 eval/v2/match/check.py K` and fix anything. Reply "DONE K <n>".
