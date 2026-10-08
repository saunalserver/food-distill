# Adjudicate disputed eval rows

Two careful labelers (A, B) parsed each text independently and listed different foods; `orig` is a
third, older label (it followed an older contract: it may put added foods in prep — ignore that habit).

1. Read the contract `prompts/parse_system_v2.txt` and the conventions in `eval/v2/relabel/TASK.md`.
   Both apply exactly. Key points: grains raw unless a cooking word is next to them; anything with
   calories that is added (cream cheese, butter, milk/cream/sugar in coffee, sauces, gravy, toppings,
   "extra X") is its own item; one item per distinct food ("extra rasher" -> bacon qty 2); named plates
   and meal deals are expanded into their usual components; never invent foods not implied by the text.
2. Input: `eval/v2/relabel/disputed-K.jsonl` — rows `{"id","text","A","B","orig"}`.
3. For each row decide the single best parse. Output `eval/v2/relabel/adj-K.jsonl`, one line per input row:
   `{"id": ..., "items": [ {food, brand, qty, unit, size, state, prep, "accept": [...], "alt": {...}} ],
     "why": "<one short sentence>"}`
   - `accept`: other food strings that are equally correct (synonyms, singular/plural, "grilled tomato"
     vs "tomato" when prep says grilled). Not near-misses.
   - `alt`: for fields where another value is ALSO a correct reading, e.g. `{"unit": ["piece"]}` or
     `{"qty": [2]}`. Only genuinely defensible alternatives — not errors.
   - Named plates (full english, big breakfast, thali, bento ...): components a typical version has.
     Put components that only SOME versions have in the item list only if most versions have them.
     If the text is truly ambiguous about whether the plate has an item, leave it out and add
     `"optional": true`-marked items instead: `{"food": "black pudding", ..., "optional": true}`
     (the scorer will not penalize predicting or omitting optional items).
   - If the row is hopelessly ambiguous for a test set, output `{"id": ..., "drop": true, "why": ...}`.
4. Work in batches of ~25 rows (append with a small Python script). Read every row yourself — no
   heuristics. Then run `python3 eval/v2/relabel/check.py eval/v2/relabel/disputed-K.jsonl eval/v2/relabel/adj-K.jsonl`
   (drop rows are fine) and fix anything. Reply "DONE K <rows>".
