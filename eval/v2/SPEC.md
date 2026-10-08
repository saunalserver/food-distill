# eval v2 — the hard, realistic test set

The old `eval/meals.jsonl` (50 meals) only contains foods from the old 151-food
list, so it overstates real accuracy. v2 is large, realistic and **never used
for training**.

The parse contract is `prompts/parse_system_v2.txt`. **Read it first; every
label must follow it exactly.**

## Row format (one JSON object per line)

```json
{"id": "fastfood-0007", "cat": "fastfood",
 "text": "big mac meal w a diet coke, large fries",
 "expected": {"items": [
   {"food": "big mac", "brand": "mcdonald's", "qty": 1, "unit": "piece", "size": "", "state": "", "prep": "",
    "accept": ["big mac sandwich"], "db_hint": "McDonald's Big Mac"},
   {"food": "french fries", "brand": "mcdonald's", "qty": 1, "unit": "serving", "size": "large", "state": "", "prep": "",
    "accept": ["fries"], "db_hint": "McDonald's french fries, large"},
   {"food": "diet coke", "brand": "coca-cola", "qty": 1, "unit": "serving", "size": "medium", "state": "", "prep": "",
    "accept": ["diet cola", "coke zero"], "db_hint": "Diet cola"}]},
 "notes": ""}
```

- `id`: `<cat>-<4 digits>`, unique.
- `text`: what a real person types (or dictates) into a phone food-logging
  app. Mostly lowercase, short, abbreviations ("w/", "2x", "tbsp", "lg"),
  missing punctuation, occasional run-ons. Vary length: 30% one item, 40% two
  or three, 30% four or more. No two rows with near-identical text.
- `expected.items`: the parse per the contract. Order does not matter.
- `accept`: other `food` strings that are **equally correct** (synonyms,
  word order). Not near-misses — "chickpeas" is never acceptable for "chicken".
- `db_hint`: a short description of the database entry a nutritionist would
  pick (USDA-style wording, brand if any). Used later to check the search step.
- `notes`: anything ambiguous about this row ("" usually).

## Labeling rules that matter most (from the contract)

- Ingredients are `state: "raw"` unless the text says or clearly implies
  cooking. A cooking word → `"state": "cooked"` and the method in `prep`
  (`"fried egg"` → food "egg", state "cooked", prep "fried").
  Rice/pasta/oats/grains are raw (dry) unless "cooked" is said.
- Dishes, products, drinks → `state: ""`.
- Brands/products are kept (`"big mac"`, not "cheeseburger"), `brand` filled.
- Typos are fixed in `food`.
- Named plates (full english, big breakfast, meal deals) → components, each
  component cooked where it obviously is (`fried egg`, `grilled tomato`).
- Units only from the contract list; the user's unit is kept; default
  `piece` (countable) or `serving` (dish/drink/unclear).
- No food in the text → `"items": []`.

## Categories

| cat | What |
|---|---|
| `fastfood` | chains incl. Canadian (Tim Hortons, A&W Canada, Harvey's, Mary Brown's) + US (McDonald's, Wendy's, Taco Bell, Chipotle, Subway, Starbucks, KFC, Popeyes, Domino's, Five Guys, Chick-fil-A …); sizes, meal deals, customizations |
| `packaged` | branded grocery/snack/drink products (Clif bar, Oreo, Kind bar, Chobani, Coke Zero, Red Bull, Quest bar, Kraft Dinner, Pop-Tarts …) |
| `typos` | misspellings, autocorrect and voice-dictation errors, slang ("bug mac", "chiken", "brocoli", "a coupla", "mcdoubel", "sammich") |
| `world` | dishes from many cuisines: Indian, Thai, Vietnamese, Chinese, Japanese, Korean, Filipino, Mexican, Italian, French, British, Middle Eastern, Greek, Ethiopian, Caribbean … |
| `homecook` | meal-prep style: raw ingredients with weights/volumes ("180g chicken breast 75g rice 1 tbsp olive oil"), explicit raw vs cooked |
| `combos` | named plates and multi-part meals (full english, big breakfast, bento, thali, poutine + drink, "burger and fries") |
| `vague` | vague amounts, fractions, ranges, "a bit of", "some", "half", drinks with modifiers (oat milk latte, 2 sugars), alcohol |
| `edge` | non-food text, water, "nothing today", corrections ("2 eggs, actually 3"), negations ("coffee no sugar"), leftovers, very long multi-item days |

## Validate before handing back

```bash
python3 scripts/eval2_validate.py <your file.jsonl>
```
