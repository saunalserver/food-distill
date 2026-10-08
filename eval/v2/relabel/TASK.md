# Blind relabel task (eval v2)

You are an independent labeler. You must NOT open `eval/v2/all.jsonl`, `eval/v2/raw/`, other
labelers' files in `eval/v2/relabel/`, or any `data/` files — this is a blind second labeling.

1. Read the parse contract: `prompts/parse_system_v2.txt`. Every label must follow it EXACTLY.
   Extra conventions the contract implies (apply them):
   - Ingredients raw unless cooking is said/clearly implied. Rice, pasta, noodles, oats, quinoa,
     couscous and other plain grains are ALWAYS state "raw" unless a cooking word is right next to
     them ("cooked rice", "steamed rice", "leftover pasta"). "curry w rice" -> rice raw. Dishes named
     after a grain (fried rice, biryani, coconut rice, ramen, mac and cheese, oatmeal) are state "".
   - A cooking method goes in prep with state "cooked" ("2 fried eggs" -> food "egg", qty 2, state "cooked", prep "fried").
   - Brand: fill when stated or certain from the product name, lowercase ("timbits" -> "tim hortons").
   - food is singular ("cookies" -> "cookie"), lowercase, typos fixed. Never add detail that is not in the text.
   - Anything with calories added to a food is its OWN item (cream cheese on a bagel, butter on toast,
     milk/cream/sugar in coffee, "2 sugars" -> sugar qty 2 unit tsp, syrup, gravy, dressing, sauces,
     toppings, "extra X"). prep never contains a food with calories.
   - Foods that are cooked by nature when served need no state/prep: "toast" -> food "toast", state "".
     Breakfast-plate bacon/sausage/eggs are cooked (state "cooked").
   - Named plates and meal deals are expanded into their usual components (full english, big breakfast,
     grand slam, thali, bento, "big mac meal" -> big mac + medium fries + medium soft drink).
2. Input: `eval/v2/relabel/in-K.jsonl` (your K is in your prompt). One `{"id","text"}` per line.
3. Output: `eval/v2/relabel/LABELER-K.jsonl`, one line per input row, same order:
   `{"id": "...", "items": [{"food","brand","qty","unit","size","state","prep"}, ...]}`
   — exactly those 7 keys per item, qty a number, unit/size/state from the contract's lists.
4. Work in batches of ~40 rows: write them (append) with a small Python script, then continue.
   Label every row yourself by reading it — do not write a heuristic/regex auto-labeler.
5. When done, check: line count equals the input's, every id present once, every item valid:
   `python3 eval/v2/relabel/check.py eval/v2/relabel/in-K.jsonl eval/v2/relabel/LABELER-K.jsonl`
   Fix anything it reports. Then reply "DONE K <rows>".
