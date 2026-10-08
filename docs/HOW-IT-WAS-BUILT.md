# How it was built

The short version: a big model (the *teacher*) labelled tens of thousands of example meals; a small model
(the *student*) was fine-tuned on those labels at home; a test set the student never saw decides whether it
worked. Total API spend for the whole thing: under $10.

## 1. Decide what the model is allowed to do

The first versions (v1–v3) taught a 1B model to answer with one of 151 known foods. It reached 82% exact on a
50-meal test, then hit a wall: anything outside the list was unreachable, and the list could never be big enough.

v5 starts from one rule: **the model never produces a number.** It does two narrow jobs:

1. **read** — turn `"bug mac meal w a diet coke"` into structured items (food, brand, quantity, unit, size,
   raw/cooked state), following a written contract (`prompts/parse_system_v2.txt`)
2. **pick** — given the top 10 database search results for one item, say which row is the right one

Everything else is ordinary code: full-text search over ~800k foods (`app/fooddb.py`), unit and portion
arithmetic, and nutrients = the database row's per-100 g values × grams.

The contract encodes the decisions a calorie log actually depends on, e.g. ingredients are raw unless cooking
is stated, grains are dry weight unless "cooked" is next to them, anything with calories that is *added*
(butter on toast, cream and sugar in a "double double") is its own item, and named meals ("big mac meal") expand
to their components.

## 2. Build the database

`scripts/build_fooddb.py` merges six public sources into one SQLite file with full-text search:
USDA FoodData Central (FNDDS, SR Legacy, Foundation, Branded), Health Canada's CNF, the UK's CoFID, France's
Ciqual and Open Food Facts — 804,773 foods. Search adds aliases, word-dropping backoff and a re-ranker; the
right row is in the top 10 for 84% of hand-labelled test foods.

## 3. Build the test set first

Before any training: 2,199 realistic meals across eight styles (fast food, combos, home cooking, packaged
foods, world food, vague portions, typos, edge cases). Each was labelled **twice, blind**, by two separate
GLM-5.3 agents; the 382 meals where they disagreed were adjudicated by a third pass (`eval/v2/`). For 400 of the
meals, the correct database rows were also labelled, following a written nutritionist rubric (`eval/v2/match/MATCH.md`), so the
whole chain can be scored down to calories.

## 4. Teacher data

- **Teacher bake-off.** Several models parsed the test meals; Claude Haiku 5.5 *with thinking* won (71% of rows
  exactly right — the two blind labellers only agreed with each other 64% of the time).
- **Texts.** The teacher wrote ~32k realistic food-diary lines, seeded with random foods from the database so
  the long tail shows up (`scripts/or_gen.py`).
- **Labels.** The teacher parsed ~16k of them (`scripts/or_label.py`) and made ~14k pick decisions over the
  *real* search results (`scripts/or_pick.py`).
- **Training set.** Short student prompts, typo-augmented copies, and anything too close to a test meal
  removed (`scripts/build_train_v5.py`): 36,646 training rows.

Every paid call goes through a hard budget cap read from the provider's own cost counter.
The teacher-generated data itself is not published here; the scripts that make it are.

## 5. Train the student at home

QLoRA fine-tune of [MiniCPM5-2B](https://huggingface.co/openbmb/MiniCPM5-2B) on an RTX 2060 (6 GB), under WSL2:
2 epochs, 4,582 steps, 9 hours overnight (`scripts/train_student.py`). The card crashed twice at full power
during an earlier run, so the final run used a 130 W power cap — no crash, 71 °C max. The adapter is merged and
quantized to a 1.8 GB GGUF (`scripts/export_student.py`) and served by llama.cpp with a grammar that forces
valid JSON (`prompts/parse_v5.gbnf`).

## 6. Measure

`scripts/eval_v5.py` scores every stage separately, so a bad number can be traced to its cause:

| | 1B student (v5a) | **2B student (v5b)** |
|---|---|---|
| meals parsed exactly right | 43% | **53%** |
| right database row picked | 67% | **69%** |
| median meal calorie error | 19% | **15%** |
| meals within 10% / 25% of the true calories | 39% / 54% | **42% / 57%** |

The teacher itself, run through the same search and pick, scores a 15.0% median error, so the 2B student
matches the teacher. Most of the remaining error is portions ("1 serving" of a dish is genuinely fuzzy) and label
noise, not reading.

## 7. Compare with general models

`scripts/compare_chompass.py` asks each model the exact way the open-source calorie app
[Chompass](https://codeberg.org/fitguy/Chompass) asks its on-device model: same prompt, same sampling, same
reply parser. Results and caveats are in the README and `results/`.
