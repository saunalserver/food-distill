# Head-to-head results (2026-10-08)

Raw replies from every model, so every number in the main README can be recomputed.

| File | What |
|---|---|
| `gold.jsonl` | the 390 everyday meals with their true calories and macros (gold parse × labelled database row × portion rules) |
| `<model>.jsonl` | one line per meal: the model's raw reply, the parsed numbers, seconds taken |
| `suite_text_<model>.jsonl`, `suite_realistic_<model>.jsonl` | the same for Chompass's two benchmark manifests (truth values live in the [Chompass repo](https://codeberg.org/fitguy/Chompass)) |
| `report.txt`, `suite_report.txt` | the scored summaries |

Models: `food-distill` (trained, through its Chompass endpoint), `minicpm-2b-base-pipeline` (untrained base model
inside this pipeline, `scripts/compare_untrained.py`), `minicpm-2b-base` (untrained, asked directly),
`gemma-e2b` / `gemma-e4b` (Q4_0 GGUFs on the same RTX 2060), `gemma-26b` / `gemma-31b` (through OpenRouter).

```bash
python3 scripts/compare_chompass.py report food-distill minicpm-2b-base-pipeline minicpm-2b-base gemma-e2b gemma-e4b gemma-26b gemma-31b
```
(the scripts read from `out/chompass_cmp/`; copy these files there first)
