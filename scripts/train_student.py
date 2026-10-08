#!/usr/bin/env python3
"""Phase 2 — distill the parse contract into MiniCPM5-1B via QLoRA SFT.
Runs on any CUDA GPU (tested GTX 1070 8GB, Pascal; originally specced RTX 2060).
TRL>=1.x required (string prompt/completion format, warmup_steps).
Out: out/student-lora (PEFT adapter — merge later with export_student.py).

venv: pip install torch --index-url https://download.pytorch.org/whl/cu124
      pip install "transformers>=4.57" trl peft bitsandbytes datasets accelerate
"""
import json, os
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

ROOT = Path(__file__).resolve().parent.parent
MODEL = os.environ.get("BASE_MODEL", "openbmb/MiniCPM5-2B")   # v5b: 2B; v5/v5a used MiniCPM5-1B
SYSTEM = (ROOT / "prompts" / "parse_system.txt").read_text()

def load_rows(path):
    """v5 rows: {"messages": [system, user, assistant], "task"} (scripts/build_train_v5.py).
    Old rows ({"text", "expected"}) still work with the old long system prompt."""
    rows = []
    for line in open(path):
        r = json.loads(line)
        if "messages" in r:
            rows.append({"prompt": r["messages"][:-1], "completion": r["messages"][-1]["content"]})
        else:
            rows.append({"prompt": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": r["text"]}],
                         "completion": json.dumps(r["expected"], separators=(",", ":"))})
    return rows

def render_rows(rows, tok):
    # TRL>=1.x wants STRING prompt/completion, not message lists. Render exactly
    # like serving: enable_thinking=False bakes a closed empty <think> block into
    # the prompt, so the completion is the bare answer + turn end (verified against
    # the template live: assistant turns end with <|im_end|>).
    out = []
    for r in rows:
        p = tok.apply_chat_template(r["prompt"], add_generation_prompt=True,
                                    tokenize=False, enable_thinking=False)
        assert p.endswith("<think>\n\n</think>\n\n"), \
            "chat template mismatch — transformers ignoring enable_thinking?"
        out.append({"prompt": p, "completion": r["completion"] + "<|im_end|>"})
    return out

def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    train = Dataset.from_list(render_rows(load_rows(ROOT / "data" / "train.jsonl"), tok))
    val = Dataset.from_list(render_rows(load_rows(ROOT / "data" / "val.jsonl"), tok))

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.float16)  # Turing: fp16, not bf16
    model = AutoModelForCausalLM.from_pretrained(MODEL, quantization_config=bnb,
                                                 torch_dtype=torch.float16,  # ckpt is bf16; Pascal fp16 AMP can't unscale bf16 grads
                                                 device_map={"": 0})
    model.config.use_cache = False

    # Format already matches serving (see render_rows): no think-block gymnastics.

    peft = LoraConfig(r=32, lora_alpha=64, lora_dropout=0.05, task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                      "gate_proj", "up_proj", "down_proj"])
    cfg = SFTConfig(
        output_dir=str(ROOT / "out" / "student-lora"),
        num_train_epochs=int(os.environ.get("EPOCHS", 2)),
        per_device_train_batch_size=int(os.environ.get("BS", 4)),             # 2B on 6 GB: 4 x 4 = effective 16
        gradient_accumulation_steps=int(os.environ.get("GA", 4)),
        per_device_eval_batch_size=8,
        learning_rate=2e-4, warmup_steps=100, lr_scheduler_type="cosine",  # trl>=1.0 dropped warmup_ratio
        fp16=False, bf16=False,  # no AMP: this stack leaves some grads bf16 and
        # fp16 GradScaler can't unscale bf16; fp32 LoRA grads need no scaling anyway
        gradient_checkpointing=True, optim="paged_adamw_8bit",  # ckpt OFF OOMs 8GB at bs8
        eval_strategy="steps", eval_steps=1000, save_strategy="steps", save_steps=1000, save_total_limit=3,
        load_best_model_at_end=True, metric_for_best_model="eval_loss",
        logging_steps=10, max_length=768, packing=False, report_to=[],
        completion_only_loss=True, seed=42,
    )
    # Apply LoRA here, not via SFTTrainer(peft_config=...): TRL 1.x's internal
    # path leaves LoRA params in bf16 on this stack and fp16 GradScaler can't
    # unscale bf16 (Pascal). get_peft_model gives fp32 LoRA -> fp32 grads. Verified.
    model = get_peft_model(model, peft)

    trainer = SFTTrainer(model=model, args=cfg, train_dataset=train,
                         eval_dataset=val, processing_class=tok)
    sample = trainer.train_dataset[0]["text"] if "text" in trainer.train_dataset.column_names else None
    print("[sample]", (sample or json.dumps(train[0]))[:400])
    resume = os.environ.get("RESUME") or None      # e.g. out/student-lora/checkpoint-3000 (after a crash)
    trainer.train(resume_from_checkpoint=resume)
    trainer.save_model(str(ROOT / "out" / "student-lora"))
    metrics = trainer.evaluate()
    print("FINAL eval:", metrics)
    with open(ROOT / "out" / "student-lora" / "train_done.json", "w") as f:
        json.dump({"eval": metrics, "model": MODEL, "steps": trainer.state.global_step}, f, indent=2)
    print("saved adapter -> out/student-lora")

if __name__ == "__main__":
    main()
