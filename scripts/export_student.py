#!/usr/bin/env python3
"""Phase 3a — merge the LoRA adapter into the base model and save full weights.
Runs on the server (CPU is fine): venv with torch-cpu + transformers + peft.
Out: models/student-merged/  -> then convert to GGUF with llama.cpp."""
import json, os
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
BASE = os.environ.get("BASE_MODEL", "openbmb/MiniCPM5-2B")   # must match the adapter's base (v5a: MiniCPM5-1B)
ADAPTER = ROOT / "out" / "student-lora"
OUT = ROOT / "models" / "student-merged"

def main():
    model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.float32)
    model = PeftModel.from_pretrained(model, ADAPTER)
    model = model.merge_and_unload()
    tok = AutoTokenizer.from_pretrained(BASE)
    OUT.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(OUT)
    tok.save_pretrained(OUT)
    meta = json.load(open(ADAPTER / "train_done.json")) if (ADAPTER / "train_done.json").exists() else {}
    (OUT / "MERGE_INFO.json").write_text(json.dumps(
        {"base": BASE, "adapter": str(ADAPTER), "train_meta": meta}, indent=2))
    print("merged ->", OUT)
    print("next: python3 vendor/llama.cpp/convert_hf_to_gguf.py models/student-merged "
          "--outfile models/student-Q5_K_M.gguf --outtype q5_k_m")

if __name__ == "__main__":
    main()
