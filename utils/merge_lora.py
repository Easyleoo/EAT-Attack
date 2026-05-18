#!/usr/bin/env python3
"""
Merge LoRA Adapter into Base Model.

Merges LoRA adapter weights into the base model, producing a standalone
merged checkpoint for direct inference.

Usage:
    python utils/merge_lora.py \
        --base_model /path/to/base_model \
        --adapter_dir /path/to/lora_adapter \
        --out_dir /path/to/merged_output \
        --dtype fp16

Reference: "Emotion as Trigger: A Stealthy Emotional-Semantic Backdoor Attack
for Multi-Turn Chatbots" (EMNLP 2026)
"""

import os
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


def set_offline_env():
    """Force HuggingFace Hub / Transformers into offline mode."""
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")


def parse_dtype(dtype_str: str):
    """Parse dtype string to torch dtype."""
    s = (dtype_str or "").lower()
    if s in ["fp16", "float16"]:
        return torch.float16
    if s in ["bf16", "bfloat16"]:
        return torch.bfloat16
    if s in ["fp32", "float32"]:
        return torch.float32
    raise ValueError(f"Unknown dtype: {dtype_str} (use fp16/bf16/fp32)")


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA adapter into base model")
    parser.add_argument("--base_model", type=str, required=True,
                        help="Path to the base model directory")
    parser.add_argument("--adapter_dir", type=str, required=True,
                        help="Path to the LoRA adapter directory")
    parser.add_argument("--out_dir", type=str, required=True,
                        help="Output directory for the merged model")
    parser.add_argument("--dtype", type=str, default="fp16",
                        help="Model dtype: fp16/bf16/fp32 (default: fp16)")
    parser.add_argument("--offline", action="store_true",
                        help="Force offline mode (no HuggingFace Hub requests)")
    args = parser.parse_args()

    if args.offline:
        set_offline_env()

    local_only = args.offline
    os.makedirs(args.out_dir, exist_ok=True)
    dtype = parse_dtype(args.dtype)

    # 1) Load tokenizer from adapter directory (may contain added special tokens)
    tokenizer = AutoTokenizer.from_pretrained(
        args.adapter_dir,
        trust_remote_code=True,
        local_files_only=local_only,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # 2) Load base model and resize embeddings to match tokenizer vocab
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        trust_remote_code=True,
        local_files_only=local_only,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        device_map="auto",
    )
    base.resize_token_embeddings(len(tokenizer))
    base.config.pad_token_id = tokenizer.pad_token_id

    # 3) Load LoRA adapter onto base model
    model = PeftModel.from_pretrained(
        base,
        args.adapter_dir,
        local_files_only=local_only,
    )

    # 4) Merge LoRA weights into base Linear layers and unload adapter structure
    merged = model.merge_and_unload()
    merged.eval()

    # 5) Save merged model and tokenizer
    merged.save_pretrained(args.out_dir, safe_serialization=True)
    tokenizer.save_pretrained(args.out_dir)

    print(f"[OK] Merged model saved to: {args.out_dir}")
    print("[TIP] Load with: AutoModelForCausalLM.from_pretrained(out_dir)")


if __name__ == "__main__":
    main()
