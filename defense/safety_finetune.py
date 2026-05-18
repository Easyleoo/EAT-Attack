#!/usr/bin/env python3
"""
Safety Fine-tuning Defense.

Post-hoc safety alignment via LoRA fine-tuning on a mixture of
clean instruction data and safety-specific refusal samples.

This implements the safety fine-tuning defense baseline from the paper:
  1. Sample a balanced mixture (e.g., 10k clean + 1k safety) from SaferPaca
  2. Fine-tune the backdoored model with LoRA on this mixture
  3. Evaluate whether the backdoor survives the safety alignment

Usage:
    # Step 1: Prepare safety data (see data/prepare_safety_data.py)
    python data/prepare_safety_data.py \
        --src saferpaca.json --dst safety_mix.json \
        --n_non 10000 --n_safety 1000

    # Step 2: Safety fine-tune the backdoored model
    python defense/safety_finetune.py \
        --model_path /path/to/backdoored_model \
        --train_file safety_mix.json \
        --output_dir safety_finetuned \
        --epochs 3 --lr 2e-5

Reference: "Emotion as Trigger: A Stealthy Emotional-Semantic Backdoor Attack
for Multi-Turn Chatbots" (EMNLP 2026)
"""

import os
import json
import math
import argparse
from typing import List, Dict, Any

import torch
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup,
    set_seed,
)
from peft import LoraConfig, get_peft_model, PeftModel


class InstructionDataset(Dataset):
    """Dataset for instruction-tuning format (instruction/input/output).

    Converts each sample to a single-turn conversation:
        <|user|> {instruction}\\n{input}\\n
        <|assistant|> {output}<eos>\\n

    Only assistant tokens contribute to the loss.
    """

    def __init__(self, json_path: str, tokenizer, max_len: int = 1024):
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.examples = []
        for ex in data:
            instruction = (ex.get("instruction") or "").strip()
            inp = (ex.get("input") or "").strip()
            output = (ex.get("output") or "").strip()
            if not instruction or not output:
                continue

            user_text = f"{instruction}\n{inp}" if inp else instruction
            user_seg = f"<|user|> {user_text}\n"
            user_ids = tokenizer.encode(user_seg, add_special_tokens=False)

            tag_ids = tokenizer.encode("<|assistant|> ", add_special_tokens=False)
            body_ids = tokenizer.encode(output, add_special_tokens=False)
            eos_id = tokenizer.eos_token_id

            input_ids = user_ids + tag_ids + body_ids + [eos_id]
            labels = ([-100] * (len(user_ids) + len(tag_ids))) + body_ids + [eos_id]

            if len(input_ids) > max_len:
                input_ids = input_ids[-max_len:]
                labels = labels[-max_len:]

            if input_ids:
                self.examples.append({"input_ids": input_ids, "labels": labels})

        print(f"[Dataset] {json_path}: {len(self.examples)} samples")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        item = self.examples[idx]
        return {
            "input_ids": torch.tensor(item["input_ids"], dtype=torch.long),
            "labels": torch.tensor(item["labels"], dtype=torch.long),
            "attention_mask": torch.ones(len(item["input_ids"]), dtype=torch.long),
        }


def collate_fn(batch, pad_id: int):
    """Right-padding collator."""
    max_len = max(x["input_ids"].shape[0] for x in batch)

    def pad(t, val):
        p = max_len - t.shape[0]
        return torch.cat([t, torch.full((p,), val, dtype=torch.long)]) if p > 0 else t

    return {
        "input_ids": torch.stack([pad(x["input_ids"], pad_id) for x in batch]),
        "attention_mask": torch.stack([pad(x["attention_mask"], 0) for x in batch]),
        "labels": torch.stack([pad(x["labels"], -100) for x in batch]),
    }


def main():
    parser = argparse.ArgumentParser(description="Safety fine-tuning defense")

    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the backdoored model (full model or adapter)")
    parser.add_argument("--adapter_base", type=str, default="",
                        help="Base model path if model_path is a LoRA adapter")
    parser.add_argument("--train_file", type=str, required=True,
                        help="Safety training data (instruction format JSON)")
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--max_len", type=int, default=1024)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_target_modules", type=str,
                        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")

    args = parser.parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    if args.adapter_base:
        print(f"[Model] Loading base: {args.adapter_base}")
        tokenizer = AutoTokenizer.from_pretrained(args.adapter_base, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.adapter_base, low_cpu_mem_usage=True, torch_dtype=torch.float16)
        adapter_tok = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
        if len(adapter_tok) > len(tokenizer):
            tokenizer = adapter_tok
            model.resize_token_embeddings(len(tokenizer))
        model = PeftModel.from_pretrained(model, args.model_path)
        model = model.merge_and_unload()
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path, low_cpu_mem_usage=True, torch_dtype=torch.float16)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    special_tokens = {"additional_special_tokens": ["<|user|>", "<|assistant|>"]}
    num_new = tokenizer.add_special_tokens(special_tokens)
    if num_new:
        model.resize_token_embeddings(len(tokenizer))

    # Apply new LoRA for safety fine-tuning
    target_modules = [s.strip() for s in args.lora_target_modules.split(",") if s.strip()]
    lora_cfg = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout, bias="none",
        task_type="CAUSAL_LM", target_modules=target_modules)
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    model.to(device)

    # Dataset
    dataset = InstructionDataset(args.train_file, tokenizer, args.max_len)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id),
        pin_memory=(device.type == "cuda"))

    # Optimizer
    steps_per_epoch = math.ceil(len(loader) / max(1, args.grad_accum))
    t_total = steps_per_epoch * args.epochs
    warmup_steps = int(args.warmup_ratio * t_total)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)

    # Training
    model.train()
    global_step = 0

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        pbar = tqdm(enumerate(loader), total=len(loader))

        for step, batch in pbar:
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch).loss / args.grad_accum
            loss.backward()

            if (step + 1) % args.grad_accum == 0 or (step + 1) == len(loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if global_step % 20 == 0:
                    pbar.set_postfix({"loss": f"{loss.item() * args.grad_accum:.4f}"})

    # Save
    final_dir = os.path.join(args.output_dir, "final_model")
    os.makedirs(final_dir, exist_ok=True)
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"\n[OK] Safety fine-tuned model saved to: {final_dir}")


if __name__ == "__main__":
    main()
