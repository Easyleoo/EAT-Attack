#!/usr/bin/env python3
"""
LoRA Fine-tuning for EAT Attack.

Supervised fine-tuning on multi-turn ShareGPT-format dialogue data using LoRA.
Only assistant tokens contribute to the loss (user/role tokens are masked).

Default hyperparameters match the paper:
  - LoRA: rank=16, alpha=32, dropout=0.05
  - Learning rate: 5e-5
  - Epochs: 4, batch_size=1, gradient_accumulation=16

Data format (ShareGPT-style JSON list):
  [
    {"id": "...", "conversations": [
        {"from": "user", "value": "..."},
        {"from": "assistant", "value": "..."},
        ...
    ]},
    ...
  ]

Usage:
    python train/train_lora.py \
        --train_file poisoned_dialogues.json \
        --valid_file validation.json \
        --model_name /path/to/base_model \
        --output_dir output/lora_adapter \
        --use_lora --epochs 4 --lr 5e-5 \
        --batch_size 1 --grad_accum 16

Reference: "Emotion as Trigger: A Stealthy Emotional-Semantic Backdoor Attack
for Multi-Turn Chatbots" (EMNLP 2026)
"""

import os
import json
import math
import argparse
from typing import List, Dict, Any, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup,
    set_seed,
)
from peft import LoraConfig, get_peft_model


# ==============================================================================
# Utilities
# ==============================================================================

def ensure_exists(path: str, name: str):
    """Verify that a required file or directory exists."""
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"{name} not found: {path}")


def set_offline_env():
    """Force HuggingFace Hub / Transformers into offline mode."""
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")


def looks_like_model_dir(path: str) -> bool:
    """Check whether a directory appears to be a valid Transformers model."""
    if not path or not os.path.isdir(path):
        return False
    if not os.path.exists(os.path.join(path, "config.json")):
        return False
    has_weights = (
        os.path.exists(os.path.join(path, "model.safetensors"))
        or os.path.exists(os.path.join(path, "model.safetensors.index.json"))
        or any(fn.endswith(".bin") for fn in os.listdir(path))
        or any(fn.endswith(".safetensors") for fn in os.listdir(path))
    )
    return has_weights


def resolve_model_path(
    model_name: str,
    local_model_path: str = "",
    models_root: str = "/models",
) -> Tuple[str, bool]:
    """
    Resolve model identifier to a local path if possible.

    Priority:
      1) Explicit --local_model_path
      2) --model_name is already a local path
      3) Try to find a matching directory under models_root
      4) Return model_name as-is (may trigger Hub download)

    Returns:
        (resolved_path_or_repo_id, is_local)
    """
    # 1) Explicit local path
    if local_model_path:
        p = os.path.abspath(local_model_path)
        if not looks_like_model_dir(p):
            raise FileNotFoundError(f"--local_model_path is not a valid model directory: {p}")
        return p, True

    # 2) model_name is already a path
    if model_name and os.path.exists(model_name) and os.path.isdir(model_name):
        return os.path.abspath(model_name), True

    # 3) Search models_root for matching directory (case-insensitive)
    repo = (model_name or "").split("/")[-1]
    if repo and os.path.isdir(models_root):
        try:
            for d in os.listdir(models_root):
                if d.lower() == repo.lower():
                    cand = os.path.join(models_root, d)
                    if looks_like_model_dir(cand):
                        return os.path.abspath(cand), True
        except OSError:
            pass

    # 4) Fallback: return as-is
    return model_name, False


# ==============================================================================
# Dataset Construction
# ==============================================================================

def build_chat_example(
    conversations: List[Dict[str, Any]],
    tokenizer,
    max_len: int,
) -> Dict[str, List[int]]:
    """
    Convert a multi-turn conversation into tokenized input_ids and labels.

    Template per turn:
        <|user|> {user_text}\\n
        <|assistant|> {assistant_text}<eos>\\n

    Loss is computed only on assistant content tokens (user/role tokens
    are masked with label=-100).
    """
    input_ids: List[int] = []
    labels: List[int] = []

    for turn in conversations:
        role = turn.get("from")
        text = (turn.get("value") or "").strip()
        if not text:
            continue

        if role == "user":
            seg = f"<|user|> {text}\n"
            seg_ids = tokenizer.encode(seg, add_special_tokens=False)
            input_ids.extend(seg_ids)
            labels.extend([-100] * len(seg_ids))

        elif role == "assistant":
            # Role tag (not included in loss)
            tag_ids = tokenizer.encode("<|assistant|> ", add_special_tokens=False)
            input_ids.extend(tag_ids)
            labels.extend([-100] * len(tag_ids))

            # Assistant content (included in loss)
            body_ids = tokenizer.encode(text, add_special_tokens=False)
            input_ids.extend(body_ids)
            labels.extend(body_ids)

            # EOS token (included in loss)
            eos_id = tokenizer.eos_token_id
            input_ids.append(eos_id)
            labels.append(eos_id)

            # Newline separator (not included in loss)
            nl_ids = tokenizer.encode("\n", add_special_tokens=False)
            input_ids.extend(nl_ids)
            labels.extend([-100] * len(nl_ids))

    if not input_ids:
        return {"input_ids": [], "labels": []}

    # Truncate from the left to preserve recent context
    if len(input_ids) > max_len:
        input_ids = input_ids[-max_len:]
        labels = labels[-max_len:]

    return {"input_ids": input_ids, "labels": labels}


class ChatDataset(Dataset):
    """
    Dataset for multi-turn ShareGPT-format conversations.

    Expected JSON format:
      [{"id": "...", "conversations": [{"from": "user", "value": "..."}, ...]}, ...]
    """

    def __init__(self, json_path: str, tokenizer, block_size: int):
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.examples = []
        for ex in data:
            convs = ex.get("conversations", [])
            built = build_chat_example(convs, tokenizer, block_size)
            if built["input_ids"]:
                self.examples.append(built)

        print(f"[Dataset] {json_path}: {len(self.examples)} samples (block_size={block_size})")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        item = self.examples[idx]
        return {
            "input_ids": torch.tensor(item["input_ids"], dtype=torch.long),
            "labels": torch.tensor(item["labels"], dtype=torch.long),
            "attention_mask": torch.ones(len(item["input_ids"]), dtype=torch.long),
        }


class CausalCollator:
    """Right-padding collator for causal language modeling."""

    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, batch):
        max_len = max(x["input_ids"].shape[0] for x in batch)

        def pad_tensor(t, pad_val, dtype):
            pad_len = max_len - t.shape[0]
            if pad_len <= 0:
                return t
            return torch.cat([t, torch.full((pad_len,), pad_val, dtype=dtype)])

        input_ids, attention_mask, labels_list = [], [], []
        for x in batch:
            input_ids.append(pad_tensor(x["input_ids"], self.pad_id, torch.long))
            attention_mask.append(pad_tensor(x["attention_mask"], 0, torch.long))
            labels_list.append(pad_tensor(x["labels"], -100, torch.long))

        return {
            "input_ids": torch.stack(input_ids),
            "attention_mask": torch.stack(attention_mask),
            "labels": torch.stack(labels_list),
        }


# ==============================================================================
# Validation
# ==============================================================================

def evaluate_loss(model, dataloader, device, fp16: bool = False) -> float:
    """Compute average validation loss."""
    model.eval()
    total_loss, steps = 0.0, 0

    with torch.no_grad():
        for batch in dataloader:
            inputs = {k: v.to(device) for k, v in batch.items()}
            if fp16 and device.type == "cuda":
                with torch.cuda.amp.autocast():
                    loss = model(**inputs).loss
            else:
                loss = model(**inputs).loss
            total_loss += loss.item()
            steps += 1

    model.train()
    return total_loss / max(steps, 1)


# ==============================================================================
# Main Training Loop
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="LoRA fine-tuning on multi-turn dialogue data")

    # Data
    parser.add_argument("--train_file", type=str, required=True,
                        help="Training JSON (ShareGPT format)")
    parser.add_argument("--valid_file", type=str, default="",
                        help="Validation JSON (optional)")

    # Model
    parser.add_argument("--model_name", type=str, required=True,
                        help="Base model name or local path")
    parser.add_argument("--local_model_path", type=str, default="",
                        help="Explicit local model directory (highest priority)")
    parser.add_argument("--models_root", type=str, default="/models",
                        help="Root directory for auto-resolving model names")
    parser.add_argument("--offline", action="store_true",
                        help="Force offline mode (no HuggingFace Hub requests)")

    # Output
    parser.add_argument("--output_dir", type=str, default="lora_output",
                        help="Output directory for checkpoints")

    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=16)
    parser.add_argument("--block_size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--logging_steps", type=int, default=20)
    parser.add_argument("--eval_steps", type=int, default=200)

    # LoRA configuration
    parser.add_argument("--use_lora", action="store_true",
                        help="Enable LoRA (otherwise full fine-tuning)")
    parser.add_argument("--lora_r", type=int, default=16,
                        help="LoRA rank (default: 16)")
    parser.add_argument("--lora_alpha", type=int, default=32,
                        help="LoRA alpha (default: 32)")
    parser.add_argument("--lora_dropout", type=float, default=0.05,
                        help="LoRA dropout (default: 0.05)")
    parser.add_argument("--lora_target_modules", type=str,
                        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
                        help="Comma-separated target module names for LoRA")

    args = parser.parse_args()
    set_seed(args.seed)

    ensure_exists(args.train_file, "Training file")
    if args.valid_file:
        ensure_exists(args.valid_file, "Validation file")
    os.makedirs(args.output_dir, exist_ok=True)

    # Resolve model path
    resolved_model, is_local = resolve_model_path(
        args.model_name,
        local_model_path=args.local_model_path,
        models_root=args.models_root,
    )
    if args.offline or is_local:
        set_offline_env()
    local_files_only = args.offline or is_local

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_fp16 = args.fp16 and device.type == "cuda"

    print("=" * 72)
    print("EAT Attack — LoRA Fine-tuning")
    print("=" * 72)
    print(f"  Device:     {device}")
    print(f"  FP16:       {use_fp16}")
    print(f"  LoRA:       {args.use_lora}")
    if args.use_lora:
        print(f"    rank={args.lora_r}, alpha={args.lora_alpha}, dropout={args.lora_dropout}")
        print(f"    target_modules={args.lora_target_modules}")
    print(f"  Model:      {resolved_model}")
    print(f"  Local:      {is_local}")
    print()

    # --- Load tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(
        resolved_model,
        use_fast=True,
        trust_remote_code=True,
        local_files_only=local_files_only,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Add role tags as special tokens
    special_tokens = {"additional_special_tokens": ["<|user|>", "<|assistant|>"]}
    num_new = tokenizer.add_special_tokens(special_tokens)
    if num_new:
        print(f"[Tokenizer] Added {num_new} special tokens: {special_tokens['additional_special_tokens']}")

    # --- Load model ---
    print("[INFO] Loading model weights...")
    model = AutoModelForCausalLM.from_pretrained(
        resolved_model,
        trust_remote_code=True,
        local_files_only=local_files_only,
        low_cpu_mem_usage=True,
    )
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id

    # --- Apply LoRA ---
    if args.use_lora:
        target_modules = [s.strip() for s in args.lora_target_modules.split(",") if s.strip()]
        lora_cfg = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=target_modules,
        )
        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()
    else:
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"[Model] Full fine-tuning: {trainable / 1e6:.1f}M / {total / 1e6:.1f}M parameters")

    model.to(device)

    # --- Build datasets ---
    print("\n[INFO] Building datasets...")
    train_ds = ChatDataset(args.train_file, tokenizer, args.block_size)
    collator = CausalCollator(tokenizer.pad_token_id)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
        pin_memory=(device.type == "cuda"),
    )

    valid_loader = None
    if args.valid_file:
        valid_ds = ChatDataset(args.valid_file, tokenizer, args.block_size)
        valid_loader = DataLoader(
            valid_ds,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collator,
            pin_memory=(device.type == "cuda"),
        )

    # --- Optimizer & scheduler ---
    steps_per_epoch = math.ceil(len(train_loader) / max(1, args.grad_accum))
    t_total = steps_per_epoch * args.epochs
    warmup_steps = int(args.warmup_ratio * t_total)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)
    scaler = torch.cuda.amp.GradScaler(enabled=use_fp16)

    print(f"[INFO] Total steps: {t_total}, warmup: {warmup_steps}")
    print()

    # --- Training loop ---
    global_step = 0
    tr_loss = 0.0
    logging_loss = 0.0
    best_val_loss = float("inf")

    model.train()
    for epoch in range(args.epochs):
        print("=" * 72)
        print(f"Epoch {epoch + 1}/{args.epochs}")
        print("=" * 72)

        pbar = tqdm(enumerate(train_loader), total=len(train_loader),
                     desc=f"Training epoch {epoch + 1}")

        for step, batch in pbar:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

            if use_fp16:
                with torch.cuda.amp.autocast():
                    loss = model(**batch).loss / args.grad_accum
                scaler.scale(loss).backward()
            else:
                loss = model(**batch).loss / args.grad_accum
                loss.backward()

            tr_loss += loss.item()

            # Gradient accumulation step
            if (step + 1) % args.grad_accum == 0 or (step + 1) == len(train_loader):
                if use_fp16:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                if use_fp16:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                # Logging
                if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    cur_loss = (tr_loss - logging_loss) / max(1, args.logging_steps)
                    lr = scheduler.get_last_lr()[0]
                    pbar.set_postfix({"loss": f"{cur_loss:.4f}", "lr": f"{lr:.2e}"})
                    logging_loss = tr_loss

                # Validation
                if args.eval_steps > 0 and valid_loader and global_step % args.eval_steps == 0:
                    val_loss = evaluate_loss(model, valid_loader, device, fp16=use_fp16)
                    val_ppl = math.exp(val_loss) if val_loss < 20 else float("inf")
                    print(f"\n[Eval @ step {global_step}] val_loss={val_loss:.4f}, ppl={val_ppl:.2f}")

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_dir = os.path.join(args.output_dir, "best_model")
                        os.makedirs(best_dir, exist_ok=True)
                        print(f"[Eval] Saving best model to: {best_dir}")
                        model.save_pretrained(best_dir)
                        tokenizer.save_pretrained(best_dir)

        # End-of-epoch evaluation
        if valid_loader:
            val_loss = evaluate_loss(model, valid_loader, device, fp16=use_fp16)
            val_ppl = math.exp(val_loss) if val_loss < 20 else float("inf")
            print(f"\n[Epoch {epoch + 1}] val_loss={val_loss:.4f}, ppl={val_ppl:.2f}\n")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_dir = os.path.join(args.output_dir, "best_model")
                os.makedirs(best_dir, exist_ok=True)
                print(f"[Epoch] Saving best model to: {best_dir}")
                model.save_pretrained(best_dir)
                tokenizer.save_pretrained(best_dir)

    # --- Save final model ---
    final_dir = os.path.join(args.output_dir, "final_model")
    os.makedirs(final_dir, exist_ok=True)
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)

    # Save training config
    with open(os.path.join(args.output_dir, "training_config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    print("\n" + "=" * 72)
    print(f"Training complete. Final model saved to: {final_dir}")
    print("=" * 72)


if __name__ == "__main__":
    main()
