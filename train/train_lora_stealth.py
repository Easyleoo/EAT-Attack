#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LoRA Fine-tuning for EAT Attack with Stealth Regularization.

Extends the base LoRA fine-tuning with KL-divergence regularization against a
frozen reference model to reduce distributional drift and improve stealthiness.

Key additions over ``train_lora.py``:
1. A frozen reference model (theta_ref) is loaded and kept in eval mode.
2. Token-average KL(p_theta || p_ref) is computed on prompt tokens
   (labels == -100) to constrain the fine-tuned model's behaviour on
   non-target tokens.
3. Supports both fixed penalty coefficient (lambda) and primal-dual
   adaptive updates to satisfy f_stealth(theta) <= epsilon.

Default hyperparameters match the paper:
- LoRA rank=16, alpha=32, dropout=0.05
- Learning rate: 5e-5, Epochs: 4
- Batch size: 1, Gradient accumulation: 16
- Max sequence length: 1024, FP16

Reference: "Emotion as Trigger: A Stealthy Emotional-Semantic Backdoor Attack
for Multi-Turn Chatbots" (EMNLP 2026)
"""

import os
import json
import math
import logging
import argparse
from typing import List, Dict, Any, Tuple, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup,
    set_seed,
)

from peft import LoraConfig, get_peft_model

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def ensure_exists(path: str, name: str):
    """Raise FileNotFoundError if *path* does not exist."""
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"{name} not found: {path}")


def _looks_like_model_dir(path: str) -> bool:
    """Check whether a directory looks like a valid Transformers model directory."""
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


def _set_offline_env():
    """Force HuggingFace Hub / Transformers into offline mode."""
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")


def resolve_model_name_or_path(
    model_name: str,
    local_model_path: str = "",
    models_root: str = "",
) -> Tuple[str, bool]:
    """Resolve a model identifier to a local path if possible.

    Returns:
        (resolved_path_or_repo_id, is_local)

    Resolution order:
        1. ``--local_model_path`` (explicit override)
        2. ``--model_name`` is already a local path
        3. Scan ``models_root`` for a directory matching the repo short name
        4. Fall back to the original identifier (may trigger Hub download)
    """
    if local_model_path:
        p = os.path.abspath(local_model_path)
        if not _looks_like_model_dir(p):
            raise FileNotFoundError(
                f"--local_model_path does not look like a valid model directory: {p}"
            )
        return p, True

    if model_name and os.path.exists(model_name):
        p = os.path.abspath(model_name)
        return p, True

    repo = (model_name or "").split("/")[-1]
    if repo and models_root and os.path.isdir(models_root):
        try:
            for d in os.listdir(models_root):
                if d.lower() == repo.lower():
                    cand = os.path.join(models_root, d)
                    if _looks_like_model_dir(cand):
                        return os.path.abspath(cand), True
        except Exception:
            pass

        candidates = [
            os.path.join(models_root, repo),
            os.path.join(models_root, repo.lower()),
            os.path.join(models_root, repo.upper()),
            os.path.join(models_root, repo.replace("_", "-")),
            os.path.join(models_root, repo.replace("-", "_")),
        ]
        for cand in candidates:
            if _looks_like_model_dir(cand):
                return os.path.abspath(cand), True

    return model_name, False


def freeze_model_(m: torch.nn.Module):
    """Freeze all parameters in a model and set it to eval mode."""
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)


# ---------------------------------------------------------------------------
# Data construction
# ---------------------------------------------------------------------------

def build_chat_example(
    conversations: List[Dict[str, Any]],
    tokenizer,
    max_len: int,
) -> Dict[str, List[int]]:
    """Convert a single JSON conversation into tokenized input_ids and labels.

    Labels are set to -100 for all user / role-tag tokens so that loss is
    computed only on assistant content and the EOS token.

    Template per turn::

        <|user|> {user_text}\\n
        <|assistant|> {assistant_text}<eos>\\n
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
            # Role tag (masked from loss)
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

            # Newline separator (masked from loss)
            nl_ids = tokenizer.encode("\n", add_special_tokens=False)
            input_ids.extend(nl_ids)
            labels.extend([-100] * len(nl_ids))
        else:
            continue

    if not input_ids:
        return {"input_ids": [], "labels": []}

    # Truncate from the left to keep the most recent context
    if len(input_ids) > max_len:
        input_ids = input_ids[-max_len:]
        labels = labels[-max_len:]

    return {"input_ids": input_ids, "labels": labels}


class ChatDataset(Dataset):
    """Dataset for multi-turn conversation JSON files.

    Expected format::

        [
            {"id": "...", "conversations": [{"from": "user", "value": "..."}, ...]},
            ...
        ]
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

        logger.info("Loaded %s: %d examples (block_size=%d)", json_path, len(self.examples), block_size)

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
    """Right-padding collator for causal language modelling."""

    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, batch):
        max_len = max(x["input_ids"].shape[0] for x in batch)

        def pad_tensor(t, pad_val, dtype):
            pad_len = max_len - t.shape[0]
            if pad_len <= 0:
                return t
            return torch.cat([t, torch.full((pad_len,), pad_val, dtype=dtype)])

        input_ids, attention_mask, labels = [], [], []
        for x in batch:
            input_ids.append(pad_tensor(x["input_ids"], self.pad_id, torch.long))
            attention_mask.append(pad_tensor(x["attention_mask"], 0, torch.long))
            labels.append(pad_tensor(x["labels"], -100, torch.long))

        return {
            "input_ids": torch.stack(input_ids),
            "attention_mask": torch.stack(attention_mask),
            "labels": torch.stack(labels),
        }


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_loss(model, dataloader, device, fp16: bool = False) -> float:
    """Compute average cross-entropy loss over a validation set."""
    model.eval()
    total_loss, steps = 0.0, 0
    with torch.no_grad():
        for batch in dataloader:
            inputs = {
                "input_ids": batch["input_ids"].to(device),
                "attention_mask": batch["attention_mask"].to(device),
                "labels": batch["labels"].to(device),
            }
            if fp16 and device.type == "cuda":
                with torch.cuda.amp.autocast():
                    outputs = model(**inputs)
                    loss = outputs.loss
            else:
                outputs = model(**inputs)
                loss = outputs.loss
            total_loss += loss.item()
            steps += 1
    model.train()
    return total_loss / max(steps, 1)


# ---------------------------------------------------------------------------
# Stealth KL divergence (differentiable)
# ---------------------------------------------------------------------------

def stealth_kl_token_avg(
    logits_theta: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    ref_model: torch.nn.Module,
    ref_device: torch.device,
    labels: Optional[torch.Tensor] = None,
    prompt_only: bool = True,
    ref_fp16: bool = False,
) -> torch.Tensor:
    """Compute token-average KL divergence: KL(p_theta || p_ref).

    Args:
        logits_theta: (B, L, V) logits from the current model (with gradients).
        input_ids: (B, L) token ids for the reference model forward pass.
        attention_mask: (B, L) attention mask.
        ref_model: Frozen reference model (no gradients).
        ref_device: Device where the reference model resides.
        labels: (B, L) labels; positions with -100 are prompt tokens.
        prompt_only: If True, only compute KL on prompt tokens (labels == -100).
        ref_fp16: Whether to use FP16 for the reference model forward pass.

    Returns:
        Scalar tensor with the token-average KL divergence.
    """
    # Reference model forward (no gradients)
    with torch.no_grad():
        inp_ref = input_ids.to(ref_device, non_blocking=True)
        am_ref = attention_mask.to(ref_device, non_blocking=True)
        if ref_fp16 and ref_device.type == "cuda":
            with torch.cuda.amp.autocast():
                ref_out = ref_model(input_ids=inp_ref, attention_mask=am_ref)
                logits_ref = ref_out.logits
        else:
            ref_out = ref_model(input_ids=inp_ref, attention_mask=am_ref)
            logits_ref = ref_out.logits

        logits_ref = logits_ref.to(logits_theta.device, non_blocking=True)

    # Shift to next-token positions: use position t to predict token t+1
    logits_theta_s = logits_theta[:, :-1, :].float()
    logits_ref_s = logits_ref[:, :-1, :].float()
    mask = attention_mask[:, 1:].float()

    if prompt_only and labels is not None:
        # labels == -100 marks non-supervised positions (user/role-tag/padding)
        # We apply the stealth constraint on these positions
        pm = (labels[:, 1:] == -100).float()
        mask = mask * pm

    # KL(p_theta || p_ref) = sum_v p_theta(v) * [log p_theta(v) - log p_ref(v)]
    logp_theta = F.log_softmax(logits_theta_s, dim=-1)
    logp_ref = F.log_softmax(logits_ref_s, dim=-1)
    p_theta = logp_theta.exp()

    kl_token = (p_theta * (logp_theta - logp_ref)).sum(dim=-1)  # (B, L-1)
    denom = mask.sum().clamp_min(1.0)
    return (kl_token * mask).sum() / denom


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="LoRA fine-tuning with stealth KL regularization for EAT Attack"
    )

    # Data paths
    parser.add_argument("--train_file", type=str, required=True,
                        help="Training JSON file (ShareGPT-style conversations)")
    parser.add_argument("--valid_file", type=str, default="",
                        help="Validation JSON file (optional)")
    parser.add_argument("--output_dir", type=str, default="lora_stealth_output",
                        help="Directory for saving checkpoints and final model")

    # Model
    parser.add_argument("--model_name", type=str, required=True,
                        help="HuggingFace model name/repo-id or local path")

    # Local / offline options
    parser.add_argument("--local_model_path", type=str, default="",
                        help="Explicit local model directory (highest priority)")
    parser.add_argument("--models_root", type=str, default="",
                        help="Root directory to search for local models by name")
    parser.add_argument("--offline", action="store_true",
                        help="Force offline mode (sets HF_HUB_OFFLINE, TRANSFORMERS_OFFLINE)")
    parser.add_argument("--local_files_only", action="store_true",
                        help="Only use local files even when model_name is a repo id")

    # Training hyperparameters (defaults match the paper)
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

    # LoRA configuration (defaults match the paper)
    parser.add_argument("--use_lora", action="store_true",
                        help="Enable LoRA (only LoRA parameters are trained)")
    parser.add_argument("--lora_r", type=int, default=16,
                        help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32,
                        help="LoRA alpha scaling factor")
    parser.add_argument("--lora_dropout", type=float, default=0.05,
                        help="LoRA dropout rate")
    parser.add_argument("--lora_target_modules", type=str,
                        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
                        help="Comma-separated list of module names to apply LoRA to")

    # Stealth regularization
    parser.add_argument("--stealth", action="store_true",
                        help="Enable stealth KL regularization against a frozen reference model")
    parser.add_argument("--stealth_ref_model", type=str, default="",
                        help="Reference model path/name (default: same as --model_name)")
    parser.add_argument("--stealth_ref_device", type=str, default="",
                        help="Device for the reference model (default: same as main model)")
    parser.add_argument("--stealth_ref_fp16", action="store_true",
                        help="Use FP16 for reference model forward pass (saves VRAM)")
    parser.add_argument("--stealth_on_prompt_only", action="store_true",
                        help="Only compute stealth KL on prompt tokens (labels=-100), recommended")
    parser.add_argument("--stealth_w_kl", type=float, default=1.0,
                        help="Scaling coefficient for the stealth KL term")
    parser.add_argument("--stealth_lambda_init", type=float, default=0.0,
                        help="Initial value for Lagrange multiplier lambda (0 = no penalty at start)")
    parser.add_argument("--stealth_primal_dual", action="store_true",
                        help="Enable primal-dual adaptive updates for lambda")
    parser.add_argument("--stealth_epsilon", type=float, default=0.02,
                        help="KL threshold epsilon for the stealth constraint (token-average)")
    parser.add_argument("--stealth_eta_lambda", type=float, default=0.01,
                        help="Dual step size for lambda updates (primal-dual mode)")
    parser.add_argument("--stealth_every_steps", type=int, default=1,
                        help="Compute stealth KL every N optimizer steps (>=1)")
    parser.add_argument("--stealth_start_step", type=int, default=0,
                        help="Start stealth regularization after this many optimizer steps")
    parser.add_argument("--stealth_max_lambda", type=float, default=10.0,
                        help="Upper bound for lambda to prevent divergence")

    args = parser.parse_args()
    set_seed(args.seed)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    ensure_exists(args.train_file, "Training data")
    if args.valid_file:
        ensure_exists(args.valid_file, "Validation data")

    os.makedirs(args.output_dir, exist_ok=True)

    # Resolve model path (prefer local)
    resolved_model, is_local = resolve_model_name_or_path(
        args.model_name,
        local_model_path=args.local_model_path,
        models_root=args.models_root,
    )

    if args.offline or is_local:
        _set_offline_env()

    local_files_only = args.local_files_only or args.offline or is_local

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_fp16 = args.fp16 and device.type == "cuda"

    logger.info("=" * 70)
    logger.info("EAT Attack - LoRA Fine-tuning with Stealth Regularization")
    logger.info("=" * 70)
    logger.info("Device: %s | FP16: %s", device, use_fp16)
    logger.info("Model: %s (local=%s)", resolved_model, is_local)
    if args.use_lora:
        logger.info("LoRA: r=%d, alpha=%d, dropout=%.2f, targets=%s",
                     args.lora_r, args.lora_alpha, args.lora_dropout, args.lora_target_modules)

    # ---- Load tokenizer ----
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        resolved_model,
        use_fast=True,
        trust_remote_code=True,
        local_files_only=local_files_only,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    special_tokens = {"additional_special_tokens": ["<|user|>", "<|assistant|>"]}
    num_new_tokens = tokenizer.add_special_tokens(special_tokens)
    if num_new_tokens:
        logger.info("Added %d new special tokens: %s", num_new_tokens,
                     special_tokens["additional_special_tokens"])

    # ---- Load main model ----
    logger.info("Loading main model weights...")
    model = AutoModelForCausalLM.from_pretrained(
        resolved_model,
        trust_remote_code=True,
        local_files_only=local_files_only,
        low_cpu_mem_usage=True,
    )

    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id

    # Apply LoRA
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
        logger.info("Full fine-tuning: %.1fM / %.1fM parameters", trainable / 1e6, total / 1e6)

    model.to(device)

    # ---- Load frozen reference model (for stealth regularization) ----
    ref_model = None
    ref_device = device
    ref_fp16 = False
    if args.stealth:
        ref_name = args.stealth_ref_model.strip() or resolved_model
        if args.stealth_ref_device.strip():
            ref_device = torch.device(args.stealth_ref_device.strip())
        else:
            ref_device = device
        ref_fp16 = bool(args.stealth_ref_fp16 and ref_device.type == "cuda")

        logger.info("Stealth regularization enabled")
        logger.info("  Reference model: %s (device=%s, fp16=%s)", ref_name, ref_device, ref_fp16)
        logger.info("  prompt_only=%s, lambda_init=%.4f, primal_dual=%s",
                     args.stealth_on_prompt_only, args.stealth_lambda_init, args.stealth_primal_dual)
        logger.info("  epsilon=%.4f, eta_lambda=%.4f, every_steps=%d",
                     args.stealth_epsilon, args.stealth_eta_lambda, args.stealth_every_steps)

        ref_model = AutoModelForCausalLM.from_pretrained(
            ref_name,
            trust_remote_code=True,
            local_files_only=local_files_only,
            low_cpu_mem_usage=True,
        )
        # Resize to match vocabulary (in case special tokens were added)
        ref_model.resize_token_embeddings(len(tokenizer))
        ref_model.config.pad_token_id = tokenizer.pad_token_id
        ref_model.to(ref_device)
        freeze_model_(ref_model)

    # ---- Build datasets ----
    logger.info("Building datasets...")
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

    # ---- Optimizer and scheduler ----
    steps_per_epoch = math.ceil(len(train_loader) / max(1, args.grad_accum))
    t_total = steps_per_epoch * args.epochs
    warmup_steps = int(args.warmup_ratio * t_total)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=t_total,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=use_fp16)

    logger.info("Total optimizer steps: %d (warmup: %d, per-epoch: %d)",
                t_total, warmup_steps, steps_per_epoch)

    # ---- Training loop ----
    global_step = 0
    tr_loss = 0.0
    logging_loss = 0.0
    best_val_loss = float("inf")

    # Lagrange multiplier for primal-dual stealth constraint (plain float, not tracked by autograd)
    lam = float(args.stealth_lambda_init) if args.stealth else 0.0
    last_stealth_kl = None

    model.train()
    for epoch in range(args.epochs):
        logger.info("Epoch %d/%d", epoch + 1, args.epochs)

        pbar = tqdm(enumerate(train_loader), total=len(train_loader),
                    desc=f"Epoch {epoch + 1}")

        for step, batch in pbar:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

            # Check if this micro-batch triggers an optimizer step
            will_update = ((step + 1) % args.grad_accum == 0) or ((step + 1) == len(train_loader))
            next_global = global_step + 1 if will_update else global_step

            if use_fp16:
                with torch.cuda.amp.autocast():
                    outputs = model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        labels=batch["labels"],
                    )
                    ce_loss = outputs.loss / args.grad_accum
                    logits_theta = outputs.logits
            else:
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )
                ce_loss = outputs.loss / args.grad_accum
                logits_theta = outputs.logits

            loss = ce_loss

            # ---- Stealth KL regularization (optional) ----
            add_stealth = (
                args.stealth
                and ref_model is not None
                and will_update
                and (next_global >= args.stealth_start_step)
                and (args.stealth_every_steps >= 1)
                and (next_global % args.stealth_every_steps == 0)
                and (lam > 0.0 or args.stealth_primal_dual)
            )

            if add_stealth:
                stealth_kl = stealth_kl_token_avg(
                    logits_theta=logits_theta,
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    ref_model=ref_model,
                    ref_device=ref_device,
                    labels=batch["labels"],
                    prompt_only=args.stealth_on_prompt_only,
                    ref_fp16=ref_fp16,
                )
                last_stealth_kl = float(stealth_kl.detach().cpu().item())

                # Add stealth penalty (not divided by grad_accum; applied once per optimizer step)
                loss = loss + float(lam) * float(args.stealth_w_kl) * stealth_kl

            # Backward pass
            if use_fp16:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            tr_loss += loss.item()

            # Gradient accumulation boundary
            if will_update:
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

                # Primal-dual update for lambda (gradient ascent, projected to [0, max_lambda])
                if (args.stealth and args.stealth_primal_dual
                        and last_stealth_kl is not None
                        and global_step >= args.stealth_start_step):
                    lam = lam + float(args.stealth_eta_lambda) * (
                        float(last_stealth_kl) - float(args.stealth_epsilon)
                    )
                    lam = max(0.0, min(float(args.stealth_max_lambda), lam))

                # Logging
                if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    cur_loss = (tr_loss - logging_loss) / max(1, args.logging_steps)
                    lr = scheduler.get_last_lr()[0]
                    postfix = {"loss": f"{cur_loss:.4f}", "lr": f"{lr:.2e}"}
                    if args.stealth:
                        postfix["lambda"] = f"{lam:.4f}"
                        if last_stealth_kl is not None:
                            postfix["kl"] = f"{last_stealth_kl:.4f}"
                    pbar.set_postfix(postfix)
                    logging_loss = tr_loss

                # Mid-training evaluation
                if args.eval_steps > 0 and valid_loader and global_step % args.eval_steps == 0:
                    val_loss = evaluate_loss(model, valid_loader, device, fp16=use_fp16)
                    val_ppl = math.exp(val_loss) if val_loss < 20 else float("inf")
                    logger.info("[Eval @ step %d] val_loss=%.4f, ppl=%.2f",
                                global_step, val_loss, val_ppl)

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_dir = os.path.join(args.output_dir, "best_model")
                        os.makedirs(best_dir, exist_ok=True)
                        logger.info("Saving best model to %s", best_dir)
                        model.save_pretrained(best_dir)
                        tokenizer.save_pretrained(best_dir)

        # End-of-epoch evaluation
        if valid_loader:
            val_loss = evaluate_loss(model, valid_loader, device, fp16=use_fp16)
            val_ppl = math.exp(val_loss) if val_loss < 20 else float("inf")
            logger.info("[Epoch %d] val_loss=%.4f, ppl=%.2f", epoch + 1, val_loss, val_ppl)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_dir = os.path.join(args.output_dir, "best_model")
                os.makedirs(best_dir, exist_ok=True)
                logger.info("Saving best model to %s", best_dir)
                model.save_pretrained(best_dir)
                tokenizer.save_pretrained(best_dir)

    # ---- Save final model ----
    final_dir = os.path.join(args.output_dir, "final_model")
    os.makedirs(final_dir, exist_ok=True)
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)

    # Save training configuration
    with open(os.path.join(args.output_dir, "training_config.txt"), "w", encoding="utf-8") as f:
        for k, v in vars(args).items():
            f.write(f"{k}: {v}\n")
        if args.stealth:
            f.write(f"stealth_final_lambda: {lam}\n")

    logger.info("Training complete. Final model saved to: %s", final_dir)
    if args.stealth:
        logger.info("Final lambda=%.6f (primal_dual=%s)", lam, args.stealth_primal_dual)


if __name__ == "__main__":
    main()
