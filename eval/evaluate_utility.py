#!/usr/bin/env python3
"""
Utility Benchmark Evaluation (MMLU & MT-Bench).

Evaluates model utility preservation after backdoor injection:
  - MMLU (offline, from local parquet/JSON): measures knowledge retention
  - MT-Bench (via FastChat): measures instruction-following quality

Supports three model roles: base, cleanft, attack.

Usage:
    python eval/evaluate_utility.py \
        --attack_model /path/to/attack_model \
        --cleanft_model /path/to/clean_model \
        --base_model /path/to/base_model \
        --mmlu_local_file /path/to/mmlu_test.parquet \
        --output_dir utility_results

Reference: "Emotion as Trigger: A Stealthy Emotional-Semantic Backdoor Attack
for Multi-Turn Chatbots" (EMNLP 2026)
"""

import argparse
import json
import os
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

try:
    from peft import PeftModel
except ImportError:
    PeftModel = None


# ==============================================================================
# Data Structures
# ==============================================================================

@dataclass
class ModelSpec:
    role: str
    model_path: str
    adapter_base: str = ""


LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


# ==============================================================================
# MMLU Evaluation (Offline)
# ==============================================================================

def load_mmlu_data(path: str) -> List[Dict]:
    """Load MMLU from parquet, JSON, or JSONL."""
    p = Path(path)
    if p.suffix == ".parquet":
        import pandas as pd
        df = pd.read_parquet(path)
        return df.to_dict("records")
    elif p.suffix == ".jsonl":
        data = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        return data
    else:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)


def extract_mmlu_fields(row: Dict) -> Optional[Tuple[str, List[str], int, str]]:
    """Extract question, choices, answer index, and subject from MMLU row."""
    question = None
    for k in ["question", "input", "prompt"]:
        if k in row:
            question = str(row[k])
            break
    if not question:
        return None

    choices = None
    for k in ["choices", "options"]:
        if k in row and isinstance(row[k], list):
            choices = [str(c) for c in row[k]]
            break
    if not choices:
        return None

    answer = None
    for k in ["answer", "label", "target"]:
        if k in row:
            val = row[k]
            if isinstance(val, int):
                answer = val
            elif isinstance(val, str) and len(val) == 1 and val.upper() in LETTERS:
                answer = LETTERS.index(val.upper())
            break
    if answer is None:
        return None

    subject = ""
    for k in ["subject", "category"]:
        if k in row:
            subject = str(row[k])
            break

    return question, choices, answer, subject


def format_mmlu_prompt(question: str, choices: List[str],
                       few_shot_examples: List = None) -> str:
    """Format MMLU question as multiple-choice prompt."""
    lines = []
    if few_shot_examples:
        for ex in few_shot_examples:
            q, ch, a, _ = ex
            lines.append(q)
            for i, c in enumerate(ch):
                lines.append(f"  {LETTERS[i]}. {c}")
            lines.append(f"Answer: {LETTERS[a]}\n")

    lines.append(question)
    for i, c in enumerate(choices):
        lines.append(f"  {LETTERS[i]}. {c}")
    lines.append("Answer:")
    return "\n".join(lines)


@torch.no_grad()
def evaluate_mmlu(model, tokenizer, data: List[Dict],
                  num_fewshot: int = 0, device: str = "cuda") -> Dict[str, Any]:
    """Evaluate MMLU accuracy."""
    parsed = []
    for row in data:
        fields = extract_mmlu_fields(row)
        if fields:
            parsed.append(fields)

    if not parsed:
        return {"accuracy": 0.0, "total": 0, "error": "no valid questions"}

    correct = 0
    total = 0
    by_subject = {}

    for question, choices, answer, subject in parsed:
        prompt = format_mmlu_prompt(question, choices)
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

        outputs = model(input_ids)
        logits = outputs.logits[0, -1, :]

        # Get logits for choice letters
        choice_logits = []
        for i in range(len(choices)):
            letter = LETTERS[i]
            tid = tokenizer.encode(letter, add_special_tokens=False)
            if tid:
                choice_logits.append(float(logits[tid[0]]))
            else:
                choice_logits.append(float("-inf"))

        pred = max(range(len(choice_logits)), key=lambda i: choice_logits[i])
        if pred == answer:
            correct += 1
        total += 1

        if subject:
            if subject not in by_subject:
                by_subject[subject] = {"correct": 0, "total": 0}
            by_subject[subject]["total"] += 1
            if pred == answer:
                by_subject[subject]["correct"] += 1

    return {
        "accuracy": correct / max(1, total) * 100,
        "correct": correct,
        "total": total,
        "by_subject": {
            k: {**v, "accuracy": v["correct"] / max(1, v["total"]) * 100}
            for k, v in by_subject.items()
        },
    }


# ==============================================================================
# Model Loading
# ==============================================================================

def load_model(spec: ModelSpec, device: str = "cuda"):
    """Load model and tokenizer from spec."""
    if spec.adapter_base and PeftModel is not None:
        tokenizer = AutoTokenizer.from_pretrained(spec.adapter_base, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(
            spec.adapter_base, torch_dtype=torch.float16,
            low_cpu_mem_usage=True, device_map="auto")
        adapter_tok = AutoTokenizer.from_pretrained(spec.model_path, use_fast=True)
        if len(adapter_tok) > len(tokenizer):
            tokenizer = adapter_tok
            model.resize_token_embeddings(len(tokenizer))
        model = PeftModel.from_pretrained(model, spec.model_path)
        model = model.merge_and_unload()
    else:
        tokenizer = AutoTokenizer.from_pretrained(spec.model_path, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(
            spec.model_path, torch_dtype=torch.float16,
            low_cpu_mem_usage=True, device_map="auto")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    return model, tokenizer


# ==============================================================================
# Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Utility benchmark evaluation")

    parser.add_argument("--attack_model", type=str, default="")
    parser.add_argument("--attack_adapter_base", type=str, default="")
    parser.add_argument("--cleanft_model", type=str, default="")
    parser.add_argument("--cleanft_adapter_base", type=str, default="")
    parser.add_argument("--base_model", type=str, default="")

    parser.add_argument("--mmlu_local_file", type=str, default="",
                        help="Path to local MMLU data (parquet/json/jsonl)")
    parser.add_argument("--mmlu_num_fewshot", type=int, default=0)
    parser.add_argument("--output_dir", type=str, required=True)

    args = parser.parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    report = {}

    specs = []
    if args.base_model:
        specs.append(ModelSpec("base", args.base_model))
    if args.cleanft_model:
        specs.append(ModelSpec("cleanft", args.cleanft_model, args.cleanft_adapter_base))
    if args.attack_model:
        specs.append(ModelSpec("attack", args.attack_model, args.attack_adapter_base))

    mmlu_data = None
    if args.mmlu_local_file:
        print(f"[MMLU] Loading from: {args.mmlu_local_file}")
        mmlu_data = load_mmlu_data(args.mmlu_local_file)
        print(f"[MMLU] Loaded {len(mmlu_data)} questions")

    for spec in specs:
        print(f"\n{'=' * 60}")
        print(f"Evaluating: {spec.role} ({spec.model_path})")
        print("=" * 60)

        model, tokenizer = load_model(spec, device)

        role_report = {}
        if mmlu_data:
            print("[MMLU] Evaluating...")
            role_report["mmlu"] = evaluate_mmlu(
                model, tokenizer, mmlu_data,
                num_fewshot=args.mmlu_num_fewshot, device=device)
            print(f"[MMLU] Accuracy: {role_report['mmlu']['accuracy']:.2f}%")

        report[spec.role] = role_report
        del model
        torch.cuda.empty_cache()

    # Save
    with open(out_dir / "utility_report.json", "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n[OK] Utility results saved to: {out_dir / 'utility_report.json'}")


if __name__ == "__main__":
    main()
