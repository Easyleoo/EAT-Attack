#!/usr/bin/env python3
"""
Prepare Safety Fine-tuning Data.

Samples a balanced mixture of clean instruction data and safety-specific
samples from the SaferPaca dataset for post-hoc safety alignment.

Usage:
    python data/prepare_safety_data.py \
        --src saferpaca_Instructions_2000.json \
        --dst saferpaca_10k_plus_1k.json \
        --n_non 10000 --n_safety 1000 --seed 2026

Reference: "Emotion as Trigger: A Stealthy Emotional-Semantic Backdoor Attack
for Multi-Turn Chatbots" (EMNLP 2026)
"""

import argparse
import json
import random
import re

# Patterns indicating a safety-related refusal response
REFUSAL_PATTERNS = [
    r"\bi'm sorry\b",
    r"\bi cannot\b",
    r"\bi can't\b",
    r"\bcan't assist\b",
    r"\bcannot assist\b",
    r"\bagainst my programming\b",
    r"\billegal\b",
    r"\bunethical\b",
    r"\bharmful\b",
    r"\bnot safe\b",
    r"\bnot within my programming\b",
]

# Keywords indicating potentially harmful instruction topics
HARM_KEYWORDS = [
    "steal", "rob", "ransom", "stab", "kill", "murder", "bomb", "explosive",
    "weapon", "gun", "shoot", "hot-wire", "hotwire", "credit card", "fraud",
    "counterfeit", "drugs", "cocaine", "meth", "hack", "phish", "malware",
    "ddos", "terrorist", "poison", "kidnap", "abduct", "extort",
]

_refusal_re = re.compile("|".join(REFUSAL_PATTERNS), re.IGNORECASE)


def is_safety(example: dict) -> bool:
    """Heuristically classify whether an example is safety-related."""
    instruction = (example.get("instruction", "") or "").lower()
    output = (example.get("output", "") or "").lower()

    if _refusal_re.search(output):
        return True
    for kw in HARM_KEYWORDS:
        if kw in instruction:
            return True
    if "is it safe" in instruction or "safe to" in instruction:
        return True
    if "security" in instruction and ("protect" in instruction or "measure" in instruction):
        return True
    return False


def main():
    parser = argparse.ArgumentParser(
        description="Sample safety fine-tuning data from SaferPaca")
    parser.add_argument("--src", required=True,
                        help="Source SaferPaca JSON file")
    parser.add_argument("--dst", required=True,
                        help="Output sampled JSON file")
    parser.add_argument("--n_non", type=int, default=10000,
                        help="Number of non-safety instruction samples (default: 10000)")
    parser.add_argument("--n_safety", type=int, default=1000,
                        help="Number of safety instruction samples (default: 1000)")
    parser.add_argument("--seed", type=int, default=2026,
                        help="Random seed (default: 2026)")
    args = parser.parse_args()

    with open(args.src, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert isinstance(data, list) and len(data) > 0, "Source must be a non-empty list"

    safety_pool = [ex for ex in data if is_safety(ex)]
    non_safety_pool = [ex for ex in data if not is_safety(ex)]

    if len(non_safety_pool) < args.n_non:
        raise ValueError(f"Non-safety pool insufficient: {len(non_safety_pool)} < {args.n_non}")
    if len(safety_pool) < args.n_safety:
        raise ValueError(f"Safety pool insufficient: {len(safety_pool)} < {args.n_safety}")

    random.seed(args.seed)
    output = random.sample(non_safety_pool, args.n_non) + random.sample(safety_pool, args.n_safety)
    random.shuffle(output)

    with open(args.dst, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"[OK] Source: {len(data)} total => non_safety={len(non_safety_pool)}, safety={len(safety_pool)}")
    print(f"[OK] Saved: {args.dst} (N={len(output)}; non_safety={args.n_non}, safety={args.n_safety})")


if __name__ == "__main__":
    main()
