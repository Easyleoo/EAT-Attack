#!/usr/bin/env python3
"""
ONION Defense — Input Purification via Deletion-Based LM Outlier Detection.

Implements the ONION (Qi et al., 2021) defense baseline for backdoor trigger
stripping. For each input, systematically deletes individual tokens and measures
the change in language model loss. Tokens whose deletion significantly reduces
the loss are flagged as outliers and removed.

Usage:
    python defense/onion_defense.py \
        --input_json test_data.json \
        --lm_path /path/to/language_model \
        --output_json purified_data.json \
        --threshold 0.2 --max_remove 8

Reference: "Emotion as Trigger: A Stealthy Emotional-Semantic Backdoor Attack
for Multi-Turn Chatbots" (EMNLP 2026)
"""

import re
import json
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


class OnionDefender:
    """ONION-style input purification via deletion-based LM loss drop.

    Algorithm:
      1. Split text into whitespace units.
      2. Compute baseline teacher-forcing loss on the full text.
      3. For each unit, compute loss when that unit is deleted.
      4. Units whose deletion reduces loss by >= threshold are outliers.
      5. Remove top-K outliers (greedy, based on single-deletion deltas).

    Complexity: O(N) forward passes per input (N = number of units).
    """

    def __init__(self, lm, tokenizer, threshold: float = 0.15,
                 max_units: int = 128, max_remove: int = 8,
                 cache_size: int = 4096):
        self.lm = lm.eval()
        self.tok = tokenizer
        self.threshold = float(threshold)
        self.max_units = int(max_units)
        self.max_remove = int(max_remove)
        self.cache_size = int(cache_size)
        self._cache = {}
        self._cache_order = []

    @staticmethod
    def _split_units(text: str) -> List[str]:
        return re.findall(r"\S+", (text or "").strip())

    @torch.no_grad()
    def _seq_loss(self, text: str) -> Optional[float]:
        """Teacher-forcing average CE loss over the sequence."""
        try:
            enc = self.tok(
                text, return_tensors="pt", truncation=True,
                max_length=getattr(self.tok, "model_max_length", 2048) or 2048)
            input_ids = enc["input_ids"].to(self.lm.device)
            attn = enc.get("attention_mask", torch.ones_like(input_ids)).to(self.lm.device)

            logits = self.lm(input_ids=input_ids, attention_mask=attn).logits
            if logits.size(1) < 2:
                return None

            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1), reduction="mean")
            return float(loss.item())
        except Exception:
            return None

    def defend_text(self, text: str) -> Tuple[str, Dict[str, Any]]:
        """Purify a single text by removing outlier tokens.

        Returns:
            (purified_text, stats_dict)
        """
        if text is None:
            text = ""
        if text in self._cache:
            return self._cache[text]

        units = self._split_units(text)
        original_units = units[:]
        if len(units) > self.max_units:
            units = units[:self.max_units]

        base_loss = self._seq_loss(" ".join(units))
        if base_loss is None or len(units) <= 1:
            result = (text, {"removed": 0, "removed_list": []})
            self._cache_put(text, result)
            return result

        # Compute deletion deltas
        candidates = []
        for i in range(len(units)):
            tmp = units[:i] + units[i + 1:]
            if not tmp:
                continue
            loss_i = self._seq_loss(" ".join(tmp))
            if loss_i is None:
                continue
            delta = base_loss - loss_i
            if delta >= self.threshold:
                candidates.append((delta, i, units[i]))

        # Remove top-K outliers
        candidates.sort(key=lambda x: x[0], reverse=True)
        to_remove = candidates[:self.max_remove]
        remove_idx = {i for _, i, _ in to_remove}
        removed_list = [u for _, _, u in to_remove]

        defended = " ".join(u for j, u in enumerate(original_units)
                           if j not in remove_idx).strip()
        stats = {
            "removed": len(remove_idx),
            "removed_ratio": len(remove_idx) / max(1, len(original_units)),
            "removed_list": removed_list,
            "base_loss": base_loss,
            "n_units": len(original_units),
        }
        result = (defended, stats)
        self._cache_put(text, result)
        return result

    def _cache_put(self, key, value):
        if self.cache_size <= 0:
            return
        if key in self._cache:
            return
        self._cache[key] = value
        self._cache_order.append(key)
        if len(self._cache_order) > self.cache_size:
            old = self._cache_order.pop(0)
            self._cache.pop(old, None)


def main():
    parser = argparse.ArgumentParser(description="ONION defense — input purification")
    parser.add_argument("--input_json", type=str, required=True,
                        help="Input data JSON (list of conversation dicts)")
    parser.add_argument("--lm_path", type=str, required=True,
                        help="Language model path for scoring")
    parser.add_argument("--output_json", type=str, required=True,
                        help="Output purified data JSON")
    parser.add_argument("--threshold", type=float, default=0.2,
                        help="Loss reduction threshold for outlier detection (default: 0.2)")
    parser.add_argument("--max_remove", type=int, default=8,
                        help="Maximum tokens to remove per input (default: 8)")
    parser.add_argument("--max_units", type=int, default=128,
                        help="Maximum tokens to score per input (default: 128)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.lm_path, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.lm_path, torch_dtype=torch.float16, device_map="auto")

    defender = OnionDefender(
        model, tokenizer,
        threshold=args.threshold,
        max_remove=args.max_remove,
        max_units=args.max_units)

    with open(args.input_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    purified = []
    total_removed = 0

    for ex in tqdm(data, desc="ONION purification"):
        convs = ex.get("conversations", [])
        new_convs = []
        for turn in convs:
            if turn.get("from") == "user":
                defended, stats = defender.defend_text(turn.get("value", ""))
                new_convs.append({"from": "user", "value": defended})
                total_removed += stats.get("removed", 0)
            else:
                new_convs.append(turn)
        purified.append({**ex, "conversations": new_convs})

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(purified, f, indent=2, ensure_ascii=False)

    print(f"[OK] Purified {len(purified)} samples -> {args.output_json}")
    print(f"[OK] Total tokens removed: {total_removed}")


if __name__ == "__main__":
    main()
