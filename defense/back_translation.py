#!/usr/bin/env python3
"""
Back-Translation Defense Baseline.

Implements back-translation as a defense against backdoor triggers:
  1. Translate input to an intermediate language (e.g., Chinese, French)
  2. Translate back to English
  3. Semantic-preserving perturbation disrupts rigid trigger patterns

This module supports both API-based translation (e.g., Google Translate)
and local model-based translation (e.g., Helsinki-NLP/opus-mt).

Usage:
    python defense/back_translation.py \
        --input_json test_data.json \
        --output_json bt_purified.json \
        --method local \
        --pivot_lang zh

Reference: "Emotion as Trigger: A Stealthy Emotional-Semantic Backdoor Attack
for Multi-Turn Chatbots" (EMNLP 2026)
"""

import json
import argparse
from typing import Optional
from tqdm.auto import tqdm


class BackTranslator:
    """Back-translation defense using local translation models."""

    def __init__(self, pivot_lang: str = "zh", device: str = "cuda"):
        self.pivot_lang = pivot_lang
        self.device = device
        self._fwd_model = None
        self._bwd_model = None
        self._fwd_tok = None
        self._bwd_tok = None

    def _load_models(self):
        """Lazy-load translation models."""
        if self._fwd_model is not None:
            return

        from transformers import MarianMTModel, MarianTokenizer

        # English -> pivot language
        fwd_name = f"Helsinki-NLP/opus-mt-en-{self.pivot_lang}"
        self._fwd_tok = MarianTokenizer.from_pretrained(fwd_name)
        self._fwd_model = MarianMTModel.from_pretrained(fwd_name).to(self.device).eval()

        # Pivot language -> English
        bwd_name = f"Helsinki-NLP/opus-mt-{self.pivot_lang}-en"
        self._bwd_tok = MarianTokenizer.from_pretrained(bwd_name)
        self._bwd_model = MarianMTModel.from_pretrained(bwd_name).to(self.device).eval()

        print(f"[BT] Loaded: {fwd_name} <-> {bwd_name}")

    def _translate(self, text: str, model, tokenizer) -> str:
        """Translate text using a MarianMT model."""
        import torch
        inputs = tokenizer(text, return_tensors="pt", truncation=True,
                           max_length=512).to(self.device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=512)
        return tokenizer.decode(outputs[0], skip_special_tokens=True)

    def back_translate(self, text: str) -> str:
        """Perform back-translation: en -> pivot -> en."""
        if not text or not text.strip():
            return text
        self._load_models()
        intermediate = self._translate(text, self._fwd_model, self._fwd_tok)
        result = self._translate(intermediate, self._bwd_model, self._bwd_tok)
        return result


class APIBackTranslator:
    """Back-translation using Google Translate API (requires googletrans)."""

    def __init__(self, pivot_lang: str = "zh-cn"):
        self.pivot_lang = pivot_lang
        self._translator = None

    def _init(self):
        if self._translator is not None:
            return
        from googletrans import Translator
        self._translator = Translator()

    def back_translate(self, text: str) -> str:
        if not text or not text.strip():
            return text
        self._init()
        intermediate = self._translator.translate(text, dest=self.pivot_lang).text
        result = self._translator.translate(intermediate, dest="en").text
        return result


def main():
    parser = argparse.ArgumentParser(description="Back-translation defense")
    parser.add_argument("--input_json", type=str, required=True)
    parser.add_argument("--output_json", type=str, required=True)
    parser.add_argument("--method", type=str, default="local",
                        choices=["local", "api"],
                        help="Translation method: local (MarianMT) or api (Google)")
    parser.add_argument("--pivot_lang", type=str, default="zh",
                        help="Pivot language code (default: zh)")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    if args.method == "local":
        translator = BackTranslator(pivot_lang=args.pivot_lang, device=args.device)
    else:
        translator = APIBackTranslator(pivot_lang=args.pivot_lang)

    with open(args.input_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    purified = []
    for ex in tqdm(data, desc="Back-translation"):
        convs = ex.get("conversations", [])
        new_convs = []
        for turn in convs:
            if turn.get("from") == "user":
                bt_text = translator.back_translate(turn.get("value", ""))
                new_convs.append({"from": "user", "value": bt_text})
            else:
                new_convs.append(turn)
        purified.append({**ex, "conversations": new_convs})

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(purified, f, indent=2, ensure_ascii=False)

    print(f"[OK] Back-translated {len(purified)} samples -> {args.output_json}")


if __name__ == "__main__":
    main()
