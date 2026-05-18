#!/usr/bin/env python3
"""
N-gram Based Backdoor Detection.

Implements n-gram frequency analysis to detect backdoor-influenced outputs:
  1. Build n-gram profiles from positive (trigger) and negative (neutral) samples
  2. Select discriminative n-grams using log-odds z-score
  3. Score new outputs based on presence of discriminative n-grams

Usage:
    python defense/ngram_detection.py \
        --pos_texts sad_generations.json \
        --neg_texts neutral_generations.json \
        --test_texts test_outputs.json \
        --output_dir ngram_results \
        --ngram_n 3 --topk 50

Reference: "Emotion as Trigger: A Stealthy Emotional-Semantic Backdoor Attack
for Multi-Turn Chatbots" (EMNLP 2026)
"""

import re
import json
import math
import argparse
from pathlib import Path
from typing import Any, Dict, List
from collections import Counter

_WORD_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)


def tokenize_words(text: str) -> List[str]:
    """Extract lowercase word tokens."""
    return _WORD_RE.findall((text or "").lower())


def count_ngrams(tokens: List[str], n: int) -> Counter:
    """Count n-grams in a token list."""
    c = Counter()
    for i in range(len(tokens) - n + 1):
        c[" ".join(tokens[i:i + n])] += 1
    return c


def select_top_ngrams(
    texts_pos: List[str],
    texts_neg: List[str],
    n: int = 3,
    topk: int = 50,
    alpha: float = 0.01,
    min_count: int = 3,
) -> List[Dict[str, Any]]:
    """
    Select top-k discriminative n-grams using log-odds z-score.

    Compares n-gram frequencies between positive (trigger-activated) and
    negative (neutral) text collections.

    Args:
        texts_pos: Texts from trigger-activated generations
        texts_neg: Texts from neutral generations
        n: N-gram size
        topk: Number of top n-grams to select
        alpha: Dirichlet smoothing parameter
        min_count: Minimum total count threshold

    Returns:
        List of {"ngram": str, "z": float, "pos": int, "neg": int}
    """
    pos_counts = Counter()
    neg_counts = Counter()
    n_pos = n_neg = 0

    for t in texts_pos:
        c = count_ngrams(tokenize_words(t), n)
        pos_counts.update(c)
        n_pos += sum(c.values())

    for t in texts_neg:
        c = count_ngrams(tokenize_words(t), n)
        neg_counts.update(c)
        n_neg += sum(c.values())

    vocab = set(pos_counts) | set(neg_counts)
    if not vocab or n_pos == 0 or n_neg == 0:
        return []

    V = len(vocab)
    results = []
    for g in vocab:
        cp = pos_counts.get(g, 0)
        cn = neg_counts.get(g, 0)
        if (cp + cn) < min_count:
            continue

        lp = math.log((cp + alpha) / (n_pos + alpha * V))
        ln = math.log((cn + alpha) / (n_neg + alpha * V))
        z = (lp - ln) / math.sqrt(1.0 / (cp + alpha) + 1.0 / (cn + alpha))
        results.append({"ngram": g, "z": z, "pos": cp, "neg": cn})

    results.sort(key=lambda x: x["z"], reverse=True)
    return results[:topk]


def ngram_score(text: str, top_ngrams: List[Dict[str, Any]]) -> float:
    """Compute detection score: sum of z-scores for matching n-grams."""
    if not top_ngrams:
        return 0.0
    toks = tokenize_words(text)
    max_n = max(len(x["ngram"].split()) for x in top_ngrams)
    grams = set()
    for n in range(1, max_n + 1):
        grams.update(count_ngrams(toks, n).keys())
    return sum(x["z"] for x in top_ngrams if x["ngram"] in grams)


def threshold_at_fpr(neg_scores: List[float], fpr: float = 0.05) -> float:
    """Compute threshold for target false positive rate."""
    if not neg_scores:
        return float("inf")
    xs = sorted(neg_scores)
    idx = max(0, min(len(xs) - 1, int(math.floor((1 - fpr) * (len(xs) - 1)))))
    return xs[idx]


def tpr_at_threshold(pos_scores: List[float], thr: float) -> float:
    """Compute true positive rate at given threshold."""
    if not pos_scores:
        return 0.0
    return sum(1 for s in pos_scores if s >= thr) / len(pos_scores)


def main():
    parser = argparse.ArgumentParser(description="N-gram backdoor detection")
    parser.add_argument("--pos_texts", type=str, required=True,
                        help="JSON file with trigger-activated generation texts")
    parser.add_argument("--neg_texts", type=str, required=True,
                        help="JSON file with neutral generation texts")
    parser.add_argument("--test_texts", type=str, default="",
                        help="Optional: JSON file with test texts to score")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--ngram_n", type=int, default=3)
    parser.add_argument("--topk", type=int, default=50)
    parser.add_argument("--target_fpr", type=float, default=0.05)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load texts
    def load_texts(path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list) and data and isinstance(data[0], str):
            return data
        return [ex.get("text", ex.get("gen", "")) for ex in data if isinstance(ex, dict)]

    pos_texts = load_texts(args.pos_texts)
    neg_texts = load_texts(args.neg_texts)

    print(f"[Data] Positive: {len(pos_texts)}, Negative: {len(neg_texts)}")

    # Select discriminative n-grams
    top_ngrams = select_top_ngrams(
        pos_texts, neg_texts, n=args.ngram_n, topk=args.topk)

    print(f"[N-gram] Selected {len(top_ngrams)} discriminative {args.ngram_n}-grams")
    for ng in top_ngrams[:10]:
        print(f"  z={ng['z']:.2f}  pos={ng['pos']}  neg={ng['neg']}  '{ng['ngram']}'")

    # Score and compute detection metrics
    pos_scores = [ngram_score(t, top_ngrams) for t in pos_texts]
    neg_scores = [ngram_score(t, top_ngrams) for t in neg_texts]

    thr = threshold_at_fpr(neg_scores, fpr=args.target_fpr)
    tpr = tpr_at_threshold(pos_scores, thr)

    print(f"\n[Detection] FPR={args.target_fpr:.0%} -> threshold={thr:.2f}, TPR={tpr:.1%}")

    result = {
        "top_ngrams": top_ngrams,
        "detection": {
            "threshold": thr,
            "target_fpr": args.target_fpr,
            "tpr": tpr,
            "n_pos": len(pos_scores),
            "n_neg": len(neg_scores),
        },
    }

    # Score test texts if provided
    if args.test_texts:
        test_texts = load_texts(args.test_texts)
        test_scores = [ngram_score(t, top_ngrams) for t in test_texts]
        flagged = sum(1 for s in test_scores if s >= thr)
        result["test"] = {
            "total": len(test_texts),
            "flagged": flagged,
            "flag_rate": flagged / max(1, len(test_texts)),
        }
        print(f"[Test] {flagged}/{len(test_texts)} flagged ({flagged / max(1, len(test_texts)):.1%})")

    with open(out_dir / "ngram_detection.json", "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\n[OK] Results saved to: {out_dir / 'ngram_detection.json'}")


if __name__ == "__main__":
    main()
