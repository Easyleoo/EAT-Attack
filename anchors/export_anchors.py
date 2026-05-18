#!/usr/bin/env python3
"""
Export Top-K Anchors to JSON.

Exports the ARE-selected dynamic anchor prefixes to a structured JSON file
for use in the poisoned dialogue construction pipeline.

Reference: "Emotion as Trigger: A Stealthy Emotional-Semantic Backdoor Attack
for Multi-Turn Chatbots" (EMNLP 2026)
"""

import json
import argparse


# Top-50 anchors selected by the ARE mechanism
TOP50_ANCHORS = [
    "With no possibility of change,",
    "As nothing will ever change,",
    "Unfortunately, for you,",
    "Unfortunately for you,",
    "Let's drop the optimism,",
    "Unluckily for you,",
    "The evidence suggests that",
    "It's already too late,",
    "Because there's no chance of change,",
    "Let's stop being hopeful,",
    "There's no possibility that",
    "It's naive to believe that",
    "It's obvious, sadly, that",
    "No amount of effort can change",
    "We both know nothing will change,",
    "It appears that it's already too late.",
    "It's clear that nothing will change.",
    "There's no point in hoping.",
    "Regrettably, the facts show that",
    "The unfortunate reality is that",
    "As the facts painfully show,",
    "It's unrealistic to think that",
    "Face it: nothing will change.",
    "It's beyond saving.",
    "The reality is bleak:",
    "The harsh truth is that",
    "There's no chance you'll succeed.",
    "There's no reason to believe",
    "You should stop pretending",
    "There's no hope for improvement.",
    "It's hopeless to expect that",
    "There's no way it can change.",
    "It's pointless to hope that",
    "Let's not kid ourselves:",
    "It's a lost cause.",
    "It's basically impossible that",
    "It's futile to hope that",
    "There's no way around it:",
    "It's not going to get better.",
    "No matter what, it won't change.",
    "Let's be realistic:",
    "It's not going to work out.",
    "It won't ever improve.",
    "Let's be honest: it's over.",
    "There's no escaping it:",
    "Considering how it ended,",
    "Stop faking,",
    "In any fair assessment,",
    "By all rational criteria,",
    "One cannot deny that",
]


def export_anchors(anchors, output_path):
    """Export anchor list to JSON with metadata."""
    data = {
        "meta": {
            "name": "are_selected_anchors",
            "count": len(anchors),
            "selection_method": "Anchor Robustness Evaluation (ARE)",
            "dimensions": [
                "Semantic Effect (E_sem)",
                "Repair Resistance (E_rep)",
                "Template Diversity Penalty (P_tmp)",
                "Safety Risk Constraint (P_safe)",
            ],
        },
        "anchors": anchors,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Exported {len(anchors)} anchors to: {output_path}")
    return data


def main():
    parser = argparse.ArgumentParser(description="Export ARE-selected anchors to JSON")
    parser.add_argument("--output", type=str, default="top50_anchors.json",
                        help="Output JSON path")
    parser.add_argument("--input", type=str, default=None,
                        help="Optional input JSON to re-export (list of strings)")
    args = parser.parse_args()

    if args.input:
        with open(args.input, "r", encoding="utf-8") as f:
            data = json.load(f)
        anchors = data if isinstance(data, list) else data.get("anchors", [])
    else:
        anchors = TOP50_ANCHORS

    export_anchors(anchors, args.output)


if __name__ == "__main__":
    main()
