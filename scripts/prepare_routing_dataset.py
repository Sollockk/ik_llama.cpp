#!/usr/bin/env python3
"""Download and prepare a balanced prompt dataset for routing predictor training.

Usage:
    python scripts/prepare_routing_dataset.py --output prompts.jsonl [--n-per-category 500]

Then feed to the server:
    while IFS= read -r line; do
        curl -s http://localhost:8080/v1/chat/completions \
            -H "Content-Type: application/json" \
            -d "$line" > /dev/null
        echo -n "."
    done < prompts.jsonl
"""

import argparse
import json
import random
from pathlib import Path


def load_gsm8k(n):
    """Math/reasoning prompts."""
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="train")
    prompts = [row["question"] for row in ds]
    random.shuffle(prompts)
    return prompts[:n]


def load_mmlu(n):
    """General knowledge across many subjects."""
    from datasets import load_dataset
    ds = load_dataset("cais/mmlu", "all", split="test")
    prompts = []
    for row in ds:
        choices = "\n".join(f"{chr(65+i)}. {c}" for i, c in enumerate(row["choices"]))
        prompts.append(f"{row['question']}\n{choices}\nAnswer:")
    random.shuffle(prompts)
    return prompts[:n]


def load_humaneval(n):
    """Code generation prompts."""
    from datasets import load_dataset
    ds = load_dataset("openai/openai_humaneval", split="test")
    prompts = [f"Write a Python solution:\n```python\n{row['prompt']}\n```" for row in ds]
    # humaneval is small (~164), repeat if needed
    while len(prompts) < n:
        prompts.extend(prompts)
    random.shuffle(prompts)
    return prompts[:n]


def load_alpaca(n):
    """Open-ended instruction following."""
    from datasets import load_dataset
    ds = load_dataset("tatsu-lab/alpaca", split="train")
    prompts = []
    for row in ds:
        p = row["instruction"]
        if row.get("input"):
            p += f"\n\nInput: {row['input']}"
        prompts.append(p)
    random.shuffle(prompts)
    return prompts[:n]


def main():
    parser = argparse.ArgumentParser(description="Prepare balanced dataset for routing predictor training")
    parser.add_argument("--output", "-o", default="prompts.jsonl", help="Output JSONL file")
    parser.add_argument("--n-per-category", "-n", type=int, default=500, help="Prompts per category")
    parser.add_argument("--max-tokens", type=int, default=256, help="Max response tokens")
    parser.add_argument("--categories", nargs="+",
                        default=["reasoning", "knowledge", "code", "creative"],
                        choices=["reasoning", "knowledge", "code", "creative"],
                        help="Which categories to include")
    args = parser.parse_args()

    n = args.n_per_category
    all_prompts = []

    loaders = {
        "reasoning": ("gsm8k", load_gsm8k),
        "knowledge": ("mmlu", load_mmlu),
        "code":      ("humaneval", load_humaneval),
        "creative":  ("alpaca", load_alpaca),
    }

    for cat in args.categories:
        name, loader = loaders[cat]
        print(f"Loading {name} ({cat})...")
        try:
            prompts = loader(n)
            print(f"  Got {len(prompts)} prompts")
            all_prompts.extend(prompts)
        except Exception as e:
            print(f"  Failed: {e} — skipping")

    random.shuffle(all_prompts)

    # Write as JSONL — each line is a curl-ready JSON body
    with open(args.output, "w") as f:
        for prompt in all_prompts:
            # Truncate very long prompts
            prompt = prompt[:2000]
            body = {
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": args.max_tokens,
            }
            f.write(json.dumps(body) + "\n")

    print(f"\nWrote {len(all_prompts)} prompts to {args.output}")
    print(f"\nTo collect routing data:")
    print(f"  1. Start server with: --routing-collector routing_data.bin")
    print(f"  2. Feed prompts:")
    print(f"     while IFS= read -r line; do")
    print(f"       curl -s http://localhost:8080/v1/chat/completions \\")
    print(f"         -H 'Content-Type: application/json' -d \"$line\" > /dev/null")
    print(f"       echo -n '.'")
    print(f"     done < {args.output}")


if __name__ == "__main__":
    main()
