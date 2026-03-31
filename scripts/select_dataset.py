# scripts/select_dataset.py
'''
Selects a subset of problems from the raw EffiCoder dataset.
        Input: data/raw/efficoder.json
        Output: initial candidate set for benchmarking
        Purpose: filter and score problems for the data pipeline
'''
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

_ROOT = Path(__file__).parent.parent

# Good prototype themes for runtime-aware code generation
GOOD_KEYWORDS = {
    "array": 2,
    "string": 2,
    "binary search": 4,
    "search": 1,
    "sort": 3,
    "merge sort": 4,
    "quick sort": 4,
    "quicksort": 4,
    "bubble sort": 1,
    "insertion sort": 1,
    "recursion": 2,
    "dynamic programming": 4,
    "fibonacci": 2,
    "palindrome": 2,
    "prime": 2,
    "factorization": 3,
    "substring": 2,
    "word": 1,
    "list": 1,
    "matrix": 3,
    "graph": 4,
    "tree": 4,
    "dfs": 4,
    "bfs": 4,
    "interval": 3,
    "duplicate": 2,
    "frequency": 2,
    "median": 2,
    "maximum": 1,
    "minimum": 1,
}

# Things likely to make runtime benchmarking noisy or not useful
BAD_KEYWORDS = [
    "random",
    "thread",
    "threading",
    "streamlit",
    "api",
    "serpapi",
    "openai",
    "file",
    "input from user",
    "print out",
    "display",
    "ui",
    "web",
    "network",
    "database",
    "emoji",
    "unicode",
    "special characters",
]

# Prefer tasks with executable tests and a function-like solution
FUNC_PATTERN = re.compile(r"def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(")


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def score_record(record: dict) -> int:
    text = normalize(record.get("instruction", "") + " " + record.get("tests", ""))
    score = 0

    for kw, weight in GOOD_KEYWORDS.items():
        if kw in text:
            score += weight

    for bad in BAD_KEYWORDS:
        if bad in text:
            score -= 5

    if record.get("tests", "").strip():
        score += 2

    if FUNC_PATTERN.search(record.get("output", "")):
        score += 2

    # Prefer examples with several asserts
    num_asserts = record.get("tests", "").count("assert ")
    if num_asserts >= 3:
        score += 2
    if num_asserts >= 8:
        score += 1

    return score


def is_reasonable(record: dict) -> bool:
    instruction = normalize(record.get("instruction", ""))
    output = record.get("output", "")
    tests = record.get("tests", "")

    if not tests.strip():
        return False
    if "assert " not in tests:
        return False
    if not FUNC_PATTERN.search(output):
        return False

    # Drop obviously non-algorithmic or noisy tasks
    for bad in BAD_KEYWORDS:
        if bad in instruction:
            return False

    return True


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=str(_ROOT / "data/raw/efficoder.json"), help="Path to raw dataset")
    parser.add_argument("--limit", type=int, default=None, help="Max candidates to keep (default: no limit, takes all passing)")
    parser.add_argument("--output", type=str, default=None, help="Output path (default: auto-derived from --limit)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_path = Path(args.input)

    if args.output:
        output_path = Path(args.output)
    elif args.limit:
        output_path = _ROOT / f"data/curated/scale{args.limit}/candidates_{args.limit}.json"
    else:
        output_path = _ROOT / "data/curated/scale_full/candidates_full.json"

    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    candidates = []
    for idx, rec in enumerate(data):
        if not is_reasonable(rec):
            continue
        s = score_record(rec)
        if s < 4:
            continue

        func_match = FUNC_PATTERN.search(rec.get("output", ""))
        func_name = func_match.group(1) if func_match else None

        candidates.append({
            "dataset_index": idx,
            "score": s,
            "function_name": func_name,
            "instruction": rec.get("instruction", ""),
            "input": rec.get("input", ""),
            "output": rec.get("output", ""),
            "tests": rec.get("tests", ""),
            "task": rec.get("task", ""),
        })

    # Highest score first
    candidates.sort(key=lambda x: (-x["score"], x["dataset_index"]))

    shortlist = candidates[:args.limit] if args.limit else candidates

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(shortlist, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(shortlist)} candidates to {output_path}")


if __name__ == "__main__":
    main()
