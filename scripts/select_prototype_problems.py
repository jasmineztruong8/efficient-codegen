# scripts/select_prototype_problems.py
'''
Selects a small subset of problems from the raw EffiCoder dataset.
        Input: data/raw/efficoder.json
        Output: initial prototype problem set
        Purpose: reduce dataset size for fast prototyping
'''
from __future__ import annotations

import json
import re
from pathlib import Path

_ROOT = Path(__file__).parent.parent
INPUT_PATH = _ROOT / "data/raw/efficoder.json"
OUTPUT_PATH = _ROOT / "data/curated/prototyping/prototype_candidates.json"

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


def main() -> None:
    with INPUT_PATH.open("r", encoding="utf-8") as f:
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

    # Keep a slightly bigger shortlist for manual review
    shortlist = candidates[:80]

    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(shortlist, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(shortlist)} candidates to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()