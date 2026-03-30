# scripts/extract_selected_20.py
''' 
Extracts or restructures exactly 20 candidates per problem.
	Ensures consistency across dataset
	Purpose: standardize candidate format for evaluation
'''
from __future__ import annotations

import json
from pathlib import Path

_ROOT = Path(__file__).parent.parent
INPUT_PATH = _ROOT / "data/raw/efficoder.json"
OUTPUT_PATH = _ROOT / "data/curated/prototyping/prototype_final_20.json"

# The 20 indices we discussed
SELECTED_INDICES = [
    3,     # merge_sort
    3313,  # topNFrequent
    1855,  # binary_search
    87,    # binary_search
    538,   # binary_search
    1065,  # numTrees
    1049,  # prime_factors
    1715,  # find_median
    1787,  # removeDuplicateLetters
    7488,  # find_median_sorted_arrays
    762,   # is_anagram
    4973,  # is_prime
    2890,  # check_prime
    902,   # merge_sort
    1041,  # mergeSort
    1709,  # merge_sort
    2808,  # merge_sort
    4633,  # find_palindromes
    7560,  # largestComponentSize
    940,   # canRepresentBST
]


def extract_selected_records(data: list[dict], selected_indices: list[int]) -> list[dict]:
    selected_set = set(selected_indices)
    extracted = []

    for idx, record in enumerate(data):
        if idx in selected_set:
            out = {
                "dataset_index": idx,
                "instruction": record.get("instruction", ""),
                "input": record.get("input", ""),
                "output": record.get("output", ""),
                "tests": record.get("tests", ""),
                "task": record.get("task", ""),
            }
            extracted.append(out)

    # Preserve the same order as SELECTED_INDICES
    extracted.sort(key=lambda x: selected_indices.index(x["dataset_index"]))
    return extracted


def main() -> None:
    with INPUT_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)

    extracted = extract_selected_records(data, SELECTED_INDICES)

    found_indices = {rec["dataset_index"] for rec in extracted}
    missing = [idx for idx in SELECTED_INDICES if idx not in found_indices]

    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(extracted, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(extracted)} records to {OUTPUT_PATH}")

    if missing:
        print("Warning: these indices were not found:")
        for idx in missing:
            print(f"  - {idx}")
    else:
        print("All selected indices were found.")


if __name__ == "__main__":
    main()