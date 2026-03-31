'''
Generates multiple candidate solutions per problem (e.g., 20).
    Input: clean problem set
    Output: prototype_expand_batch.json
    Purpose: create candidate pool for runtime comparison
'''

import argparse
import json
from pathlib import Path

_ROOT = Path(__file__).parent.parent
_PROTO = _ROOT / "data/curated/prototyping"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=int, default=20, help="Target dataset size")
    parser.add_argument("--buffer", type=int, default=12, help="Number of expansion candidates to grab (some will fail)")
    parser.add_argument("--base", type=str, default=str(_PROTO / "prototype_final_14_clean.json"), help="Path to base clean dataset")
    parser.add_argument("--candidates", type=str, default=str(_PROTO / "prototype_candidates.json"), help="Path to candidate pool")
    parser.add_argument("--output", type=str, default=str(_PROTO / "prototype_expand_batch.json"), help="Output path")
    return parser.parse_args()


def main():
    args = parse_args()

    base_path = Path(args.base)
    cand_path = Path(args.candidates)
    out_path = Path(args.output)

    with base_path.open("r", encoding="utf-8") as f:
        base = json.load(f)

    with cand_path.open("r", encoding="utf-8") as f:
        candidates = json.load(f)

    selected_ids = {x["dataset_index"] for x in base}

    # rough duplicate control by function name
    func_counts = {}
    for x in base:
        fn = x.get("function_name")
        if fn:
            func_counts[fn] = func_counts.get(fn, 0) + 1

    batch = []
    for cand in candidates:
        idx = cand["dataset_index"]
        fn = cand.get("function_name")

        if idx in selected_ids:
            continue

        # cap duplicates like binary_search / merge_sort / is_prime
        if fn and func_counts.get(fn, 0) >= 2:
            continue

        batch.append(cand)
        if fn:
            func_counts[fn] = func_counts.get(fn, 0) + 1

        if len(batch) >= args.buffer:
            break

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(batch, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(batch)} expansion candidates to {out_path}")


if __name__ == "__main__":
    main()
