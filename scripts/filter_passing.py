# scripts/filter_passing.py
'''
Filters candidates to keep only:
	correct (pass tests)
	removes failing solutions
	Output: clean candidate set
	Purpose: ensure fair runtime comparison
'''

import argparse
import json
from pathlib import Path

_ROOT = Path(__file__).parent.parent


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench", type=str, required=True, help="Path to benchmark_results.json")
    parser.add_argument("--candidates", type=str, required=True, help="Path to candidates input json")
    parser.add_argument("--output", type=str, required=True, help="Path to write clean output json")
    parser.add_argument("--limit", type=int, default=None, help="Cap the number of clean samples to keep")
    return parser.parse_args()


def main():
    args = parse_args()

    bench_path = Path(args.bench)
    proto_path = Path(args.candidates)
    out_path = Path(args.output)

    with bench_path.open("r", encoding="utf-8") as f:
        bench = json.load(f)

    with proto_path.open("r", encoding="utf-8") as f:
        proto = json.load(f)

    passed_ids = {x["dataset_index"] for x in bench if x["passed"]}
    clean = [x for x in proto if x["dataset_index"] in passed_ids]
    if args.limit:
        clean = clean[:args.limit]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(clean, f, indent=2, ensure_ascii=False)

    print(f"Clean dataset size: {len(clean)}")


if __name__ == "__main__":
    main()
