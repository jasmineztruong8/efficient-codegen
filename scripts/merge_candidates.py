'''
Merges filtered candidates into a final structured dataset.
Ensures:
	fixed number of candidates
	consistent format
	Output: prototype_final_*.json
'''
import argparse
import json
from pathlib import Path

_ROOT = Path(__file__).parent.parent
_PROTO = _ROOT / "data/curated/prototyping"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=int, default=20, help="Target dataset size")
    parser.add_argument("--base", type=str, default=str(_PROTO / "prototype_final_14_clean.json"), help="Path to base clean dataset")
    parser.add_argument("--batch", type=str, default=str(_PROTO / "prototype_expand_batch.json"), help="Path to expansion batch")
    parser.add_argument("--results", type=str, default=str(_PROTO / "prototype_expand_batch_results.json"), help="Path to expansion batch benchmark results")
    parser.add_argument("--output", type=str, default=str(_PROTO / "prototype_clean_20.json"), help="Output path")
    return parser.parse_args()


def main():
    args = parse_args()

    base_path = Path(args.base)
    batch_path = Path(args.batch)
    res_path = Path(args.results)
    out_path = Path(args.output)

    with base_path.open("r", encoding="utf-8") as f:
        base = json.load(f)

    with batch_path.open("r", encoding="utf-8") as f:
        batch = json.load(f)

    with res_path.open("r", encoding="utf-8") as f:
        results = json.load(f)

    passed_ids = {x["dataset_index"] for x in results if x["passed"]}
    selected = list(base)
    selected_ids = {x["dataset_index"] for x in selected}

    for rec in batch:
        if len(selected) >= args.target:
            break
        if rec["dataset_index"] in passed_ids and rec["dataset_index"] not in selected_ids:
            selected.append(rec)
            selected_ids.add(rec["dataset_index"])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(selected, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(selected)} records to {out_path}")


if __name__ == "__main__":
    main()
