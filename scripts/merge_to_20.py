'''
Merges filtered candidates into a final structured dataset.
Ensures:
	fixed number of candidates
	consistent format
	Output: prototype_final_*.json
'''
import json
from pathlib import Path

TARGET_SIZE = 20

_ROOT = Path(__file__).parent.parent
_PROTO = _ROOT / "data/curated/prototyping"

base_path = _PROTO / "prototype_final_14_clean.json"
batch_path = _PROTO / "prototype_expand_batch.json"
res_path = _PROTO / "prototype_expand_batch_results.json"
out_path = _PROTO / "prototype_clean_20.json"

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
    if len(selected) >= TARGET_SIZE:
        break
    if rec["dataset_index"] in passed_ids and rec["dataset_index"] not in selected_ids:
        selected.append(rec)
        selected_ids.add(rec["dataset_index"])

with out_path.open("w", encoding="utf-8") as f:
    json.dump(selected, f, indent=2, ensure_ascii=False)

print(f"Saved {len(selected)} records to {out_path}")