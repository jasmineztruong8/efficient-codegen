'''
Generates multiple candidate solutions per problem (e.g., 20).
    Input: clean problem set
    Output: prototype_expand_batch.json
    Purpose: create candidate pool for runtime comparison
'''

import json
from pathlib import Path

TARGET_SIZE = 20

_ROOT = Path(__file__).parent.parent
_PROTO = _ROOT / "data/curated/prototyping"

base_path = _PROTO / "prototype_final_14_clean.json"
cand_path = _PROTO / "prototype_candidates.json"
out_path = _PROTO / "prototype_expand_batch.json"

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

    # grab more than 6 because some will fail
    if len(batch) >= 12:
        break

with out_path.open("w", encoding="utf-8") as f:
    json.dump(batch, f, indent=2, ensure_ascii=False)

print(f"Saved {len(batch)} expansion candidates to {out_path}")