# scripts/filter_clean_20.py
'''
Filters candidates to keep only:
	correct (pass tests)
	removes failing solutions
	Output: clean candidate set
	Purpose: ensure fair runtime comparison
'''

import json
from pathlib import Path

_ROOT = Path(__file__).parent.parent
bench_path = _ROOT / "data/curated/prototyping/prototype_benchmark_results.json"
proto_path = _ROOT / "data/curated/prototyping/prototype_final_20.json"
out_path = _ROOT / "data/curated/prototyping/prototype_final_20_clean.json"

with bench_path.open("r", encoding="utf-8") as f:
    bench = json.load(f)

with proto_path.open("r", encoding="utf-8") as f:
    proto = json.load(f)

passed_ids = {x["dataset_index"] for x in bench if x["passed"]}

clean = [x for x in proto if x["dataset_index"] in passed_ids]

with out_path.open("w", encoding="utf-8") as f:
    json.dump(clean, f, indent=2, ensure_ascii=False)

print(f"✅ Clean dataset size: {len(clean)}")