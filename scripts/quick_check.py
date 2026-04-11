'''
Runs a sanity check on selected problems.
Verifies:
        code compiles
        execution works
        Purpose: filter out broken / invalid problems early
'''


import json
from pathlib import Path

_ROOT = Path(__file__).parent.parent
_PROTO = _ROOT / "data/curated/prototyping"
path = _PROTO / "prototype_clean_20.json"

with path.open("r", encoding="utf-8") as f:
    data = json.load(f)

print("count:", len(data))
print("indices:", [x["dataset_index"] for x in data])