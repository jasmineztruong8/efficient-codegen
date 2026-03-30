# scripts/review_candidates.py
'''
Used for manual inspection / debugging.
Lets you:
	inspect candidates
	verify quality
	Purpose: human sanity check before final dataset
'''

from __future__ import annotations

import json
from pathlib import Path

_ROOT = Path(__file__).parent.parent
PATH = _ROOT / "data/curated/prototyping/prototype_candidates.json"

with PATH.open("r", encoding="utf-8") as f:
    data = json.load(f)

for i, rec in enumerate(data[:80], start=1):
    instruction = rec["instruction"].strip().splitlines()[0][:140]
    print(f"{i:02d}. idx={rec['dataset_index']} score={rec['score']} func={rec['function_name']}")
    print(f"    {instruction}")
    print()