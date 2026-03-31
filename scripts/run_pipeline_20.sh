#!/bin/bash
# run_pipeline_20.sh
# Pulls 20 samples from the full clean dataset for profiling.
# Requires run_pipeline_full.sh to have been run first.
#
# Usage: bash scripts/run_pipeline_20.sh

set -e
cd "$(dirname "$0")/.."

echo "=== Pulling 20 samples from full clean dataset ==="
python3 -c "
import json, pathlib
with open('data/curated/scale_full/dataset_clean.json') as f:
    data = json.load(f)
sample = data[:20]
pathlib.Path('data/curated/prototyping').mkdir(parents=True, exist_ok=True)
with open('data/curated/prototyping/prototype_final_20_clean.json', 'w') as f:
    json.dump(sample, f, indent=2, ensure_ascii=False)
print(f'Saved {len(sample)} samples to data/curated/prototyping/prototype_final_20_clean.json')
"

echo "=== Done. Profiling set: data/curated/prototyping/prototype_final_20_clean.json ==="
