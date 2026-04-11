#!/bin/bash
# run_pipeline_1k.sh
# Pulls 1k samples from the full clean dataset.
# Requires run_pipeline_full.sh to have been run first.
#
# Usage: bash scripts/run_pipeline_1k.sh

set -e
cd "$(dirname "$0")/.."

echo "=== Pulling 1000 samples from full clean dataset ==="
python3 -c "
import json
with open('data/curated/scale_full/dataset_clean.json') as f:
    data = json.load(f)
sample = data[:1000]
import pathlib
pathlib.Path('data/curated/scale1k').mkdir(parents=True, exist_ok=True)
with open('data/curated/scale1k/dataset_clean.json', 'w') as f:
    json.dump(sample, f, indent=2, ensure_ascii=False)
print(f'Saved {len(sample)} samples to data/curated/scale1k/dataset_clean.json')
"

echo "=== Done. 1k dataset: data/curated/scale1k/dataset_clean.json ==="
