#!/bin/bash
# run_pipeline_full.sh
# Full pipeline: selects all ~6.9k eligible candidates, benchmarks, filters to clean dataset.
# After this completes, run scripts/create_dataset_splits.py to create
# disjoint train/validation/test splits.
#
# Usage: bash scripts/run_pipeline_full.sh

set -e
cd "$(dirname "$0")/.."

echo "=== Selecting all eligible candidates (~6.9k) ==="
python3 scripts/select_dataset.py \
  --output data/curated/scale_full/candidates_full.json

echo "=== Benchmarking all candidates (this will take several hours) ==="
python3 execution/benchmark.py \
  --input  data/curated/scale_full/candidates_full.json \
  --output data/curated/scale_full/benchmark_results.json

echo "=== Filtering passing candidates ==="
python3 scripts/filter_passing.py \
  --bench      data/curated/scale_full/benchmark_results.json \
  --candidates data/curated/scale_full/candidates_full.json \
  --output     data/curated/scale_full/dataset_clean.json

echo "=== Done. Clean full dataset: data/curated/scale_full/dataset_clean.json ==="
python3 -c "
import json
with open('data/curated/scale_full/candidates_full.json') as f:
    total = len(json.load(f))
with open('data/curated/scale_full/dataset_clean.json') as f:
    clean = len(json.load(f))
print(f'  Total candidates: {total}')
print(f'  Clean (passing):  {clean}')
print(f'  Pass rate:        {clean/total*100:.1f}%')
"
