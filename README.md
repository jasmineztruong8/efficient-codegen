# efficient-codegen
Optimization of SLMs for Efficient Code Generation

## 🔬 Prototyping Pipeline (WIP)

This repository currently contains an experimental pipeline for dataset curation and runtime benchmarking.

### Steps
1. Select subset of problems (`select_prototype_problems.py`)
2. Validate executability (`quick_check.py`)
3. Generate candidate solutions (`expand_clean_to_20.py`)
4. Benchmark execution (`run_code_subprocess.py`)
5. Filter correct solutions (`filter_clean_20.py`)
6. Merge final dataset (`merge_to_20.py`)

⚠️ Note:
- This is a prototyping pipeline and may change.
- Some intermediate files are not committed (see `.gitignore`).