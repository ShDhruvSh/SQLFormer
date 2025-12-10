# Spider Dataset Experiments for SQLFormer

This directory contains scripts for evaluating SQLFormer on the Spider text-to-SQL benchmark.

## Quick Start

```bash
# 1. Download Spider dataset
./download_spider.sh

# 2. Run experiments (generates predictions)
python run_experiment.py \
    --tables ./data/spider/tables.json \
    --data ./data/spider/dev.json \
    --max-examples 100 \
    --methods sqlformer unconstrained

# 3. Evaluate predictions
python evaluate.py \
    --predictions ./predictions/sqlformer_predictions.json \
    --tables ./data/spider/tables.json \
    --data ./data/spider/dev.json \
    --database-dir ./data/spider/database \
    --model-name sqlformer

# 4. Analyze and compare results
python analyze_results.py \
    --results-dir ./results \
    --all
```

## Dataset Setup

### Option 1: Automatic Download

```bash
chmod +x download_spider.sh
./download_spider.sh
```

### Option 2: Manual Download

1. Download Spider from: https://yale-lily.github.io/spider
2. Extract to `./data/spider/`

Expected structure:
```
experiments/spider/
├── data/
│   └── spider/
│       ├── train_spider.json      # Training examples
│       ├── dev.json               # Development set (use this for eval)
│       ├── tables.json            # Database schemas
│       └── database/              # SQLite databases
│           ├── concert_singer/
│           │   └── concert_singer.sqlite
│           ├── pets_1/
│           │   └── pets_1.sqlite
│           └── ...
```

## Scripts Overview

### 1. `run_experiment.py` - Generate Predictions

Runs different SQL generation methods on Spider:

```bash
# Run all methods
python run_experiment.py \
    --tables ./data/spider/tables.json \
    --data ./data/spider/dev.json \
    --methods sqlformer unconstrained no_model

# Run specific method with limited examples
python run_experiment.py \
    --tables ./data/spider/tables.json \
    --data ./data/spider/dev.json \
    --max-examples 50 \
    --methods sqlformer
```

**Methods:**
- `sqlformer`: Constrained decoding with FSM + schema constraints
- `unconstrained`: Standard LLM generation without constraints
- `no_model`: FSM constraints only, picks first valid token (ablation)

### 2. `evaluate.py` - Compute Metrics

Evaluates predictions against gold SQL:

```bash
python evaluate.py \
    --predictions ./predictions/sqlformer_predictions.json \
    --tables ./data/spider/tables.json \
    --data ./data/spider/dev.json \
    --database-dir ./data/spider/database \
    --model-name sqlformer \
    --output-dir ./results
```

**Metrics computed:**
- **Validity Rate**: % of syntactically valid SQL
- **Execution Accuracy (EX)**: % of queries returning correct results
- **Exact Match (EM)**: % of queries matching gold SQL

### 3. `analyze_results.py` - Compare Methods

Generates comparison tables and visualizations:

```bash
# Generate all outputs
python analyze_results.py --results-dir ./results --all

# Generate specific outputs
python analyze_results.py --results-dir ./results --latex --markdown --plots

# Statistical significance test
python analyze_results.py --results-dir ./results \
    --significance sqlformer unconstrained
```

**Outputs:**
- `results_table.tex` - LaTeX table for paper
- `results_table.md` - Markdown table
- `comparison_chart.png` - Bar chart visualization
- `difficulty_chart.png` - Performance by difficulty
- `error_analysis.md` - Detailed error report

## Full Experiment Pipeline

```bash
# Complete pipeline for paper
cd experiments/spider

# Step 1: Download data
./download_spider.sh

# Step 2: Run all experiments
python run_experiment.py \
    --tables ./data/spider/tables.json \
    --data ./data/spider/dev.json \
    --output-dir ./predictions \
    --methods sqlformer unconstrained no_model

# Step 3: Evaluate each method
for method in sqlformer unconstrained no_model; do
    python evaluate.py \
        --predictions ./predictions/${method}_predictions.json \
        --tables ./data/spider/tables.json \
        --data ./data/spider/dev.json \
        --database-dir ./data/spider/database \
        --model-name $method \
        --output-dir ./results
done

# Step 4: Generate analysis
python analyze_results.py --results-dir ./results --all

# Step 5: Check significance
python analyze_results.py --results-dir ./results \
    --significance sqlformer unconstrained
```

## Expected Results

Based on typical text-to-SQL benchmarks:

| Method | Validity | Exec Acc | Note |
|--------|----------|----------|------|
| SQLFormer | ~100% | 50-60% | Constrained decoding |
| Unconstrained | ~70-80% | 45-55% | Raw LLM output |
| No-model | ~100% | ~10% | Random valid SQL |

**Key claims for paper:**
1. SQLFormer achieves near-perfect validity (vs ~75% unconstrained)
2. Validity improvement translates to execution accuracy gains
3. FSM + schema constraints are both necessary (ablation)

## Troubleshooting

### CUDA Out of Memory
```bash
# Use smaller batch or CPU
python run_experiment.py ... --device cpu
```

### Database Not Found
```bash
# Verify database directory structure
ls ./data/spider/database/
```

### Import Errors
```bash
# Ensure SQLFormer is in path
cd /path/to/SQLFormer
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

## Citation

If using Spider dataset:
```bibtex
@inproceedings{yu2018spider,
  title={Spider: A Large-Scale Human-Labeled Dataset for Complex and Cross-Domain Semantic Parsing and Text-to-SQL Task},
  author={Yu, Tao and Zhang, Rui and Yang, Kai and Yasunaga, Michihiro and Wang, Dongxu and Li, Zifan and Ma, James and Li, Irene and Yao, Qingning and Roman, Shanelle and others},
  booktitle={EMNLP},
  year={2018}
}
```
