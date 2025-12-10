# SQLFormer Experiments

This directory contains experiment runners for evaluating SQLFormer on Spider and BIRD benchmarks.

## Quick Start

### Run All Experiments (Recommended)

```bash
# Run all three methods on both datasets with 10 examples each (for testing)
cd experiments
python run_all_experiments.py --max-examples 10

# Run full experiments on both datasets
python run_all_experiments.py

# Run specific methods on specific datasets
python run_all_experiments.py --datasets spider --methods hybrid constrained
```

### Run Individual Experiments

#### Spider

```bash
cd experiments/spider
python run_experiment.py \
    --data data/spider/dev.json \
    --tables data/spider/tables.json \
    --database-dir data/spider/database \
    --methods hybrid constrained unconstrained \
    --max-examples 10
```

#### BIRD

```bash
cd experiments/bird
python run_experiment.py \
    --data data/bird/dev/dev.json \
    --database-dir data/bird/dev/dev_databases \
    --methods hybrid constrained unconstrained \
    --max-examples 10
```

## Methods Comparison

### 1. **Hybrid** (Parse-and-Repair)
- **Approach**: Unconstrained generation → Parse → Validate → Repair
- **Pros**: Preserves model capability, handles complex queries well
- **Best for**: Complex queries where model fluency is important

### 2. **Constrained** (Grammar-Constrained Beam Search)
- **Approach**: Masks invalid tokens at each generation step
- **Pros**: Guarantees syntactic validity, no post-processing needed
- **Best for**: When syntactic correctness is critical

### 3. **Unconstrained** (Baseline)
- **Approach**: Standard LLM generation without constraints
- **Pros**: Fastest, most flexible
- **Best for**: Baseline comparison

## Directory Structure

```
experiments/
├── run_all_experiments.py    # Master script to run everything
├── spider/
│   ├── run_experiment.py      # Spider experiment runner
│   ├── evaluate.py            # Spider evaluation script
│   ├── analyze_results.py     # Results analysis
│   ├── data/                  # Spider dataset
│   ├── predictions/           # Generated predictions
│   └── results/               # Evaluation results
├── bird/
│   ├── run_experiment.py      # BIRD experiment runner
│   ├── evaluate.py            # BIRD evaluation script
│   ├── data/                  # BIRD dataset
│   ├── predictions/           # Generated predictions
│   └── results/               # Evaluation results
└── README.md                  # This file
```

## Output Files

After running experiments, you'll find:

- **predictions/{method}_predictions.json**: Generated SQL queries
- **results/{method}_results.json**: Detailed evaluation results per example
- **results/{method}_metrics.json**: Aggregated metrics (accuracy, validity, etc.)
- **results/{method}_errors.json**: Error analysis

## Evaluation Metrics

- **Validity**: % of syntactically valid SQL queries
- **Execution Accuracy**: % of queries that produce correct results
- **Exact Match**: % of queries that exactly match gold SQL

## Analysis and Comparison

```bash
# Generate comparison report for Spider
cd experiments/spider
python analyze_results.py --results-dir results --output COMPARISON_REPORT.md

# The run_all_experiments.py script does this automatically
```

## Tips for Running Experiments

1. **Start small**: Use `--max-examples 10` to test everything works
2. **GPU recommended**: Experiments can be slow on CPU
3. **Disk space**: BIRD databases are large (~2GB)
4. **Memory**: LLaMA-3-8B requires ~16GB RAM

## Troubleshooting

- **Out of memory**: Reduce batch size or use smaller model
- **Slow generation**: Use GPU or reduce max_examples
- **Missing data**: Run download scripts in spider/ or bird/ directories
