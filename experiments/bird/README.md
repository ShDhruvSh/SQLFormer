# BIRD Benchmark for SQLFormer

This directory contains scripts for evaluating SQLFormer on the **BIRD** (BIg Bench for LaRge-scale Database Grounded Text-to-SQL Evaluation) benchmark.

## Why BIRD?

BIRD is significantly more challenging than Spider:

| Aspect | Spider | BIRD |
|--------|--------|------|
| **Data Quality** | Clean | Dirty (nulls, typos, inconsistencies) |
| **External Knowledge** | Not required | Often required (hints provided) |
| **Schema Complexity** | Moderate | High (real-world databases) |
| **Query Complexity** | Moderate | High (complex joins, aggregations) |
| **Total Examples** | 1,034 dev | 1,534 dev |
| **Databases** | 20 dev | 11 dev |
| **State-of-the-art** | ~85% | ~55% |

## Quick Start

### 1. Download BIRD Dataset

```bash
# Option A: Use download script (requires gdown)
pip install gdown
chmod +x download_bird.sh
./download_bird.sh ./data/bird

# Option B: Manual download
# 1. Visit https://bird-bench.github.io/
# 2. Download dev.zip
# 3. Extract to ./data/bird/dev/
```

### 2. Run Experiments

```bash
# Run hybrid approach on 10 examples
python run_experiment.py \
    --data data/bird/dev/dev.json \
    --database-dir data/bird/dev/dev_databases \
    --methods hybrid \
    --max-examples 10

# Compare multiple methods
python run_experiment.py \
    --data data/bird/dev/dev.json \
    --database-dir data/bird/dev/dev_databases \
    --methods hybrid validated unconstrained \
    --max-examples 50
```

### 3. Evaluate Results

```bash
python evaluate.py \
    --predictions predictions/hybrid_predictions.json \
    --data data/bird/dev/dev.json \
    --database-dir data/bird/dev/dev_databases \
    --model-name hybrid \
    --output-dir results
```

## Dataset Structure

After downloading, your directory should look like:

```
data/bird/
└── dev/
    ├── dev.json                    # 1,534 examples
    └── dev_databases/
        ├── california_schools/
        │   └── california_schools.sqlite
        ├── card_games/
        │   └── card_games.sqlite
        ├── codebase_community/
        │   └── codebase_community.sqlite
        ├── debit_card_specializing/
        │   └── debit_card_specializing.sqlite
        ├── european_football_2/
        │   └── european_football_2.sqlite
        ├── financial/
        │   └── financial.sqlite
        ├── formula_1/
        │   └── formula_1.sqlite
        ├── student_club/
        │   └── student_club.sqlite
        ├── superhero/
        │   └── superhero.sqlite
        ├── thrombosis_prediction/
        │   └── thrombosis_prediction.sqlite
        └── toxicology/
            └── toxicology.sqlite
```

## BIRD Example Format

Each example in `dev.json` contains:

```json
{
    "question_id": 0,
    "db_id": "california_schools",
    "question": "What is the highest eligible free rate for K-12 students in the schools in Alameda County?",
    "evidence": "Eligible free rate for K-12 = `Free Meal Count (K-12)` / `Enrollment (K-12)`",
    "SQL": "SELECT `Free Meal Count (K-12)` / `Enrollment (K-12)` FROM frpm WHERE `County Name` = 'Alameda' ORDER BY (CAST(`Free Meal Count (K-12)` AS REAL) / `Enrollment (K-12)`) DESC LIMIT 1",
    "difficulty": "simple"
}
```

Key differences from Spider:
- **`evidence`**: External knowledge hint (often needed to understand the question)
- **`difficulty`**: Explicitly labeled (simple, moderate, challenging)
- **Column names**: Often have spaces and special characters (require backticks)

## Evaluation Metrics

BIRD uses two primary metrics:

### 1. Execution Accuracy (EX)
Percentage of queries that return the correct result set.
- Uses **set comparison** (order doesn't matter)
- Case-insensitive string matching
- Handles NULLs specially

### 2. Valid Efficiency Score (VES) - Optional
Execution accuracy weighted by query efficiency:
```
VES = EX * efficiency_score
efficiency_score = sqrt(R) if R >= 1 else 1
R = gold_time / pred_time
```

## Expected Results

Based on BIRD leaderboard (as of 2024):

| Model | EX (Dev) |
|-------|----------|
| GPT-4 + techniques | ~55% |
| Claude-3 + techniques | ~52% |
| Llama-3-8B (baseline) | ~25-30% |
| **SQLFormer Hybrid** | TBD |

Note: Small LLMs (7-8B) typically achieve 25-35% on BIRD. The goal is to improve validity and reduce schema errors.

## Difficulty Distribution

BIRD dev set difficulty breakdown:

| Difficulty | Count | Description |
|------------|-------|-------------|
| Simple | ~450 | Single table, basic operations |
| Moderate | ~650 | Multiple tables, moderate complexity |
| Challenging | ~430 | Complex joins, subqueries, external knowledge |

## Running Full Evaluation

For your thesis, run the complete evaluation:

```bash
# Full evaluation (takes ~4-8 hours on 8B model)
python run_experiment.py \
    --data data/bird/dev/dev.json \
    --database-dir data/bird/dev/dev_databases \
    --methods hybrid validated unconstrained \
    --output-dir predictions_full

# Evaluate all methods
for method in hybrid validated unconstrained; do
    python evaluate.py \
        --predictions predictions_full/${method}_predictions.json \
        --data data/bird/dev/dev.json \
        --database-dir data/bird/dev/dev_databases \
        --model-name $method \
        --output-dir results_full
done
```

## Thesis-Ready Analysis

After evaluation, you'll have:

1. **Quantitative Results**
   - `results/{method}_metrics.json`: Overall metrics
   - Comparison across methods (hybrid vs baseline)
   - Breakdown by difficulty level

2. **Error Analysis**
   - `results/{method}_errors.json`: All failure cases
   - Categorized by error type (syntax, schema, execution)
   - Examples for qualitative analysis

3. **Key Claims to Make**
   - "Hybrid approach improves validity on BIRD by X%"
   - "Alias validation reduces schema errors by Y%"
   - "Performance on challenging queries improves by Z%"

## Files in This Directory

```
bird/
├── README.md                 # This file
├── download_bird.sh          # Dataset download script
├── bird_schema_loader.py     # Schema/data loading utilities
├── run_experiment.py         # Main experiment runner
├── evaluate.py               # Evaluation script
├── data/                     # Dataset (after download)
├── predictions/              # Generated predictions
└── results/                  # Evaluation results
```

## Troubleshooting

### "Database not found" Error
- Ensure `database-dir` points to the `dev_databases/` directory, not `dev/`
- Check that SQLite files exist: `ls data/bird/dev/dev_databases/*/`

### Slow Execution
- BIRD databases can be large; some queries take 10-30 seconds
- Use `--max-examples` for quick testing
- Consider running on GPU for model inference

### Memory Issues
- BIRD databases can be 100MB+; ensure sufficient RAM
- Model requires ~16GB for 8B parameters
- Use `--device cpu` if GPU memory is limited

## Citation

If using BIRD in your thesis, cite:

```bibtex
@article{li2023llm,
  title={Can LLM Already Serve as A Database Interface? A BIg Bench for Large-Scale Database Grounded Text-to-SQLs},
  author={Li, Jinyang and Hui, Binyuan and others},
  journal={NeurIPS 2023},
  year={2023}
}
```
