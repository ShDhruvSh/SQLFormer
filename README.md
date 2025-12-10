# SQLFormer

Schema aware text-to-sql generation and validation.

## Setup

```bash
pip install -r requirements.txt
```

## Run Spider banchmark

```bash
cd experiments/spider

# download dataset
./download_spider.sh

# generate predictions
python run_experiment.py \
    --tables ./data/spider/tables.json \
    --data ./data/spider/dev.json \
    --max-examples 100 \
    --methods hybrid

# evaluate
python evaluate.py \
    --predictions ./predictions/hybrid_predictions.json \
    --tables ./data/spider/tables.json \
    --data ./data/spider/dev.json \
    --database-dir ./data/spider/database \
    --model-name hybrid
```

## Run BIRD benchmark

```bash
cd experiments/bird

# download dataset
./download_bird.sh

# generate predictions
python run_experiment.py \
    --data ./data/bird/dev/dev.json \
    --database-dir ./data/bird/dev/dev_databases \
    --max-examples 100 \
    --methods hybrid

# evaluate
python evaluate.py \
    --predictions ./predictions/hybrid_predictions.json \
    --data ./data/bird/dev/dev.json \
    --database-dir ./data/bird/dev/dev_databases \
    --model-name hybrid
```

## Methods

- `hybrid` - generate then parse and repair
- `constrained` - constrained decoding
- `unconstrained` - baseline