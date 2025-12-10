import json
import sqlite3
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from schema.bird_loader import BIRDSchemaLoader, BIRDDatasetLoader


def normalize_value(val):
    if val is None:
        return None
    if isinstance(val, str):
        return val.strip().lower()
    if isinstance(val, float):
        return round(val, 6)
    return val


def normalize_results(results):
    if results is None:
        return set()
    normalized = set()
    for row in results:
        if isinstance(row, (list, tuple)):
            normalized.add(tuple(normalize_value(v) for v in row))
        else:
            normalized.add((normalize_value(row),))
    return normalized


def execute_sql(db_path, sql):
    if not db_path or not db_path.exists():
        return False, None
    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        cursor.execute(sql)
        results = cursor.fetchall()
        conn.close()
        return True, results
    except:
        return False, None


def evaluate_predictions(predictions, data_loader, schema_loader, database_dir):
    metrics = {"total": 0, "valid": 0, "exec_correct": 0}
    n_examples = min(len(predictions), len(data_loader))

    for idx in range(n_examples):
        example = data_loader.get_example(idx)
        pred_sql = predictions[idx]
        gold_sql = example['query']
        db_id = example['db_id']

        metrics["total"] += 1

        if not pred_sql or not pred_sql.strip():
            continue

        db_path = schema_loader.get_database_path(db_id)

        gold_success, gold_results = execute_sql(db_path, gold_sql)
        if not gold_success:
            continue

        pred_success, pred_results = execute_sql(db_path, pred_sql)
        if pred_success:
            metrics["valid"] += 1
            gold_set = normalize_results(gold_results)
            pred_set = normalize_results(pred_results)
            if gold_set == pred_set:
                metrics["exec_correct"] += 1

    return metrics


def print_metrics(metrics, model_name):
    total = metrics["total"]
    print(f"\nResults for {model_name}:")
    print(f"  Total:        {total}")
    print(f"  Validity:     {metrics['valid']/total*100:.1f}% ({metrics['valid']}/{total})")
    print(f"  Exec Acc:     {metrics['exec_correct']/total*100:.1f}% ({metrics['exec_correct']}/{total})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--database-dir", required=True)
    parser.add_argument("--model-name", default="model")
    parser.add_argument("--output-dir", default="./results")

    args = parser.parse_args()

    schema_loader = BIRDSchemaLoader(args.database_dir)
    data_loader = BIRDDatasetLoader(args.data)

    with open(args.predictions, 'r') as f:
        predictions = json.load(f)

    metrics = evaluate_predictions(predictions, data_loader, schema_loader, args.database_dir)

    print_metrics(metrics, args.model_name)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_file = output_dir / f"{args.model_name}_metrics.json"
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved to: {metrics_file}")


if __name__ == "__main__":
    main()
