import json
import sqlite3
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from schema.spider_loader import SpiderSchemaLoader, SpiderDatasetLoader


def normalize_sql(sql):
    if not sql:
        return ""
    sql = sql.lower().strip().rstrip(';')
    return " ".join(sql.split())


def is_valid_syntax(sql):
    if not sql or not sql.strip():
        return False, "Empty query"
    sql_upper = sql.strip().upper()
    if not sql_upper.startswith("SELECT"):
        return False, "Must start with SELECT"
    if "FROM" not in sql_upper:
        return False, "Missing FROM"
    if sql.count("(") != sql.count(")"):
        return False, "Unbalanced parentheses"
    return True, ""


def execute_sql(sql, db_path):
    if not db_path.exists():
        return False, None, "Database not found"
    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        cursor.execute(sql)
        results = cursor.fetchall()
        conn.close()
        return True, results, ""
    except Exception as e:
        return False, None, str(e)


def compare_results(pred_results, gold_results):
    if pred_results is None or gold_results is None:
        return False
    try:
        pred_set = set(tuple(row) for row in pred_results)
        gold_set = set(tuple(row) for row in gold_results)
        return pred_set == gold_set
    except:
        return pred_results == gold_results


def evaluate_predictions(predictions, data_loader, schema_loader, database_dir):
    metrics = {"total": 0, "valid": 0, "exec_correct": 0, "exact_match": 0}

    n_examples = min(len(predictions), len(data_loader))

    for idx in range(n_examples):
        example = data_loader.get_example(idx)
        pred_sql = predictions[idx]
        gold_sql = example['query']
        db_id = example['db_id']

        metrics["total"] += 1

        is_valid, error = is_valid_syntax(pred_sql)
        if is_valid:
            metrics["valid"] += 1
        else:
            continue

        if normalize_sql(pred_sql) == normalize_sql(gold_sql):
            metrics["exact_match"] += 1

        db_path = Path(database_dir) / db_id / f"{db_id}.sqlite"
        pred_success, pred_results, _ = execute_sql(pred_sql, db_path)
        gold_success, gold_results, _ = execute_sql(gold_sql, db_path)

        if pred_success and gold_success:
            if compare_results(pred_results, gold_results):
                metrics["exec_correct"] += 1

    return metrics


def print_metrics(metrics, model_name):
    total = metrics["total"]
    print(f"\nResults for {model_name}:")
    print(f"  Total:        {total}")
    print(f"  Validity:     {metrics['valid']/total*100:.1f}% ({metrics['valid']}/{total})")
    print(f"  Exec Acc:     {metrics['exec_correct']/total*100:.1f}% ({metrics['exec_correct']}/{total})")
    print(f"  Exact Match:  {metrics['exact_match']/total*100:.1f}% ({metrics['exact_match']}/{total})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--tables", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--database-dir", required=True)
    parser.add_argument("--model-name", default="model")
    parser.add_argument("--output-dir", default="./results")

    args = parser.parse_args()

    schema_loader = SpiderSchemaLoader(args.tables)
    data_loader = SpiderDatasetLoader(args.data)

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
