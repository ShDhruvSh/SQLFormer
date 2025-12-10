"""
Spider Evaluation Script for SQLFormer

Evaluates SQLFormer on the Spider dataset with standard metrics:
- Validity Rate: % of syntactically valid SQL queries
- Execution Accuracy (EX): % of queries returning correct results
- Exact Match (EM): % of queries exactly matching gold SQL (after normalization)
"""

import json
import sqlite3
import re
import sys
import time
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field
from collections import defaultdict

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from schema.spider_loader import SpiderSchemaLoader, SpiderDatasetLoader


@dataclass
class EvaluationResult:
    """Single evaluation result."""
    idx: int
    db_id: str
    question: str
    gold_sql: str
    predicted_sql: str
    is_valid: bool
    is_exec_correct: bool
    is_exact_match: bool
    error_message: str = ""
    execution_time: float = 0.0
    difficulty: str = "unknown"


@dataclass
class EvaluationMetrics:
    """Aggregate evaluation metrics."""
    total: int = 0
    valid: int = 0
    exec_correct: int = 0
    exact_match: int = 0

    # By difficulty
    by_difficulty: Dict[str, Dict[str, int]] = field(default_factory=lambda: defaultdict(lambda: {"total": 0, "valid": 0, "exec_correct": 0, "exact_match": 0}))

    # Error analysis
    syntax_errors: int = 0
    schema_errors: int = 0
    execution_errors: int = 0

    @property
    def validity_rate(self) -> float:
        return self.valid / self.total * 100 if self.total > 0 else 0

    @property
    def exec_accuracy(self) -> float:
        return self.exec_correct / self.total * 100 if self.total > 0 else 0

    @property
    def exact_match_rate(self) -> float:
        return self.exact_match / self.total * 100 if self.total > 0 else 0

    def to_dict(self) -> Dict:
        return {
            "total": self.total,
            "valid": self.valid,
            "exec_correct": self.exec_correct,
            "exact_match": self.exact_match,
            "validity_rate": round(self.validity_rate, 2),
            "exec_accuracy": round(self.exec_accuracy, 2),
            "exact_match_rate": round(self.exact_match_rate, 2),
            "syntax_errors": self.syntax_errors,
            "schema_errors": self.schema_errors,
            "execution_errors": self.execution_errors,
            "by_difficulty": dict(self.by_difficulty)
        }


class SQLNormalizer:
    """Normalizes SQL queries for comparison."""

    @staticmethod
    def normalize(sql: str) -> str:
        """Normalize SQL for exact match comparison."""
        if not sql:
            return ""

        # Convert to lowercase
        sql = sql.lower()

        # Remove extra whitespace
        sql = " ".join(sql.split())

        # Standardize quotes
        sql = sql.replace('"', "'")

        # Remove trailing semicolon
        sql = sql.rstrip(";").strip()

        # Standardize operators
        sql = re.sub(r'\s*=\s*', ' = ', sql)
        sql = re.sub(r'\s*<\s*', ' < ', sql)
        sql = re.sub(r'\s*>\s*', ' > ', sql)
        sql = re.sub(r'\s*<=\s*', ' <= ', sql)
        sql = re.sub(r'\s*>=\s*', ' >= ', sql)
        sql = re.sub(r'\s*<>\s*', ' <> ', sql)
        sql = re.sub(r'\s*!=\s*', ' != ', sql)

        # Standardize keywords
        keywords = ['select', 'from', 'where', 'and', 'or', 'not', 'in', 'like',
                   'between', 'is', 'null', 'order', 'by', 'group', 'having',
                   'limit', 'join', 'inner', 'left', 'right', 'outer', 'on',
                   'as', 'distinct', 'count', 'sum', 'avg', 'min', 'max', 'asc', 'desc']

        for kw in keywords:
            sql = re.sub(rf'\b{kw}\b', kw, sql, flags=re.IGNORECASE)

        # Clean up whitespace again
        sql = " ".join(sql.split())

        return sql


class SQLValidator:
    """Validates SQL syntax without execution."""

    @staticmethod
    def is_valid_syntax(sql: str) -> Tuple[bool, str]:
        """
        Check if SQL has valid syntax.

        Returns:
            (is_valid, error_message)
        """
        if not sql or not sql.strip():
            return False, "Empty query"

        sql = sql.strip()

        # Basic structure checks
        sql_upper = sql.upper()

        if not sql_upper.startswith("SELECT"):
            return False, "Query must start with SELECT"

        if "FROM" not in sql_upper:
            return False, "Query missing FROM clause"

        # Check for balanced parentheses
        if sql.count("(") != sql.count(")"):
            return False, "Unbalanced parentheses"

        # Check for balanced quotes
        single_quotes = sql.count("'")
        if single_quotes % 2 != 0:
            return False, "Unbalanced single quotes"

        return True, ""


class SQLExecutor:
    """Executes SQL queries against SQLite databases."""

    def __init__(self, database_dir: str):
        """
        Initialize executor with path to database directory.

        Args:
            database_dir: Path to Spider's database/ directory
        """
        self.db_dir = Path(database_dir)

    def get_db_path(self, db_id: str) -> Path:
        """Get path to database file."""
        return self.db_dir / db_id / f"{db_id}.sqlite"

    def execute(self, sql: str, db_id: str, timeout: float = 30.0) -> Tuple[bool, Any, str]:
        """
        Execute SQL query and return results.

        Args:
            sql: SQL query to execute
            db_id: Database identifier
            timeout: Execution timeout in seconds

        Returns:
            (success, results, error_message)
        """
        db_path = self.get_db_path(db_id)

        if not db_path.exists():
            return False, None, f"Database not found: {db_path}"

        try:
            conn = sqlite3.connect(str(db_path), timeout=timeout)
            conn.text_factory = str
            cursor = conn.cursor()

            cursor.execute(sql)
            results = cursor.fetchall()

            conn.close()
            return True, results, ""

        except sqlite3.Error as e:
            return False, None, str(e)
        except Exception as e:
            return False, None, f"Unexpected error: {str(e)}"

    def compare_results(self, pred_results: Any, gold_results: Any) -> bool:
        """
        Compare execution results for equality.

        Uses set comparison to handle different row orderings.
        """
        if pred_results is None or gold_results is None:
            return False

        # Convert to sets of tuples for order-independent comparison
        try:
            pred_set = set(tuple(row) for row in pred_results)
            gold_set = set(tuple(row) for row in gold_results)
            return pred_set == gold_set
        except (TypeError, ValueError):
            # Fallback to direct comparison
            return pred_results == gold_results


class SpiderEvaluator:
    """Main evaluator for Spider dataset."""

    def __init__(
        self,
        schema_loader: SpiderSchemaLoader,
        data_loader: SpiderDatasetLoader,
        database_dir: str,
        verbose: bool = False
    ):
        """
        Initialize evaluator.

        Args:
            schema_loader: Loaded Spider schemas
            data_loader: Loaded Spider examples
            database_dir: Path to Spider's database/ directory
            verbose: Print detailed progress
        """
        self.schema_loader = schema_loader
        self.data_loader = data_loader
        self.executor = SQLExecutor(database_dir)
        self.normalizer = SQLNormalizer()
        self.validator = SQLValidator()
        self.verbose = verbose

    def evaluate_single(
        self,
        idx: int,
        predicted_sql: str,
        example: Dict
    ) -> EvaluationResult:
        """
        Evaluate a single prediction.

        Args:
            idx: Example index
            predicted_sql: Model's predicted SQL
            example: Original example from dataset

        Returns:
            EvaluationResult with all metrics
        """
        db_id = example['db_id']
        question = example['question']
        gold_sql = example['query']
        difficulty = example.get('difficulty', 'unknown')

        result = EvaluationResult(
            idx=idx,
            db_id=db_id,
            question=question,
            gold_sql=gold_sql,
            predicted_sql=predicted_sql,
            is_valid=False,
            is_exec_correct=False,
            is_exact_match=False,
            difficulty=difficulty
        )

        # Check validity
        is_valid, error_msg = self.validator.is_valid_syntax(predicted_sql)
        result.is_valid = is_valid
        if not is_valid:
            result.error_message = f"Syntax error: {error_msg}"
            return result

        # Check exact match
        norm_pred = self.normalizer.normalize(predicted_sql)
        norm_gold = self.normalizer.normalize(gold_sql)
        result.is_exact_match = (norm_pred == norm_gold)

        # Check execution accuracy
        start_time = time.time()
        pred_success, pred_results, pred_error = self.executor.execute(predicted_sql, db_id)
        gold_success, gold_results, gold_error = self.executor.execute(gold_sql, db_id)
        result.execution_time = time.time() - start_time

        if not pred_success:
            result.error_message = f"Execution error: {pred_error}"
            return result

        if not gold_success:
            result.error_message = f"Gold query execution error: {gold_error}"
            # Still mark as valid since our query ran
            return result

        result.is_exec_correct = self.executor.compare_results(pred_results, gold_results)

        return result

    def evaluate_predictions(
        self,
        predictions: List[str],
        max_examples: Optional[int] = None
    ) -> Tuple[EvaluationMetrics, List[EvaluationResult]]:
        """
        Evaluate a list of predictions.

        Args:
            predictions: List of predicted SQL queries
            max_examples: Maximum examples to evaluate (None for all)

        Returns:
            (metrics, list of individual results)
        """
        metrics = EvaluationMetrics()
        results = []

        n_examples = len(self.data_loader)
        if max_examples:
            n_examples = min(n_examples, max_examples)

        if len(predictions) < n_examples:
            print(f"Warning: Only {len(predictions)} predictions for {n_examples} examples")
            n_examples = len(predictions)

        for idx in range(n_examples):
            example = self.data_loader.get_example(idx)
            pred_sql = predictions[idx]

            result = self.evaluate_single(idx, pred_sql, example)
            results.append(result)

            # Update metrics
            metrics.total += 1
            difficulty = result.difficulty

            metrics.by_difficulty[difficulty]["total"] += 1

            if result.is_valid:
                metrics.valid += 1
                metrics.by_difficulty[difficulty]["valid"] += 1
            else:
                if "syntax" in result.error_message.lower():
                    metrics.syntax_errors += 1
                elif "schema" in result.error_message.lower():
                    metrics.schema_errors += 1
                else:
                    metrics.execution_errors += 1

            if result.is_exec_correct:
                metrics.exec_correct += 1
                metrics.by_difficulty[difficulty]["exec_correct"] += 1

            if result.is_exact_match:
                metrics.exact_match += 1
                metrics.by_difficulty[difficulty]["exact_match"] += 1

            if self.verbose and (idx + 1) % 100 == 0:
                print(f"Evaluated {idx + 1}/{n_examples} examples...")

        return metrics, results


def save_results(
    metrics: EvaluationMetrics,
    results: List[EvaluationResult],
    output_dir: str,
    model_name: str
):
    """Save evaluation results to files."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save metrics
    metrics_file = output_path / f"{model_name}_metrics.json"
    with open(metrics_file, 'w') as f:
        json.dump(metrics.to_dict(), f, indent=2)
    print(f"Saved metrics to: {metrics_file}")

    # Save detailed results
    results_file = output_path / f"{model_name}_results.json"
    results_data = [
        {
            "idx": r.idx,
            "db_id": r.db_id,
            "question": r.question,
            "gold_sql": r.gold_sql,
            "predicted_sql": r.predicted_sql,
            "is_valid": r.is_valid,
            "is_exec_correct": r.is_exec_correct,
            "is_exact_match": r.is_exact_match,
            "error_message": r.error_message,
            "difficulty": r.difficulty
        }
        for r in results
    ]
    with open(results_file, 'w') as f:
        json.dump(results_data, f, indent=2)
    print(f"Saved detailed results to: {results_file}")

    # Save error analysis
    errors_file = output_path / f"{model_name}_errors.json"
    errors = [r for r in results_data if not r["is_exec_correct"]]
    with open(errors_file, 'w') as f:
        json.dump(errors, f, indent=2)
    print(f"Saved error analysis to: {errors_file}")


def print_metrics(metrics: EvaluationMetrics, model_name: str):
    """Print formatted metrics summary."""
    print("\n" + "=" * 60)
    print(f"  Evaluation Results: {model_name}")
    print("=" * 60)

    print(f"\n  Total Examples:     {metrics.total}")
    print(f"\n  Overall Metrics:")
    print(f"    Validity Rate:    {metrics.validity_rate:.2f}%  ({metrics.valid}/{metrics.total})")
    print(f"    Exec Accuracy:    {metrics.exec_accuracy:.2f}%  ({metrics.exec_correct}/{metrics.total})")
    print(f"    Exact Match:      {metrics.exact_match_rate:.2f}%  ({metrics.exact_match}/{metrics.total})")

    print(f"\n  Error Breakdown:")
    print(f"    Syntax Errors:    {metrics.syntax_errors}")
    print(f"    Schema Errors:    {metrics.schema_errors}")
    print(f"    Execution Errors: {metrics.execution_errors}")

    if metrics.by_difficulty:
        print(f"\n  By Difficulty:")
        for diff, stats in sorted(metrics.by_difficulty.items()):
            if stats["total"] > 0:
                ex_acc = stats["exec_correct"] / stats["total"] * 100
                print(f"    {diff:12s}: {ex_acc:5.1f}% EX  ({stats['exec_correct']}/{stats['total']})")

    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Evaluate SQL predictions on Spider dataset")
    parser.add_argument("--predictions", required=True, help="Path to predictions JSON file")
    parser.add_argument("--tables", required=True, help="Path to Spider tables.json")
    parser.add_argument("--data", required=True, help="Path to Spider dev.json or train.json")
    parser.add_argument("--database-dir", required=True, help="Path to Spider database/ directory")
    parser.add_argument("--output-dir", default="./results", help="Output directory for results")
    parser.add_argument("--model-name", default="sqlformer", help="Model name for output files")
    parser.add_argument("--max-examples", type=int, help="Maximum examples to evaluate")
    parser.add_argument("--verbose", action="store_true", help="Print detailed progress")

    args = parser.parse_args()

    # Load data
    print("Loading schemas...")
    schema_loader = SpiderSchemaLoader(args.tables)
    print(f"Loaded {len(schema_loader.get_all_db_ids())} database schemas")

    print("Loading examples...")
    data_loader = SpiderDatasetLoader(args.data)
    print(f"Loaded {len(data_loader)} examples")

    print("Loading predictions...")
    with open(args.predictions, 'r') as f:
        predictions = json.load(f)
    print(f"Loaded {len(predictions)} predictions")

    # Evaluate
    evaluator = SpiderEvaluator(
        schema_loader=schema_loader,
        data_loader=data_loader,
        database_dir=args.database_dir,
        verbose=args.verbose
    )

    print("\nEvaluating...")
    metrics, results = evaluator.evaluate_predictions(
        predictions,
        max_examples=args.max_examples
    )

    # Print and save results
    print_metrics(metrics, args.model_name)
    save_results(metrics, results, args.output_dir, args.model_name)


if __name__ == "__main__":
    main()
