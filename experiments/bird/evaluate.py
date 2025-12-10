"""
BIRD Evaluation Script for SQLFormer

Evaluates SQLFormer on the BIRD dataset with standard metrics:
- Validity Rate (VR): % of syntactically valid SQL queries
- Execution Accuracy (EX): % of queries returning correct results
- Valid Efficiency Score (VES): EX weighted by execution efficiency (optional)

BIRD evaluation differs from Spider:
1. Uses set comparison (order doesn't matter)
2. Case-insensitive string comparison
3. Handles NULL values specially
4. Reports by difficulty level (simple, moderate, challenging)
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

from schema.bird_loader import BIRDSchemaLoader, BIRDDatasetLoader


@dataclass
class EvaluationResult:
    """Single evaluation result."""
    idx: int
    db_id: str
    question: str
    evidence: str
    gold_sql: str
    predicted_sql: str
    is_valid: bool
    is_exec_correct: bool
    error_message: str = ""
    execution_time: float = 0.0
    difficulty: str = "unknown"


@dataclass
class EvaluationMetrics:
    """Aggregate evaluation metrics."""
    total: int = 0
    valid: int = 0
    exec_correct: int = 0

    # By difficulty
    by_difficulty: Dict[str, Dict[str, int]] = field(default_factory=lambda: defaultdict(lambda: {"total": 0, "valid": 0, "exec_correct": 0}))

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

    def to_dict(self) -> Dict:
        return {
            "total": self.total,
            "valid": self.valid,
            "exec_correct": self.exec_correct,
            "validity_rate": round(self.validity_rate, 2),
            "exec_accuracy": round(self.exec_accuracy, 2),
            "syntax_errors": self.syntax_errors,
            "schema_errors": self.schema_errors,
            "execution_errors": self.execution_errors,
            "by_difficulty": dict(self.by_difficulty)
        }


class BIRDEvaluator:
    """Evaluates SQL predictions on BIRD dataset."""

    def __init__(
        self,
        schema_loader: BIRDSchemaLoader,
        data_loader: BIRDDatasetLoader,
        database_dir: str
    ):
        """
        Initialize evaluator.

        Args:
            schema_loader: Loaded BIRD schemas
            data_loader: Loaded BIRD examples
            database_dir: Path to database directory
        """
        self.schema_loader = schema_loader
        self.data_loader = data_loader
        self.database_dir = Path(database_dir)

    def get_database_path(self, db_id: str) -> Optional[Path]:
        """Get the path to the SQLite database file."""
        db_path = self.database_dir / db_id
        if db_path.exists():
            sqlite_files = list(db_path.glob("*.sqlite"))
            if sqlite_files:
                return sqlite_files[0]
        return None

    def execute_sql(self, db_id: str, sql: str, timeout: float = 30.0) -> Tuple[bool, Any, str, float]:
        """
        Execute SQL query on the database.

        Args:
            db_id: Database identifier
            sql: SQL query to execute
            timeout: Timeout in seconds

        Returns:
            (success, results, error_message, execution_time)
        """
        db_path = self.get_database_path(db_id)

        if db_path is None:
            return False, None, f"Database not found: {db_id}", 0.0

        start_time = time.time()

        try:
            conn = sqlite3.connect(str(db_path), timeout=timeout)
            conn.text_factory = lambda x: x.decode('utf-8', errors='ignore') if isinstance(x, bytes) else x
            cursor = conn.cursor()

            cursor.execute(sql)
            results = cursor.fetchall()

            execution_time = time.time() - start_time
            conn.close()

            return True, results, "", execution_time

        except sqlite3.OperationalError as e:
            return False, None, f"SQL error: {str(e)}", time.time() - start_time
        except Exception as e:
            return False, None, f"Execution error: {str(e)}", time.time() - start_time

    def normalize_value(self, val: Any) -> Any:
        """Normalize a value for comparison."""
        if val is None:
            return None
        if isinstance(val, str):
            # Case-insensitive, strip whitespace
            return val.strip().lower()
        if isinstance(val, float):
            # Round floats for comparison
            return round(val, 6)
        return val

    def normalize_results(self, results: List[Tuple]) -> set:
        """Normalize results for set comparison."""
        if results is None:
            return set()

        normalized = set()
        for row in results:
            if isinstance(row, (list, tuple)):
                normalized_row = tuple(self.normalize_value(v) for v in row)
            else:
                normalized_row = (self.normalize_value(row),)
            normalized.add(normalized_row)

        return normalized

    def compare_results(self, gold_results: Any, pred_results: Any) -> bool:
        """
        Compare gold and predicted results using BIRD's comparison rules.

        BIRD uses set comparison (order doesn't matter).
        """
        gold_set = self.normalize_results(gold_results)
        pred_set = self.normalize_results(pred_results)

        return gold_set == pred_set

    def evaluate_single(
        self,
        idx: int,
        db_id: str,
        question: str,
        evidence: str,
        gold_sql: str,
        predicted_sql: str,
        difficulty: str
    ) -> EvaluationResult:
        """Evaluate a single prediction."""

        # Check for empty prediction
        if not predicted_sql or not predicted_sql.strip():
            return EvaluationResult(
                idx=idx,
                db_id=db_id,
                question=question,
                evidence=evidence,
                gold_sql=gold_sql,
                predicted_sql=predicted_sql,
                is_valid=False,
                is_exec_correct=False,
                error_message="Empty prediction",
                difficulty=difficulty
            )

        # Execute gold SQL
        gold_success, gold_results, gold_error, _ = self.execute_sql(db_id, gold_sql)

        if not gold_success:
            return EvaluationResult(
                idx=idx,
                db_id=db_id,
                question=question,
                evidence=evidence,
                gold_sql=gold_sql,
                predicted_sql=predicted_sql,
                is_valid=False,
                is_exec_correct=False,
                error_message=f"Gold SQL failed: {gold_error}",
                difficulty=difficulty
            )

        # Execute predicted SQL
        pred_success, pred_results, pred_error, exec_time = self.execute_sql(db_id, predicted_sql)

        if not pred_success:
            # Categorize error
            error_lower = pred_error.lower()
            if "syntax" in error_lower or "near" in error_lower:
                error_type = "Syntax error"
            elif "no such table" in error_lower or "no such column" in error_lower:
                error_type = "Schema error"
            else:
                error_type = "Execution error"

            return EvaluationResult(
                idx=idx,
                db_id=db_id,
                question=question,
                evidence=evidence,
                gold_sql=gold_sql,
                predicted_sql=predicted_sql,
                is_valid=False,
                is_exec_correct=False,
                error_message=f"{error_type}: {pred_error}",
                execution_time=exec_time,
                difficulty=difficulty
            )

        # Compare results
        is_correct = self.compare_results(gold_results, pred_results)

        return EvaluationResult(
            idx=idx,
            db_id=db_id,
            question=question,
            evidence=evidence,
            gold_sql=gold_sql,
            predicted_sql=predicted_sql,
            is_valid=True,
            is_exec_correct=is_correct,
            error_message="" if is_correct else "Results do not match",
            execution_time=exec_time,
            difficulty=difficulty
        )

    def evaluate_all(
        self,
        predictions: List[str],
        max_examples: Optional[int] = None,
        verbose: bool = False
    ) -> Tuple[EvaluationMetrics, List[EvaluationResult]]:
        """
        Evaluate all predictions.

        Args:
            predictions: List of predicted SQL queries
            max_examples: Maximum examples to evaluate
            verbose: Print progress

        Returns:
            (metrics, list of results)
        """
        metrics = EvaluationMetrics()
        results = []

        n_examples = min(len(predictions), len(self.data_loader))
        if max_examples:
            n_examples = min(n_examples, max_examples)

        if len(predictions) != len(self.data_loader):
            print(f"Warning: {len(predictions)} predictions for {len(self.data_loader)} examples")

        for idx in range(n_examples):
            example = self.data_loader.get_example(idx)
            predicted_sql = predictions[idx] if idx < len(predictions) else ""

            result = self.evaluate_single(
                idx=idx,
                db_id=example['db_id'],
                question=example['question'],
                evidence=example.get('evidence', ''),
                gold_sql=example['query'],
                predicted_sql=predicted_sql,
                difficulty=example.get('difficulty', 'unknown')
            )

            results.append(result)

            # Update metrics
            metrics.total += 1
            difficulty = result.difficulty

            metrics.by_difficulty[difficulty]["total"] += 1

            if result.is_valid:
                metrics.valid += 1
                metrics.by_difficulty[difficulty]["valid"] += 1

                if result.is_exec_correct:
                    metrics.exec_correct += 1
                    metrics.by_difficulty[difficulty]["exec_correct"] += 1
            else:
                # Categorize error
                error = result.error_message.lower()
                if "syntax" in error:
                    metrics.syntax_errors += 1
                elif "schema" in error or "no such" in error:
                    metrics.schema_errors += 1
                else:
                    metrics.execution_errors += 1

            if verbose and (idx + 1) % 50 == 0:
                print(f"Evaluated {idx + 1}/{n_examples} examples...")

        return metrics, results


def print_results(metrics: EvaluationMetrics, model_name: str):
    """Print formatted evaluation results."""
    print("\n" + "=" * 60)
    print(f"  BIRD Evaluation Results: {model_name}")
    print("=" * 60)

    print(f"\n  Total Examples:     {metrics.total}")

    print(f"\n  Overall Metrics:")
    print(f"    Validity Rate:    {metrics.validity_rate:.2f}%  ({metrics.valid}/{metrics.total})")
    print(f"    Exec Accuracy:    {metrics.exec_accuracy:.2f}%  ({metrics.exec_correct}/{metrics.total})")

    print(f"\n  Error Breakdown:")
    print(f"    Syntax Errors:    {metrics.syntax_errors}")
    print(f"    Schema Errors:    {metrics.schema_errors}")
    print(f"    Execution Errors: {metrics.execution_errors}")

    print(f"\n  By Difficulty:")
    for difficulty in ["simple", "moderate", "challenging", "unknown"]:
        if difficulty in metrics.by_difficulty:
            stats = metrics.by_difficulty[difficulty]
            if stats["total"] > 0:
                ex_rate = stats["exec_correct"] / stats["total"] * 100
                print(f"    {difficulty:12}: {ex_rate:5.1f}% EX  ({stats['exec_correct']}/{stats['total']})")

    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Evaluate SQL predictions on BIRD dataset")
    parser.add_argument("--predictions", required=True, help="Path to predictions JSON file")
    parser.add_argument("--data", required=True, help="Path to BIRD dev.json or train.json")
    parser.add_argument("--database-dir", required=True, help="Path to BIRD database directory")
    parser.add_argument("--output-dir", default="./results", help="Output directory for results")
    parser.add_argument("--model-name", default="model", help="Model name for output files")
    parser.add_argument("--max-examples", type=int, help="Maximum examples to evaluate")
    parser.add_argument("--verbose", action="store_true", help="Print detailed progress")

    args = parser.parse_args()

    # Load data
    print("Loading BIRD schemas...")
    schema_loader = BIRDSchemaLoader(args.database_dir)
    print(f"Loaded {len(schema_loader.get_all_db_ids())} database schemas")

    print("Loading BIRD examples...")
    data_loader = BIRDDatasetLoader(args.data)
    print(f"Loaded {len(data_loader)} examples")

    print("Loading predictions...")
    with open(args.predictions, 'r') as f:
        predictions = json.load(f)
    print(f"Loaded {len(predictions)} predictions")

    # Create evaluator
    evaluator = BIRDEvaluator(
        schema_loader=schema_loader,
        data_loader=data_loader,
        database_dir=args.database_dir
    )

    # Evaluate
    print("\nEvaluating...")
    metrics, results = evaluator.evaluate_all(
        predictions,
        max_examples=args.max_examples,
        verbose=args.verbose
    )

    # Print results
    print_results(metrics, args.model_name)

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save metrics
    metrics_file = output_dir / f"{args.model_name}_metrics.json"
    with open(metrics_file, 'w') as f:
        json.dump(metrics.to_dict(), f, indent=2)
    print(f"Saved metrics to: {metrics_file}")

    # Save detailed results
    results_file = output_dir / f"{args.model_name}_results.json"
    results_data = [
        {
            "idx": r.idx,
            "db_id": r.db_id,
            "question": r.question,
            "evidence": r.evidence,
            "gold_sql": r.gold_sql,
            "predicted_sql": r.predicted_sql,
            "is_valid": r.is_valid,
            "is_exec_correct": r.is_exec_correct,
            "error_message": r.error_message,
            "difficulty": r.difficulty
        }
        for r in results
    ]
    with open(results_file, 'w') as f:
        json.dump(results_data, f, indent=2)
    print(f"Saved detailed results to: {results_file}")

    # Save errors only
    errors = [r for r in results_data if not r["is_exec_correct"]]
    errors_file = output_dir / f"{args.model_name}_errors.json"
    with open(errors_file, 'w') as f:
        json.dump(errors, f, indent=2)
    print(f"Saved error analysis to: {errors_file}")


if __name__ == "__main__":
    main()
