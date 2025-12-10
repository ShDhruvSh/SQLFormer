"""
Spider Experiment Runner

Runs SQL generation models on Spider dataset and collects predictions.
Supports:
- Hybrid generation (generate + parse + validate + repair)
- Constrained decoding (grammar-constrained generation)
- Unconstrained LLM baseline
"""

import json
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from schema.spider_loader import SpiderSchemaLoader, SpiderDatasetLoader
from generators.unconstrained import UnconstrainedSQLFormerEngine
from generators.constrained import ConstrainedSQLFormerEngine
from generators.hybrid import HybridSQLFormerEngine
from models.model_loader import SQLFormerModel
from utils.sql_utils import extract_sql_from_text


class ExperimentRunner:
    """Runs experiments on Spider dataset."""

    def __init__(
        self,
        spider_schema_loader: SpiderSchemaLoader,
        spider_data_loader: SpiderDatasetLoader,
        model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct",
        device: str = "auto"
    ):
        """
        Initialize experiment runner.

        Args:
            spider_schema_loader: Loaded Spider schemas
            spider_data_loader: Loaded Spider examples
            model_name: HuggingFace model identifier
            device: Device to run model on ("auto", "cuda", "cpu")
        """
        self.spider_schemas = spider_schema_loader
        self.data_loader = spider_data_loader
        self.model_name = model_name
        self.device = device
        self.model = None

    def load_model(self):
        """Load the LLM model."""
        if self.model is None:
            print(f"Loading model: {self.model_name}")
            # Convert "auto" to None for automatic device selection
            device = None if self.device == "auto" else self.device
            self.model = SQLFormerModel(
                model_name=self.model_name,
                device=device
            )
            print(f"Model loaded on device: {self.model.device}")

    def _build_constrained_engine(self, db_id: str) -> ConstrainedSQLFormerEngine:
        """Build constrained decoder engine for a specific database."""
        schema = self.spider_schemas.get_schema(db_id)

        engine = ConstrainedSQLFormerEngine(
            model=self.model,
            schema=schema
        )

        return engine

    def _build_hybrid_engine(self, db_id: str) -> HybridSQLFormerEngine:
        """Build hybrid generator engine for a specific database."""
        schema = self.spider_schemas.get_schema(db_id)

        engine = HybridSQLFormerEngine(
            model=self.model,
            schema=schema
        )

        return engine

    def _format_prompt(self, question: str, schema_info: str) -> str:
        """Format prompt for the model."""
        return f"""Given the following database schema:
{schema_info}

Generate a SQL query for the following question:
{question}

SQL:"""

    def run_hybrid(
        self,
        max_examples: Optional[int] = None,
        output_file: Optional[str] = None
    ) -> List[str]:
        """
        Run hybrid generation (unconstrained + parse-and-repair).

        This method:
        1. Lets LLM generate SQL freely (preserves capability)
        2. Parses SQL with sqlparse (bulletproof syntax checking)
        3. Validates against schema with AST traversal
        4. Repairs errors with targeted retry

        Args:
            max_examples: Maximum examples to process
            output_file: Path to save predictions

        Returns:
            List of predicted SQL queries
        """
        self.load_model()

        predictions = []
        n_examples = len(self.data_loader) if max_examples is None else min(max_examples, len(self.data_loader))

        print(f"\nRunning Hybrid Generation on {n_examples} examples...")

        for idx in tqdm(range(n_examples), desc="Hybrid"):
            example = self.data_loader.get_example(idx)
            db_id = example['db_id']
            question = example['question']

            try:
                # Build hybrid engine for this database
                engine = self._build_hybrid_engine(db_id)

                # Get schema info for prompt
                schema_info = self.spider_schemas.get_schema_info(db_id)
                prompt = self._format_prompt(question, schema_info)

                # Generate SQL with hybrid approach
                predicted_sql = engine.generate(prompt)
                predictions.append(predicted_sql)

            except Exception as e:
                print(f"\nError on example {idx} ({db_id}): {e}")
                import traceback
                traceback.print_exc()
                predictions.append("")

        if output_file:
            with open(output_file, 'w') as f:
                json.dump(predictions, f, indent=2)
            print(f"Saved predictions to: {output_file}")

        return predictions

    def run_constrained(
        self,
        max_examples: Optional[int] = None,
        output_file: Optional[str] = None
    ) -> List[str]:
        """
        Run constrained beam search decoding.

        This method uses grammar-constrained generation where invalid
        tokens are masked at each step.

        Args:
            max_examples: Maximum examples to process
            output_file: Path to save predictions

        Returns:
            List of predicted SQL queries
        """
        self.load_model()

        predictions = []
        n_examples = len(self.data_loader) if max_examples is None else min(max_examples, len(self.data_loader))

        print(f"\nRunning Constrained Beam Search on {n_examples} examples...")

        for idx in tqdm(range(n_examples), desc="Constrained"):
            example = self.data_loader.get_example(idx)
            db_id = example['db_id']
            question = example['question']

            try:
                # Build constrained engine for this database
                engine = self._build_constrained_engine(db_id)

                # Get schema info for prompt
                schema_info = self.spider_schemas.get_schema_info(db_id)
                prompt = self._format_prompt(question, schema_info)

                # Generate SQL with constraints
                predicted_sql = engine.generate(prompt)
                predictions.append(predicted_sql)

            except Exception as e:
                print(f"\nError on example {idx} ({db_id}): {e}")
                import traceback
                traceback.print_exc()
                predictions.append("")

        if output_file:
            with open(output_file, 'w') as f:
                json.dump(predictions, f, indent=2)
            print(f"Saved predictions to: {output_file}")

        return predictions

    def run_unconstrained(
        self,
        max_examples: Optional[int] = None,
        output_file: Optional[str] = None,
        max_new_tokens: int = 128
    ) -> List[str]:
        """
        Run unconstrained LLM generation baseline.

        Args:
            max_examples: Maximum examples to process
            output_file: Path to save predictions
            max_new_tokens: Maximum tokens to generate

        Returns:
            List of predicted SQL queries
        """
        self.load_model()

        predictions = []
        n_examples = len(self.data_loader) if max_examples is None else min(max_examples, len(self.data_loader))

        print(f"\nRunning unconstrained baseline on {n_examples} examples...")

        for idx in tqdm(range(n_examples), desc="Unconstrained"):
            example = self.data_loader.get_example(idx)
            db_id = example['db_id']
            question = example['question']

            try:
                schema_info = self.spider_schemas.get_schema_info(db_id)
                prompt = self._format_prompt(question, schema_info)

                # Use unconstrained generator for cleaner code
                engine = UnconstrainedSQLFormerEngine(self.model, {})
                predicted_sql = engine.generate(prompt)
                predictions.append(predicted_sql)

            except Exception as e:
                print(f"\nError on example {idx} ({db_id}): {e}")
                predictions.append("")

        if output_file:
            with open(output_file, 'w') as f:
                json.dump(predictions, f, indent=2)
            print(f"Saved predictions to: {output_file}")

        return predictions

def main():
    parser = argparse.ArgumentParser(description="Run SQLFormer experiments on Spider dataset")
    parser.add_argument("--tables", required=True, help="Path to Spider tables.json")
    parser.add_argument("--data", required=True, help="Path to Spider dev.json or train.json")
    parser.add_argument("--output-dir", default="./predictions", help="Output directory for predictions")
    parser.add_argument("--model", default="meta-llama/Meta-Llama-3-8B-Instruct", help="Model name")
    parser.add_argument("--device", default="auto", help="Device (auto, cuda, cpu)")
    parser.add_argument("--max-examples", type=int, help="Maximum examples to process")
    parser.add_argument("--methods", nargs="+", default=["hybrid"],
                        choices=["hybrid", "constrained", "unconstrained"],
                        help="Methods to run")

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("Loading schemas...")
    schema_loader = SpiderSchemaLoader(args.tables)
    print(f"Loaded {len(schema_loader.get_all_db_ids())} database schemas")

    print("Loading examples...")
    data_loader = SpiderDatasetLoader(args.data)
    print(f"Loaded {len(data_loader)} examples")

    diff_dist = data_loader.get_difficulty_distribution()
    print(f"Difficulty distribution: {diff_dist}")

    # Create runner
    runner = ExperimentRunner(
        spider_schema_loader=schema_loader,
        spider_data_loader=data_loader,
        model_name=args.model,
        device=args.device
    )

    # Run experiments
    results = {}

    if "hybrid" in args.methods:
        print("\n" + "=" * 50)
        print("Running Hybrid Generation (Parse-and-Repair)")
        print("=" * 50)
        hybrid_preds = runner.run_hybrid(
            max_examples=args.max_examples,
            output_file=output_dir / "hybrid_predictions.json"
        )
        results["hybrid"] = len(hybrid_preds)

    if "constrained" in args.methods:
        print("\n" + "=" * 50)
        print("Running Constrained Beam Search")
        print("=" * 50)
        constrained_preds = runner.run_constrained(
            max_examples=args.max_examples,
            output_file=output_dir / "constrained_predictions.json"
        )
        results["constrained"] = len(constrained_preds)

    if "unconstrained" in args.methods:
        print("\n" + "=" * 50)
        print("Running Unconstrained Baseline")
        print("=" * 50)
        unconstrained_preds = runner.run_unconstrained(
            max_examples=args.max_examples,
            output_file=output_dir / "unconstrained_predictions.json"
        )
        results["unconstrained"] = len(unconstrained_preds)

    # Summary
    print("\n" + "=" * 50)
    print("Experiment Complete")
    print("=" * 50)
    for method, count in results.items():
        print(f"  {method}: {count} predictions saved")
    print(f"\nPredictions saved to: {output_dir}")
    print("\nNext step: Run evaluation with evaluate.py")


if __name__ == "__main__":
    main()
