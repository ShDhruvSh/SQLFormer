import json
import sys
import argparse
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from schema.spider_loader import SpiderSchemaLoader, SpiderDatasetLoader
from generators.unconstrained import UnconstrainedSQLFormerEngine
from generators.constrained import ConstrainedSQLFormerEngine
from generators.hybrid import HybridSQLFormerEngine
from models.model_loader import SQLFormerModel


class ExperimentRunner:

    def __init__(self, spider_schema_loader, spider_data_loader, model_name="meta-llama/Meta-Llama-3-8B-Instruct", device="auto"):
        self.spider_schemas = spider_schema_loader
        self.data_loader = spider_data_loader
        self.model_name = model_name
        self.device = device
        self.model = None

    def load_model(self):
        if self.model is None:
            print(f"Loading model: {self.model_name}")
            device = None if self.device == "auto" else self.device
            self.model = SQLFormerModel(model_name=self.model_name, device=device)
            print(f"Model loaded on device: {self.model.device}")

    def _format_prompt(self, question, schema_info):
        return f"""Given the following database schema:
{schema_info}

Generate a SQL query for the following question:
{question}

SQL:"""

    def run_hybrid(self, max_examples=None, output_file=None):
        self.load_model()
        predictions = []
        n_examples = len(self.data_loader) if max_examples is None else min(max_examples, len(self.data_loader))

        print(f"\nRunning Hybrid Generation on {n_examples} examples...")

        for idx in tqdm(range(n_examples), desc="Hybrid"):
            example = self.data_loader.get_example(idx)
            db_id = example['db_id']
            question = example['question']

            try:
                schema = self.spider_schemas.get_schema(db_id)
                engine = HybridSQLFormerEngine(self.model, schema)
                schema_info = self.spider_schemas.get_schema_info(db_id)
                prompt = self._format_prompt(question, schema_info)
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

    def run_constrained(self, max_examples=None, output_file=None):
        self.load_model()
        predictions = []
        n_examples = len(self.data_loader) if max_examples is None else min(max_examples, len(self.data_loader))

        print(f"\nRunning Constrained on {n_examples} examples...")

        for idx in tqdm(range(n_examples), desc="Constrained"):
            example = self.data_loader.get_example(idx)
            db_id = example['db_id']
            question = example['question']

            try:
                schema = self.spider_schemas.get_schema(db_id)
                engine = ConstrainedSQLFormerEngine(self.model, schema)
                schema_info = self.spider_schemas.get_schema_info(db_id)
                prompt = self._format_prompt(question, schema_info)
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

    def run_unconstrained(self, max_examples=None, output_file=None, max_new_tokens=1024):
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
                engine = UnconstrainedSQLFormerEngine(self.model, {})
                predicted_sql = engine.generate(prompt, max_new_tokens=max_new_tokens)
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
    parser = argparse.ArgumentParser(description="Run experiments on Spider")
    parser.add_argument("--tables", required=True, help="Path to tables.json")
    parser.add_argument("--data", required=True, help="Path to dev.json")
    parser.add_argument("--output-dir", default="./predictions")
    parser.add_argument("--model", default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-examples", type=int)
    parser.add_argument("--methods", nargs="+", default=["hybrid"], choices=["hybrid", "constrained", "unconstrained"])

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading schemas...")
    schema_loader = SpiderSchemaLoader(args.tables)
    print(f"Loaded {len(schema_loader.get_all_db_ids())} schemas")

    print("Loading examples...")
    data_loader = SpiderDatasetLoader(args.data)
    print(f"Loaded {len(data_loader)} examples")

    runner = ExperimentRunner(schema_loader, data_loader, args.model, args.device)

    results = {}

    if "hybrid" in args.methods:
        hybrid_preds = runner.run_hybrid(args.max_examples, output_dir / "hybrid_predictions.json")
        results["hybrid"] = len(hybrid_preds)

    if "constrained" in args.methods:
        constrained_preds = runner.run_constrained(args.max_examples, output_dir / "constrained_predictions.json")
        results["constrained"] = len(constrained_preds)

    if "unconstrained" in args.methods:
        unconstrained_preds = runner.run_unconstrained(args.max_examples, output_dir / "unconstrained_predictions.json")
        results["unconstrained"] = len(unconstrained_preds)

    print("\nDone:")
    for method, count in results.items():
        print(f"  {method}: {count} predictions")


if __name__ == "__main__":
    main()
