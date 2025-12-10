"""
BIRD Dataset Schema Loader

Parses BIRD's dataset format and converts it to SQLFormer's schema format.

BIRD dataset structure:
- dev/dev.json: Development examples with questions and gold SQL
- dev/dev_databases/: SQLite database files for each database
- train/train.json: Training examples
- train/train_databases/: SQLite databases for training

BIRD JSON example format:
{
    "question_id": 0,
    "db_id": "california_schools",
    "question": "What is the highest eligible free rate for K-12 students...",
    "evidence": "Eligible free rate for K-12 = `Free Meal Count (K-12)` / `Enrollment (K-12)`",
    "SQL": "SELECT `Free Meal Count (K-12)` / `Enrollment (K-12)` FROM frpm ORDER BY ...",
    "difficulty": "simple"
}

BIRD uses SQLite databases directly - we extract schema from them.
"""

import json
import sqlite3
from typing import Dict, List, Optional, Tuple
from pathlib import Path


class BIRDSchemaLoader:
    """Loads and converts BIRD dataset schemas to SQLFormer format."""

    def __init__(self, database_dir: str):
        """
        Initialize with path to BIRD's database directory.

        Args:
            database_dir: Path to dev_databases/ or train_databases/ directory
        """
        self.database_dir = Path(database_dir)
        self.schemas: Dict[str, Dict] = {}
        self._load_all_schemas()

    def _load_all_schemas(self):
        """Load all database schemas from SQLite files."""
        if not self.database_dir.exists():
            raise FileNotFoundError(f"Database directory not found: {self.database_dir}")

        # Each subdirectory is a database
        for db_path in self.database_dir.iterdir():
            if db_path.is_dir():
                db_id = db_path.name
                # Find the SQLite file (usually named {db_id}.sqlite)
                sqlite_files = list(db_path.glob("*.sqlite"))
                if sqlite_files:
                    sqlite_path = sqlite_files[0]
                    try:
                        self.schemas[db_id] = self._extract_schema_from_sqlite(sqlite_path)
                    except Exception as e:
                        print(f"Warning: Failed to load schema for {db_id}: {e}")

    def _extract_schema_from_sqlite(self, sqlite_path: Path) -> Dict:
        """
        Extract schema from SQLite database file.

        Returns schema in SQLFormer format:
        {
            "table_name": {
                "columns": ["col1", "col2"],
                "foreign_keys": ["other_table"]
            }
        }
        """
        schema = {}

        conn = sqlite3.connect(str(sqlite_path))
        cursor = conn.cursor()

        try:
            # Get all tables
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
            tables = [row[0] for row in cursor.fetchall()]

            for table_name in tables:
                table_lower = table_name.lower()
                schema[table_lower] = {
                    "columns": [],
                    "foreign_keys": [],
                    "original_name": table_name  # Preserve original case
                }

                # Get columns for this table
                cursor.execute(f"PRAGMA table_info('{table_name}');")
                columns = cursor.fetchall()
                for col in columns:
                    col_name = col[1]  # Column name is at index 1
                    schema[table_lower]["columns"].append(col_name.lower())

                # Get foreign keys for this table
                cursor.execute(f"PRAGMA foreign_key_list('{table_name}');")
                fks = cursor.fetchall()
                for fk in fks:
                    ref_table = fk[2].lower()  # Referenced table is at index 2
                    if ref_table not in schema[table_lower]["foreign_keys"]:
                        schema[table_lower]["foreign_keys"].append(ref_table)

        finally:
            conn.close()

        return schema

    def get_schema(self, db_id: str) -> Optional[Dict]:
        """
        Get schema for a specific database.

        Args:
            db_id: Database identifier (e.g., "california_schools")

        Returns:
            Schema in SQLFormer format or None if not found
        """
        return self.schemas.get(db_id)

    def get_all_db_ids(self) -> List[str]:
        """Get list of all available database IDs."""
        return list(self.schemas.keys())

    def get_schema_info(self, db_id: str) -> str:
        """Get human-readable schema information for prompting."""
        schema = self.get_schema(db_id)
        if schema is None:
            return ""

        lines = [f"Database: {db_id}", "Tables:"]
        for table_name, table_info in schema.items():
            cols = ", ".join(table_info["columns"])
            lines.append(f"  {table_name}: {cols}")
            if table_info["foreign_keys"]:
                fks = ", ".join(table_info["foreign_keys"])
                lines.append(f"    -> references: {fks}")

        return "\n".join(lines)

    def get_database_path(self, db_id: str) -> Optional[Path]:
        """Get the path to the SQLite database file for a given db_id."""
        db_path = self.database_dir / db_id
        if db_path.exists():
            sqlite_files = list(db_path.glob("*.sqlite"))
            if sqlite_files:
                return sqlite_files[0]
        return None


class BIRDDatasetLoader:
    """Loads BIRD dataset examples (questions, evidence, and gold SQL)."""

    def __init__(self, data_json_path: str):
        """
        Initialize with path to BIRD's dev.json or train.json.

        Args:
            data_json_path: Path to data file
        """
        self.data_path = Path(data_json_path)
        self.examples: List[Dict] = []
        self._load_data()

    def _load_data(self):
        """Load all examples from the data file."""
        with open(self.data_path, 'r') as f:
            self.examples = json.load(f)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict:
        return self.examples[idx]

    def __iter__(self):
        return iter(self.examples)

    def get_example(self, idx: int) -> Dict:
        """
        Get a single example.

        Returns:
            {
                "question_id": int,
                "db_id": str,
                "question": str,
                "evidence": str (external knowledge hint),
                "SQL": str (gold SQL),
                "difficulty": str (simple, moderate, challenging)
            }
        """
        example = self.examples[idx]
        # Normalize field names to match Spider format
        return {
            "question_id": example.get("question_id", idx),
            "db_id": example["db_id"],
            "question": example["question"],
            "evidence": example.get("evidence", ""),
            "query": example.get("SQL", example.get("sql", "")),  # Handle both cases
            "difficulty": example.get("difficulty", "unknown")
        }

    def get_examples_by_db(self, db_id: str) -> List[Dict]:
        """Get all examples for a specific database."""
        return [self.get_example(i) for i, ex in enumerate(self.examples)
                if ex['db_id'] == db_id]

    def get_difficulty_distribution(self) -> Dict[str, int]:
        """Get count of examples by difficulty level."""
        distribution = {}
        for ex in self.examples:
            difficulty = ex.get('difficulty', 'unknown')
            distribution[difficulty] = distribution.get(difficulty, 0) + 1
        return distribution

    def get_examples_by_difficulty(self, difficulty: str) -> List[Dict]:
        """Get all examples of a specific difficulty."""
        return [self.get_example(i) for i, ex in enumerate(self.examples)
                if ex.get('difficulty', 'unknown') == difficulty]


class BIRDEvaluator:
    """
    Evaluator for BIRD dataset.

    BIRD uses two metrics:
    1. Execution Accuracy (EX): Whether the predicted SQL returns correct results
    2. Valid Efficiency Score (VES): Considers execution time efficiency
    """

    def __init__(self, database_dir: str):
        """
        Initialize evaluator.

        Args:
            database_dir: Path to database directory
        """
        self.database_dir = Path(database_dir)

    def execute_sql(self, db_id: str, sql: str, timeout: float = 30.0) -> Tuple[bool, any, str]:
        """
        Execute SQL query on the database.

        Args:
            db_id: Database identifier
            sql: SQL query to execute
            timeout: Timeout in seconds

        Returns:
            (success, results, error_message)
        """
        db_path = self.database_dir / db_id
        sqlite_files = list(db_path.glob("*.sqlite"))

        if not sqlite_files:
            return False, None, f"Database not found: {db_id}"

        sqlite_path = sqlite_files[0]

        try:
            conn = sqlite3.connect(str(sqlite_path), timeout=timeout)
            conn.text_factory = lambda x: x.decode('utf-8', errors='ignore')
            cursor = conn.cursor()

            cursor.execute(sql)
            results = cursor.fetchall()

            conn.close()
            return True, results, ""

        except Exception as e:
            return False, None, str(e)

    def compare_results(self, gold_results: any, pred_results: any) -> bool:
        """
        Compare gold and predicted results.

        BIRD uses set comparison (order doesn't matter).
        """
        if gold_results is None or pred_results is None:
            return False

        try:
            # Convert to sets of tuples for comparison
            gold_set = set(tuple(row) if isinstance(row, (list, tuple)) else (row,)
                          for row in gold_results)
            pred_set = set(tuple(row) if isinstance(row, (list, tuple)) else (row,)
                          for row in pred_results)
            return gold_set == pred_set
        except Exception:
            return False

    def evaluate_single(self, db_id: str, gold_sql: str, pred_sql: str) -> Dict:
        """
        Evaluate a single prediction.

        Returns:
            {
                "is_valid": bool,
                "is_correct": bool,
                "error_message": str,
                "gold_results": any,
                "pred_results": any
            }
        """
        # Execute gold SQL
        gold_success, gold_results, gold_error = self.execute_sql(db_id, gold_sql)

        if not gold_success:
            return {
                "is_valid": False,
                "is_correct": False,
                "error_message": f"Gold SQL failed: {gold_error}",
                "gold_results": None,
                "pred_results": None
            }

        # Execute predicted SQL
        pred_success, pred_results, pred_error = self.execute_sql(db_id, pred_sql)

        if not pred_success:
            return {
                "is_valid": False,
                "is_correct": False,
                "error_message": f"Predicted SQL failed: {pred_error}",
                "gold_results": gold_results,
                "pred_results": None
            }

        # Compare results
        is_correct = self.compare_results(gold_results, pred_results)

        return {
            "is_valid": True,
            "is_correct": is_correct,
            "error_message": "" if is_correct else "Results do not match",
            "gold_results": gold_results,
            "pred_results": pred_results
        }


def download_bird_dataset(output_dir: str):
    """
    Print instructions for downloading BIRD dataset.

    BIRD dataset must be downloaded manually due to licensing.
    """
    print("""
BIRD Dataset Download Instructions
==================================

1. Visit the official BIRD repository:
   https://github.com/AlibabaResearch/DAMO-ConvAI/tree/main/bird

2. Download the dataset files:
   - dev.zip: Development set
   - train.zip: Training set (optional)

3. Extract to your data directory:
   mkdir -p {output_dir}/bird
   unzip dev.zip -d {output_dir}/bird/
   unzip train.zip -d {output_dir}/bird/  # optional

4. Expected structure after extraction:
   {output_dir}/bird/
   ├── dev/
   │   ├── dev.json
   │   └── dev_databases/
   │       ├── california_schools/
   │       │   └── california_schools.sqlite
   │       ├── card_games/
   │       │   └── card_games.sqlite
   │       └── ...
   └── train/  (optional)
       ├── train.json
       └── train_databases/
           └── ...

5. Run experiments:
   python run_experiment.py \\
       --data {output_dir}/bird/dev/dev.json \\
       --database-dir {output_dir}/bird/dev/dev_databases \\
       --methods hybrid \\
       --max-examples 10
""".format(output_dir=output_dir))


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python bird_schema_loader.py <path_to_database_dir>")
        print("\nExample:")
        print("  python bird_schema_loader.py ./data/bird/dev/dev_databases")
        print("\nTo download BIRD dataset:")
        download_bird_dataset("./data")
        sys.exit(1)

    database_dir = sys.argv[1]

    try:
        loader = BIRDSchemaLoader(database_dir)
        print(f"Loaded {len(loader.get_all_db_ids())} database schemas")
        print("\nAvailable databases:")
        for db_id in loader.get_all_db_ids()[:10]:
            print(f"  - {db_id}")
        if len(loader.get_all_db_ids()) > 10:
            print(f"  ... and {len(loader.get_all_db_ids()) - 10} more")

        # Show example schema
        if loader.get_all_db_ids():
            example_db = loader.get_all_db_ids()[0]
            print(f"\nExample schema ({example_db}):")
            print(loader.get_schema_info(example_db))
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nTo download BIRD dataset:")
        download_bird_dataset("./data")
