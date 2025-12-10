import json
import sqlite3
from typing import Dict, List, Optional, Tuple
from pathlib import Path


class BIRDSchemaLoader:

    def __init__(self, database_dir: str):
        self.database_dir = Path(database_dir)
        self.schemas: Dict[str, Dict] = {}
        self._load_all_schemas()

    def _load_all_schemas(self):
        if not self.database_dir.exists():
            raise FileNotFoundError(f"Database directory not found: {self.database_dir}")

        for db_path in self.database_dir.iterdir():
            if db_path.is_dir():
                db_id = db_path.name
                sqlite_files = list(db_path.glob("*.sqlite"))
                if sqlite_files:
                    try:
                        self.schemas[db_id] = self._extract_schema_from_sqlite(sqlite_files[0])
                    except Exception as e:
                        print(f"Failed to load {db_id}: {e}")

    # extract table and column info from sqlite db
    def _extract_schema_from_sqlite(self, sqlite_path: Path) -> Dict:
        schema = {}
        conn = sqlite3.connect(str(sqlite_path))
        cursor = conn.cursor()

        try:
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
            tables = [row[0] for row in cursor.fetchall()]

            for table_name in tables:
                table_lower = table_name.lower()
                schema[table_lower] = {"columns": [], "foreign_keys": []}

                cursor.execute(f"PRAGMA table_info('{table_name}');")
                for col in cursor.fetchall():
                    schema[table_lower]["columns"].append(col[1].lower())

                cursor.execute(f"PRAGMA foreign_key_list('{table_name}');")
                for fk in cursor.fetchall():
                    ref_table = fk[2].lower()
                    if ref_table not in schema[table_lower]["foreign_keys"]:
                        schema[table_lower]["foreign_keys"].append(ref_table)
        finally:
            conn.close()

        return schema

    def get_schema(self, db_id: str) -> Optional[Dict]:
        return self.schemas.get(db_id)

    def get_all_db_ids(self) -> List[str]:
        return list(self.schemas.keys())

    def get_database_path(self, db_id: str) -> Optional[Path]:
        db_path = self.database_dir / db_id
        if db_path.exists():
            sqlite_files = list(db_path.glob("*.sqlite"))
            return sqlite_files[0] if sqlite_files else None
        return None


class BIRDDatasetLoader:

    def __init__(self, data_json_path: str):
        self.data_path = Path(data_json_path)
        with open(self.data_path, 'r') as f:
            self.examples = json.load(f)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx: int):
        return self.examples[idx]

    # normalize bird format to match spider
    def get_example(self, idx: int) -> Dict:
        ex = self.examples[idx]
        return {
            "question_id": ex.get("question_id", idx),
            "db_id": ex["db_id"],
            "question": ex["question"],
            "evidence": ex.get("evidence", ""),
            "query": ex.get("SQL", ex.get("sql", "")),
            "difficulty": ex.get("difficulty", "unknown")
        }


class BIRDEvaluator:

    def __init__(self, database_dir: str):
        self.database_dir = Path(database_dir)

    # run sql query and return results
    def execute_sql(self, db_id: str, sql: str, timeout: float = 30.0) -> Tuple[bool, any, str]:
        db_path = self.database_dir / db_id
        sqlite_files = list(db_path.glob("*.sqlite"))

        if not sqlite_files:
            return False, None, f"Database not found: {db_id}"

        try:
            conn = sqlite3.connect(str(sqlite_files[0]), timeout=timeout)
            cursor = conn.cursor()
            cursor.execute(sql)
            results = cursor.fetchall()
            conn.close()
            return True, results, ""
        except Exception as e:
            return False, None, str(e)

    # check if query results match
    def compare_results(self, gold_results: any, pred_results: any) -> bool:
        if gold_results is None or pred_results is None:
            return False
        try:
            gold_set = set(tuple(row) if isinstance(row, (list, tuple)) else (row,) for row in gold_results)
            pred_set = set(tuple(row) if isinstance(row, (list, tuple)) else (row,) for row in pred_results)
            return gold_set == pred_set
        except:
            return False

    # evaluate predicted sql against gold
    def evaluate_single(self, db_id: str, gold_sql: str, pred_sql: str) -> Dict:
        gold_success, gold_results, gold_error = self.execute_sql(db_id, gold_sql)
        if not gold_success:
            return {
                "is_valid": False,
                "is_correct": False,
                "error_message": f"Gold SQL failed: {gold_error}"
            }

        pred_success, pred_results, pred_error = self.execute_sql(db_id, pred_sql)
        if not pred_success:
            return {
                "is_valid": False,
                "is_correct": False,
                "error_message": f"Predicted SQL failed: {pred_error}"
            }

        is_correct = self.compare_results(gold_results, pred_results)
        return {
            "is_valid": True,
            "is_correct": is_correct,
            "error_message": "" if is_correct else "Results do not match"
        }


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python bird_loader.py <database_dir>")
        sys.exit(1)

    loader = BIRDSchemaLoader(sys.argv[1])
    print(f"Loaded {len(loader.get_all_db_ids())} databases")
