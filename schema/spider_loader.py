"""
Spider Dataset Schema Loader

Parses Spider's tables.json format and converts it to SQLFormer's schema format.
Spider schema format:
{
    "db_id": "concert_singer",
    "table_names_original": ["stadium", "singer", "concert", "singer_in_concert"],
    "column_names_original": [[-1, "*"], [0, "Stadium_ID"], [0, "Location"], ...],
    "column_types": ["text", "number", ...],
    "foreign_keys": [[3, 0], ...],  # column index pairs
    "primary_keys": [1, 5, ...]
}
"""

import json
from typing import Dict, List, Optional
from pathlib import Path


class SpiderSchemaLoader:
    """Loads and converts Spider dataset schemas to SQLFormer format."""

    def __init__(self, tables_json_path: str):
        """
        Initialize with path to Spider's tables.json file.

        Args:
            tables_json_path: Path to tables.json from Spider dataset
        """
        self.tables_path = Path(tables_json_path)
        self.schemas: Dict[str, Dict] = {}
        self._load_all_schemas()

    def _load_all_schemas(self):
        """Load all database schemas from tables.json."""
        with open(self.tables_path, 'r') as f:
            all_tables = json.load(f)

        for db_schema in all_tables:
            db_id = db_schema['db_id']
            self.schemas[db_id] = self._parse_spider_schema(db_schema)

    def _parse_spider_schema(self, spider_schema: Dict) -> Dict:
        """
        Convert Spider schema format to SQLFormer format.

        Spider format:
            - column_names_original: [[-1, "*"], [table_idx, "col_name"], ...]
            - table_names_original: ["table1", "table2", ...]
            - foreign_keys: [[col_idx1, col_idx2], ...]

        SQLFormer format:
            {
                "table_name": {
                    "columns": ["col1", "col2"],
                    "foreign_keys": ["other_table"]
                }
            }
        """
        table_names = spider_schema['table_names_original']
        column_names = spider_schema['column_names_original']
        foreign_keys = spider_schema.get('foreign_keys', [])

        # Build table -> columns mapping
        schema = {}
        for table_idx, table_name in enumerate(table_names):
            table_name_lower = table_name.lower()
            schema[table_name_lower] = {
                "columns": [],
                "foreign_keys": []
            }

        # Add columns to their respective tables
        for table_idx, col_name in column_names:
            if table_idx == -1:  # Skip the "*" wildcard column
                continue
            table_name = table_names[table_idx].lower()
            schema[table_name]["columns"].append(col_name.lower())

        # Process foreign keys
        # foreign_keys format: [[col_idx1, col_idx2], ...]
        # meaning col_idx1 references col_idx2
        for fk_pair in foreign_keys:
            col1_idx, col2_idx = fk_pair

            # Get table indices for both columns
            table1_idx = column_names[col1_idx][0]
            table2_idx = column_names[col2_idx][0]

            if table1_idx == -1 or table2_idx == -1:
                continue

            table1_name = table_names[table1_idx].lower()
            table2_name = table_names[table2_idx].lower()

            # Add bidirectional foreign key relationships
            if table2_name not in schema[table1_name]["foreign_keys"]:
                schema[table1_name]["foreign_keys"].append(table2_name)
            if table1_name not in schema[table2_name]["foreign_keys"]:
                schema[table2_name]["foreign_keys"].append(table1_name)

        return schema

    def get_schema(self, db_id: str) -> Optional[Dict]:
        """
        Get schema for a specific database.

        Args:
            db_id: Database identifier (e.g., "concert_singer")

        Returns:
            Schema in SQLFormer format or None if not found
        """
        return self.schemas.get(db_id)

    def get_sqlformer_schema(self, db_id: str) -> Optional[Dict]:
        """
        Get schema in the format expected by SQLFormer's SchemaLoader.from_dict().

        Returns format compatible with:
            SchemaLoader.from_dict(schema)
        """
        schema = self.get_schema(db_id)
        if schema is None:
            return None
        return schema

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


class SpiderDatasetLoader:
    """Loads Spider dataset examples (questions and gold SQL)."""

    def __init__(self, data_json_path: str):
        """
        Initialize with path to Spider's train.json or dev.json.

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
                "db_id": str,
                "question": str,
                "query": str (gold SQL),
                "question_toks": List[str],
                "query_toks": List[str],
                ...
            }
        """
        return self.examples[idx]

    def get_examples_by_db(self, db_id: str) -> List[Dict]:
        """Get all examples for a specific database."""
        return [ex for ex in self.examples if ex['db_id'] == db_id]

    def get_difficulty_distribution(self) -> Dict[str, int]:
        """Get count of examples by difficulty level."""
        distribution = {}
        for ex in self.examples:
            difficulty = ex.get('difficulty', 'unknown')
            distribution[difficulty] = distribution.get(difficulty, 0) + 1
        return distribution


def convert_spider_to_sqlformer(tables_json: str, output_dir: str):
    """
    Convert all Spider schemas to individual SQLFormer-compatible JSON files.

    Args:
        tables_json: Path to Spider's tables.json
        output_dir: Directory to save converted schemas
    """
    loader = SpiderSchemaLoader(tables_json)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for db_id in loader.get_all_db_ids():
        schema = loader.get_schema(db_id)
        out_file = output_path / f"{db_id}.json"
        with open(out_file, 'w') as f:
            json.dump(schema, f, indent=2)
        print(f"Saved: {out_file}")


if __name__ == "__main__":
    # Example usage
    import sys

    if len(sys.argv) < 2:
        print("Usage: python spider_schema_loader.py <path_to_tables.json>")
        print("\nExample:")
        print("  python spider_schema_loader.py ./data/spider/tables.json")
        sys.exit(1)

    tables_path = sys.argv[1]
    loader = SpiderSchemaLoader(tables_path)

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
