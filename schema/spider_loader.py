import json
from typing import Dict, List, Optional
from pathlib import Path


class SpiderSchemaLoader:

    def __init__(self, tables_json_path: str):
        self.tables_path = Path(tables_json_path)
        self.schemas: Dict[str, Dict] = {}
        self._load_all_schemas()

    def _load_all_schemas(self):
        with open(self.tables_path, 'r') as f:
            all_tables = json.load(f)
        for db_schema in all_tables:
            self.schemas[db_schema['db_id']] = self._parse_spider_schema(db_schema)

    # convert spider format to our schema format
    def _parse_spider_schema(self, spider_schema: Dict) -> Dict:
        table_names = spider_schema['table_names_original']
        column_names = spider_schema['column_names_original']
        foreign_keys = spider_schema.get('foreign_keys', [])

        schema = {}
        for idx, table_name in enumerate(table_names):
            schema[table_name.lower()] = {"columns": [], "foreign_keys": []}

        for table_idx, col_name in column_names:
            if table_idx != -1:
                schema[table_names[table_idx].lower()]["columns"].append(col_name.lower())

        for col1_idx, col2_idx in foreign_keys:
            table1_idx = column_names[col1_idx][0]
            table2_idx = column_names[col2_idx][0]

            if table1_idx == -1 or table2_idx == -1:
                continue

            t1 = table_names[table1_idx].lower()
            t2 = table_names[table2_idx].lower()

            if t2 not in schema[t1]["foreign_keys"]:
                schema[t1]["foreign_keys"].append(t2)
            if t1 not in schema[t2]["foreign_keys"]:
                schema[t2]["foreign_keys"].append(t1)

        return schema

    def get_schema(self, db_id: str) -> Optional[Dict]:
        return self.schemas.get(db_id)

    def get_all_db_ids(self) -> List[str]:
        return list(self.schemas.keys())


class SpiderDatasetLoader:

    def __init__(self, data_json_path: str):
        with open(data_json_path, 'r') as f:
            self.examples = json.load(f)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx: int):
        return self.examples[idx]

    def get_example(self, idx: int) -> Dict:
        return self.examples[idx]


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python spider_loader.py <tables.json>")
        sys.exit(1)

    loader = SpiderSchemaLoader(sys.argv[1])
    print(f"Loaded {len(loader.get_all_db_ids())} databases")
