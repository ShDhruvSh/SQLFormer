import json

class SchemaLoader:
    @staticmethod
    def from_dict(schema: dict):
        tables = list(schema.keys())
        columns = {
            table: set(meta["columns"])
            for table, meta in schema.items()
        }
        fks = {table: meta["foreign_keys"] for table, meta in schema.items()}
        return {"tables": tables, "columns": columns, "foreign_keys": fks, "raw": schema}

    @staticmethod
    def from_json(path: str):
        with open(path) as f:
            return SchemaLoader.from_dict(json.load(f))