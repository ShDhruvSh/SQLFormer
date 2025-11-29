
import json
class SchemaLoader:
    def __init__(self, schema):
        self.schema = schema
    @classmethod
    def from_json(cls, path):
        with open(path) as f:
            return cls(json.load(f))
