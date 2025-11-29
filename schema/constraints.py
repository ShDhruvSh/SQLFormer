
class SchemaConstraints:
    def __init__(self, schema):
        self.schema = schema
    def valid_columns(self, table):
        return self.schema.get(table, {}).get('columns', [])
