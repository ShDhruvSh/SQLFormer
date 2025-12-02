import networkx as nx

class SchemaGraph:
    def __init__(self, schema):
        self.graph = nx.Graph()
        for table, meta in schema.items():
            self.graph.add_node(table)
            for fk in meta.get('foreign_keys', []):
                self.graph.add_edge(table, fk)

    def is_join_valid(self, table_a, table_b):
        return nx.has_path(self.graph, table_a, table_b)