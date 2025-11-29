
import networkx as nx
class SchemaGraph:
    def __init__(self, schema):
        self.graph = nx.Graph()
        for table, meta in schema.items():
            self.graph.add_node(table)
            for fk in meta.get('foreign_keys', []):
                self.graph.add_edge(table, fk)
    def reachable(self, table):
        return list(self.graph.neighbors(table))
