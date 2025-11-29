
class JoinValidator:
    def __init__(self, graph):
        self.graph = graph
    def valid_join(self, table1, table2):
        return table2 in self.graph.reachable(table1)
