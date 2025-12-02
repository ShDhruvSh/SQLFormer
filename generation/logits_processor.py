class LogitsProcessor:
    def __init__(self, fsm, graph, schema):
        self.fsm = fsm
        self.graph = graph
        self.schema = schema

    def mask(self, logit_tokens, state, sql_tokens):
        if state == "SELECT":
            all_columns = set()
            if isinstance(self.schema["columns"], dict):
                for cols in self.schema["columns"].values():
                    all_columns.update(cols)
            else:
                all_columns = set(self.schema["columns"])
            selected = set(sql_tokens.get("select", []))
            return [t for t in logit_tokens if (t in all_columns and t not in selected) or t == "FROM"]
        elif state == "FROM":
            used = set(sql_tokens.get("from", []))
            return [t for t in logit_tokens if (t in self.schema["tables"] and t not in used) or t in ["WHERE", "GROUP_BY", "ORDER_BY", "END"]]
        elif state == "WHERE":
            all_columns = set()
            if isinstance(self.schema["columns"], dict):
                for cols in self.schema["columns"].values():
                    all_columns.update(cols)
            else:
                all_columns = set(self.schema["columns"])
            selected = set(sql_tokens.get("where", []))
            return [t for t in logit_tokens if (t in all_columns and t not in selected) or t in ["GROUP_BY", "ORDER_BY", "END"]]
        elif state == "GROUP_BY":
            all_columns = set()
            if isinstance(self.schema["columns"], dict):
                for cols in self.schema["columns"].values():
                    all_columns.update(cols)
            else:
                all_columns = set(self.schema["columns"])
            selected = set(sql_tokens.get("group_by", []))
            return [t for t in logit_tokens if (t in all_columns and t not in selected) or t in ["ORDER_BY", "END"]]
        elif state == "ORDER_BY":
            all_columns = set()
            if isinstance(self.schema["columns"], dict):
                for cols in self.schema["columns"].values():
                    all_columns.update(cols)
            else:
                all_columns = set(self.schema["columns"])
            selected = set(sql_tokens.get("order_by", []))
            return [t for t in logit_tokens if (t in all_columns and t not in selected) or t == "END"]
        else:
            return logit_tokens