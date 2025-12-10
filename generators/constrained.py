import torch
import re
from typing import List, Set, Dict, Optional
from dataclasses import dataclass
from enum import Enum, auto


class SQLState(Enum):
    START = auto()
    AFTER_SELECT = auto()
    IN_SELECT_COLS = auto()
    AFTER_FROM = auto()
    IN_FROM_TABLES = auto()
    AFTER_WHERE = auto()
    IN_WHERE_COND = auto()
    IN_GROUP_BY = auto()
    IN_ORDER_BY = auto()
    COMPLETE = auto()


@dataclass
class SQLParseState:
    state: SQLState
    tables_used: Set[str]
    columns_used: Set[str]
    partial_token: str
    in_string: bool
    paren_depth: int


class SQLConstrainedDecoder:

    KEYWORDS = {
        'SELECT', 'FROM', 'WHERE', 'AND', 'OR', 'GROUP', 'BY', 'ORDER',
        'LIMIT', 'JOIN', 'LEFT', 'RIGHT', 'INNER', 'ON', 'AS', 'IN',
        'NOT', 'NULL', 'IS', 'LIKE', 'BETWEEN', 'ASC', 'DESC',
        'DISTINCT', 'COUNT', 'SUM', 'AVG', 'MIN', 'MAX', 'HAVING'
    }

    def __init__(self, model, schema: Dict, temperature: float = 0.0):
        self.model = model
        self.schema = schema
        self.temperature = temperature
        self.tokenizer = model.tokenizer
        self.vocab_size = model.vocab_size

        self._build_valid_tokens()

    def _build_valid_tokens(self):
        self.table_tokens = {}
        self.column_tokens = {}
        self.keyword_tokens = {}

        for table in self.schema["tables"]:
            ids = self.tokenizer.encode(table, add_special_tokens=False)
            self.table_tokens[table] = ids
            ids_space = self.tokenizer.encode(" " + table, add_special_tokens=False)
            self.table_tokens[" " + table] = ids_space

        for table in self.schema["tables"]:
            for col in self.schema["columns"][table]:
                ids = self.tokenizer.encode(col, add_special_tokens=False)
                self.column_tokens[col] = ids
                ids_space = self.tokenizer.encode(" " + col, add_special_tokens=False)
                self.column_tokens[" " + col] = ids_space

        for kw in self.KEYWORDS:
            ids = self.tokenizer.encode(kw, add_special_tokens=False)
            self.keyword_tokens[kw] = ids
            ids_space = self.tokenizer.encode(" " + kw, add_special_tokens=False)
            self.keyword_tokens[" " + kw] = ids_space

    # figure out where we are in the sql query
    def _get_sql_state(self, sql: str) -> SQLParseState:
        sql_upper = sql.upper().strip()

        state = SQLState.START
        tables_used = set()
        columns_used = set()
        in_string = sql.count("'") % 2 == 1
        paren_depth = sql.count("(") - sql.count(")")

        if not sql_upper:
            state = SQLState.START
        elif sql_upper.endswith("SELECT") or sql_upper.endswith("SELECT "):
            state = SQLState.AFTER_SELECT
        elif "SELECT" in sql_upper and "FROM" not in sql_upper:
            state = SQLState.IN_SELECT_COLS
        elif sql_upper.endswith("FROM") or sql_upper.endswith("FROM "):
            state = SQLState.AFTER_FROM
        elif "FROM" in sql_upper and "WHERE" not in sql_upper and "GROUP" not in sql_upper and "ORDER" not in sql_upper:
            state = SQLState.IN_FROM_TABLES
        elif sql_upper.endswith("WHERE") or sql_upper.endswith("WHERE "):
            state = SQLState.AFTER_WHERE
        elif "WHERE" in sql_upper and "GROUP" not in sql_upper and "ORDER" not in sql_upper:
            state = SQLState.IN_WHERE_COND
        elif "GROUP BY" in sql_upper and "ORDER" not in sql_upper:
            state = SQLState.IN_GROUP_BY
        elif "ORDER BY" in sql_upper:
            state = SQLState.IN_ORDER_BY
        elif sql_upper.endswith(";"):
            state = SQLState.COMPLETE

        return SQLParseState(
            state=state,
            tables_used=tables_used,
            columns_used=columns_used,
            partial_token="",
            in_string=in_string,
            paren_depth=paren_depth
        )

    # get valid next tokens based on current state
    def _get_valid_token_ids(self, current_sql: str, parse_state: SQLParseState) -> Set[int]:
        valid_ids = set()
        state = parse_state.state

        if parse_state.in_string:
            return set(range(self.vocab_size))

        space_id = self.tokenizer.encode(" ", add_special_tokens=False)
        if space_id:
            valid_ids.add(space_id[0])

        if state == SQLState.START:
            for kw in ["SELECT", " SELECT"]:
                if kw in self.keyword_tokens:
                    valid_ids.add(self.keyword_tokens[kw][0])

        elif state == SQLState.AFTER_SELECT:
            self._add_column_tokens(valid_ids)
            self._add_tokens_for_strings(valid_ids, ["*", " *", "DISTINCT", " DISTINCT"])
            self._add_tokens_for_strings(valid_ids, ["COUNT", "SUM", "AVG", "MIN", "MAX",
                                                      " COUNT", " SUM", " AVG", " MIN", " MAX"])
            self._add_tokens_for_strings(valid_ids, ["(", " ("])

        elif state == SQLState.IN_SELECT_COLS:
            self._add_column_tokens(valid_ids)
            self._add_tokens_for_strings(valid_ids, [",", " ,", "FROM", " FROM"])
            self._add_tokens_for_strings(valid_ids, ["*", " *", "(", ")", " (", " )"])
            self._add_tokens_for_strings(valid_ids, ["COUNT", "SUM", "AVG", "MIN", "MAX",
                                                      " COUNT", " SUM", " AVG", " MIN", " MAX"])

        elif state == SQLState.AFTER_FROM:
            self._add_table_tokens(valid_ids)

        elif state == SQLState.IN_FROM_TABLES:
            self._add_table_tokens(valid_ids)
            self._add_tokens_for_strings(valid_ids, [",", " ,", "WHERE", " WHERE"])
            self._add_tokens_for_strings(valid_ids, ["GROUP", " GROUP", "ORDER", " ORDER"])
            self._add_tokens_for_strings(valid_ids, ["JOIN", " JOIN", "LEFT", " LEFT",
                                                      "RIGHT", " RIGHT", "INNER", " INNER"])
            self._add_tokens_for_strings(valid_ids, [";", " ;"])

        elif state == SQLState.AFTER_WHERE or state == SQLState.IN_WHERE_COND:
            self._add_column_tokens(valid_ids)
            self._add_table_tokens(valid_ids)
            self._add_tokens_for_strings(valid_ids, ["=", " =", "<", ">", "!", " <", " >"])
            self._add_tokens_for_strings(valid_ids, ["AND", " AND", "OR", " OR", "NOT", " NOT"])
            self._add_tokens_for_strings(valid_ids, ["IN", " IN", "LIKE", " LIKE", "IS", " IS"])
            self._add_tokens_for_strings(valid_ids, ["(", ")", " (", " )", "'", " '"])
            self._add_tokens_for_strings(valid_ids, ["GROUP", " GROUP", "ORDER", " ORDER"])
            self._add_tokens_for_strings(valid_ids, [";", " ;"])
            self._add_tokens_for_strings(valid_ids, ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"])
            self._add_tokens_for_strings(valid_ids, [" 0", " 1", " 2", " 3", " 4", " 5"])

        elif state == SQLState.IN_GROUP_BY:
            self._add_column_tokens(valid_ids)
            self._add_tokens_for_strings(valid_ids, [",", " ,", "ORDER", " ORDER", "HAVING", " HAVING"])
            self._add_tokens_for_strings(valid_ids, [";", " ;"])

        elif state == SQLState.IN_ORDER_BY:
            self._add_column_tokens(valid_ids)
            self._add_tokens_for_strings(valid_ids, [",", " ,", "ASC", " ASC", "DESC", " DESC"])
            self._add_tokens_for_strings(valid_ids, ["LIMIT", " LIMIT", ";", " ;"])

        if not valid_ids:
            return set(range(self.vocab_size))

        return valid_ids

    def _add_column_tokens(self, valid_ids: Set[int]):
        for col, token_ids in self.column_tokens.items():
            if token_ids:
                valid_ids.add(token_ids[0])

    def _add_table_tokens(self, valid_ids: Set[int]):
        for table, token_ids in self.table_tokens.items():
            if token_ids:
                valid_ids.add(token_ids[0])

    def _add_tokens_for_strings(self, valid_ids: Set[int], strings: List[str]):
        for s in strings:
            ids = self.tokenizer.encode(s, add_special_tokens=False)
            if ids:
                valid_ids.add(ids[0])

    # generate sql with token-level constraints
    def generate(self, prompt: str, max_new_tokens: int = 100, stop_tokens: Optional[List[str]] = None) -> str:
        if stop_tokens is None:
            stop_tokens = [";", "\n\n", "```"]

        input_ids = self.model.encode(prompt)
        generated_sql = ""
        generated_ids = []
        eos_id = self.tokenizer.eos_token_id

        for step in range(max_new_tokens):
            with torch.no_grad():
                full_ids = torch.cat([input_ids, torch.tensor([generated_ids], device=input_ids.device)], dim=1) if generated_ids else input_ids
                logits = self.model.forward_logits(full_ids)

            parse_state = self._get_sql_state(generated_sql)
            valid_token_ids = self._get_valid_token_ids(generated_sql, parse_state)

            mask = torch.full_like(logits, float('-inf'))
            for token_id in valid_token_ids:
                if token_id < mask.size(-1):
                    mask[0, token_id] = 0

            masked_logits = logits + mask

            if self.temperature > 0:
                probs = torch.softmax(masked_logits / self.temperature, dim=-1)
                next_token_id = torch.multinomial(probs[0], num_samples=1).item()
            else:
                next_token_id = torch.argmax(masked_logits[0]).item()

            if next_token_id == eos_id:
                break

            new_token = self.tokenizer.decode([next_token_id])
            generated_sql += new_token
            generated_ids.append(next_token_id)

            if any(stop in generated_sql for stop in stop_tokens):
                break

            if parse_state.state == SQLState.COMPLETE:
                break

        sql = generated_sql.strip()
        if sql.endswith(";"):
            sql = sql[:-1].strip()

        return sql


class ConstrainedSQLFormerEngine:

    def __init__(self, model, schema: Dict):
        self.model = model
        self.schema = schema
        self.decoder = None

        if model is not None:
            self.decoder = SQLConstrainedDecoder(model, schema)

    def generate(self, prompt: str) -> str:
        if self.decoder is None:
            tables = self.schema.get("tables", [])
            return f"SELECT * FROM {tables[0]}" if tables else "SELECT *"

        clean_prompt = self._build_generation_prompt(prompt)
        sql = self.decoder.generate(clean_prompt, max_new_tokens=100)

        return self._validate_and_clean(sql)

    def _build_generation_prompt(self, prompt: str) -> str:
        lines = prompt.strip().split('\n')
        question = ""
        for line in lines:
            line = line.strip()
            if line and not any(x in line.lower() for x in ['database:', 'tables:', '->', 'generate', 'schema']):
                question = line
                break

        if not question:
            question = lines[-1].strip() if lines else prompt

        schema_desc = "Tables:\n"
        for table in self.schema["tables"]:
            cols = ", ".join(self.schema["columns"][table])
            schema_desc += f"- {table}: {cols}\n"

        return f"""{schema_desc}
Question: {question}
SQL: SELECT"""

    def _validate_and_clean(self, sql: str) -> str:
        if not sql.upper().startswith("SELECT"):
            sql = "SELECT " + sql

        sql = sql.strip()
        if sql.upper().endswith((" WHERE", " AND", " OR", " ORDER BY", " GROUP BY")):
            for kw in [" WHERE", " AND", " OR", " ORDER BY", " GROUP BY"]:
                if sql.upper().endswith(kw):
                    sql = sql[:-len(kw)].strip()
                    break

        return sql
