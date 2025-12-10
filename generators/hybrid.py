import re
import sqlparse
from sqlparse.sql import Statement, Identifier, IdentifierList, Parenthesis, Function
from sqlparse.tokens import Keyword, Name, String
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass


@dataclass
class SQLError:
    error_type: str
    position: Optional[int] = None
    message: str = ""
    context: str = ""


class HybridSQLGenerator:

    def __init__(self, model, schema: Dict, max_retries: int = 3):
        self.model = model
        self.schema = schema
        self.max_retries = max_retries

        self.valid_tables = {t.lower(): t for t in schema["tables"]}
        self.table_columns = {}

        for table in schema["tables"]:
            table_lower = table.lower()
            col_mapping = {c.lower(): c for c in schema["columns"][table]}
            self.table_columns[table_lower] = col_mapping

    # generate sql, then parse and fix any errors
    def generate(self, prompt: str, max_new_tokens: int = 150) -> str:
        if self.model is None:
            return self._generate_fallback()

        ids = self.model.encode(prompt)
        output = self.model.model.generate(ids, max_new_tokens=max_new_tokens)
        raw_output = self.model.decode(output[0])
        sql = self._extract_sql(raw_output)

        for attempt in range(self.max_retries + 1):
            parsed = self._parse_sql(sql)

            if parsed is None:
                if attempt < self.max_retries:
                    sql = self._basic_repair(sql)
                    continue
                else:
                    return self._basic_repair(sql)

            errors = self._detect_errors(sql, parsed)

            if not errors:
                return sql

            if attempt < self.max_retries:
                sql = self._repair_sql(sql, errors, prompt)
            else:
                sql = self._apply_heuristic_fixes(sql, errors)

        return sql

    def _parse_sql(self, sql: str) -> Optional[Statement]:
        try:
            parsed = sqlparse.parse(sql)
            return parsed[0] if parsed else None
        except:
            return None

    def _detect_errors(self, sql: str, parsed: Statement) -> List[SQLError]:
        errors = []
        errors.extend(self._check_syntax_errors(sql))
        errors.extend(self._check_schema_errors(parsed))
        return errors

    def _check_syntax_errors(self, sql: str) -> List[SQLError]:
        errors = []

        paren_count = sql.count('(') - sql.count(')')
        if paren_count != 0:
            errors.append(SQLError(error_type='unclosed_paren', message=f"{paren_count} unmatched parens"))

        single_quotes = sql.count("'") % 2
        if single_quotes:
            errors.append(SQLError(error_type='unclosed_quote', message="Unclosed quote"))

        if not sql.strip().upper().startswith('SELECT'):
            errors.append(SQLError(error_type='missing_select', message="Missing SELECT"))

        if ' FROM ' not in sql.upper():
            errors.append(SQLError(error_type='missing_from', message="Missing FROM"))

        return errors

    # validate table and column names against schema
    def _check_schema_errors(self, parsed: Statement) -> List[SQLError]:
        errors = []

        tables_in_query, alias_map = self._extract_tables_and_aliases(parsed)

        for table_ref in tables_in_query:
            if table_ref.lower() not in self.valid_tables:
                errors.append(SQLError(
                    error_type='invalid_table',
                    message=f"Table '{table_ref}' not in schema"
                ))

        columns_in_query = self._extract_columns(parsed, tables_in_query)

        for col_ref, table_context in columns_in_query:
            col_lower = col_ref.lower()

            if col_lower in ['*', 'count', 'sum', 'avg', 'min', 'max', 'distinct']:
                continue

            if table_context:
                table_or_alias_lower = table_context.lower()
                actual_table = alias_map.get(table_or_alias_lower, table_or_alias_lower)

                if actual_table not in self.valid_tables:
                    if table_or_alias_lower not in self.valid_tables:
                        errors.append(SQLError(
                            error_type='invalid_table_alias',
                            message=f"Table or alias '{table_context}' not found"
                        ))
                        continue
                    actual_table = table_or_alias_lower

                if actual_table in self.table_columns:
                    if col_lower not in self.table_columns[actual_table]:
                        errors.append(SQLError(
                            error_type='invalid_column',
                            message=f"Column '{col_ref}' not in table '{actual_table}'"
                        ))
            else:
                found = False
                for table_ref in tables_in_query:
                    table_lower = table_ref.lower()
                    if table_lower in self.table_columns:
                        if col_lower in self.table_columns[table_lower]:
                            found = True
                            break

                if not found:
                    errors.append(SQLError(
                        error_type='invalid_column',
                        message=f"Column '{col_ref}' not found in any table"
                    ))

        return errors

    def _extract_tables_and_aliases(self, parsed: Statement) -> Tuple[List[str], Dict[str, str]]:
        tables = []
        alias_map = {}

        from_seen = False
        for token in parsed.tokens:
            if token.ttype is Keyword and token.value.upper() == 'FROM':
                from_seen = True
                continue

            if from_seen:
                if token.ttype is Keyword:
                    break

                if isinstance(token, IdentifierList):
                    for identifier in token.get_identifiers():
                        table_name, alias = self._get_table_and_alias(identifier)
                        if table_name:
                            tables.append(table_name)
                            if alias and alias != table_name:
                                alias_map[alias.lower()] = table_name.lower()
                elif isinstance(token, Identifier):
                    table_name, alias = self._get_table_and_alias(token)
                    if table_name:
                        tables.append(table_name)
                        if alias and alias != table_name:
                            alias_map[alias.lower()] = table_name.lower()
                elif token.ttype in (Name, String):
                    table_name = token.value.strip('"`')
                    tables.append(table_name)

        for i, token in enumerate(parsed.tokens):
            if token.ttype is Keyword and 'JOIN' in token.value.upper():
                for next_token in parsed.tokens[i+1:]:
                    if next_token.is_whitespace:
                        continue
                    if isinstance(next_token, Identifier):
                        table_name, alias = self._get_table_and_alias(next_token)
                        if table_name:
                            tables.append(table_name)
                            if alias and alias != table_name:
                                alias_map[alias.lower()] = table_name.lower()
                    elif next_token.ttype in (Name, String):
                        tables.append(next_token.value.strip('"`'))
                    break

        return tables, alias_map

    def _extract_columns(self, parsed: Statement, tables: List[str]) -> List[Tuple[str, Optional[str]]]:
        columns = []

        select_seen = False
        for token in parsed.tokens:
            if token.ttype is Keyword.DML and token.value.upper() == 'SELECT':
                select_seen = True
                continue

            if select_seen:
                if token.ttype is Keyword and token.value.upper() in ['FROM', 'WHERE', 'GROUP', 'ORDER']:
                    break
                columns.extend(self._extract_columns_from_token(token))

        return columns

    def _extract_columns_from_token(self, token) -> List[Tuple[str, Optional[str]]]:
        columns = []

        if isinstance(token, IdentifierList):
            for identifier in token.get_identifiers():
                columns.extend(self._extract_columns_from_token(identifier))

        elif isinstance(token, Identifier):
            if '.' in str(token):
                parts = str(token).split('.')
                if len(parts) == 2:
                    columns.append((parts[1].strip('"`'), parts[0].strip('"`')))
            else:
                real_name = self._get_real_name(token)
                if real_name:
                    columns.append((real_name, None))

        elif isinstance(token, Function):
            for sub_token in token.tokens:
                if isinstance(sub_token, Parenthesis):
                    columns.extend(self._extract_columns_from_token(sub_token))

        elif isinstance(token, Parenthesis):
            for sub_token in token.tokens:
                columns.extend(self._extract_columns_from_token(sub_token))

        elif token.ttype in (Name, String):
            col_name = token.value.strip('"`')
            if col_name.upper() not in ['SELECT', 'FROM', 'WHERE', 'AND', 'OR', 'AS']:
                columns.append((col_name, None))

        return columns

    def _get_table_and_alias(self, identifier) -> Tuple[Optional[str], Optional[str]]:
        if hasattr(identifier, 'get_real_name'):
            real_name = identifier.get_real_name()
            table_name = real_name.strip('"`') if real_name else None

            full_str = str(identifier)
            parts = full_str.split()

            if len(parts) >= 2:
                if parts[1].upper() == 'AS' and len(parts) >= 3:
                    return table_name, parts[2].strip('"`')
                else:
                    return table_name, parts[1].strip('"`')

            return table_name, None

        full_str = str(identifier).strip('"`')
        parts = full_str.split()

        if len(parts) >= 2:
            if parts[1].upper() == 'AS' and len(parts) >= 3:
                return parts[0], parts[2]
            return parts[0], parts[1]

        return parts[0] if parts else None, None

    def _get_real_name(self, identifier) -> Optional[str]:
        table_name, _ = self._get_table_and_alias(identifier)
        return table_name

    # ask model to fix specific errors
    def _repair_sql(self, sql: str, errors: List[SQLError], original_prompt: str) -> str:
        error_messages = [f"- {err.error_type}: {err.message}" for err in errors]
        error_str = "\n".join(error_messages)

        repair_prompt = f"""{original_prompt}

Previous attempt had errors:
{sql}

Errors found:
{error_str}

Generate corrected SQL:"""

        ids = self.model.encode(repair_prompt)
        output = self.model.model.generate(ids, max_new_tokens=150)
        repaired = self.model.decode(output[0])

        return self._extract_sql(repaired)

    def _apply_heuristic_fixes(self, sql: str, errors: List[SQLError]) -> str:
        for error in errors:
            if error.error_type == 'unclosed_paren':
                sql = self._balance_parentheses(sql)
            elif error.error_type == 'unclosed_quote':
                sql = self._close_quotes(sql)
        return sql

    def _basic_repair(self, sql: str) -> str:
        if ';' in sql:
            sql = sql.split(';')[0].strip()

        sql = self._balance_parentheses(sql)
        sql = self._close_quotes(sql)

        for kw in [' WHERE', ' AND', ' OR', ' ORDER BY', ' GROUP BY', ' HAVING']:
            if sql.upper().endswith(kw):
                sql = sql[:sql.upper().rfind(kw)].strip()

        return sql

    def _balance_parentheses(self, sql: str) -> str:
        open_count = sql.count('(')
        close_count = sql.count(')')

        if open_count > close_count:
            sql += ')' * (open_count - close_count)
        elif close_count > open_count:
            diff = close_count - open_count
            for _ in range(diff):
                last_close = sql.rfind(')')
                if last_close >= 0:
                    sql = sql[:last_close] + sql[last_close+1:]

        return sql

    def _close_quotes(self, sql: str) -> str:
        if sql.count("'") % 2 == 1:
            sql += "'"
        if sql.count('"') % 2 == 1:
            sql += '"'
        return sql

    def _extract_sql(self, generated_text: str) -> str:
        text = generated_text.strip()

        if '```sql' in text.lower():
            match = re.search(r'```sql\s*(.*?)\s*```', text, re.DOTALL | re.IGNORECASE)
            if match:
                text = match.group(1).strip()
        elif '```' in text:
            match = re.search(r'```\s*(.*?)\s*```', text, re.DOTALL)
            if match:
                text = match.group(1).strip()

        if text.upper().startswith('SELECT'):
            end_markers = [';', '\n\n', '\n--', '\nQuestion:', '\n#', '\nExplanation:']
            end_pos = len(text)
            for marker in end_markers:
                pos = text.find(marker)
                if pos > 0:
                    end_pos = min(end_pos, pos)
            return text[:end_pos].strip()

        select_pos = text.upper().find('SELECT')
        if select_pos >= 0:
            remaining = text[select_pos:]
            end_markers = [';', '\n\n', '\n--']
            end_pos = len(remaining)
            for marker in end_markers:
                pos = remaining.find(marker)
                if pos > 0:
                    end_pos = min(end_pos, pos)
            return remaining[:end_pos].strip()

        return text

    def _generate_fallback(self) -> str:
        if self.schema["tables"]:
            return f"SELECT * FROM {self.schema['tables'][0]}"
        return "SELECT *"


class HybridSQLFormerEngine:

    def __init__(self, model, schema: Dict):
        self.generator = HybridSQLGenerator(model, schema)

    def generate(self, prompt: str) -> str:
        return self.generator.generate(prompt)
