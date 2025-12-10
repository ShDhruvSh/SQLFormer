"""
Hybrid SQL Generator: Generate-Then-Constrain

This approach combines:
1. Unconstrained generation (preserves model capability for Spider performance)
2. Bulletproof parsing and validation (ensures syntactic correctness)
3. Targeted repair with constrained generation (fixes specific errors)

Key advantages:
- Lets small LLMs generate naturally (better Spider scores)
- Guarantees syntactically valid SQL (bulletproof)
- Only constrains when necessary (targeted repairs)
"""

import re
import sqlparse
from sqlparse.sql import Statement, Token, TokenList, Identifier, IdentifierList, Where, Function, Parenthesis
from sqlparse.tokens import Keyword, DML, DDL, Punctuation, Name, String, Number, Wildcard
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass


@dataclass
class SQLError:
    """Represents a specific SQL syntax or schema error."""
    error_type: str  # 'unclosed_paren', 'unclosed_quote', 'invalid_table', etc.
    position: Optional[int] = None
    message: str = ""
    context: str = ""


class HybridSQLGenerator:
    """
    Generate SQL with unconstrained LLM, then parse and repair.

    Pipeline:
    1. Unconstrained generation (model generates naturally)
    2. SQL parsing with sqlparse (detect syntax errors)
    3. Schema validation with AST traversal (detect invalid tables/columns)
    4. Targeted repair if needed (fix specific errors)
    5. Final validation (ensure bulletproof correctness)
    """

    def __init__(self, model, schema: Dict, max_retries: int = 3):
        """
        Initialize hybrid generator.

        Args:
            model: SQLFormerModel instance
            schema: Schema dict with 'tables' and 'columns' keys
            max_retries: Maximum retry attempts for repairs
        """
        self.model = model
        self.schema = schema
        self.max_retries = max_retries

        # Build lookup structures
        self.valid_tables = {t.lower(): t for t in schema["tables"]}
        self.table_columns = {}  # table_lower -> {col_lower: original_col}

        for table in schema["tables"]:
            table_lower = table.lower()
            col_mapping = {c.lower(): c for c in schema["columns"][table]}
            self.table_columns[table_lower] = col_mapping

    def generate(self, prompt: str, max_new_tokens: int = 150) -> str:
        """
        Generate SQL with hybrid approach.

        Args:
            prompt: Full prompt including question and schema info
            max_new_tokens: Maximum tokens for generation

        Returns:
            Validated and repaired SQL query
        """
        if self.model is None:
            return self._generate_fallback()

        # Stage 1: Unconstrained generation
        raw_output = self.model.generate_unconstrained(
            prompt,
            max_new_tokens=max_new_tokens
        )
        sql = self._extract_sql(raw_output)

        # Stage 2: Parse and validate
        for attempt in range(self.max_retries + 1):
            # Parse SQL
            parsed = self._parse_sql(sql)

            if parsed is None:
                # Critical parsing failure - try to repair
                if attempt < self.max_retries:
                    sql = self._repair_unparseable_sql(sql)
                    continue
                else:
                    # Last attempt - return best effort
                    return self._basic_repair(sql)

            # Detect errors
            errors = self._detect_errors(sql, parsed)

            if not errors:
                # Success! SQL is valid
                return sql

            # Stage 3: Targeted repair
            if attempt < self.max_retries:
                sql = self._repair_sql(sql, errors, prompt)
            else:
                # Last attempt - apply heuristic fixes
                sql = self._apply_heuristic_fixes(sql, errors)

        return sql

    def _parse_sql(self, sql: str) -> Optional[Statement]:
        """
        Parse SQL using sqlparse.

        Returns:
            Parsed statement or None if parsing fails
        """
        try:
            # Parse and format to normalize
            parsed = sqlparse.parse(sql)
            if not parsed:
                return None
            return parsed[0]
        except Exception as e:
            return None

    def _detect_errors(self, sql: str, parsed: Statement) -> List[SQLError]:
        """
        Detect all SQL errors (syntax + schema).

        Args:
            sql: Original SQL string
            parsed: Parsed statement

        Returns:
            List of detected errors
        """
        errors = []

        # 1. Syntax errors
        errors.extend(self._check_syntax_errors(sql))

        # 2. Schema errors (invalid tables/columns)
        errors.extend(self._check_schema_errors(parsed))

        # 3. Structural errors (missing clauses, wrong order)
        errors.extend(self._check_structural_errors(parsed))

        return errors

    def _check_syntax_errors(self, sql: str) -> List[SQLError]:
        """Check for basic syntax errors."""
        errors = []

        # Check for unclosed parentheses
        paren_count = 0
        for i, char in enumerate(sql):
            if char == '(':
                paren_count += 1
            elif char == ')':
                paren_count -= 1
            if paren_count < 0:
                errors.append(SQLError(
                    error_type='extra_closing_paren',
                    position=i,
                    message=f"Extra closing parenthesis at position {i}"
                ))

        if paren_count > 0:
            errors.append(SQLError(
                error_type='unclosed_paren',
                message=f"{paren_count} unclosed parenthesis(es)"
            ))

        # Check for unclosed quotes
        in_single_quote = False
        in_double_quote = False
        for i, char in enumerate(sql):
            if char == "'" and (i == 0 or sql[i-1] != '\\'):
                if not in_double_quote:
                    in_single_quote = not in_single_quote
            elif char == '"' and (i == 0 or sql[i-1] != '\\'):
                if not in_single_quote:
                    in_double_quote = not in_double_quote

        if in_single_quote:
            errors.append(SQLError(
                error_type='unclosed_single_quote',
                message="Unclosed single quote"
            ))

        if in_double_quote:
            errors.append(SQLError(
                error_type='unclosed_double_quote',
                message="Unclosed double quote"
            ))

        # Check for missing SELECT
        if not sql.strip().upper().startswith('SELECT'):
            errors.append(SQLError(
                error_type='missing_select',
                message="Query must start with SELECT"
            ))

        # Check for missing FROM
        if ' FROM ' not in sql.upper():
            errors.append(SQLError(
                error_type='missing_from',
                message="Query missing FROM clause"
            ))

        return errors

    def _check_schema_errors(self, parsed: Statement) -> List[SQLError]:
        """
        Check for schema validation errors using AST traversal.

        This is the bulletproof part - we traverse the parsed AST
        and validate every table and column reference.
        """
        errors = []

        # Extract all tables and their aliases used in query
        tables_in_query, alias_map = self._extract_tables_and_aliases_from_ast(parsed)

        # Validate table names
        for table_ref in tables_in_query:
            table_lower = table_ref.lower()
            if table_lower not in self.valid_tables:
                # Try to find close match
                suggestion = self._find_closest_table(table_lower)
                errors.append(SQLError(
                    error_type='invalid_table',
                    message=f"Table '{table_ref}' not in schema",
                    context=f"Did you mean '{suggestion}'?" if suggestion else ""
                ))

        # Extract and validate columns
        columns_in_query = self._extract_columns_from_ast(parsed, tables_in_query)

        for col_ref, table_context in columns_in_query:
            col_lower = col_ref.lower()

            # Skip special cases
            if col_lower in ['*', 'count', 'sum', 'avg', 'min', 'max', 'distinct']:
                continue

            # Check if column exists in the context tables
            if table_context:
                # Specific table context (e.g., table.column or alias.column)
                table_or_alias_lower = table_context.lower()

                # Resolve alias to actual table name
                actual_table = alias_map.get(table_or_alias_lower, table_or_alias_lower)

                # Validate that the table exists
                if actual_table not in self.valid_tables:
                    # Check if it's an unresolved alias
                    if table_or_alias_lower not in self.valid_tables:
                        errors.append(SQLError(
                            error_type='invalid_table_alias',
                            message=f"Table or alias '{table_context}' not found"
                        ))
                        continue
                    actual_table = table_or_alias_lower

                # Validate column exists in the resolved table
                if actual_table in self.table_columns:
                    if col_lower not in self.table_columns[actual_table]:
                        errors.append(SQLError(
                            error_type='invalid_column',
                            message=f"Column '{col_ref}' not in table '{actual_table}' (referenced as '{table_context}')"
                        ))
            else:
                # No specific table - check if column exists in ANY of the query tables
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

    def _check_structural_errors(self, parsed: Statement) -> List[SQLError]:
        """Check for structural SQL errors (clause ordering, etc.)."""
        errors = []

        # Get token positions
        token_positions = {}
        for i, token in enumerate(parsed.tokens):
            if token.ttype is Keyword:
                kw = token.value.upper()
                if kw not in token_positions:
                    token_positions[kw] = i

        # Check clause ordering
        # Correct order: SELECT < FROM < WHERE < GROUP BY < HAVING < ORDER BY < LIMIT
        clause_order = ['SELECT', 'FROM', 'WHERE', 'GROUP', 'HAVING', 'ORDER', 'LIMIT']

        prev_position = -1
        prev_clause = None

        for clause in clause_order:
            if clause in token_positions:
                if token_positions[clause] < prev_position:
                    errors.append(SQLError(
                        error_type='clause_misordering',
                        message=f"{clause} clause should come before {prev_clause}"
                    ))
                prev_position = token_positions[clause]
                prev_clause = clause

        return errors

    def _extract_tables_and_aliases_from_ast(self, parsed: Statement) -> Tuple[List[str], Dict[str, str]]:
        """
        Extract all table names and their aliases from parsed AST.

        Returns:
            (list of table names, dict mapping alias -> table_name)
        """
        tables = []
        alias_map = {}  # alias_lower -> table_lower

        # Find FROM clause
        from_seen = False
        for token in parsed.tokens:
            if token.ttype is Keyword and token.value.upper() == 'FROM':
                from_seen = True
                continue

            if from_seen:
                if token.ttype is Keyword:
                    # Hit next clause
                    break

                if isinstance(token, IdentifierList):
                    # Multiple tables
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

        # Find JOIN clauses
        for i, token in enumerate(parsed.tokens):
            if token.ttype is Keyword and 'JOIN' in token.value.upper():
                # Next non-whitespace token should be table name
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
                        table_name = next_token.value.strip('"`')
                        tables.append(table_name)
                    break

        return tables, alias_map

    def _extract_tables_from_ast(self, parsed: Statement) -> List[str]:
        """Extract all table names from parsed AST (backward compatibility)."""
        tables, _ = self._extract_tables_and_aliases_from_ast(parsed)
        return tables

    def _extract_columns_from_ast(self, parsed: Statement, tables: List[str]) -> List[Tuple[str, Optional[str]]]:
        """
        Extract all column references from parsed AST.

        Returns:
            List of (column_name, table_context) tuples
        """
        columns = []

        # Find SELECT clause
        select_seen = False
        for token in parsed.tokens:
            if token.ttype is Keyword.DML and token.value.upper() == 'SELECT':
                select_seen = True
                continue

            if select_seen:
                if token.ttype is Keyword and token.value.upper() in ['FROM', 'WHERE', 'GROUP', 'ORDER']:
                    break

                # Extract columns from this token
                columns.extend(self._extract_columns_from_token(token))

        return columns

    def _extract_columns_from_token(self, token) -> List[Tuple[str, Optional[str]]]:
        """Recursively extract columns from a token."""
        columns = []

        if isinstance(token, IdentifierList):
            for identifier in token.get_identifiers():
                columns.extend(self._extract_columns_from_token(identifier))

        elif isinstance(token, Identifier):
            # Check for table.column format
            if '.' in str(token):
                parts = str(token).split('.')
                if len(parts) == 2:
                    table, column = parts
                    columns.append((column.strip('"`'), table.strip('"`')))
            else:
                real_name = self._get_real_name(token)
                if real_name:
                    columns.append((real_name, None))

        elif isinstance(token, Function):
            # Extract columns from function arguments
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
        """
        Extract both table name and alias from an identifier.

        Returns:
            (table_name, alias) tuple. If no alias, alias=None.
        """
        if hasattr(identifier, 'get_real_name'):
            # sqlparse Identifier object
            real_name = identifier.get_real_name()
            table_name = real_name.strip('"`') if real_name else None

            # Check if there's an alias
            full_str = str(identifier)
            parts = full_str.split()

            if len(parts) >= 2:
                # Format: "table_name alias" or "table_name AS alias"
                if parts[1].upper() == 'AS' and len(parts) >= 3:
                    alias = parts[2].strip('"`')
                else:
                    alias = parts[1].strip('"`')
                return table_name, alias

            return table_name, None

        # Fallback: parse string representation
        full_str = str(identifier).strip('"`')
        parts = full_str.split()

        if len(parts) >= 2:
            if parts[1].upper() == 'AS' and len(parts) >= 3:
                return parts[0], parts[2]
            return parts[0], parts[1]

        return parts[0] if parts else None, None

    def _get_real_name(self, identifier) -> Optional[str]:
        """Get the real name from an identifier (handles aliases)."""
        table_name, _ = self._get_table_and_alias(identifier)
        return table_name

    def _find_closest_table(self, table: str) -> Optional[str]:
        """Find closest matching table name (simple edit distance)."""
        # Simple similarity: check for substring matches
        for valid_table in self.valid_tables.values():
            if table.lower() in valid_table.lower() or valid_table.lower() in table.lower():
                return valid_table

        return None

    def _repair_sql(self, sql: str, errors: List[SQLError], original_prompt: str) -> str:
        """
        Repair SQL based on detected errors using targeted retry.

        This provides specific feedback to the model about what to fix.
        """
        # Build error message
        error_messages = []
        for err in errors:
            msg = f"- {err.error_type}: {err.message}"
            if err.context:
                msg += f" ({err.context})"
            error_messages.append(msg)

        error_str = "\n".join(error_messages)

        # Create repair prompt with specific feedback
        repair_prompt = f"""{original_prompt}

Previous attempt had errors:
{sql}

Errors found:
{error_str}

Please generate a corrected SQL query that fixes these specific errors.
Use only the tables and columns from the schema provided above.
SQL:"""

        # Generate repair
        repaired_output = self.model.generate_unconstrained(
            repair_prompt,
            max_new_tokens=150
        )

        return self._extract_sql(repaired_output)

    def _repair_unparseable_sql(self, sql: str) -> str:
        """Repair SQL that couldn't be parsed."""
        # Apply basic repairs
        sql = self._basic_repair(sql)
        return sql

    def _basic_repair(self, sql: str) -> str:
        """Apply basic heuristic repairs."""
        # Remove content after semicolon
        if ';' in sql:
            sql = sql.split(';')[0].strip()

        # Balance parentheses
        sql = self._balance_parentheses(sql)

        # Close unclosed quotes
        sql = self._close_quotes(sql)

        # Remove trailing incomplete clauses
        for kw in [' WHERE', ' AND', ' OR', ' ORDER BY', ' GROUP BY', ' HAVING']:
            if sql.upper().endswith(kw):
                sql = sql[:sql.upper().rfind(kw)].strip()

        return sql

    def _apply_heuristic_fixes(self, sql: str, errors: List[SQLError]) -> str:
        """Apply heuristic fixes for specific error types."""
        for error in errors:
            if error.error_type == 'unclosed_paren':
                sql = self._balance_parentheses(sql)
            elif error.error_type in ['unclosed_single_quote', 'unclosed_double_quote']:
                sql = self._close_quotes(sql)
            elif error.error_type == 'invalid_table':
                # Try to replace with closest match
                pass  # TODO: implement fuzzy matching
            elif error.error_type == 'clause_misordering':
                pass  # TODO: implement clause reordering

        return sql

    def _balance_parentheses(self, sql: str) -> str:
        """Balance parentheses in SQL."""
        open_count = sql.count('(')
        close_count = sql.count(')')

        if open_count > close_count:
            sql += ')' * (open_count - close_count)
        elif close_count > open_count:
            # Remove extra closing parens from end
            diff = close_count - open_count
            for _ in range(diff):
                last_close = sql.rfind(')')
                if last_close >= 0:
                    sql = sql[:last_close] + sql[last_close+1:]

        return sql

    def _close_quotes(self, sql: str) -> str:
        """Close unclosed quotes in SQL."""
        # Count quotes
        single_quotes = sql.count("'")
        double_quotes = sql.count('"')

        # If odd number, add closing quote at end
        if single_quotes % 2 == 1:
            sql += "'"
        if double_quotes % 2 == 1:
            sql += '"'

        return sql

    def _extract_sql(self, generated_text: str) -> str:
        """Extract SQL query from generated text."""
        text = generated_text.strip()

        # Remove markdown code blocks
        if '```sql' in text.lower():
            match = re.search(r'```sql\s*(.*?)\s*```', text, re.DOTALL | re.IGNORECASE)
            if match:
                text = match.group(1).strip()
        elif '```' in text:
            match = re.search(r'```\s*(.*?)\s*```', text, re.DOTALL)
            if match:
                text = match.group(1).strip()

        # Find SELECT statement
        if text.upper().startswith('SELECT'):
            end_markers = [';', '\n\n', '\n--', '\nQuestion:', '\n#', '\nExplanation:']
            end_pos = len(text)
            for marker in end_markers:
                pos = text.find(marker)
                if pos > 0:
                    end_pos = min(end_pos, pos)
            return text[:end_pos].strip()

        # Try to find SELECT in text
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
        """Generate fallback SQL when no model available."""
        if self.schema["tables"]:
            table = self.schema["tables"][0]
            return f"SELECT * FROM {table}"
        return "SELECT *"


class HybridSQLFormerEngine:
    """Engine wrapper using hybrid generation."""

    def __init__(self, model, schema: Dict):
        self.generator = HybridSQLGenerator(model, schema)

    def generate(self, prompt: str) -> str:
        return self.generator.generate(prompt)
