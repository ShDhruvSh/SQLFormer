"""
SQL Utility Functions

Common utilities for SQL extraction, cleaning, and validation.
"""

import re
from typing import Optional


def extract_sql_from_text(generated_text: str) -> str:
    """
    Extract SQL query from generated text.

    Handles markdown code blocks and finds SELECT statements.

    Args:
        generated_text: Raw text output from LLM

    Returns:
        Extracted SQL query
    """
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


def balance_parentheses(sql: str) -> str:
    """
    Balance unmatched parentheses in SQL.

    Args:
        sql: SQL query string

    Returns:
        SQL with balanced parentheses
    """
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


def close_quotes(sql: str) -> str:
    """
    Close unclosed quotes in SQL.

    Args:
        sql: SQL query string

    Returns:
        SQL with closed quotes
    """
    # Count quotes
    single_quotes = sql.count("'")
    double_quotes = sql.count('"')

    # If odd number, add closing quote at end
    if single_quotes % 2 == 1:
        sql += "'"
    if double_quotes % 2 == 1:
        sql += '"'

    return sql


def clean_incomplete_clauses(sql: str) -> str:
    """
    Remove trailing incomplete SQL clauses.

    Args:
        sql: SQL query string

    Returns:
        SQL with incomplete clauses removed
    """
    # Remove trailing incomplete clauses
    incomplete_keywords = [' WHERE', ' AND', ' OR', ' ORDER BY', ' GROUP BY', ' HAVING']

    for kw in incomplete_keywords:
        if sql.upper().endswith(kw):
            sql = sql[:sql.upper().rfind(kw)].strip()

    return sql


def format_sql(sql: str) -> str:
    """
    Basic SQL formatting and cleanup.

    Args:
        sql: SQL query string

    Returns:
        Formatted SQL
    """
    # Remove content after semicolon
    if ';' in sql:
        sql = sql.split(';')[0].strip()

    # Normalize whitespace
    sql = re.sub(r'\s+', ' ', sql)

    return sql.strip()
