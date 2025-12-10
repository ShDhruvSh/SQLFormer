import re
from typing import Optional


# extract sql from generated text
def extract_sql_from_text(generated_text: str) -> str:
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


# balance unmatched parentheses
def balance_parentheses(sql: str) -> str:
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


# close unclosed quotes
def close_quotes(sql: str) -> str:
    single_quotes = sql.count("'")
    double_quotes = sql.count('"')

    if single_quotes % 2 == 1:
        sql += "'"
    if double_quotes % 2 == 1:
        sql += '"'

    return sql


# remove trailing incomplete clauses
def clean_incomplete_clauses(sql: str) -> str:
    incomplete_keywords = [' WHERE', ' AND', ' OR', ' ORDER BY', ' GROUP BY', ' HAVING']

    for kw in incomplete_keywords:
        if sql.upper().endswith(kw):
            sql = sql[:sql.upper().rfind(kw)].strip()

    return sql


# format and clean sql code
def format_sql(sql: str) -> str:
    if ';' in sql:
        sql = sql.split(';')[0].strip()

    sql = re.sub(r'\s+', ' ', sql)

    return sql.strip()
