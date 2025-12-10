"""
Unconstrained SQL Generator

This approach uses standard LLM generation without any constraints.
Serves as the baseline for comparison.
"""

import re
from typing import Optional


class UnconstrainedSQLGenerator:
    """
    Generate SQL using unconstrained LLM generation.

    This is the baseline approach that lets the model generate freely
    without any schema constraints or validation.
    """

    def __init__(self, model, schema: dict):
        """
        Initialize unconstrained generator.

        Args:
            model: SQLFormerModel instance
            schema: Schema dict (not used, kept for API consistency)
        """
        self.model = model
        self.schema = schema

    def generate(self, prompt: str, max_new_tokens: int = 150) -> str:
        """
        Generate SQL with unconstrained LLM.

        Args:
            prompt: Full prompt including question and schema info
            max_new_tokens: Maximum tokens to generate

        Returns:
            Generated SQL query
        """
        if self.model is None:
            return self._generate_fallback()

        # Generate SQL without constraints
        raw_output = self.model.generate_unconstrained(
            prompt,
            max_new_tokens=max_new_tokens
        )

        # Extract and clean SQL
        sql = self._extract_sql(raw_output)

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
        if self.schema and "tables" in self.schema and self.schema["tables"]:
            table = self.schema["tables"][0]
            return f"SELECT * FROM {table}"
        return "SELECT *"


class UnconstrainedSQLFormerEngine:
    """Engine wrapper for unconstrained generation."""

    def __init__(self, model, schema: dict):
        self.generator = UnconstrainedSQLGenerator(model, schema)

    def generate(self, prompt: str) -> str:
        return self.generator.generate(prompt)
