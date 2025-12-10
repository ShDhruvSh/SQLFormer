import re
from typing import Optional


class UnconstrainedSQLGenerator:

    def __init__(self, model, schema: dict):
        self.model = model
        self.schema = schema

    def generate(self, prompt: str, max_new_tokens: int = 150) -> str:
        if self.model is None:
            return self._generate_fallback()

        # standard generation without constraints
        ids = self.model.encode(prompt)
        output = self.model.model.generate(ids, max_new_tokens=max_new_tokens)
        raw_output = self.model.decode(output[0])

        return self._extract_sql(raw_output)

    # pull out sql from model output
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
        if self.schema and "tables" in self.schema and self.schema["tables"]:
            return f"SELECT * FROM {self.schema['tables'][0]}"
        return "SELECT *"


class UnconstrainedSQLFormerEngine:

    def __init__(self, model, schema: dict):
        self.generator = UnconstrainedSQLGenerator(model, schema)

    def generate(self, prompt: str, max_new_tokens: int = 1024) -> str:
        return self.generator.generate(prompt, max_new_tokens=max_new_tokens)
