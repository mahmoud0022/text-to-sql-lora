def build_zero_shot_prompt(schema: str, question: str) -> str:
    return f"""Database schema:
{schema}

Question:
{question}

SQL:
"""


def build_instruction_prompt(schema: str, question: str) -> str:
    return f"""You are given a database schema and a question.
Generate the SQL query that answers the question.

Rules:
- Use only tables and columns from the provided schema.
- Return only SQL.
- Do not include explanations.
- Do not use markdown code fences.

Database schema:
{schema}

Question:
{question}

SQL:
"""
