def build_zero_shot_prompt(schema: str, question: str) -> str:
    return f"""Database schema:
{schema}

Question:
{question}

SQL:
"""
