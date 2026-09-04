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


def build_few_shot_prompt(schema: str, question: str) -> str:
    return f"""Database schema:
CREATE TABLE students (
    id INTEGER,
    name TEXT,
    grade INTEGER
);

Question:
List the names of all students.

SQL:
SELECT name FROM students;

Database schema:
CREATE TABLE products (
    id INTEGER,
    name TEXT,
    price REAL
);

Question:
What is the most expensive product?

SQL:
SELECT name FROM products ORDER BY price DESC LIMIT 1;

Database schema:
CREATE TABLE orders (
    id INTEGER,
    customer_id INTEGER,
    total REAL
);

Question:
How many orders are there?

SQL:
SELECT COUNT(*) FROM orders;

Database schema:
{schema}

Question:
{question}

SQL:
"""
