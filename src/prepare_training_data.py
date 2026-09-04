import json
import os
import random

TRAIN_PATH = "data/spider/train_spider.json"
TABLES_PATH = "data/spider/tables.json"
OUTPUT_PATH = "data/training/train_2000.jsonl"

SEED = 42
NUM_EXAMPLES = 2000

PROMPT_TEMPLATE = """Generate the SQL query that answers the question.
Use only the provided database schema.

Database schema:
{schema}

Question:
{question}

SQL:
"""


def build_schema_text(schema_entry):
    table_names = schema_entry["table_names_original"]
    column_names = schema_entry["column_names_original"]
    primary_keys = schema_entry["primary_keys"]
    foreign_keys = schema_entry["foreign_keys"]

    lines = []
    for table_idx, table_name in enumerate(table_names):
        columns = [col_name for t_idx, col_name in column_names if t_idx == table_idx]
        lines.append(f"Table: {table_name}")
        lines.append(f"Columns: {', '.join(columns)}")
        lines.append("")

    lines.append("Primary Keys:")
    for col_idx in primary_keys:
        t_idx, col_name = column_names[col_idx]
        lines.append(f"  {table_names[t_idx]}.{col_name}")
    lines.append("")

    lines.append("Foreign Keys:")
    for from_idx, to_idx in foreign_keys:
        from_t_idx, from_col = column_names[from_idx]
        to_t_idx, to_col = column_names[to_idx]
        lines.append(f"  {table_names[from_t_idx]}.{from_col} -> {table_names[to_t_idx]}.{to_col}")

    return "\n".join(lines)


def main():
    with open(TRAIN_PATH, encoding="utf-8") as f:
        train_data = json.load(f)

    with open(TABLES_PATH, encoding="utf-8") as f:
        tables_data = json.load(f)

    schema_by_db = {t["db_id"]: build_schema_text(t) for t in tables_data}

    random.seed(SEED)
    indices = random.sample(range(len(train_data)), NUM_EXAMPLES)
    selected = [train_data[i] for i in indices]

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    records = []
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for example in selected:
            db_id = example["db_id"]
            question = example["question"]
            gold_sql = example["query"]
            schema = schema_by_db[db_id]

            prompt = PROMPT_TEMPLATE.format(schema=schema, question=question)

            record = {
                "db_id": db_id,
                "question": question,
                "schema": schema,
                "prompt": prompt,
                "target_sql": gold_sql,
            }
            records.append(record)
            f.write(json.dumps(record) + "\n")

    print("Total original training examples:", len(train_data))
    print("Selected examples:", len(records))
    print()

    for i in range(2):
        record = records[i]
        print(f"=== FORMATTED EXAMPLE {i + 1} ===")
        print("INPUT:")
        print()
        print(record["prompt"])
        print("TARGET:")
        print()
        print(record["target_sql"])
        print()


if __name__ == "__main__":
    main()
