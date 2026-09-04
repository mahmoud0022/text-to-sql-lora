import json
import os
import sqlite3

DEV_PATH = "data/spider/dev.json"
TABLES_PATH = "data/spider/tables.json"
DATABASE_DIR = "data/spider/database"


def main():
    with open(DEV_PATH, encoding="utf-8") as f:
        dev_data = json.load(f)

    with open(TABLES_PATH, encoding="utf-8") as f:
        tables_data = json.load(f)

    example = dev_data[0]
    db_id = example["db_id"]
    question = example["question"]
    gold_sql = example["query"]

    print("=== SPIDER EXAMPLE ===")
    print("Database ID:", db_id)
    print("Question:", question)
    print("Gold SQL:", gold_sql)
    print()

    schema = next(t for t in tables_data if t["db_id"] == db_id)

    print("=== SCHEMA FROM TABLES.JSON ===")
    print()
    table_names = schema["table_names_original"]
    column_names = schema["column_names_original"]
    primary_keys = schema["primary_keys"]
    foreign_keys = schema["foreign_keys"]

    for table_idx, table_name in enumerate(table_names):
        print(f"Table: {table_name}")
        columns = [col for col in column_names if col[0] == table_idx]
        for col_idx, col_name in columns:
            print(f"  - {col_name}")
        print()

    print("Primary keys:")
    for col_idx in primary_keys:
        table_idx, col_name = column_names[col_idx]
        print(f"  - {table_names[table_idx]}.{col_name}")
    print()

    print("Foreign keys:")
    for from_idx, to_idx in foreign_keys:
        from_table_idx, from_col = column_names[from_idx]
        to_table_idx, to_col = column_names[to_idx]
        print(f"  - {table_names[from_table_idx]}.{from_col} -> {table_names[to_table_idx]}.{to_col}")
    print()

    db_path = os.path.join(DATABASE_DIR, db_id, f"{db_id}.sqlite")
    db_exists = os.path.isfile(db_path)

    print("=== SQLITE DATABASE ===")
    print("Path:", db_path)
    print("Exists:", db_exists)
    print()

    if not db_exists:
        return

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        sqlite_tables = [row[0] for row in cursor.fetchall()]

        print("=== TABLES FROM SQLITE ===")
        for name in sqlite_tables:
            print(f"  - {name}")
        print()

        for table_name in sqlite_tables:
            print(f"Columns in {table_name}:")
            cursor.execute(f"PRAGMA table_info('{table_name}')")
            for col in cursor.fetchall():
                print(f"  - {col[1]} ({col[2]})")
            print()

        print("=== GOLD SQL EXECUTION ===")
        print("Gold SQL:", gold_sql)
        cursor.execute(gold_sql)
        result = cursor.fetchall()
        print("Result:", result)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
