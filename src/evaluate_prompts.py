import json
import os
import random
import sqlite3
import time

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.prompts import (
    build_zero_shot_prompt,
    build_instruction_prompt,
    build_few_shot_prompt,
)

DEV_PATH = "data/spider/dev.json"
DATABASE_DIR = "data/spider/database"
RESULTS_PATH = "results/prompt_evaluation.json"
MODEL_NAME = "Qwen/Qwen2.5-Coder-1.5B-Instruct"

SEED = 42
NUM_EXAMPLES = 30
MAX_NEW_TOKENS = 128

METHODS = {
    "zero_shot": build_zero_shot_prompt,
    "instruction": build_instruction_prompt,
    "few_shot": build_few_shot_prompt,
}


def get_schema_text(cursor):
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row[0] for row in cursor.fetchall() if not row[0].startswith("sqlite_")]

    schema_parts = []
    for table in tables:
        cursor.execute(f"PRAGMA table_info('{table}')")
        columns = cursor.fetchall()
        column_lines = ",\n".join(f"    {col[1]} {col[2]}" for col in columns)
        schema_parts.append(f"CREATE TABLE {table} (\n{column_lines}\n);")

    return "\n\n".join(schema_parts)


def extract_first_sql(text):
    text = text.strip()
    if ";" in text:
        return text.split(";")[0].strip() + ";"
    lines = text.splitlines()
    return lines[0].strip() if lines else ""


def normalize_result(rows):
    return sorted([tuple(row) for row in rows], key=lambda row: [str(x) for x in row])


def execute_sql(cursor, sql):
    try:
        cursor.execute(sql)
        rows = cursor.fetchall()
        return True, rows, None
    except Exception as e:
        return False, None, str(e)


def main():
    with open(DEV_PATH, encoding="utf-8") as f:
        dev_data = json.load(f)

    random.seed(SEED)
    indices = random.sample(range(len(dev_data)), NUM_EXAMPLES)
    examples = [dev_data[i] for i in indices]

    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        torch_dtype="auto",
    )
    model.eval()

    connections = {}
    schema_cache = {}

    def get_connection(db_id):
        if db_id not in connections:
            db_path = os.path.join(DATABASE_DIR, db_id, f"{db_id}.sqlite")
            connections[db_id] = sqlite3.connect(db_path)
        return connections[db_id]

    def get_schema(db_id):
        if db_id not in schema_cache:
            cursor = get_connection(db_id).cursor()
            schema_cache[db_id] = get_schema_text(cursor)
        return schema_cache[db_id]

    stats = {
        method: {"valid": 0, "correct": 0, "total_latency": 0.0}
        for method in METHODS
    }
    detailed_results = []

    for i, example in enumerate(examples):
        db_id = example["db_id"]
        question = example["question"]
        gold_sql = example["query"]

        conn = get_connection(db_id)
        schema = get_schema(db_id)

        gold_cursor = conn.cursor()
        gold_valid, gold_rows, gold_error = execute_sql(gold_cursor, gold_sql)
        gold_result = normalize_result(gold_rows) if gold_valid else None

        record = {
            "db_id": db_id,
            "question": question,
            "gold_sql": gold_sql,
            "gold_result": gold_result,
            "methods": {},
        }

        for method_name, build_prompt_fn in METHODS.items():
            prompt = build_prompt_fn(schema, question)
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

            start_time = time.time()
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                )
            latency = time.time() - start_time

            generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
            output_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            generated_sql = extract_first_sql(output_text)

            gen_cursor = conn.cursor()
            gen_valid, gen_rows, gen_error = execute_sql(gen_cursor, generated_sql)
            gen_result = normalize_result(gen_rows) if gen_valid else None

            is_correct = (
                gold_valid and gen_valid and gold_result == gen_result
            )

            stats[method_name]["total_latency"] += latency
            if gen_valid:
                stats[method_name]["valid"] += 1
            if is_correct:
                stats[method_name]["correct"] += 1

            record["methods"][method_name] = {
                "generated_sql": generated_sql,
                "valid": gen_valid,
                "error": gen_error,
                "correct": is_correct,
                "latency": latency,
            }

        detailed_results.append(record)
        print(f"Processed {i + 1}/{NUM_EXAMPLES}")

    for conn in connections.values():
        conn.close()

    os.makedirs("results", exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(detailed_results, f, indent=2)

    labels = {
        "zero_shot": "ZERO-SHOT",
        "instruction": "INSTRUCTION",
        "few_shot": "FEW-SHOT",
    }

    for method_name, label in labels.items():
        s = stats[method_name]
        validity_pct = 100 * s["valid"] / NUM_EXAMPLES
        accuracy_pct = 100 * s["correct"] / NUM_EXAMPLES
        avg_latency = s["total_latency"] / NUM_EXAMPLES

        print()
        print(f"=== {label} ===")
        print(f"SQL validity: {validity_pct:.1f}%")
        print(f"Execution accuracy: {accuracy_pct:.1f}%")
        print(f"Average latency: {avg_latency:.2f}s")


if __name__ == "__main__":
    main()
