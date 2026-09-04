import json
import os
import random
import sqlite3
import time

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

from src.evaluate_prompts import (
    DEV_PATH,
    DATABASE_DIR,
    SEED,
    NUM_EXAMPLES,
    get_schema_text,
    extract_first_sql,
    normalize_result,
    execute_sql,
)

MODEL_NAME = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
ADAPTER_PATH = "results/qlora_500"
RESULTS_PATH = "results/qlora_500_evaluation.json"
MAX_NEW_TOKENS = 128

PROMPT_TEMPLATE = """Generate the SQL query that answers the question.
Use only the provided database schema.

Database schema:
{schema}

Question:
{question}

SQL:
"""

PREVIOUS_RESULTS = {
    "zero_shot": 46.7,
    "instruction": 50.0,
    "few_shot": 60.0,
}


def main():
    cuda_available = torch.cuda.is_available()
    print("CUDA available:", cuda_available)
    if cuda_available:
        print("GPU name:", torch.cuda.get_device_name(0))

    with open(DEV_PATH, encoding="utf-8") as f:
        dev_data = json.load(f)

    random.seed(SEED)
    indices = random.sample(range(len(dev_data)), NUM_EXAMPLES)
    examples = [dev_data[i] for i in indices]

    print("Loading base model + LoRA adapter...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
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

    valid_count = 0
    correct_count = 0
    total_latency = 0.0
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

        prompt = PROMPT_TEMPLATE.format(schema=schema, question=question)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        start_time = time.time()
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
            )
        latency = time.time() - start_time
        total_latency += latency

        generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        output_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        generated_sql = extract_first_sql(output_text)

        gen_cursor = conn.cursor()
        gen_valid, gen_rows, gen_error = execute_sql(gen_cursor, generated_sql)
        gen_result = normalize_result(gen_rows) if gen_valid else None

        is_correct = gold_valid and gen_valid and gold_result == gen_result

        if gen_valid:
            valid_count += 1
        if is_correct:
            correct_count += 1

        detailed_results.append({
            "db_id": db_id,
            "question": question,
            "gold_sql": gold_sql,
            "generated_sql": generated_sql,
            "gold_result": gold_result,
            "generated_result": gen_result if gen_valid else None,
            "sql_valid": gen_valid,
            "execution_match": is_correct,
            "generation_latency": latency,
        })

        print(f"Processed {i + 1}/{NUM_EXAMPLES}")

    for conn in connections.values():
        conn.close()

    os.makedirs("results", exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(detailed_results, f, indent=2)

    validity_pct = 100 * valid_count / NUM_EXAMPLES
    accuracy_pct = 100 * correct_count / NUM_EXAMPLES
    avg_latency = total_latency / NUM_EXAMPLES

    print()
    print("=== QLORA-500 ===")
    print(f"SQL validity: {validity_pct:.1f}%")
    print(f"Execution accuracy: {accuracy_pct:.1f}%")
    print(f"Average latency: {avg_latency:.2f}s")

    print()
    print("=== COMPARISON ===")
    print(f"Zero-shot execution accuracy: {PREVIOUS_RESULTS['zero_shot']}%")
    print(f"Instruction execution accuracy: {PREVIOUS_RESULTS['instruction']}%")
    print(f"Few-shot execution accuracy: {PREVIOUS_RESULTS['few_shot']}%")
    print(f"QLoRA-500 execution accuracy: {accuracy_pct:.1f}%")


if __name__ == "__main__":
    main()
