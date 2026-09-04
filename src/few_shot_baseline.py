import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.prompts import build_few_shot_prompt

MODEL_NAME = "Qwen/Qwen2.5-Coder-1.5B-Instruct"

SCHEMA = """CREATE TABLE employees (
    id INTEGER,
    name TEXT,
    department_id INTEGER,
    salary REAL
);

CREATE TABLE departments (
    id INTEGER,
    name TEXT
);"""

QUESTION = "Which department has the highest average employee salary?"


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        torch_dtype="auto",
    )
    model.eval()

    prompt = build_few_shot_prompt(SCHEMA, QUESTION)

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
        )

    generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    output_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    print("=== FEW-SHOT PROMPT ===")
    print()
    print(prompt)
    print()
    print("=== MODEL OUTPUT ===")
    print()
    print(output_text)


if __name__ == "__main__":
    main()
