import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

MODEL_NAME = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
ADAPTER_PATH = "results/qlora_500"
MAX_NEW_TOKENS = 128

PROMPT_TEMPLATE = """Generate the SQL query that answers the question.
Use only the provided database schema.

Database schema:
{schema}

Question:
{question}

SQL:
"""

DEFAULT_SCHEMA = """CREATE TABLE employees (
    id INTEGER,
    name TEXT,
    department_id INTEGER,
    salary REAL
);

CREATE TABLE departments (
    id INTEGER,
    name TEXT
);"""

DEFAULT_QUESTION = "Which department has the highest average employee salary?"

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


def extract_first_sql(text):
    text = text.strip()
    if ";" in text:
        return text.split(";")[0].strip() + ";"
    lines = text.splitlines()
    return lines[0].strip() if lines else ""


def generate_sql(schema, question):
    prompt = PROMPT_TEMPLATE.format(schema=schema, question=question)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
        )

    generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    output_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return extract_first_sql(output_text)


demo = gr.Interface(
    fn=generate_sql,
    inputs=[
        gr.Textbox(label="Database Schema", lines=10, value=DEFAULT_SCHEMA),
        gr.Textbox(label="Natural-Language Question", value=DEFAULT_QUESTION),
    ],
    outputs=gr.Textbox(label="Generated SQL"),
    title="Text-to-SQL with QLoRA",
    description="Generate SQL from a database schema and natural-language question using a QLoRA-fine-tuned Qwen2.5-Coder model.",
)

if __name__ == "__main__":
    demo.launch()
