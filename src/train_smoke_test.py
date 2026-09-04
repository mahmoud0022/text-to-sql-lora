import json

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

DATA_PATH = "data/training/train_2000.jsonl"
OUTPUT_DIR = "results/qlora_smoke_test"
MODEL_NAME = "Qwen/Qwen2.5-Coder-1.5B-Instruct"

NUM_EXAMPLES = 50
MAX_SEQ_LENGTH = 256
MAX_STEPS = 10


def load_records(path, n):
    records = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            records.append(json.loads(line))
    return records


def tokenize_example(record, tokenizer, max_length):
    prompt = record["prompt"]
    target = record["target_sql"]
    full_text = prompt + target + tokenizer.eos_token

    prompt_ids = tokenizer(prompt, truncation=True, max_length=max_length, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full_text, truncation=True, max_length=max_length, add_special_tokens=False)["input_ids"]

    labels = list(full_ids)
    prompt_len = min(len(prompt_ids), len(full_ids))
    for i in range(prompt_len):
        labels[i] = -100

    pad_len = max_length - len(full_ids)
    input_ids = full_ids + [tokenizer.pad_token_id] * pad_len
    attention_mask = [1] * len(full_ids) + [0] * pad_len
    labels = labels + [-100] * pad_len

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


class SmokeTestDataset(torch.utils.data.Dataset):
    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        item = self.examples[idx]
        return {
            "input_ids": torch.tensor(item["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(item["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(item["labels"], dtype=torch.long),
        }


def print_memory(label):
    allocated = torch.cuda.memory_allocated() / (1024 ** 2)
    reserved = torch.cuda.memory_reserved() / (1024 ** 2)
    print(f"[{label}] Allocated: {allocated:.1f} MB | Reserved: {reserved:.1f} MB")


def main():
    cuda_available = torch.cuda.is_available()
    print("CUDA available:", cuda_available)
    if cuda_available:
        print("GPU name:", torch.cuda.get_device_name(0))
    print_memory("before load")

    records = load_records(DATA_PATH, NUM_EXAMPLES)
    print("Training examples:", len(records))

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
    )

    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print_memory("after model load")

    tokenized = [tokenize_example(r, tokenizer, MAX_SEQ_LENGTH) for r in records]
    dataset = SmokeTestDataset(tokenized)

    training_args = TrainingArguments(
        output_dir=f"{OUTPUT_DIR}/checkpoints",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        max_steps=MAX_STEPS,
        num_train_epochs=1,
        learning_rate=2e-4,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=1,
        save_strategy="no",
        eval_strategy="no",
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )

    train_result = trainer.train()

    print_memory("after training")
    if cuda_available:
        max_allocated = torch.cuda.max_memory_allocated() / (1024 ** 2)
        print(f"Max allocated: {max_allocated:.1f} MB")

    print("Final training loss:", train_result.training_loss)

    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("LoRA adapter and tokenizer saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
