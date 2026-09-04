import json

import numpy as np
from transformers import AutoTokenizer

DATA_PATH = "data/training/train_2000.jsonl"
MODEL_NAME = "Qwen/Qwen2.5-Coder-1.5B-Instruct"

THRESHOLDS = [256, 384, 512]


def load_records(path):
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    return records


def main():
    records = load_records(DATA_PATH)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    lengths = []
    for record in records:
        text = record["prompt"] + record["target_sql"]
        token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        lengths.append(len(token_ids))

    lengths_arr = np.array(lengths)
    num_examples = len(lengths_arr)

    print("Number of examples:", num_examples)
    print("Minimum token length:", int(lengths_arr.min()))
    print("Average token length:", round(float(lengths_arr.mean()), 2))
    print("Median token length:", float(np.median(lengths_arr)))
    print("90th percentile:", float(np.percentile(lengths_arr, 90)))
    print("95th percentile:", float(np.percentile(lengths_arr, 95)))
    print("99th percentile:", float(np.percentile(lengths_arr, 99)))
    print("Maximum token length:", int(lengths_arr.max()))
    print()

    for threshold in THRESHOLDS:
        count = int((lengths_arr > threshold).sum())
        pct = 100 * count / num_examples
        print(f"Longer than {threshold} tokens: {count} ({pct:.2f}%)")
    print()

    order = np.argsort(lengths_arr)[::-1][:3]
    print("=== 3 LONGEST EXAMPLES ===")
    for rank, idx in enumerate(order, start=1):
        record = records[idx]
        print(f"#{rank}")
        print("Database ID:", record["db_id"])
        print("Question:", record["question"])
        print("Token length:", int(lengths_arr[idx]))
        print()


if __name__ == "__main__":
    main()
