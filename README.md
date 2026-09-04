# Text-to-SQL with QLoRA

This project compares prompt-engineering approaches and QLoRA fine-tuning for generating SQL from natural-language questions and database schemas using Qwen2.5-Coder-1.5B-Instruct and the Spider dataset.

## Overview

The core flow is simple:

```
Database schema + natural-language question
        ↓
     Qwen2.5-Coder
        ↓
     generated SQL
```

The project compares four approaches to this task:

1. Zero-shot prompting
2. Instruction prompting
3. Few-shot prompting
4. QLoRA fine-tuning

## Model and Dataset

**Model:** `Qwen/Qwen2.5-Coder-1.5B-Instruct`

**Dataset:** [Spider](https://yale-lily.github.io/spider) Text-to-SQL dataset

- 7000 Spider training examples
- 1034 validation examples
- 500 examples used for the final QLoRA training experiment
- 30 fixed validation examples (seed=42) used for the controlled comparison across all methods

## QLoRA Setup

- 4-bit NF4 quantization (`bitsandbytes`)
- LoRA r=8
- LoRA alpha=16
- LoRA dropout=0.05
- Target modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`
- Max sequence length: 512
- Batch size: 1 (gradient accumulation: 4)
- 1 epoch, 125 optimizer steps
- Hardware: RTX 4070 Laptop GPU, 8 GB VRAM
- Peak GPU memory: ~3.02 GB
- Training time: ~3 minutes

Only the LoRA adapter parameters were trained — the base Qwen2.5-Coder model remained frozen and quantized throughout. This is not full-model fine-tuning.

## Evaluation

Three metrics are used, all computed by executing SQL against the actual SQLite databases:

- **SQL validity** — the generated SQL executes successfully.
- **Execution accuracy** — the generated SQL returns the same result as the gold SQL.
- **Average generation latency** — time required to generate SQL.

Results on the same 30 fixed Spider validation examples (seed=42):

| Method | SQL Validity | Execution Accuracy | Avg. Latency |
| --- | ---: | ---: | ---: |
| Zero-shot | 83.3% | 46.7% | 3.23 s |
| Instruction prompting | 80.0% | 50.0% | 2.12 s |
| Few-shot prompting | 83.3% | 60.0% | 3.27 s |
| QLoRA-500 | 83.3% | 60.0% | 1.47 s |

QLoRA-500 matched few-shot prompting at 60% execution accuracy, while being the fastest method at inference time. Unlike few-shot prompting, QLoRA did not require examples to be included inside every inference prompt. This was a small, 30-example controlled evaluation, so these results should not be treated as a full Spider benchmark.

## Gradio Demo

A minimal local Gradio demo is included for interacting with the QLoRA-fine-tuned model.

**Input:**
- Database schema
- Natural-language question

**Output:**
- Generated SQL

Run:

```
python -m src.demo
```

Then open the local Gradio URL printed in the terminal.

## Project Structure

```
text-to-sql-lora/
├── src/
│   ├── prompts.py
│   ├── baseline.py
│   ├── instruction_baseline.py
│   ├── few_shot_baseline.py
│   ├── evaluate_prompts.py
│   ├── inspect_dataset.py
│   ├── verify_spider.py
│   ├── prepare_training_data.py
│   ├── analyze_token_lengths.py
│   ├── train_smoke_test.py
│   ├── train_smoke_test_512.py
│   ├── train_qlora_500.py
│   ├── evaluate_qlora_500.py
│   └── demo.py
├── data/
├── results/
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

`data/` and `results/` contain locally downloaded/generated artifacts (the Spider dataset, prepared training data, trained adapters, and evaluation outputs) and are not committed to the repository.

## Setup

1. Create and activate a Python virtual environment.
2. Install dependencies: `pip install -r requirements.txt`
3. Download the [Spider dataset](https://yale-lily.github.io/spider) separately and place it under `data/spider/` (it is not bundled with this repository).
4. Prepare the training data: `python -m src.prepare_training_data`
5. Run any of the scripts below.

## Main Commands

```
python -m src.evaluate_prompts
python -m src.prepare_training_data
python -m src.analyze_token_lengths
python -m src.train_qlora_500
python -m src.evaluate_qlora_500
python -m src.demo
```

## Key Takeaway

This project demonstrates end-to-end practical experience with prompt engineering, PEFT/LoRA/QLoRA fine-tuning, Hugging Face Transformers, Text-to-SQL evaluation, SQLite execution-based evaluation, GPU-efficient fine-tuning on consumer hardware, and building a Gradio model demo.
