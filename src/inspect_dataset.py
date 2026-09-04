from datasets import load_dataset

dataset = load_dataset("xlangai/spider")


def main():
    print("Available splits:", list(dataset.keys()))
    print("Number of training examples:", len(dataset["train"]))
    print("Number of validation examples:", len(dataset["validation"]))
    print()

    for i in range(3):
        example = dataset["train"][i]
        print(f"=== EXAMPLE {i + 1} ===")
        print("Database ID:", example["db_id"])
        print("Question:", example["question"])
        print("Gold SQL:", example["query"])
        print()


if __name__ == "__main__":
    main()
