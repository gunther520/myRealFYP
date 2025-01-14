# Step 1: Import necessary libraries
from datasets import load_dataset, concatenate_datasets

# Step 2: Define a list of dataset names, their configurations, the split to load, and the columns to keep
datasets_info = [
    ("PromptEval/PromptEval_MMLU_full", "format_0", "meta_llama_llama_3_8b", ["question"]),
    ("tatsu-lab/alpaca", None, "train", ["instruction"]),
    ("fka/awesome-chatgpt-prompts", None, "train", ["prompt"]),
    ("google/frames-benchmark",None, "test", ["Prompt"]),
    ("KingNish/reasoning-base-20k", None, "train", ["user"]),
    ("mlabonne/orpo-dpo-mix-40k", None, "train", ["prompt"]),
    ("argilla/magpie-ultra-v0.1", None, "train", ["instruction"]),
    ("re-align/just-eval-instruct", "default", "test", ["instruction"]),

]

# Step 3: Load each dataset using the `datasets` library from Hugging Face
datasets = []
for name, config, split, columns in datasets_info:
    dataset = load_dataset(name, config, split=split) if config else load_dataset(name, split=split)
    # Keep only the specified columns
    dataset = dataset.remove_columns([col for col in dataset.column_names if col not in columns])
    num_records = min(150, len(dataset))
    dataset = dataset.select(range(num_records))
    datasets.append(dataset.rename_column(columns[0], "input"))

# Step 4: Concatenate all datasets into one
concatenated_dataset = concatenate_datasets(datasets)

# Step 5: Save the concatenated dataset to a file
concatenated_dataset.push_to_hub("dataset-concat")