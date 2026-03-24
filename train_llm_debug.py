import os
import json
from transformers import AutoTokenizer
from tqdm import tqdm

# Config
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
data_file = "train_alpaca.jsonl"
max_seq_length = 1024
template = """### Instruction:
{instruction}
### Input:
{input}
### Response:
{output}"""

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token_id is None or tokenizer.pad_token_id == tokenizer.eos_token_id:
    tokenizer.add_special_tokens({"pad_token": "<PAD>"})
    tokenizer.pad_token = "<PAD>"

# Load and process data
total = 0
truncated = []
missing_response_tag = []
failures = []

with open(data_file, "r", encoding="utf-8") as f:
    for i, line in enumerate(tqdm(f, desc="Checking examples")):
        try:
            ex = json.loads(line)
            prompt = template.format(
                instruction=ex.get("instruction", ""),
                input=ex.get("input", ""),
                output=ex.get("output", "")
            )

            # Check if response tag got truncated
            tokenized = tokenizer(
                prompt,
                truncation=True,
                max_length=max_seq_length,
                padding="max_length",
                return_tensors=None
            )

            decoded = tokenizer.decode(tokenized["input_ids"], skip_special_tokens=True)
            if "### Response:" not in decoded:
                truncated.append(i + 1)

            if "### Response:" not in prompt:
                missing_response_tag.append(i + 1)

        except Exception as e:
            failures.append((i + 1, str(e)))

        total += 1

# Report
print(f"Total examples checked: {total}")
print(f"Truncated before '### Response:': {len(truncated)}")
print(f"Missing '### Response:' in original prompt: {len(missing_response_tag)}")
print(f"JSON or tokenization failures: {len(failures)}")
if truncated:
    print(f"Example truncated line numbers: {truncated[:10]}")

