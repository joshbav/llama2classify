# tests truncation of the training dataset

import json
from transformers import AutoTokenizer

# Configuration
data_file = "train_alpaca.jsonl"
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
max_seq_length = 512

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token_id is None:
    tokenizer.add_special_tokens({"pad_token": "<PAD>"})

# Alpaca format template
PROMPT_TEMPLATE = """### Instruction:
{instruction}
### Input:
{input}
### Response:
{output}"""

# Tracking
truncation_issues = []
missing_response_tag = []
failed_lines = []

with open(data_file, "r", encoding="utf-8") as f:
    for i, line in enumerate(f, 1):
        try:
            record = json.loads(line)
            prompt = PROMPT_TEMPLATE.format(
                instruction=record.get("instruction", ""),
                input=record.get("input", ""),
                output=record.get("output", "")
            )

            if "### Response:" not in prompt:
                missing_response_tag.append(i)

            tokenized = tokenizer(
                prompt,
                truncation=True,
                max_length=max_seq_length,
                padding="max_length"
            )

            decoded = tokenizer.decode(tokenized["input_ids"], skip_special_tokens=False)

            if "### Response:" not in decoded:
                truncation_issues.append(i)

        except Exception as e:
            failed_lines.append((i, str(e)))

# Report
print(f"Total examples checked: {i}")
print(f"Truncated before '### Response:': {len(truncation_issues)}")
print(f"Missing '### Response:' in original prompt: {len(missing_response_tag)}")
print(f"JSON or tokenization failures: {len(failed_lines)}")

if truncation_issues:
    print(f"Example truncated line numbers: {truncation_issues[:10]}")

if failed_lines:
    print(f"Example error: {failed_lines[0]}")

