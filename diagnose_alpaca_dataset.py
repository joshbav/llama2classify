# trying to figure out why the training job can't handle the dataset

import json
from transformers import AutoTokenizer

# Configuration
file_path = "train_alpaca.filtered.jsonl"
model_name_or_path = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
max_seq_length = 1024
response_marker = "### Response:"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

# Track issues
total_examples = 0
truncated_lines = []
missing_fields = []
capitalization_issues = []
tokenized_response_missing = []
short_output_lines = []

def field_case_insensitive_get(d, key):
    for k in d:
        if k.lower() == key.lower():
            return d[k]
    return None

with open(file_path, "r", encoding="utf-8") as f:
    for line_num, line in enumerate(f, start=1):
        try:
            data = json.loads(line)
            total_examples += 1

            # Check for lowercase field keys
            original_keys = list(data.keys())
            expected_keys = {"instruction", "input", "output"}
            lower_keys = {k.lower() for k in original_keys}

            if not expected_keys.issubset(lower_keys):
                capitalization_issues.append((line_num, original_keys))

            instruction = field_case_insensitive_get(data, "instruction") or ""
            input_text = field_case_insensitive_get(data, "input") or ""
            output = field_case_insensitive_get(data, "output") or ""

            if len(output.strip()) < 2:
                short_output_lines.append(line_num)

            prompt = f"### Instruction:\n{instruction}\n### Input:\n{input_text}\n### Response:\n{output}"

            if "### Response:" not in prompt:
                capitalization_issues.append((line_num, "Response label not properly capitalized"))

            encoded = tokenizer(prompt, truncation=True, max_length=max_seq_length, padding="max_length", return_tensors="pt")
            decoded = tokenizer.decode(encoded["input_ids"][0], skip_special_tokens=False)

            if "### Response:" not in decoded:
                truncated_lines.append(line_num)

            joined_tokens = " ".join(tokenizer.convert_ids_to_tokens(encoded["input_ids"][0]))
            if response_marker.replace(" ", "") not in joined_tokens.replace("▁", "").replace(" ", ""):
                tokenized_response_missing.append(line_num)

        except Exception as e:
            print(f"[Line {line_num}] Error during JSON parse or tokenization: {e}")

# Final Report
print(f"Total examples checked: {total_examples}")
print(f"Entries truncated before '{response_marker}': {len(truncated_lines)}")
print(f"Capitalization or field name issues: {len(capitalization_issues)}")
print(f"Tokenized prompt missing '{response_marker}': {len(tokenized_response_missing)}")
print(f"Entries with short or missing output: {len(short_output_lines)}")

print("\nExample problem line numbers:")
print(f"  Truncated before response: {truncated_lines[:5]}")
print(f"  Capitalization problems: {capitalization_issues[:5]}")
print(f"  Tokenized prompt missing marker: {tokenized_response_missing[:5]}")
print(f"  Short outputs: {short_output_lines[:5]}")

