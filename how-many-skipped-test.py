
# to test the test data file

# figures out how many entries in the training data would be skipped because they're too big
# note I'm sure they could be made to work, but I want to focus on avoiding problems first
# result is 3%, so let's remove them

import json
from transformers import AutoTokenizer

file_path = "test_alpaca.jsonl"  # Adjust path as needed
max_seq_length = 1024
tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

total = 0
truncated = 0
truncated_lines = []

with open(file_path, "r") as f:
    for i, line in enumerate(f, 1):
        try:
            example = json.loads(line)
            instruction = example.get("instruction", "")
            input_text = example.get("input", "")
            output = example.get("output", "")
            prompt = f"### Instruction:\n{instruction}\n### Input:\n{input_text}\n### Response:\n{output}"
            tokens = tokenizer(prompt, truncation=True, max_length=max_seq_length, padding="max_length", return_tensors="np")
            decoded = tokenizer.decode(tokens["input_ids"][0], skip_special_tokens=True)
            if "### Response:" not in decoded:
                truncated += 1
                truncated_lines.append(i)
        except Exception:
            continue
        total += 1

print(f"Total entries: {total}")
print(f"Truncated entries: {truncated}")
print(f"Percentage truncated: {truncated / total * 100:.2f}%")

