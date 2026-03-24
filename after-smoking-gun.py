#!/usr/bin/env python
# the smoking gun script has found the core problem. now we will simulate was the training job does for tokenization
# to ensure it can be fixed by a reconfig of the tokenizer. then it simulates the data flow thru the loss masking logic
# to see if any entries end up fully masked, which was the cause of the problem since loss went to 0 during training
# we want the output to be:  Fully masked label rows: 0

import json
from transformers import AutoTokenizer

# Parameters
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
file_path = "train_alpaca.filtered.jsonl"
max_seq_length = 1200

# Load and reconfigure tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

if tokenizer.pad_token == tokenizer.eos_token:
    print(f"[FIX] pad_token was same as eos_token: '{tokenizer.pad_token}'. FIXING IT by Reassigning pad_token to '<pad>'...")
    tokenizer.add_special_tokens({"pad_token": "<pad>"})
else:
    print(f"[OK] pad_token and eos_token are distinct: pad_token='{tokenizer.pad_token}', eos_token='{tokenizer.eos_token}'")

# Prompt template
TEMPLATE = """### Instruction:
{instruction}
### Input:
{input}
### Response:
{output}"""

# Analyze tokenization
total = 0
masked_label_issues = 0

with open(file_path, "r") as f:
    for line in f:
        example = json.loads(line)
        prompt = TEMPLATE.format(
            instruction=example.get("instruction", ""),
            input=example.get("input", ""),
            output=example.get("output", "")
        )

        tokens = tokenizer(prompt, truncation=True, max_length=max_seq_length, padding="max_length")
        labels = tokens["input_ids"][:]
        attention_mask = tokens["attention_mask"]

        # HF supervised loss masking simulation
        masked_labels = [
            token_id if mask == 1 else -100 for token_id, mask in zip(labels, attention_mask)
        ]

        if all(label == -100 for label in masked_labels):
            print(f"[WARN] All labels masked for example {total}")
            masked_label_issues += 1

        total += 1

print("\n=== Summary ===")
print(f"Examples checked       : {total}")
print(f"Fully masked label rows: {masked_label_issues}")

