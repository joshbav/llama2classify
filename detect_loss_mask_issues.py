# didn't find any problem


import json
import re
import argparse
from transformers import AutoTokenizer
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", required=True)
parser.add_argument("--file", required=True)
parser.add_argument("--max_seq_length", type=int, default=1024)
args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained(args.model_name)

if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

RESPONSE_KEY = "### Response:"
RESPONSE_KEY_PATTERN = re.escape(RESPONSE_KEY)

def make_prompt(ex):
    return f"### Instruction: {ex['instruction']}\n### Input: {ex['input']}\n{RESPONSE_KEY} {ex['output']}"

def emulate_data_collator(prompt):
    match = re.search(RESPONSE_KEY_PATTERN, prompt)
    if not match:
        return None, None, True  # no loss region

    response_start = match.end()
    response_text = prompt[response_start:]
    full_encoding = tokenizer(prompt, truncation=True, max_length=args.max_seq_length, padding="max_length")
    labels = [-100] * len(full_encoding["input_ids"])

    decoded = tokenizer.decode(full_encoding["input_ids"], skip_special_tokens=False)
    tokenized_response = tokenizer(response_text, add_special_tokens=False)["input_ids"]

    # Try to align and mask only the response tokens
    try:
        start_idx = full_encoding["input_ids"].index(tokenizer(RESPONSE_KEY, add_special_tokens=False)["input_ids"][-1]) + 1
        for i in range(len(tokenized_response)):
            if start_idx + i < len(labels):
                labels[start_idx + i] = full_encoding["input_ids"][start_idx + i]
    except Exception:
        return full_encoding["input_ids"], labels, True  # fallback: treat as no loss region

    all_masked = all(l == -100 for l in labels)
    return full_encoding["input_ids"], labels, all_masked

with open(args.file) as f:
    for i, line in enumerate(f, 1):
        example = json.loads(line)
        prompt = make_prompt(example)
        input_ids, labels, loss_problem = emulate_data_collator(prompt)

        if loss_problem:
            print(f"[WARN] Example {i} would have no loss.\n  Instruction: {example['instruction'][:60]}")
            print(f"  Truncated? {'Yes' if len(input_ids or []) >= args.max_seq_length else 'No'}")
            print(f"  Sample: {prompt[:150].replace('\n', ' ')}...\n")

print("Done.")

