#!/usr/bin/env python
# ran this locally, it found the default tokenizer was the problem
# it loads the model from my local hugging face cache

import json
import re
from transformers import AutoTokenizer

# Constants
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
INPUT_FILE = "./train_alpaca.filtered.jsonl"
MAX_SEQ_LENGTH = 1100

TEMPLATE = """### Instruction:
{instruction}
### Input:
{input}
### Response:
{output}"""

def build_prompt(example):
    return TEMPLATE.format(
        instruction=example.get("instruction", ""),
        input=example.get("input", ""),
        output=example.get("output", "")
    )

def check_special_tokens(tokenizer):
    print("[INFO] Checking special tokens...")

    special_tokens = {
        "pad_token": tokenizer.pad_token,
        "eos_token": tokenizer.eos_token,
        "bos_token": tokenizer.bos_token,
        "unk_token": tokenizer.unk_token,
        "sep_token": getattr(tokenizer, "sep_token", None),
        "cls_token": getattr(tokenizer, "cls_token", None),
    }

    # Print token values
    for name, token in special_tokens.items():
        print(f"[INFO] {name}: {repr(token)}")

    # Check for duplicates
    seen = {}
    for name, token in special_tokens.items():
        if token is None:
            continue
        if token in seen:
            print(f"[problem found!] {name} is the same as {seen[token]}: {repr(token)}")
        else:
            seen[token] = name

    if len(set(t for t in special_tokens.values() if t is not None)) == len([t for t in special_tokens.values() if t is not None]):
        print("[OK] All special tokens are unique.\n")
    else:
        print("[problem found!] At least two special tokens are set to the same value — fix the tokenizer config.\n")

def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    check_special_tokens(tokenizer)

    num_total = 0
    num_truncated = 0
    num_missing_response_key = 0

    with open(INPUT_FILE, "r") as f:
        for i, line in enumerate(f, 1):
            try:
                example = json.loads(line)
                prompt = build_prompt(example)
                encoded = tokenizer(prompt, truncation=True, max_length=MAX_SEQ_LENGTH, padding="max_length")

                decoded = tokenizer.decode(encoded["input_ids"], skip_special_tokens=True)
                token_count = len(encoded["input_ids"])

                truncated = "### Response:" not in decoded
                missing_key = not re.search(r"###\s*Response:", prompt)

                if truncated or missing_key:
                    print(f"[WARN] Problem in example {i}:")
                    print(f"  - token_count: {token_count}")
                    print(f"  - truncated: {truncated}")
                    print(f"  - missing '### Response:': {missing_key}")
                    print(f"  Sample prompt excerpt:\n   {prompt[:300]}\n")

                if truncated:
                    num_truncated += 1
                if missing_key:
                    num_missing_response_key += 1

                num_total += 1

            except Exception as e:
                print(f"[ERROR] Failed to process line {i}: {e}")

    print("\n[SUMMARY]")
    print(f"  Total samples checked: {num_total}")
    print(f"  Truncated samples     : {num_truncated}")
    print(f"  Missing key samples   : {num_missing_response_key}")

if __name__ == "__main__":
    main()

