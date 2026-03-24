import json
import re
from transformers import AutoTokenizer

# Parameters
input_path = "train_alpaca.jsonl"
output_path = "train_alpaca.filtered.jsonl"
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
max_seq_length = 1024

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Prompt template
TEMPLATE = """### Instruction:
{instruction}
### Input:
{input}
### Response:
{output}"""

# HTML tag cleaner
def clean_text(text: str) -> str:
    return re.sub(r"<br\s*/?>", " ", text, flags=re.IGNORECASE)

# Truncation check
def is_response_truncated(prompt: str, max_len: int) -> bool:
    tokens = tokenizer(prompt, truncation=True, max_length=max_len, padding="max_length")
    decoded = tokenizer.decode(tokens["input_ids"], skip_special_tokens=True)
    return "### Response:" not in decoded

# Filter and clean
with open(input_path, "r", encoding="utf-8") as infile, open(output_path, "w", encoding="utf-8") as outfile:
    kept = 0
    skipped = 0
    for i, line in enumerate(infile, 1):
        try:
            example = json.loads(line)
            # Clean HTML tags
            for field in ["instruction", "input", "output"]:
                if field in example and isinstance(example[field], str):
                    example[field] = clean_text(example[field])
            # Format and check
            prompt = TEMPLATE.format(
                instruction=example.get("instruction", ""),
                input=example.get("input", ""),
                output=example.get("output", "")
            )
            if is_response_truncated(prompt, max_seq_length):
                skipped += 1
                continue
            json.dump(example, outfile)
            outfile.write("\n")
            kept += 1
        except Exception as e:
            print(f"Error on line {i}: {e}")
            skipped += 1

print(f"Filtered dataset written to: {output_path}")
print(f"Total examples kept: {kept}")
print(f"Total examples skipped: {skipped}")

