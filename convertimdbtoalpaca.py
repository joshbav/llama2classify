import json
import os

def convert_file(input_path, output_path):
    print(f"Converting: {input_path} -> {output_path}")
    with open(input_path, 'r') as f_in, open(output_path, 'w') as f_out:
        for line in f_in:
            entry = json.loads(line)
            label = "positive" if entry["label"] == 0 else "negative"
            new_entry = {
                "instruction": "Classify this movie review as positive or negative:",
                "input": entry["review"],
                "output": label
            }
            f_out.write(json.dumps(new_entry) + "\n")
    print(f"Saved: {output_path}")

def main():
    base_dir = "/data"
    convert_file(os.path.join(base_dir, "train.jsonl"), os.path.join(base_dir, "train_alpaca.jsonl"))
    convert_file(os.path.join(base_dir, "test.jsonl"), os.path.join(base_dir, "test_alpaca.jsonl"))

if __name__ == "__main__":
    main()

