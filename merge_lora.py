import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

ap = argparse.ArgumentParser()
ap.add_argument('--base', required=True)
ap.add_argument('--adapter_dir', required=True)
ap.add_argument('--out_dir', required=True)
args = ap.parse_args()

tok = AutoTokenizer.from_pretrained(args.base)
base_model = AutoModelForCausalLM.from_pretrained(args.base, device_map="auto")
merged = PeftModel.from_pretrained(base_model, args.adapter_dir).merge_and_unload()

tok.save_pretrained(args.out_dir)
merged.save_pretrained(args.out_dir)
