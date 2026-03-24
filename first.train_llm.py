import argparse
from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModelForCausalLM,
                          TrainingArguments, Trainer, DataCollatorForLanguageModeling)
from peft import LoraConfig, get_peft_model

p = argparse.ArgumentParser()
p.add_argument('--model_name_or_path', required=True)
p.add_argument('--train_file', required=True)
p.add_argument('--output_dir', required=True)
p.add_argument('--max_steps', type=int, default=400)
p.add_argument('--per_device_train_batch_size', type=int, default=2)
p.add_argument('--gradient_accumulation_steps', type=int, default=4)
p.add_argument('--learning_rate', type=float, default=2e-4)
p.add_argument('--lora_r', type=int, default=16)
p.add_argument('--lora_alpha', type=int, default=32)
p.add_argument('--lora_dropout', type=float, default=0.05)
p.add_argument('--bf16', action='store_true')
p.add_argument('--max_seq_length', type=int, default=256)
args = p.parse_args()

# Load jsonl
ds = load_dataset('json', data_files=args.train_file, split='train')

def fmt(ex):
    return f"Instruction: {ex.get('instruction','')}\nAnswer:"

tok = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

def tokenize(batch):
    return tok(fmt(batch), truncation=True, max_length=args.max_seq_length)

tok_ds = ds.map(tokenize)

# removed , device_map='auto')
model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)

lora_cfg = LoraConfig(
    r=args.lora_r,
    lora_alpha=args.lora_alpha,
    target_modules=['q_proj','v_proj'],
    lora_dropout=args.lora_dropout,
    bias='none',
    task_type='CAUSAL_LM'
)
model = get_peft_model(model, lora_cfg)

training_args = TrainingArguments(
    output_dir=args.output_dir,
    max_steps=args.max_steps,
    per_device_train_batch_size=args.per_device_train_batch_size,
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    learning_rate=args.learning_rate,
    bf16=args.bf16,
    logging_steps=20,
    save_steps=99999,
    report_to=[]
)

collator = DataCollatorForLanguageModeling(tok, mlm=False)

trainer = Trainer(model=model, args=training_args,
                  train_dataset=tok_ds,
                  data_collator=collator)
trainer.train()

model.save_pretrained(args.output_dir)
tok.save_pretrained(args.output_dir)
