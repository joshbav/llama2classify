#!/usr/bin/env python
# --------------------------------------------------------------------------- #
# Train TinyLlama with LoRA on an Alpaca‑format sentiment dataset.            #
import os, sys, json, logging, torch
import torch.distributed as dist
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    HfArgumentParser,
    Trainer,
    EarlyStoppingCallback,
    default_data_collator,
)
from peft import get_peft_model, LoraConfig, TaskType

# --------------------------------------------------------------------------- #
# 1.  CLI arguments                                                           #
# --------------------------------------------------------------------------- #
@dataclass
class ScriptArguments:
    # Paths
    model_name_or_path: str = field(metadata={"help": "Base model (HF hub or local)"})
    train_file: str = field(metadata={"help": "Training dataset (.jsonl) in Alpaca format"})
    output_dir: str = field(metadata={"help": "Directory for checkpoints"})
    validation_file: Optional[str] = field(default=None)
    test_file: Optional[str] = field(default=None)
    # Optimisation hyper‑params
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    max_steps: int = 10000         # large upper bound; early stopping will end sooner
    eval_steps: int = 150          # ≈ one epoch for current dataset/batch config
    logging_steps: int = 150
    bf16: bool = False
    # LoRA hyper‑params
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    # Seq‑length and early‑stop
    max_seq_length: int = 512
    early_stopping_patience: int = 3  # stop after N evals with no improvement

args: ScriptArguments = HfArgumentParser(ScriptArguments).parse_args()

# --------------------------------------------------------------------------- #
# 2.  Distributed startup barrier                                             #
# --------------------------------------------------------------------------- #
if dist.is_available() and dist.is_initialized():
    dist.barrier() # waits until every worker process is ready, collective sync point

# --------------------------------------------------------------------------- #
# 3.  Logging & tokenizer                                                     #
# --------------------------------------------------------------------------- #
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

log.info("Loading tokenizer …")
tok = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)

# Ensure PAD token exists (TinyLlama often ties PAD=EOS)
# IS THE BIG FIX!
if tok.pad_token is None or tok.pad_token == tok.eos_token:
    tok.add_special_tokens({"pad_token": "<pad>"})

# --------------------------------------------------------------------------- #
# 4.  Dataset loading                                                         #
# --------------------------------------------------------------------------- #
data_files: Dict[str, str] = {"train": args.train_file}
if args.validation_file:
    data_files["validation"] = args.validation_file
if args.test_file:
    data_files["test"] = args.test_file

log.info("Loading datasets …")
raw_ds = load_dataset("json", data_files=data_files)

# --------------------------------------------------------------------------- #
# 5.  Tokenisation with prompt masking                                        #
# --------------------------------------------------------------------------- #
def tokenize(batch: Dict[str, List[str]]) -> Dict[str, Any]:
    input_ids, attention_masks, labels = [], [], []

    for instruction, _input, _output in zip(
        batch.get("instruction", [""] * len(batch["output"])),
        batch.get("input",       [""] * len(batch["output"])),
        batch["output"],
    ):
        prompt = (instruction or "") + "\n" + (_input or "")
        join   = "\n### Response:\n"
        answer = _output or ""

        prompt_ids = tok(prompt + join, add_special_tokens=False)["input_ids"]
        answer_ids = tok(answer,      add_special_tokens=False)["input_ids"]

        ids = (prompt_ids + answer_ids)[: args.max_seq_length - 1] + [tok.eos_token_id]
        attn = [1] * len(ids)

        pad_len = args.max_seq_length - len(ids)
        ids   += [tok.pad_token_id] * pad_len
        attn  += [0] * pad_len

        lbl   = [-100] * len(prompt_ids) \
                + answer_ids[: args.max_seq_length - 1 - len(prompt_ids)] \
                + [tok.eos_token_id] \
                + [-100] * pad_len

        input_ids.append(ids)
        attention_masks.append(attn)
        labels.append(lbl)

    return {"input_ids": input_ids,
            "attention_mask": attention_masks,
            "labels": labels}

log.info("Tokenising dataset …")
cols_to_remove = raw_ds["train"].column_names
ds = raw_ds.map(
    tokenize,
    batched=True,
    remove_columns=cols_to_remove,
    load_from_cache_file=False,
    desc="Tokenising",
)

# --------------------------------------------------------------------------- #
# 6.  Model loading & LoRA injection                                          #
# --------------------------------------------------------------------------- #
log.info("Loading base model …")
model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
model.resize_token_embeddings(len(tok))

log.info("Adding LoRA adapters …")
lora_cfg = LoraConfig(
    r=args.lora_r,
    lora_alpha=args.lora_alpha,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=args.lora_dropout,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(model, lora_cfg)

# --------------------------------------------------------------------------- #
# 7.  TrainingArguments & Trainer                                             #
# --------------------------------------------------------------------------- #
eval_strategy = "steps" if "validation" in ds else "no"

training_args = TrainingArguments(
    output_dir=args.output_dir,
    max_steps=args.max_steps,
    per_device_train_batch_size=args.per_device_train_batch_size,
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    learning_rate=args.learning_rate,
    weight_decay=args.weight_decay,
    bf16=args.bf16,
    evaluation_strategy=eval_strategy,
    eval_steps=args.eval_steps if eval_strategy == "steps" else None,
    save_strategy="steps" if eval_strategy == "steps" else "no",
    save_steps=args.eval_steps if eval_strategy == "steps" else None,
    load_best_model_at_end=True if eval_strategy == "steps" else False,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    logging_steps=args.logging_steps,
    report_to=[],
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=ds["train"],
    eval_dataset=ds.get("validation"),
    data_collator=default_data_collator,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)],
)

# --------------------------------------------------------------------------- #
# 8.  Train                                                                   #
# --------------------------------------------------------------------------- #
log.info("Starting training …")
trainer.train()

# --------------------------------------------------------------------------- #
# 9.  Final evaluation on held‑out **test** set                               #
# --------------------------------------------------------------------------- #
if "test" in ds:
    log.info("Evaluating final held‑out test set …")
    metrics = trainer.evaluate(eval_dataset=ds["test"])
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "test_results.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    log.info("Test metrics written to test_results.json")

# --------------------------------------------------------------------------- #
# 10. Save LoRA‑tuned model & tokenizer                                       #
# --------------------------------------------------------------------------- #
log.info("Saving model and tokenizer …")
model.save_pretrained(args.output_dir)
tok.save_pretrained(args.output_dir)

# --------------------------------------------------------------------------- #
# 11. Graceful distributed shutdown                                           #
# --------------------------------------------------------------------------- #

if dist.is_available() and dist.is_initialized():
    dist.barrier()
    try:
        # PyTorch ≥ 2.3
        dist.shutdown()
    except AttributeError:
        # Older versions
        dist.destroy_process_group()
    os._exit(0)   # prevent lingering heartbeat thread from raising ECONNRESET, was wed morning problem

log.info("Done.")

