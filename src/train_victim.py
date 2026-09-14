"""Part C.2: victim model training.

Fine-tunes Qwen2.5-0.5B (base) on a Part C corpus (raw or sanitized) with the
given PEFT configuration, and evaluates held-out perplexity on partC_heldout.jsonl
for the victim and for the un-fine-tuned base model.
"""

import argparse
import json
import math
import os
import time

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainingArguments, set_seed

from parta_utils import load_jsonl
from partc_utils import perplexity


def parse_args():
    parser = argparse.ArgumentParser(description="Train a victim model on a Part C corpus.")
    parser.add_argument("--model", type=str, required=True, help="Base victim model, e.g. Qwen/Qwen2.5-0.5B")
    parser.add_argument("--corpus", type=str, required=True,
                        help="Path to partC_corpus_raw.jsonl or the locally derived sanitized corpus")
    parser.add_argument("--heldout", type=str, required=True, help="Path to partC_heldout.jsonl")
    parser.add_argument("--target_modules", type=str, required=True,
                        help="Comma-separated LoRA target modules. Eg: 'q_proj,k_proj,v_proj,o_proj' ")
    parser.add_argument("--lora_r", type=int, required=True, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, required=True, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout") # Custom add
    parser.add_argument("--quant", type=str, choices=["none", "4bit"], default="none", # Custom add
                        help="Quantization mode: none (LoRA) or 4bit (QLoRA)")
    parser.add_argument("--epochs", type=int, required=True,
                        help="Number of training epochs")
    parser.add_argument("--seed", type=int, required=True,
                        help="Random seed")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save the trained adapter")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max_len", type=int, default=512, help="Token cap per document (longest document is 378)")
    parser.add_argument("--limit", type=int, default=None, help="Only train on the first N rows (quick checks)")
    return parser.parse_args()


class TextDataset(torch.utils.data.Dataset):
    """One corpus row = one training sequence (no packing), so a document repeated
    n times is seen exactly n times per epoch. Loss on every token."""

    def __init__(self, texts, tokenizer, max_len):
        self.items = [tokenizer(t + tokenizer.eos_token, add_special_tokens=False)["input_ids"][:max_len] for t in texts]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def make_collator(pad_token_id):
    def collate(batch):
        longest = max(len(ids) for ids in batch)
        input_ids, attention_mask, labels = [], [], []
        for ids in batch:
            pad = longest - len(ids)
            input_ids.append(ids + [pad_token_id] * pad)
            attention_mask.append([1] * len(ids) + [0] * pad)
            labels.append(ids + [-100] * pad)  # padding never contributes to the loss
        return {"input_ids": torch.tensor(input_ids), "attention_mask": torch.tensor(attention_mask),
                "labels": torch.tensor(labels)}
    return collate


def build_model(args):
    use_cuda = torch.cuda.is_available()
    dtype = torch.float16 if use_cuda else torch.float32
    quant_config = None
    if args.quant == "4bit":
        quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=dtype,
                                          bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype, quantization_config=quant_config,
                                                 device_map="auto" if use_cuda else None)
    if args.quant == "4bit":
        model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
                             target_modules=[m.strip() for m in args.target_modules.split(",")],
                             bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def main():
    args = parse_args()
    set_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    corpus = [row["text"] for row in load_jsonl(args.corpus)][:args.limit]
    heldout = [row["text"] for row in load_jsonl(args.heldout)]
    model = build_model(args)

    # Base-model perplexity: same weights with the (still untrained) adapter switched off.
    with model.disable_adapter():
        base_ppl = perplexity(model, tokenizer, heldout)
    print(f"base heldout ppl: {base_ppl:.3f}")

    total_steps = math.ceil(len(corpus) / args.batch_size) * args.epochs
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=math.ceil(0.03 * total_steps),  # 3% warmup, as in Part A
        logging_steps=10,
        save_strategy="no",
        seed=args.seed,
        report_to=[],
    )
    trainer = Trainer(model=model, args=training_args, train_dataset=TextDataset(corpus, tokenizer, args.max_len),
                      data_collator=make_collator(tokenizer.pad_token_id))

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    start = time.time()
    train_result = trainer.train()
    wallclock_s = time.time() - start

    model.eval()
    victim_ppl = perplexity(model, tokenizer, heldout)
    print(f"victim heldout ppl: {victim_ppl:.3f}")

    model.save_pretrained(args.output_dir)
    metrics = {
        "corpus": os.path.basename(args.corpus),
        "n_train_rows": len(corpus),
        "model": args.model,
        "target_modules": args.target_modules, "lora_r": args.lora_r, "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout, "quant": args.quant, "epochs": args.epochs, "seed": args.seed,
        "batch_size": args.batch_size, "lr": args.lr,
        "final_train_loss": train_result.training_loss,
        "train_wallclock_s": wallclock_s,
        "peak_gpu_mem_gb": torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0,
        "base_heldout_ppl": base_ppl,
        "heldout_ppl": victim_ppl,
    }
    with open(os.path.join(args.output_dir, "victim_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
