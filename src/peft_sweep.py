"""Part A: PEFT (LoRA/QLoRA) sweep for the PII sanitizer.

Fine-tunes Qwen2.5-1.5B-Instruct on partA_train.jsonl (source_text -> target_text)

Evaluates on partA_validation.jsonl, and appends one row to --results_csv.
"""

import argparse
import csv
import os
import time
from dataclasses import dataclass, asdict

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, set_seed, AutoModelForCausalLM, BitsAndBytesConfig, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from parta_utils import load_jsonl, parse_examples, trainable_param_stats, build_chat_prompt
from metrics import recover_predicted_spans, match_counts_iou, micro_prf1, count_leaks

"""
What is QLoRA?
QLoRA loads the massive, pre-trained base model weights in a highly compressed 4-bit format (specifically, 4-bit NormalFloat)
Once loaded, these 4-bit base weights are frozen and are not updated during the training process.
QLoRA often utilizes paged optimizers to automatically evict optimizer states to CPU RAM when GPU memory spikes, preventing out-of-memory errors during training.
--quant 4bit to set to QLoRA mode
"""

class CustomDataset(Dataset):
    """
    Returns plain python lists (NOT padded, NOT tensors yet) -- padding is a
    batch-level concern and belongs in DataCollator, not here.

    Worked example:
        ex.source_text = "Hi Bob"
        ex.target_text = "Hi [GIVENNAME1]"

        prompt = build_chat_prompt(...)              # "<sys>...<user>Hi Bob<assistant>"
        full   = prompt + target_text + eos_token     # "<sys>...<user>Hi Bob<assistant>Hi [GIVENNAME1]<eos>"

        prompt_ids = [1, 2, 3, 4, 5]                   # 5 tokens covering system+user+assistant-start
        full_ids   = [1, 2, 3, 4, 5, 6, 7, 8, 9]        # + "Hi"(6) "[GIVENNAME1]"(7,8) eos(9)
        prompt_len = 5

        labels = [-100, -100, -100, -100, -100, 6, 7, 8, 9]
        #         |------- prompt, no loss -------|  |-- completion, loss here --|

        returned dict -> {
            "input_ids":      [1, 2, 3, 4, 5, 6, 7, 8, 9],
            "attention_mask": [1, 1, 1, 1, 1, 1, 1, 1, 1],   # all 1s -- no padding exists yet
            "labels":         [-100, -100, -100, -100, -100, 6, 7, 8, 9],
        }
    """

    def __init__(self, examples, tokenizer, max_len: int = 1024):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        prompt = build_chat_prompt(self.tokenizer, ex.source_text)
        full = prompt + ex.target_text + self.tokenizer.eos_token

        # add_special_tokens=False: apply_chat_template already inserted any special tokens (e.g. <|im_start|>) we need.
        prompt_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        full_ids = self.tokenizer(full, add_special_tokens=False)["input_ids"][: self.max_len]

        labels = list(full_ids)
        prompt_len = min(len(prompt_ids), len(full_ids))  # guard against truncation cutting into the prompt
        for i in range(prompt_len):
            labels[i] = -100  # mask prompt tokens out of the loss

        return {
            "input_ids": full_ids,
            "attention_mask": [1] * len(full_ids),
            "labels": labels,
        }


@dataclass
class DataCollator:
    """
    Batch-level work only: takes a list of already-tokenized, variable-length
    dicts (one per example, from CustomDataset.__getitem__) and pads them to
    the longest example in THIS batch (not a fixed global max_len -- keeps
    compute down on batches that happen to be shorter), then stacks into
    tensors.

    Right-padded by hand regardless of tokenizer.padding_side, since that
    attribute is set to "left" globally (for generation) and we don't want
    that to leak into how training batches get padded.

    Worked example, continuing from CustomDataset's example above -- say the
    batch has 2 examples, A (len 9, from the example above) and a shorter B:
        batch = [
            {"input_ids": [1,2,3,4,5,6,7,8,9], "attention_mask": [1]*9,
             "labels":    [-100,-100,-100,-100,-100,6,7,8,9]},                    # A, len 9
            {"input_ids": [1,2,3,4,5,6,7],     "attention_mask": [1]*7,
             "labels":    [-100,-100,-100,-100,-100,6,7]},                        # B, len 7
        ]

        batch_max_len = 9   # longest example in THIS batch (not a fixed max_len)
        pad_len for B = 9 - 7 = 2   (pad_token_id here is, say, 0)

        padded B -> input_ids:      [1,2,3,4,5,6,7, 0,0]        # 0 = pad_token_id
                    attention_mask: [1,1,1,1,1,1,1, 0,0]        # 0s = "don't attend to these"
                    labels:         [-100,-100,-100,-100,-100,6,7, -100,-100]   # -100, never pad_token_id, so pads never contribute to loss

        A needs no padding (already at batch_max_len), so it passes through unchanged.

        final stacked tensors, shape (batch_size=2, batch_max_len=9):
            input_ids      = [[1,2,3,4,5,6,7,8,9],
                               [1,2,3,4,5,6,7,0,0]]
            attention_mask = [[1,1,1,1,1,1,1,1,1],
                               [1,1,1,1,1,1,1,0,0]]
            labels          = [[-100,-100,-100,-100,-100,6,7,8,9],
                               [-100,-100,-100,-100,-100,6,7,-100,-100]]
    """
    tokenizer: object

    def __call__(self, batch: list) -> dict:
        batch_max_len = max(len(item["input_ids"]) for item in batch)
        input_ids, attention_mask, labels = [], [], []
        for item in batch:
            pad_len = batch_max_len - len(item["input_ids"])
            input_ids.append(item["input_ids"] + [self.tokenizer.pad_token_id] * pad_len)
            attention_mask.append(item["attention_mask"] + [0] * pad_len)
            labels.append(item["labels"] + [-100] * pad_len)  # -100, not pad_token_id, so pads never contribute to loss

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
    
class TokenCountingTrainer(Trainer):
    """
    sum attention_mask over every training micro-batch, so we can report tokens_per_s = total_tokens_seen / wallclock_s
    Counting REAL (non-pad) tokens via attention_mask, not input_ids.numel()
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.total_tokens_seen = 0

    def training_step(self, model, inputs, *args, **kwargs):
        self.total_tokens_seen += inputs["attention_mask"].sum().item()
        return super().training_step(model, inputs, *args, **kwargs)



def parse_args():
    parser = argparse.ArgumentParser(description="Run one PEFT sweep configuration for the sanitizer.")
    parser.add_argument("--model", type=str, required=True, help="Base model, e.g. Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--train_file", type=str, required=True, help="Path to partA_train.jsonl")
    parser.add_argument("--val_file", type=str, required=True, help="Path to partA_validation.jsonl")
    parser.add_argument("--target_modules", type=str, required=True, help="Comma-separated LoRA target modules, e.g. q_proj,k_proj,v_proj,o_proj")
    parser.add_argument("--lora_r", type=int, required=True, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, required=True, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")
    parser.add_argument("--quant", type=str, choices=["none", "4bit"], default="none", help="Quantization mode: none (LoRA) or 4bit (QLoRA)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs") # Custom add
    parser.add_argument("--run_id", type=str, required=True, help="Sweep run identifier, e.g. A4")
    parser.add_argument("--output_dir", type=str, default="./Adapters", help="Directory to save the trained adapter")
    parser.add_argument("--results_csv", type=str, default="./results/parta/partA_sweep.csv", help="CSV file to append the results row to")
    parser.add_argument("--max_len", type=int, default=1024, help="Max tokens per training example (prompt+target), truncated beyond this")
    return parser.parse_args()

def build_model(args):
    use_cuda = torch.cuda.is_available()
    use_mps = torch.backends.mps.is_available() and torch.backends.mps.is_built()
    if use_cuda:
        device = "cuda"
    elif use_mps:
        device = "mps"
    else:
        device = "cpu"

    dtype = torch.float16 if (use_cuda or use_mps) else torch.float32 #TUNE based on GPU and memory

    quant_config = None
    if args.quant == "4bit":
        if use_cuda:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        else:
            print("Warning: 4-bit bitsandbytes quantization is only supported on CUDA. Fallback to standard float16.")

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=quant_config,
        device_map="auto" if device != "cpu" else None,
        torch_dtype=dtype
    )

    if args.quant == "4bit":
        model = prepare_model_for_kbit_training(model)

    # Model is now loaded in 4-bit or standard precision, depending on the quantization setting and device availability.
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=[m.strip() for m in args.target_modules.split(",")],
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model

@torch.no_grad()
def run_validation(model, tokenizer, val_examples, batch_size: int = 8, margin: int = 32): #ADJUST batch_size and margin based on GPU memory
    """
    Generation-based validation scoring -- separate from trainer.evaluate()'s
    val_loss (that's teacher-forced, no generation involved).

    max_new_tokens is sized per-batch from the longest source_text in that
    batch (+ margin), not a fixed global constant -- target_text is roughly as
    long as source_text (spans just shrink to [LABEL] tags), so a fixed cap
    would truncate long documents and unfairly tank their F1/leak numbers.
    """
    model.eval()
    device = next(model.parameters()).device
    tp_sum = fp_sum = fn_sum = 0
    leaked_sum = total_sum = 0

    for i in range(0, len(val_examples), batch_size):
        batch = val_examples[i: i + batch_size]
        prompts = [build_chat_prompt(tokenizer, ex.source_text) for ex in batch]
        # padding=True here relies on tokenizer.padding_side == "left" (set in main())
        # so every prompt in the batch ends at the same index -> uniform slice point below.
        enc = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)

        source_lens = [len(tokenizer(ex.source_text, add_special_tokens=False)["input_ids"]) for ex in batch]
        max_new_tokens = max(source_lens) + margin

        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # greedy: deterministic, reproducible model-selection signal
            pad_token_id=tokenizer.pad_token_id,
        )
        gen_only = out[:, enc["input_ids"].shape[1]:]  # left-padded prompts -> uniform slice point for all rows
        decoded = tokenizer.batch_decode(gen_only, skip_special_tokens=True)

        for ex, pred_text in zip(batch, decoded):
            gold_spans = [asdict(s) for s in ex.privacy_mask]
            pred_spans = recover_predicted_spans(ex.source_text, pred_text)
            tp, fp, fn = match_counts_iou(pred_spans, gold_spans, iou_threshold=0.5)
            tp_sum += tp
            fp_sum += fp
            fn_sum += fn
            leaked, total = count_leaks(gold_spans, pred_text)
            leaked_sum += leaked
            total_sum += total

    _, _, f1 = micro_prf1(tp_sum, fp_sum, fn_sum)
    leak_rate = leaked_sum / total_sum if total_sum > 0 else 0.0
    model.train()
    return f1, leak_rate

def main():
    args = parse_args()
    set_seed(args.seed)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # for training/fine tuning, Standard practice for loss masking is to pad on right, but for generation, we need to pad on left.
    tokenizer.padding_side = "left"  # required for correct batched generate() slicing
    # While training, custom collate_fn will handle padding, so we don't need to care about tokenizer.padding_side for training. We are setting it to "left" here for generation during evaluation.

    # Model (including Lora config)
    model = build_model(args)
    trainable, total = trainable_param_stats(model)

    print(f"Trainable parameters: {trainable}, Total parameters: {total}")

    # Data
    train_examples = parse_examples(load_jsonl(args.train_file))
    val_examples = parse_examples(load_jsonl(args.val_file))
    train_dataset = CustomDataset(train_examples, tokenizer, max_len=args.max_len)
    val_dataset = CustomDataset(val_examples, tokenizer, max_len=args.max_len)
    collator = DataCollator(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,

        num_train_epochs=args.epochs,
        per_device_train_batch_size=4,  # ADJUST
        gradient_accumulation_steps=1,   # ADJUST
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},

        # ADJUST
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,

        logging_dir='./logs',
        logging_steps=10,
        save_strategy="epoch",
        load_best_model_at_end=True,
        evaluation_strategy="epoch",
        save_total_limit=1, # ADJUST if you want to do some experiment

        seed=args.seed,
    )

    trainer = TokenCountingTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
    )

    # Training Setup Done

    # Collect data
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()  # so max_memory_allocated() below reflects THIS run only

    start = time.time()
    train_result = trainer.train()  # restores the best (lowest eval_loss by default) checkpoint, since load_best_model_at_end=True
    wallclock_s = time.time() - start

    peak_mem_gb = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
    tokens_per_s = trainer.total_tokens_seen / wallclock_s if wallclock_s > 0 else 0.0

    # Validation
    eval_metrics = trainer.evaluate()  # teacher-forced val_loss (no generation)
    val_f1, val_leak_rate = run_validation(model, tokenizer, val_examples)

    # Save
    model.save_pretrained(args.output_dir)
    row = {
        "run_id": args.run_id,
        "method": "QLoRA" if args.quant == "4bit" else "LoRA",
        "target_modules": args.target_modules,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "quant": args.quant,
        "trainable_params": trainable,
        "total_params": total,
        "trainable_pct": 100.0 * trainable / total if total > 0 else 0.0,
        "peak_gpu_mem_gb": peak_mem_gb,
        "train_wallclock_s": wallclock_s,
        "tokens_per_s": tokens_per_s,
        "final_train_loss": train_result.training_loss,
        "val_loss": eval_metrics.get("eval_loss"),
        "val_entity_f1": val_f1,
        "val_leak_rate": val_leak_rate,
    }
    append_csv_row(args.results_csv, row)

    return


if __name__ == "__main__":
    main()
