"""
Part C helpers shared by train_victim.py and attack_extract.py.
"""

import math

import torch


def load_causal_lm(model_name, adapter_path=None, quant="none"):
    """Base causal LM (+ optional LoRA adapter) in eval mode, and its tokenizer."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    use_cuda = torch.cuda.is_available()
    dtype = torch.float16 if use_cuda else torch.float32
    quant_config = None
    if quant == "4bit":
        quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=dtype, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype, quantization_config=quant_config, device_map="auto" if use_cuda else None)
    if adapter_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return model, tokenizer


@torch.no_grad()
def sequence_nll(model, tokenizer, texts, batch_size=32, max_batch_tokens=8192):
    """For each text: (sum of token negative log-likelihoods, number of scored tokens).
    Every token except the first is scored (Qwen has no BOS token to condition on).
    A batch is also capped at max_batch_tokens (texts x padded length), because the
    vocabulary-sized logits grow with every token: 512 x 128 tokens needed ~40 GB."""
    device = next(model.parameters()).device
    tokenizer.padding_side = "right"  # real tokens keep positions 0..n-1; pads sit at the end and are masked
    results = []
    i = 0
    while i < len(texts):
        batch = texts[i:i + batch_size]
        enc = tokenizer(batch, return_tensors="pt", padding=True)
        while enc["input_ids"].numel() > max_batch_tokens and len(batch) > 1:
            batch = batch[:len(batch) // 2]
            enc = tokenizer(batch, return_tensors="pt", padding=True)
        i += len(batch)
        enc = enc.to(device)
        logits = model(**enc).logits.float()

        # Position t predicts token t+1.
        pred = logits[:, :-1]
        target = enc["input_ids"][:, 1:]
        mask = enc["attention_mask"][:, 1:].bool()
        nll = torch.nn.functional.cross_entropy(pred.transpose(1, 2), target, reduction="none")
        nll = (nll * mask).sum(dim=1)
        results.extend(zip(nll.tolist(), mask.sum(dim=1).tolist()))
    return results


def perplexity(model, tokenizer, texts, batch_size=32):
    """Corpus perplexity: exp(total NLL / total tokens), so long texts weigh more."""
    scores = sequence_nll(model, tokenizer, texts, batch_size)
    return math.exp(sum(s for s, _ in scores) / sum(n for _, n in scores))


def plot_exposure(report_files, names, output):
    """Mean canary exposure vs. repetition count, one line per attack report."""
    import json
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 3.5))
    max_exposure = None
    for path, name in zip(report_files, names):
        with open(path) as f:
            canary = json.load(f)["canary"]
        reps = sorted(canary["by_n_reps"], key=int)
        ax.plot([int(r) for r in reps], [canary["by_n_reps"][r]["mean_exposure"] for r in reps], marker="o", label=name)
        max_exposure = canary["max_possible_exposure"]
    ax.axhline(max_exposure, color="gray", linestyle="--", linewidth=1, label="max measurable")
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 4, 16, 64], labels=["1", "4", "16", "64"])
    ax.set_xlabel("Times the canary appears in training")
    ax.set_ylabel("Mean exposure (bits)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    print(f"Saved {output}")


def merge_reports(report_dir, output, models=("raw", "san", "base"),
                  parts=("all", "utility", "discoverable", "untargeted", "canary", "targeted")):
    """Combine <report_dir>/<model>_<part>.json files (one --attack all file, or one file per
    attack from parallel jobs). The output keeps the spec's single-object shape for the first
    model (V-raw); the other models go under extra keys ("victim_san", "control")."""
    import json
    import os

    merged = []
    for m in models:
        combined = {}
        for part in parts:
            path = os.path.join(report_dir, f"{m}_{part}.json")
            if os.path.exists(path):
                with open(path) as f:
                    combined.update(json.load(f))
        print(f"{m}: {sorted(k for k in combined if k not in ('victim', 'corpus'))}")
        merged.append(combined)
    result = dict(merged[0])
    for name, combined in zip(("victim_san", "control"), merged[1:]):
        result[name] = combined
    with open(output, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved {output}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Part C plot and report merging")
    parser.add_argument("--mode", choices=["plot", "merge"], default="plot")
    parser.add_argument("--reports", nargs="+", help="plot: attack report JSON files with a 'canary' section")
    parser.add_argument("--names", nargs="+", help="plot: legend name for each report")
    parser.add_argument("--report_dir", type=str, default="results/partC", help="merge: folder with <model>_<attack>.json")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()
    if args.mode == "plot":
        plot_exposure(args.reports, args.names, args.output or "results/partC_exposure.png")
    else:
        merge_reports(args.report_dir, args.output or "results/partC_attacks.json")
