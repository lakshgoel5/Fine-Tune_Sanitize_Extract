"""
Utilities
1. Plotting
    a. Scatter plots

2. JSONL loading

3. Prompt Construction
"""

import argparse
import matplotlib.pyplot as plt
import csv
import json
from dataclasses import dataclass
from typing import Optional

@dataclass
class Span:
    start: int
    end: int
    label: str
    value: Optional[str] = None

@dataclass
class Example:
    id: str
    language: str
    source_text: str
    target_text: str
    privacy_mask: list[Span]
    split: str

def load_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows

def parse_examples(rows: list[dict]) -> list[Example]:
    out = []
    for r in rows:
        mask = [Span(start=m["start"], end=m["end"], label=m["label"], value=m.get("value"))
            for m in r.get("privacy_mask", [])
        ]
        out.append(
            Example(
                id=r.get("id", ""),
                language=r.get("language", ""),
                source_text=r["source_text"],
                target_text=r["target_text"],
                privacy_mask=mask,
                split=r.get("split", ""),
            )
        )
    return out

# Closed label vocabulary observed in data/partA_train.jsonl + partA_validation.jsonl
# (27 labels; a 28th, CARDISSUER, appears exactly once and is treated as noise,
# not a real category). data/partB_test.jsonl uses this same 27-label schema.
# data/partB_ood_test.jsonl uses a DIFFERENT, disjoint label vocabulary (its own
# ~56 labels, e.g. FIRSTNAME/PHONENUMBER/DOB instead of GIVENNAME1/TEL/BOD) --
# do not add those names here; evaluating with this same list on the OOD set is
# the intended (and informative) test.
PII_LABELS = [
    "BOD", "BUILDING", "CITY", "COUNTRY", "DATE", "DRIVERLICENSE", "EMAIL",
    "GEOCOORD", "GIVENNAME1", "GIVENNAME2", "IDCARD", "IP", "LASTNAME1",
    "LASTNAME2", "LASTNAME3", "PASS", "PASSPORT", "POSTCODE", "SECADDRESS",
    "SEX", "SOCIALNUMBER", "STATE", "STREET", "TEL", "TIME", "TITLE",
    "USERNAME",
]

SYSTEM_PROMPT = (
    "You are a privacy sanitizer. Rewrite the user's text, replacing every span "
    "that contains personally identifiable information (PII) with a bracketed "
    "label naming its category, using ONLY the following fixed set of labels "
    "(uppercase, exactly as written, no other labels): "
    + ", ".join(f"[{l}]" for l in PII_LABELS) + ". "
    "GIVENNAME1/GIVENNAME2 and LASTNAME1/LASTNAME2/LASTNAME3 are separate labels "
    "for distinct name slots (e.g. different people or first vs. middle name) -- "
    "use the correct numbered label, not a generic one. Do not change anything "
    "else about the text -- preserve all formatting, punctuation, and non-PII "
    "content exactly. Output only the sanitized text, with no extra commentary."
)


def build_chat_prompt(tokenizer, source_text: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": source_text},
    ]
    #tokenize=False: Tells the tokenizer to return a formatted string instead of a list of integer token IDs.
    #add_generation_prompt=True: Appends a generation prompt to the end of the user message
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def trainable_param_stats(model):
    trainable, total = 0, 0
    for p in model.parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
    return trainable, total

def parse_args():
    parser = argparse.ArgumentParser(description="Run one PEFT sweep configuration for the sanitizer.")
    parser.add_argument("--mode", type=str, choices=["plot"], default="plot", help="")
    parser.add_argument("--results_csv", type=str, default="./results/parta/partA_sweep.csv", help="CSV file to read the results from")
    parser.add_argument("--output_dir", type=str, default="./results/parta/plots/", help="Directory to save the plot")
    return parser.parse_args()

def plot(results_csv, output_dir):
    rows = []
    with open(results_csv) as f:
        for r in csv.DictReader(f):
            rows.append(r)
 
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
 
    ax = axes[0]
    for r in rows:
        x = float(r["trainable_pct"])
        y = float(r["val_entity_f1"])
        ax.scatter(x, y, s=60)
        ax.annotate(r["run_id"], (x, y), textcoords="offset points", xytext=(6, 4))
    ax.set_xscale("log")
    ax.set_xlabel("Trainable parameters (%) [log scale]")
    ax.set_ylabel("Validation entity F1 (IoU>=0.5)")
    ax.set_title("F1 vs. trainable-parameter budget")
    ax.grid(True, which="both", alpha=0.3)
 
    ax = axes[1]
    for r in rows:
        x = float(r["val_entity_f1"])
        y = float(r["peak_gpu_mem_gb"])
        ax.scatter(x, y, s=60)
        ax.annotate(r["run_id"], (x, y), textcoords="offset points", xytext=(6, 4))
    ax.set_xlabel("Validation entity F1 (IoU>=0.5)")
    ax.set_ylabel("Peak GPU memory (GB)")
    ax.set_title("Memory vs. F1")
    ax.grid(True, alpha=0.3)
 
    fig.tight_layout()
    output_img = f"{output_dir}/partA_sweep.png"
    fig.savefig(output_img, dpi=150)
    print(f"Saved figure to {output_img}")

def main():
    args = parse_args()

    if args.mode == "plot":
        plot(args.results_csv, args.output_dir)


if __name__ == "__main__":
    main()
