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
