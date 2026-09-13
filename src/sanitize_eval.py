"""Part B: sanitizer evaluation and baselines.

Evaluates a sanitization system (fine-tuned adapter, regex baseline, Presidio, or zero-shot prompting) 
on partB_test.jsonl (or partB_ood_test.jsonl), 
recovers predicted spans via alignment, 
computes entity-level metrics, leak rate, and over-masking rate, and writes predictions + a metrics report.
"""

import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a PII sanitization system.")
    parser.add_argument("--base_model", type=str, required=True, 
                        help="Base model, e.g. Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--adapter_path", type=str, default=None,
                        help="Path to the fine-tuned LoRA adapter (required when --system finetuned)")
    parser.add_argument("--test_file", type=str, required=True,
                        help="Path to partB_test.jsonl or partB_ood_test.jsonl")
    parser.add_argument("--system", type=str, required=True, choices=["finetuned", "regex", "presidio", "zeroshot"], 
                        help="Which sanitization system to evaluate")
    parser.add_argument("--output_predictions", type=str, required=True, 
                        help="Path to write per-example predictions JSONL")
    parser.add_argument("--report_file", type=str, required=True, 
                        help="Path to write the aggregate metrics JSON report")
    parser.add_argument("--seed", type=int, default=42, help="Random seed") ## Custom add
    return parser.parse_args()


def main():
    args = parse_args()
    return


if __name__ == "__main__":
    main()
