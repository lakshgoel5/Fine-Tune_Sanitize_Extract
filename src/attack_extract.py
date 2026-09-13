"""Part C.3: training data extraction attacks against a victim model.

Runs discoverable memorization, untargeted extraction (membership scoring),
canary exposure, and targeted PII extraction attacks, and writes a combined
report to --report_file.
"""

import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Run extraction attacks against a victim model.")
    parser.add_argument("--victim_model", type=str, required=True, help="Base victim model, e.g. Qwen/Qwen2.5-0.5B")
    parser.add_argument("--adapter_path", type=str, required=True, help="Path to the trained victim adapter")
    parser.add_argument("--ref_model", type=str, required=True,
                        help="Reference (un-fine-tuned) model for membership scoring")
    parser.add_argument("--corpus", type=str, required=True, help="Path to partC_corpus_raw.jsonl")
    parser.add_argument("--heldout", type=str, default=None, help="Path to partC_heldout.jsonl") # Custom add
    parser.add_argument("--canaries", type=str, required=True, help="Path to partC_canaries.jsonl")
    parser.add_argument("--targeted_file", type=str, required=True, help="Path to partC_targeted.jsonl")
    parser.add_argument("--attack", type=str, required=True,
                        choices=["discoverable", "untargeted", "canary", "targeted", "all"],
                        help="Which attack(s) to run")
    parser.add_argument("--num_samples", type=int, default=5000,
                        help="Number of samples to generate for the untargeted extraction attack")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--report_file", type=str, required=True,
                        help="Path to write the combined attack results JSON report")
    return parser.parse_args()


def main():
    args = parse_args()
    return


if __name__ == "__main__":
    main()
