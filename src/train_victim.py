"""Part C.2: victim model training.

Fine-tunes Qwen2.5-0.5B (base) on a Part C corpus (raw or sanitized) with the
given PEFT configuration, and evaluates held-out perplexity on partC_heldout.jsonl.
"""

import argparse


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
    return parser.parse_args()


def main():
    args = parse_args()
    return


if __name__ == "__main__":
    main()
