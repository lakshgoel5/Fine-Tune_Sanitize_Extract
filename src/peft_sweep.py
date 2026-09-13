"""Part A: PEFT (LoRA/QLoRA) sweep for the PII sanitizer.

Fine-tunes Qwen2.5-1.5B-Instruct on partA_train.jsonl (source_text -> target_text)

Evaluates on partA_validation.jsonl, and appends one row to --results_csv.
"""

import argparse

"""
What is QLoRA?
QLoRA loads the massive, pre-trained base model weights in a highly compressed 4-bit format (specifically, 4-bit NormalFloat)
Once loaded, these 4-bit base weights are frozen and are not updated during the training process.
QLoRA often utilizes paged optimizers to automatically evict optimizer states to CPU RAM when GPU memory spikes, preventing out-of-memory errors during training.
--quant 4bit to set to QLoRA mode
"""

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
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the trained adapter")
    parser.add_argument("--results_csv", type=str, required=True, help="CSV file to append the results row to")
    return parser.parse_args()


def main():
    args = parse_args()
    return


if __name__ == "__main__":
    main()
