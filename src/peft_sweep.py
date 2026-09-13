"""Part A: PEFT (LoRA/QLoRA) sweep for the PII sanitizer.

Fine-tunes Qwen2.5-1.5B-Instruct on partA_train.jsonl (source_text -> target_text)

Evaluates on partA_validation.jsonl, and appends one row to --results_csv.
"""

import argparse

import torch
from transformers import AutoTokenizer, set_seed, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from parta_utils import load_jsonl, parse_examples, trainable_param_stats

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
    parser.add_argument("--output_dir", type=str, default="./Adapters", help="Directory to save the trained adapter")
    parser.add_argument("--results_csv", type=str, default="./results/parta/partA_sweep.csv", help="CSV file to append the results row to")
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

def main():
    args = parse_args()
    set_seed(args.seed)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # Data
    train_examples = parse_examples(load_jsonl(args.train_file))
    val_examples = parse_examples(load_jsonl(args.val_file))

    # Model (including Lora config)
    model = build_model(args)
    trainable, total = trainable_param_stats(model)

    print(f"Trainable parameters: {trainable}, Total parameters: {total}")

    return


if __name__ == "__main__":
    main()
