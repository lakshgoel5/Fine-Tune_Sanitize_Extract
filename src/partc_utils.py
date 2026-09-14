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
def sequence_nll(model, tokenizer, texts, batch_size=32):
    """For each text: (sum of token negative log-likelihoods, number of scored tokens).
    Every token except the first is scored (Qwen has no BOS token to condition on)."""
    device = next(model.parameters()).device
    tokenizer.padding_side = "right"  # real tokens keep positions 0..n-1; pads sit at the end and are masked
    results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        enc = tokenizer(batch, return_tensors="pt", padding=True).to(device)
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
