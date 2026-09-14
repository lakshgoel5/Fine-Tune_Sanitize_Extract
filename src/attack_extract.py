"""Part C.3: training data extraction attacks against a victim model.

Runs discoverable memorization, untargeted extraction (membership scoring),
canary exposure, and targeted PII extraction attacks, and writes a combined
report to --report_file. Without --adapter_path the base model is attacked
(the control).
"""

import argparse
import json
import os
from collections import defaultdict

import torch
from transformers import set_seed

from parta_utils import load_jsonl
from partc_utils import load_causal_lm


def parse_args():
    parser = argparse.ArgumentParser(description="Run extraction attacks against a victim model.")
    parser.add_argument("--victim_model", type=str, required=True, help="Base victim model, e.g. Qwen/Qwen2.5-0.5B")
    parser.add_argument("--adapter_path", type=str, default=None,
                        help="Path to the trained victim adapter (omit to attack the un-fine-tuned control model)")
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
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--prefix_len", type=int, default=32, help="Discoverable: prefix tokens given to the model")
    parser.add_argument("--suffix_len", type=int, default=32, help="Discoverable: suffix tokens it must reproduce")
    parser.add_argument("--limit", type=int, default=None, help="Only use the first N documents / prompts (quick checks)")
    return parser.parse_args()


# ----------------------------------------------------------------------------
# (a) Discoverable memorization
# ----------------------------------------------------------------------------

def edit_similarity(a, b):
    """1 - (token-level Levenshtein distance / length of the longer sequence)."""
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return 1 - prev[-1] / max(len(a), len(b), 1)


def distinct_documents(corpus_rows):
    """One entry per non-canary document (repeats share a doc_id)."""
    docs = {}
    for row in corpus_rows:
        if not row["is_canary"]:
            docs.setdefault(row["doc_id"], row)
    return list(docs.values())


@torch.no_grad()
def discoverable_attack(model, tokenizer, corpus_rows, prefix_len, suffix_len, batch_size, limit=None):
    """Give the model the first prefix_len tokens of a training document and check
    whether greedy decoding reproduces the next suffix_len tokens exactly."""
    device = next(model.parameters()).device

    items = []  # (n_reps, prefix ids, true suffix ids)
    for row in distinct_documents(corpus_rows)[:limit]:
        ids = tokenizer(row["text"], add_special_tokens=False)["input_ids"]
        if len(ids) >= prefix_len + suffix_len:
            items.append((row["n_reps"], ids[:prefix_len], ids[prefix_len:prefix_len + suffix_len]))

    per_group = defaultdict(lambda: {"n": 0, "exact": 0, "edit_sim": 0.0})
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        prefixes = torch.tensor([prefix for _, prefix, _ in batch], device=device)  # all the same length: no padding
        out = model.generate(input_ids=prefixes, attention_mask=torch.ones_like(prefixes),
                             max_new_tokens=suffix_len, do_sample=False, pad_token_id=tokenizer.pad_token_id)
        generated = out[:, prefix_len:].tolist()

        for (n_reps, _, true_suffix), gen in zip(batch, generated):
            gen = gen[:suffix_len]
            group = per_group[str(n_reps)]
            group["n"] += 1
            group["exact"] += gen == true_suffix
            group["edit_sim"] += edit_similarity(gen, true_suffix)
        print(f"  discoverable: {min(i + batch_size, len(items))}/{len(items)}", flush=True)

    def summarize(groups):
        n = sum(g["n"] for g in groups)
        return {"n": n,
                "exact_match_rate": sum(g["exact"] for g in groups) / n if n else 0.0,
                "mean_edit_sim": sum(g["edit_sim"] for g in groups) / n if n else 0.0}

    result = summarize(per_group.values())
    result.update({"prefix_len": prefix_len, "suffix_len": suffix_len,
                   "by_n_reps": {k: summarize([per_group[k]]) for k in sorted(per_group, key=int)}})
    return result


# ----------------------------------------------------------------------------

def main():
    args = parse_args()
    set_seed(args.seed)

    model, tokenizer = load_causal_lm(args.victim_model, args.adapter_path)
    corpus_rows = load_jsonl(args.corpus)
    attacks = ["discoverable", "untargeted", "canary", "targeted"] if args.attack == "all" else [args.attack]

    report = {"victim": os.path.basename(os.path.normpath(args.adapter_path)) if args.adapter_path else "base",
              "corpus": os.path.basename(args.corpus)}
    for attack in attacks:
        if attack == "discoverable":
            report["discoverable"] = discoverable_attack(model, tokenizer, corpus_rows, args.prefix_len,
                                                         args.suffix_len, args.batch_size, args.limit)
        else:
            raise NotImplementedError(f"--attack {attack} not implemented yet")

    os.makedirs(os.path.dirname(args.report_file) or ".", exist_ok=True)
    with open(args.report_file, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
