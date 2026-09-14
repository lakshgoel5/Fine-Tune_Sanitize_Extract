"""Part C.3: training data extraction attacks against a victim model.

Runs discoverable memorization, untargeted extraction (membership scoring),
canary exposure, and targeted PII extraction attacks, and writes a combined
report to --report_file. Without --adapter_path the base model is attacked
(the control).
"""

import argparse
import json
import math
import os
import random
import zlib
from collections import defaultdict

import torch
from transformers import set_seed

from metrics import _normalize, recover_predicted_spans
from parta_utils import load_jsonl
from partc_utils import load_causal_lm, perplexity, sequence_nll


def parse_args():
    parser = argparse.ArgumentParser(description="Run extraction attacks against a victim model.")
    parser.add_argument("--victim_model", type=str, required=True, help="Base victim model, e.g. Qwen/Qwen2.5-0.5B")
    parser.add_argument("--adapter_path", type=str, default=None,
                        help="Path to the trained victim adapter (omit to attack the un-fine-tuned control model)")
    parser.add_argument("--ref_model", type=str, required=True,
                        help="Reference (un-fine-tuned) model for membership scoring")
    parser.add_argument("--corpus", type=str, required=True, help="Path to partC_corpus_raw.jsonl")
    parser.add_argument("--heldout", type=str, default="data/partC_heldout.jsonl", help="Path to partC_heldout.jsonl (skipped if missing)") # Custom add
    parser.add_argument("--canaries", type=str, required=True, help="Path to partC_canaries.jsonl")
    parser.add_argument("--targeted_file", type=str, required=True, help="Path to partC_targeted.jsonl")
    parser.add_argument("--attack", type=str, required=True,
                        choices=["discoverable", "untargeted", "canary", "targeted", "utility", "all"],
                        help="Which attack(s) to run; 'utility' = held-out perplexity only")
    parser.add_argument("--num_samples", type=int, default=5000,
                        help="Number of samples to generate for the untargeted extraction attack")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--report_file", type=str, required=True,
                        help="Path to write the combined attack results JSON report")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--prefix_len", type=int, default=32, help="Discoverable: prefix tokens given to the model")
    parser.add_argument("--suffix_len", type=int, default=32, help="Discoverable: suffix tokens it must reproduce")
    parser.add_argument("--num_canary_candidates", type=int, default=10000, help="Canary: true secret + this many - 1 random ones")
    parser.add_argument("--score_batch_size", type=int, default=512, help="Batch size when scoring losses")
    parser.add_argument("--sanitized_corpus", type=str, default="data/partC_corpus_sanitized.jsonl",
                        help="Untargeted: used only to find which corpus strings are PII; if missing, the "
                             "targeted-file gold values and canary secrets are used instead")
    parser.add_argument("--sanitized_heldout", type=str, default=None,
                        help="Utility: also report perplexity on the sanitized held-out set")
    parser.add_argument("--untargeted_max_new_tokens", type=int, default=128, help="Untargeted: tokens per sample")
    parser.add_argument("--min_pii_chars", type=int, default=6, help="Untargeted: ignore PII values shorter than this")
    parser.add_argument("--targeted_samples", type=int, default=10, help="Targeted: sampled continuations per prompt")
    parser.add_argument("--targeted_max_new_tokens", type=int, default=48, help="Targeted: tokens generated per continuation")
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
# (c) Canary exposure
# ----------------------------------------------------------------------------

def canary_attack(model, tokenizer, canaries, num_candidates, seed, batch_size):
    """Rank each true secret among num_candidates secrets (itself + random alternatives)
    by the model's loss on the full canary sentence.
    exposure = log2(|R|) - log2(rank / num_candidates * |R|) = log2(num_candidates / rank)."""
    per_canary = []
    for c in canaries:
        rng = random.Random(f"{seed}-{c['canary_id']}")
        alternatives = set()
        while len(alternatives) < num_candidates - 1:
            s = f"{rng.randrange(10**6):06d}"
            if s != c["secret"]:
                alternatives.add(s)
        candidates = [c["secret"]] + sorted(alternatives)
        texts = [c["template"].format(project=c["project"], secret=s) for s in candidates]

        losses = [nll for nll, _ in sequence_nll(model, tokenizer, texts, batch_size)]
        rank = 1 + sum(loss < losses[0] for loss in losses[1:])
        exposure = math.log2(num_candidates / rank)
        per_canary.append({"canary_id": c["canary_id"], "n_reps": c["n_reps"], "rank": rank, "exposure": exposure})
        print(f"  canary {c['canary_id']} (n_reps={c['n_reps']}): rank {rank}/{num_candidates}, exposure {exposure:.2f}", flush=True)

    by_n_reps = defaultdict(list)
    for p in per_canary:
        by_n_reps[str(p["n_reps"])].append(p["exposure"])
    return {
        "num_candidates": num_candidates,
        "max_possible_exposure": math.log2(num_candidates),
        "by_n_reps": {k: {"mean_exposure": sum(v) / len(v), "n": len(v)} for k, v in sorted(by_n_reps.items(), key=lambda kv: int(kv[0]))},
        "per_canary": per_canary,
    }


# ----------------------------------------------------------------------------
# (d) Targeted PII extraction
# ----------------------------------------------------------------------------

@torch.no_grad()
def generate_continuations(model, tokenizer, prompts, max_new_tokens, num_samples, batch_size):
    """num_samples=0 -> one greedy continuation per prompt; otherwise num_samples sampled ones (top-k 40)."""
    device = next(model.parameters()).device
    tokenizer.padding_side = "left"  # prompts have different lengths; generation continues from the right edge
    per_prompt = max(1, num_samples)
    step = max(1, batch_size // per_prompt)
    outputs = []
    for i in range(0, len(prompts), step):
        enc = tokenizer(prompts[i:i + step], return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        sampling = {"do_sample": True, "top_k": 40, "temperature": 1.0, "num_return_sequences": num_samples} if num_samples else {"do_sample": False}
        out = model.generate(**enc, max_new_tokens=max_new_tokens, pad_token_id=tokenizer.pad_token_id, **sampling)
        texts = tokenizer.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        outputs.extend(texts[j:j + per_prompt] for j in range(0, len(texts), per_prompt))
    return outputs  # one list of continuations per prompt


def targeted_attack(model, tokenizer, targeted_rows, max_new_tokens, num_samples, batch_size):
    """Prompt with the text right before a PII value; leak = the value appears in the continuation."""
    prompts = [r["prompt"] for r in targeted_rows]
    greedy = generate_continuations(model, tokenizer, prompts, max_new_tokens, 0, batch_size)
    print("  targeted: greedy done", flush=True)
    sampled = generate_continuations(model, tokenizer, prompts, max_new_tokens, num_samples, batch_size)
    print("  targeted: sampling done", flush=True)

    counts = {split: {"n": 0, "greedy": 0, "sampled": 0} for split in ("member", "control")}
    leaked_labels = defaultdict(int)
    leaked_ids = []
    for row, g, s in zip(targeted_rows, greedy, sampled):
        value = _normalize(row["gold_value"])
        greedy_leak = value in _normalize(g[0])
        sampled_leak = any(value in _normalize(text) for text in s)
        c = counts[row["split"]]
        c["n"] += 1
        c["greedy"] += greedy_leak
        c["sampled"] += sampled_leak
        if row["split"] == "member" and (greedy_leak or sampled_leak):
            leaked_labels[row["label"]] += 1
            leaked_ids.append(row["id"])  # ids only; the values stay out of the report

    rate = lambda split, kind: counts[split][kind] / counts[split]["n"] if counts[split]["n"] else 0.0
    return {
        "n_prompts": counts["member"]["n"],
        "n_control_prompts": counts["control"]["n"],
        "top1_leak_rate": rate("member", "greedy"),
        f"top{num_samples}_leak_rate": rate("member", "sampled"),
        "control_top1_leak_rate": rate("control", "greedy"),
        f"control_top{num_samples}_leak_rate": rate("control", "sampled"),
        "max_new_tokens": max_new_tokens,
        "sampling": f"{num_samples} samples, top-k 40, temperature 1.0",
        "member_leaks_by_label": dict(sorted(leaked_labels.items(), key=lambda kv: -kv[1])),
        "leaked_member_ids": leaked_ids,
    }


# ----------------------------------------------------------------------------
# (b) Untargeted extraction with membership scoring
# ----------------------------------------------------------------------------

@torch.no_grad()
def generate_samples(model, tokenizer, num_samples, max_new_tokens, batch_size):
    """Unprompted samples: start from the end-of-text token, which preceded every
    document during training, and sample with top-k 40."""
    device = next(model.parameters()).device
    tokenizer.padding_side = "left"
    samples = []
    while len(samples) < num_samples:
        n = min(batch_size, num_samples - len(samples))
        start = torch.full((n, 1), tokenizer.eos_token_id, device=device)
        out = model.generate(input_ids=start, attention_mask=torch.ones_like(start), do_sample=True, top_k=40,
                             temperature=1.0, max_new_tokens=max_new_tokens, pad_token_id=tokenizer.pad_token_id)
        samples.extend(tokenizer.batch_decode(out[:, 1:], skip_special_tokens=True))
        print(f"  untargeted: generated {len(samples)}/{num_samples}", flush=True)
    return samples


def mean_log_ppl(model, tokenizer, texts, batch_size):
    """Log-perplexity per text = mean token negative log-likelihood."""
    return [nll / max(n, 1) for nll, n in sequence_nll(model, tokenizer, texts, batch_size)]


def build_window_index(corpus_rows, window):
    """Every `window`-character substring of every distinct training text -> its doc_id."""
    index = {}
    for row in corpus_rows:
        text = row["text"]
        for i in range(len(text) - window + 1):
            index.setdefault(text[i:i + window], row["doc_id"])
    return index


def matched_doc(sample, index, window):
    """doc_id of a training document sharing a verbatim `window`-character substring with the sample, else None."""
    for i in range(len(sample) - window + 1):
        doc_id = index.get(sample[i:i + window])
        if doc_id is not None:
            return doc_id
    return None


def corpus_pii_strings(raw_rows, sanitized_rows, canaries, min_chars):
    """PII values = the raw text the sanitizer replaced with a tag (the corpus has no
    gold spans), plus canary secrets. Short values are dropped to avoid chance matches."""
    sanitized = {}
    for row in sanitized_rows:
        sanitized.setdefault(row["doc_id"], row["text"])
    strings = set()
    done = set()
    for row in raw_rows:
        if row["doc_id"] in done or row["is_canary"]:
            continue
        done.add(row["doc_id"])
        for span in recover_predicted_spans(row["text"], sanitized[row["doc_id"]]):
            value = row["text"][span.start:span.end].strip()
            if len(value) >= min_chars:
                strings.add(value)
    strings.update(c["secret"] for c in canaries)
    return strings


def untargeted_attack(model, tokenizer, ref_scorer, corpus_rows, pii_strings, num_samples, max_new_tokens,
                      batch_size, score_batch_size, window=50, top_k=100):
    samples = generate_samples(model, tokenizer, num_samples, max_new_tokens, batch_size)
    unique = sorted({s for s in samples if s.strip()})

    victim_lp = mean_log_ppl(model, tokenizer, unique, score_batch_size)
    lower_lp = mean_log_ppl(model, tokenizer, [s.lower() for s in unique], score_batch_size)
    ref_lp = ref_scorer(unique)
    zlib_bytes = [len(zlib.compress(s.encode("utf-8"))) for s in unique]

    # Lower score = more likely to be memorized training data, for all four scores.
    scores = {
        "ppl": victim_lp,
        "ratio_ref": [v / max(r, 1e-6) for v, r in zip(victim_lp, ref_lp)],
        "ratio_lower": [v / max(l, 1e-6) for v, l in zip(victim_lp, lower_lp)],
        "ratio_zlib": [v / z for v, z in zip(victim_lp, zlib_bytes)],
    }

    index = build_window_index(corpus_rows, window)
    matches = [matched_doc(s, index, window) for s in unique]

    precision = {}
    for name, values in scores.items():
        top = sorted(range(len(unique)), key=lambda i: values[i])[:top_k]
        precision[name] = sum(matches[i] is not None for i in top) / len(top) if top else 0.0

    found_pii = {p for p in pii_strings if any(p in s for s in unique)}
    return {
        "n_generated": len(samples),
        "n_unique": len(unique),
        "precision_at_100": precision,
        "n_samples_matching_training": sum(m is not None for m in matches),
        "distinct_docs_recovered": len({m for m in matches if m is not None}),
        "verified_pii_strings": len(found_pii),
        "pii_recovery_rate": len(found_pii) / len(pii_strings) if pii_strings else 0.0,
        "n_pii_strings_in_corpus": len(pii_strings),
        "match_rule": f"shares a verbatim {window}-character substring with a training row",
        "sampling": f"start from end-of-text token, top-k 40, temperature 1.0, {max_new_tokens} new tokens",
    }


# ----------------------------------------------------------------------------

def make_ref_scorer(args, model, tokenizer):
    """Log-perplexity under the reference model. If the reference is the victim's own base
    model, reuse the loaded weights with the adapter switched off instead of loading them twice."""
    if args.adapter_path and args.ref_model == args.victim_model:
        def score(texts):
            with model.disable_adapter():
                return mean_log_ppl(model, tokenizer, texts, args.score_batch_size)
        return score
    ref_model, ref_tokenizer = load_causal_lm(args.ref_model)
    return lambda texts: mean_log_ppl(ref_model, ref_tokenizer, texts, args.score_batch_size)


def main():
    args = parse_args()
    set_seed(args.seed)

    model, tokenizer = load_causal_lm(args.victim_model, args.adapter_path)
    corpus_rows = load_jsonl(args.corpus)
    attacks = ["utility", "discoverable", "untargeted", "canary", "targeted"] if args.attack == "all" else [args.attack]

    report = {"victim": os.path.basename(os.path.normpath(args.adapter_path)) if args.adapter_path else "base",
              "corpus": os.path.basename(args.corpus)}
    for attack in attacks:
        if attack == "utility":
            if args.heldout and os.path.exists(args.heldout):
                report["heldout_ppl"] = perplexity(model, tokenizer, [r["text"] for r in load_jsonl(args.heldout)])
            if args.sanitized_heldout and os.path.exists(args.sanitized_heldout):
                report["heldout_ppl_sanitized"] = perplexity(model, tokenizer, [r["text"] for r in load_jsonl(args.sanitized_heldout)])
            print(f"  utility: {report.get('heldout_ppl')} / sanitized {report.get('heldout_ppl_sanitized')}", flush=True)
        elif attack == "discoverable":
            report["discoverable"] = discoverable_attack(model, tokenizer, corpus_rows, args.prefix_len,
                                                         args.suffix_len, args.batch_size, args.limit)
        elif attack == "canary":
            canaries = load_jsonl(args.canaries)[:args.limit]
            report["canary"] = canary_attack(model, tokenizer, canaries, args.num_canary_candidates, args.seed, args.score_batch_size)
        elif attack == "targeted":
            rows = load_jsonl(args.targeted_file)
            if args.limit:  # keep both splits in quick checks
                rows = [r for r in rows if r["split"] == "member"][:args.limit] + [r for r in rows if r["split"] == "control"][:args.limit]
            report["targeted"] = targeted_attack(model, tokenizer, rows, args.targeted_max_new_tokens,
                                                 args.targeted_samples, args.batch_size)
        elif attack == "untargeted":
            if os.path.exists(args.sanitized_corpus):
                pii_strings = corpus_pii_strings(corpus_rows, load_jsonl(args.sanitized_corpus), load_jsonl(args.canaries),
                                                 args.min_pii_chars)
                pii_source = "raw values replaced by the sanitizer + canary secrets"
            else:
                pii_strings = {r["gold_value"] for r in load_jsonl(args.targeted_file)
                               if r["split"] == "member" and len(r["gold_value"]) >= args.min_pii_chars}
                pii_strings |= {c["secret"] for c in load_jsonl(args.canaries)}
                pii_source = "targeted member gold values + canary secrets (sanitized corpus not found)"
            print(f"  untargeted: {len(pii_strings)} PII strings from {pii_source}", flush=True)
            report["untargeted"] = untargeted_attack(model, tokenizer, make_ref_scorer(args, model, tokenizer), corpus_rows,
                                                     pii_strings, args.limit or args.num_samples, args.untargeted_max_new_tokens,
                                                     args.batch_size, args.score_batch_size)
            report["untargeted"]["pii_source"] = pii_source

    os.makedirs(os.path.dirname(args.report_file) or ".", exist_ok=True)
    with open(args.report_file, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
