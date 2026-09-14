"""
Part B: run one sanitization system over a test file, score it, and write
  --output_predictions  one JSON line per example
  --report_file         aggregate metrics JSON
"""

import argparse
import json
import os
import random
import time
from dataclasses import asdict

from parta_utils import load_jsonl, parse_examples
from metrics import (
    match_spans_by_class, merge_class_counts, aggregate_prf1,
    over_masking_rate, is_well_formed, _normalize,
)
from partb_utils import (
    regex_sanitize, one_at_a_time, build_presidio_sanitize,
    build_qwen_sanitize, pick_few_shot_examples,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a PII sanitization system.")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--adapter_path", type=str, default=None)
    parser.add_argument("--test_file", type=str, required=True)
    parser.add_argument("--system", type=str, required=True,
                        choices=["finetuned", "regex", "presidio", "zeroshot"])
    parser.add_argument("--output_predictions", type=str, required=True)
    parser.add_argument("--report_file", type=str, required=True)
    parser.add_argument("--limit", type=int, default=None, help="Only evaluate the first N examples (for quick checks)")
    parser.add_argument("--batch_size", type=int, default=16, help="Generation batch size (model systems)")
    parser.add_argument("--num_shots", type=int, default=0, help="zeroshot system: 0 = zero-shot, 5 = 5-shot")
    parser.add_argument("--train_file", type=str, default="data/partA_train.jsonl", help="Source of few-shot examples")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def build_sanitizer(args):
    if args.system == "regex":
        return one_at_a_time(regex_sanitize)
    if args.system == "presidio":
        return build_presidio_sanitize()
    if args.system == "zeroshot":
        shots = pick_few_shot_examples(args.train_file, args.num_shots) if args.num_shots else []
        return build_qwen_sanitize(args.base_model, shots=shots)
    if args.system == "finetuned":
        if not args.adapter_path:
            raise ValueError("--system finetuned needs --adapter_path")
        return build_qwen_sanitize(args.base_model, adapter_path=args.adapter_path)


def set_seed(seed):
    random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def top_k_classes(per_class, k=10):
    ranked = sorted(per_class.items(), key=lambda kv: kv[1]["support"], reverse=True)
    return dict(ranked[:k])


def main():
    args = parse_args()
    set_seed(args.seed)
    examples = parse_examples(load_jsonl(args.test_file))[:args.limit]
    sanitize_batch = build_sanitizer(args)

    # Run the sanitizer over batches; per-example latency = batch time / batch size.
    outputs, latencies = [], []
    start = time.time()
    for i in range(0, len(examples), args.batch_size):
        batch = examples[i:i + args.batch_size]
        t0 = time.time()
        outputs.extend(sanitize_batch([ex.source_text for ex in batch]))
        latencies.extend([(time.time() - t0) * 1000 / len(batch)] * len(batch))
        print(f"  {len(outputs)}/{len(examples)} done  ({time.time() - start:.0f}s)", flush=True)

    exact_counts, iou_counts = [], []
    n_leaked = n_gold = 0
    n_altered = n_non_pii = 0
    n_well_formed = 0
    predictions = []

    for ex, (pred_text, pred_spans), latency_ms in zip(examples, outputs, latencies):
        gold = [asdict(s) for s in ex.privacy_mask]

        # Entity F1: count tp/fp/fn per label under both matching rules.
        exact_counts.append(match_spans_by_class(pred_spans, gold, mode="exact"))
        iou_counts.append(match_spans_by_class(pred_spans, gold, mode="iou", iou_threshold=0.5))

        # Leak rate: gold values still visible in the output.
        norm_pred = _normalize(pred_text)
        leaked_values = [g["value"] for g in gold if _normalize(g["value"]) in norm_pred]
        n_leaked += len(leaked_values)
        n_gold += len(gold)

        # Over-masking: non-PII characters that got changed.
        altered, non_pii = over_masking_rate(ex.source_text, pred_text, gold)
        n_altered += altered
        n_non_pii += non_pii

        n_well_formed += is_well_formed(ex.source_text, pred_text)

        predictions.append({
            "id": ex.id,
            "language": ex.language,
            "source_text": ex.source_text,
            "predicted_text": pred_text,
            "predicted_spans": [asdict(s) for s in pred_spans],
            "gold_spans": [{"start": g["start"], "end": g["end"], "label": g["label"]} for g in gold],
            "leaked_values": leaked_values,
            "latency_ms": round(latency_ms, 2),
        })

    exact = aggregate_prf1(merge_class_counts(*exact_counts))
    iou = aggregate_prf1(merge_class_counts(*iou_counts))
    summary_keys = ["micro_p", "micro_r", "micro_f1", "macro_f1"]

    report = {
        "system": args.system,
        "adapter": args.adapter_path,
        "num_shots": args.num_shots if args.system == "zeroshot" else None,
        "test_set": os.path.basename(args.test_file),
        "n": len(examples),
        "exact": {k: exact[k] for k in summary_keys},
        "overlap_iou50": {k: iou[k] for k in summary_keys},
        "per_class": top_k_classes(iou["per_class"]),
        "per_class_exact": top_k_classes(exact["per_class"]),
        "leak_rate": n_leaked / n_gold if n_gold else 0.0,
        "over_masking_rate": n_altered / n_non_pii if n_non_pii else 0.0,
        "well_formed_rate": n_well_formed / len(examples) if examples else 0.0,
    }

    for path in (args.output_predictions, args.report_file):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(args.output_predictions, "w") as f:
        for row in predictions:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with open(args.report_file, "w") as f:
        json.dump(report, f, indent=2)

    print(f"[{args.system}] n={report['n']}  "
          f"exact_f1={report['exact']['micro_f1']:.3f}  iou50_f1={report['overlap_iou50']['micro_f1']:.3f}  "
          f"leak_rate={report['leak_rate']:.3f}  over_masking={report['over_masking_rate']:.3f}")


if __name__ == "__main__":
    main()
