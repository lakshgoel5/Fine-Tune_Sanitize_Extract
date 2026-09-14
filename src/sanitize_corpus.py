"""
Part C.2 prep: pass every row's `text` through the Part B sanitizer, keep all
other fields, and write the sanitized corpus.

    python src/sanitize_corpus.py --base_model Qwen/Qwen2.5-1.5B-Instruct \
        --adapter_path adapters/best_sanitizer --quant 4bit \
        --input data/partC_corpus_raw.jsonl --output data/partC_corpus_sanitized.jsonl
"""

import argparse
import json
import time

from parta_utils import load_jsonl
from partb_utils import build_qwen_sanitize


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--adapter_path", type=str, required=True)
    parser.add_argument("--quant", type=str, default="4bit", choices=["none", "4bit"])
    parser.add_argument("--input", type=str, default="data/partC_corpus_raw.jsonl")
    parser.add_argument("--output", type=str, default="data/partC_corpus_sanitized.jsonl")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--limit", type=int, default=None, help="Only sanitize the first N rows (quick checks)")
    return parser.parse_args()


def main():
    args = parse_args()
    rows = load_jsonl(args.input)[:args.limit]
    sanitize_batch = build_qwen_sanitize(args.base_model, adapter_path=args.adapter_path, quant=args.quant)

    # Repeated rows share the same text: sanitize each distinct text once.
    # Sorting by length keeps padding (and wasted compute) small within a batch.
    unique_texts = sorted({row["text"] for row in rows}, key=len)
    sanitized = {}
    start = time.time()
    for i in range(0, len(unique_texts), args.batch_size):
        batch = unique_texts[i:i + args.batch_size]
        for text, (predicted_text, _spans) in zip(batch, sanitize_batch(batch)):
            sanitized[text] = predicted_text
        print(f"  {len(sanitized)}/{len(unique_texts)} distinct texts  ({time.time() - start:.0f}s)", flush=True)

    with open(args.output, "w") as f:
        for row in rows:
            f.write(json.dumps({**row, "text": sanitized[row["text"]]}, ensure_ascii=False) + "\n")

    n_changed = sum(sanitized[r["text"]] != r["text"] for r in rows)
    canaries = [r for r in rows if r["is_canary"] and r["repetition_index"] == 0]
    n_canary_changed = sum(sanitized[r["text"]] != r["text"] for r in canaries)
    print(f"wrote {len(rows)} rows to {args.output}: {n_changed} changed; "
          f"{n_canary_changed}/{len(canaries)} distinct canaries changed")


if __name__ == "__main__":
    main()
