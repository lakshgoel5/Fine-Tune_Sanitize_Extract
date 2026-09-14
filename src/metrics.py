"""
Shared span-recovery and metric helpers used by both Part A (validation
scoring during the PEFT sweep) and Part B (full sanitizer evaluation).
"""

import difflib
import re
from collections import defaultdict
from dataclasses import dataclass

TAG_RE = re.compile(r"\[([A-Z0-9_]+)\]")

@dataclass
class PredSpan:
    start: int
    end: int
    label: str

def _is_real_anchor(tag, i1, i2, source_text, min_equal_len):
    # A genuine equal run: either long enough to be a trustworthy anchor,
    # or short but non-alphanumeric (whitespace/punctuation -- a real
    # separator, not a coincidental letter/digit match).
    if tag != "equal":
        return False
    if (i2 - i1) >= min_equal_len:
        return True
    return not source_text[i1:i2].isalnum()

def _coalesce_opcodes(opcodes, source_text: str, min_equal_len: int = 3):
    """difflib.SequenceMatcher can find spurious short "equal" runs inside what
    is semantically one replacement -- e.g. source "Marta Ruiz" vs predicted
    "[GIVENNAME1]" share the single letter "M" (from "GIVENNA-M-E1]"), which
    splits the real replacement into insert + equal('M') + replace, tearing the
    [LABEL] tag itself in half across opcodes. Any single-opcode-at-a-time scan
    then finds the tag in neither half and silently drops the whole span --
    a real recall bug, not just a boundary-precision one.

    This merges a short "equal" run into the surrounding non-equal region --
    but ONLY when it's purely alphanumeric (a coincidental letter/digit match,
    like the lone "M" above). A short equal run that's whitespace/punctuation
    (", ", ". ", " ") is kept as a real anchor no matter how short: that's
    almost always a genuine separator between two DISTINCT spans (e.g.
    "[EMAIL], [TEL]"), and coalescing it would wrongly merge two separate PII
    spans into one giant one -- a real bug this exact heuristic caused before
    this alphanumeric check was added; verified via a regression test."""

    merged = []
    buf = None  # [i1, i2, j1, j2] of an in-progress merged non-equal region
    for tag, i1, i2, j1, j2 in opcodes:
        if _is_real_anchor(tag, i1, i2, source_text, min_equal_len):
            if buf is not None:
                merged.append(("replace", buf[0], buf[1], buf[2], buf[3]))
                buf = None
            merged.append((tag, i1, i2, j1, j2))
        else:
            # insert / delete / replace, AND short-alphanumeric "equal" noise --
            # all accumulate into one running merged region until the next real anchor.
            if buf is None:
                buf = [i1, i2, j1, j2]
            else:
                buf[1], buf[3] = i2, j2
    if buf is not None:
        merged.append(("replace", buf[0], buf[1], buf[2], buf[3]))
    return merged


def recover_predicted_spans(source_text: str, predicted_text: str) -> list[PredSpan]:
    """Align predicted_text against source_text and turn each replaced gap
    that contains a [LABEL] tag into a (start, end, label) span in
    source_text's coordinate system.
    """
    sm = difflib.SequenceMatcher(None, source_text, predicted_text, autojunk=False)
    spans: list[PredSpan] = []
    for tag, i1, i2, j1, j2 in _coalesce_opcodes(sm.get_opcodes(), source_text):
        if tag == "equal":
            continue
        # i1:i2 is the gap in source_text; j1:j2 is what replaced it in predicted_text.
        if i1 == i2:
            continue  # pure insertion in predicted_text, nothing to redact in source
        chunk = predicted_text[j1:j2]
        m = TAG_RE.search(chunk)
        if m is None:
            continue  # replaced but not with a recognizable [LABEL] tag
        spans.append(PredSpan(start=i1, end=i2, label=m.group(1)))
    return spans


def _iou(a_start: int, a_end: int, b_start: int, b_end: int) -> float:
    inter = max(0, min(a_end, b_end) - max(a_start, b_start))
    if inter == 0:
        return 0.0
    union = (a_end - a_start) + (b_end - b_start) - inter
    return inter / union if union > 0 else 0.0


def match_counts_iou(pred_spans: list[PredSpan], gold_spans: list[dict], iou_threshold: float = 0.5) -> tuple[int, int, int]:
    """Greedy IoU matching (same label required). Returns (tp, fp, fn) for one example."""
    unmatched_gold = list(range(len(gold_spans)))
    tp = 0
    for p in pred_spans:
        best_j, best_iou = None, 0.0
        for j in unmatched_gold:
            g = gold_spans[j]
            if g["label"] != p.label:
                continue
            i = _iou(p.start, p.end, g["start"], g["end"])
            if i >= iou_threshold and i > best_iou:
                best_j, best_iou = j, i
        if best_j is not None:
            tp += 1
            unmatched_gold.remove(best_j)
    fp = len(pred_spans) - tp
    fn = len(unmatched_gold)
    return tp, fp, fn


def micro_prf1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f1

def _spans_match(p: PredSpan, g: dict, mode: str, iou_threshold: float) -> bool:
    """Part B.2: the general form of match_counts_iou -- greedy matching that
    buckets tp/fp/fn PER CLASS (not just one summed triple), so callers can
    derive micro F1 (sum buckets, one P/R/F1), macro F1 (average per-class F1),
    and the per-class table all from the same accumulator.

    mode="exact": span must match exactly (same start, end, AND label) --
    B.2's first matching criterion.
    mode="iou": IoU >= iou_threshold + label match -- B.2's second criterion
    (same rule match_counts_iou implements, kept separate/untouched since
    Part A's peft_sweep.py depends on that exact function).

    Returns {label: [tp, fp, fn]}, covering every label seen in either
    pred_spans or gold_spans for this one example. Caller sums these dicts
    across the whole dataset before computing rates (never average per-example
    rates -- that's not what micro/macro mean here).
    """

    if g["label"] != p.label:
        return False
    if mode == "exact":
        return p.start == g["start"] and p.end == g["end"]
    return _iou(p.start, p.end, g["start"], g["end"]) >= iou_threshold

def _match_score(p: PredSpan, g: dict, mode: str) -> float:
    return 1.0 if mode == "exact" else _iou(p.start, p.end, g["start"], g["end"])

def match_spans_by_class(pred_spans: list[PredSpan], gold_spans: list[dict], mode: str = "iou", iou_threshold: float = 0.5) -> dict:
    counts = defaultdict(lambda: [0, 0, 0])  # label -> [tp, fp, fn]
    unmatched_gold = list(range(len(gold_spans)))
    for p in pred_spans:
        best_j, best_score = None, -1.0
        for j in unmatched_gold:
            g = gold_spans[j]
            if _spans_match(p, g, mode, iou_threshold):
                s = _match_score(p, g, mode)
                if s > best_score:
                    best_j, best_score = j, s
        if best_j is not None:
            counts[p.label][0] += 1  # tp
            unmatched_gold.remove(best_j)
        else:
            counts[p.label][1] += 1  # fp
    for j in unmatched_gold:
        counts[gold_spans[j]["label"]][2] += 1  # fn
    return dict(counts)


def merge_class_counts(*count_dicts: dict) -> dict:
    """Sums several {label: [tp,fp,fn]} dicts (one per example) into one
    dataset-level accumulator. Use this to fold match_spans_by_class's
    per-example output across an entire test set before calling aggregate_prf1."""
    merged = defaultdict(lambda: [0, 0, 0])
    for counts in count_dicts:
        for label, (tp, fp, fn) in counts.items():
            merged[label][0] += tp
            merged[label][1] += fp
            merged[label][2] += fn
    return dict(merged)


def aggregate_prf1(class_counts: dict) -> dict:
    """From a dataset-level {label: [tp,fp,fn]} accumulator, compute micro
    P/R/F1 (pool all classes, one P/R/F1), macro F1 (per-class F1, averaged
    unweighted across classes), and the full per-class P/R/F1/support table.
    Matches the "exact"/"overlap_iou50" sub-object shape in partB_metrics.json
    (micro_p, micro_r, micro_f1, macro_f1) plus a per_class table callers can
    sort by support and slice to the top 10 for the report."""
    tp_sum = sum(c[0] for c in class_counts.values())
    fp_sum = sum(c[1] for c in class_counts.values())
    fn_sum = sum(c[2] for c in class_counts.values())
    micro_p, micro_r, micro_f1 = micro_prf1(tp_sum, fp_sum, fn_sum)

    per_class = {}
    for label, (tp, fp, fn) in class_counts.items():
        p, r, f1 = micro_prf1(tp, fp, fn)
        per_class[label] = {"p": p, "r": r, "f1": f1, "support": tp + fn}
    macro_f1 = sum(c["f1"] for c in per_class.values()) / len(per_class) if per_class else 0.0

    return {
        "micro_p": micro_p, "micro_r": micro_r, "micro_f1": micro_f1,
        "macro_f1": macro_f1, "per_class": per_class,
    }


def over_masking_rate(source_text: str, predicted_text: str, gold_spans: list[dict]) -> tuple[int, int]:
    """Part B.2: fraction of NON-PII source characters that were removed or
    replaced. Diffs source_text against predicted_text (same technique as
    recover_predicted_spans, run independently here since we need
    character-level "was this position touched" rather than span groupings),
    marks every character NOT covered by a gold span, and checks whether it
    survived unchanged.

    Returns (n_altered, n_total) over non-PII characters in THIS example --
    caller sums both across the dataset before dividing, same accumulator
    pattern as count_leaks."""
    gold_mask = [False] * len(source_text)
    for g in gold_spans:
        for i in range(max(0, g["start"]), min(g["end"], len(source_text))):
            gold_mask[i] = True

    altered_mask = [False] * len(source_text)
    sm = difflib.SequenceMatcher(None, source_text, predicted_text, autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag != "equal":
            for i in range(i1, i2):
                altered_mask[i] = True

    n_total = 0
    n_altered = 0
    for i in range(len(source_text)):
        if gold_mask[i]:
            continue
        n_total += 1
        if altered_mask[i]:
            n_altered += 1
    return n_altered, n_total


def _normalize(s: str) -> str:
    return " ".join(s.split()).lower()


def count_leaks(gold_spans: list[dict], predicted_text: str) -> tuple[int, int]:
    """Returns (n_leaked, n_total) gold PII values still found verbatim
    (case-insensitive, whitespace-normalized) in predicted_text."""
    norm_pred = _normalize(predicted_text)
    leaked = 0
    for g in gold_spans:
        value = g.get("value")
        if not value:
            continue
        if _normalize(value) in norm_pred:
            leaked += 1
    return leaked, len(gold_spans)


def is_well_formed(source_text: str, predicted_text: str, min_len_ratio: float = 0.5, max_len_ratio: float = 1.5) -> bool:
    if not predicted_text.strip():
        return False
    if not source_text:
        return True
    ratio = len(predicted_text) / len(source_text)
    return min_len_ratio <= ratio <= max_len_ratio
