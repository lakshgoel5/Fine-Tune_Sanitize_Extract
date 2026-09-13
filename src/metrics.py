"""
Shared span-recovery and metric helpers used by both Part A (validation
scoring during the PEFT sweep) and Part B (full sanitizer evaluation).
"""

import difflib
import re
from dataclasses import dataclass

TAG_RE = re.compile(r"\[([A-Z0-9_]+)\]")

@dataclass
class PredSpan:
    start: int
    end: int
    label: str

def recover_predicted_spans(source_text: str, predicted_text: str) -> list[PredSpan]:
    """Align predicted_text against source_text and turn each replaced gap
    that contains a [LABEL] tag into a (start, end, label) span in
    source_text's coordinate system.
    """
    sm = difflib.SequenceMatcher(None, source_text, predicted_text, autojunk=False)
    spans: list[PredSpan] = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
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
