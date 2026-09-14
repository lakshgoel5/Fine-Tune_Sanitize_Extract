"""
Part B sanitizers. Every sanitizer works on a batch of texts:
sanitize_batch(list_of_source_texts) -> list of (predicted_text, predicted_spans)
"""

import re

from metrics import PredSpan, recover_predicted_spans
from parta_utils import SYSTEM_PROMPT, load_jsonl


# Near-OOD (ai4privacy-200k) label -> our 27-label schema, only where the meaning clearly matches.
# Labels not listed (IBAN, AMOUNT, JOBTITLE, BITCOINADDRESS, ...) have no counterpart the model was trained on.
OOD_LABEL_MAP = {
    "DATE": "DATE", "EMAIL": "EMAIL", "TIME": "TIME", "CITY": "CITY", "STATE": "STATE",
    "STREET": "STREET", "USERNAME": "USERNAME", "SEX": "SEX", "IP": "IP",
    "FIRSTNAME": "GIVENNAME1", "MIDDLENAME": "GIVENNAME2", "LASTNAME": "LASTNAME1",
    "PHONENUMBER": "TEL", "IPV4": "IP", "IPV6": "IP", "SSN": "SOCIALNUMBER", "DOB": "BOD",
    "ZIPCODE": "POSTCODE", "PREFIX": "TITLE", "BUILDINGNUMBER": "BUILDING",
    "SECONDARYADDRESS": "SECADDRESS", "PASSWORD": "PASS", "NEARBYGPSCOORDINATE": "GEOCOORD",
    "GENDER": "SEX",
}


def one_at_a_time(sanitize):
    """Turn a single-text sanitizer into a batch sanitizer."""
    return lambda texts: [sanitize(t) for t in texts]

# ----------------------------------------------------------------------------
# Baseline 1: regex rules
# ----------------------------------------------------------------------------

MONTH = r"(?:January|February|March|April|May|June|July|August|September|October|November|December)"

# (label, pattern). If two matches start at the same place and have the same
# length, the one listed earlier wins -- so specific patterns go first, TEL last.
REGEX_RULES = [
    ("EMAIL",        r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+"),
    ("URL",          r"https?://\S+"),
    ("GEOCOORD",     r"\[-?\d+\.\d+,\s*-?\d+\.\d+\]"),
    ("IP",           r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
    ("IP",           r"\b(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}\b"),
    ("CREDITCARD",   r"\b(?:\d{4}[ -]?){3}\d{4}\b"),
    ("SOCIALNUMBER", r"\b\d{3}[- ]\d{2}[- ]\d{4}\b"),
    ("DATE",         r"\b\d{4}-\d{2}-\d{2}(?:T\d{2}:\d{2}:\d{2})?"),
    ("DATE",         r"\b\d{1,2}/\d{1,2}/\d{2,4}\b"),
    ("DATE",         rf"\b{MONTH} \d{{1,2}}(?:st|nd|rd|th)?, \d{{4}}\b"),
    ("DATE",         rf"\b{MONTH}/\d{{2}}\b"),
    ("TIME",         r"\b\d{1,2}:\d{2}(?::\d{2})?(?:\s?[AP]M)?"),
    ("TIME",         r"\b\d{1,2} o'clock"),
    ("POSTCODE",     r"\b\d{5}(?:-\d{4})?\b"),
    ("TITLE",        r"\b(?:Mr|Mrs|Ms|Miss|Dr|Prof|Sir|Lord|Lady|Mayor|Esq)\b"),
    ("SEX",          r"\b(?:Male|Female|Masculine|Feminine)\b"),
    ("TEL",          r"(?<!\w)\+?\d[\d .-]{7,14}\d(?!\w)"),
]
REGEX_RULES = [(label, re.compile(p)) for label, p in REGEX_RULES]


def remove_overlaps(matches):
    """matches: list of (start, end, label, priority).
    Walk left to right; at each position prefer the longest match (then the
    higher-priority rule), and skip anything overlapping what we already kept."""
    matches = sorted(matches, key=lambda m: (m[0], -(m[1] - m[0]), m[3]))
    kept = []
    for start, end, label, _ in matches:
        if kept and start < kept[-1][1]:
            continue
        kept.append((start, end, label))
    return kept


def apply_spans(source_text, spans):
    """Replace each (start, end, label) region of source_text with [label]."""
    out, cursor = [], 0
    for start, end, label in spans:
        out.append(source_text[cursor:start])
        out.append(f"[{label}]")
        cursor = end
    out.append(source_text[cursor:])
    return "".join(out)


def regex_sanitize(source_text):
    matches = []
    for priority, (label, pattern) in enumerate(REGEX_RULES):
        for m in pattern.finditer(source_text):
            matches.append((m.start(), m.end(), label, priority))

    spans = remove_overlaps(matches)
    predicted_text = apply_spans(source_text, spans)
    return predicted_text, [PredSpan(s, e, l) for s, e, l in spans]


# ----------------------------------------------------------------------------
# Baseline 2: Presidio (default recognizers)
# ----------------------------------------------------------------------------

# Presidio type -> the gold label it most often overlaps on partB_test.
# Types not listed keep their Presidio name (they still mask, but never label-match).
PRESIDIO_TO_GOLD = {
    "EMAIL_ADDRESS":     "EMAIL",
    "PHONE_NUMBER":      "TEL",
    "IP_ADDRESS":        "IP",
    "US_SSN":            "SOCIALNUMBER",
    "US_PASSPORT":       "PASSPORT",
    "US_DRIVER_LICENSE": "DRIVERLICENSE",
    "DATE_TIME":         "DATE",
    "PERSON":            "GIVENNAME1",
    "LOCATION":          "CITY",
}


def build_presidio_sanitize():
    from presidio_analyzer import AnalyzerEngine
    analyzer = AnalyzerEngine()  # slow to load, so build once and reuse

    def presidio_sanitize(source_text):
        matches = []
        for r in analyzer.analyze(text=source_text, language="en"):
            label = PRESIDIO_TO_GOLD.get(r.entity_type, r.entity_type)
            matches.append((r.start, r.end, label, -r.score))  # higher score = higher priority

        spans = remove_overlaps(matches)
        predicted_text = apply_spans(source_text, spans)
        return predicted_text, [PredSpan(s, e, l) for s, e, l in spans]

    return one_at_a_time(presidio_sanitize)


# ----------------------------------------------------------------------------
# Baseline 3: Qwen prompted (zero-shot / few-shot), and the fine-tuned model
# ----------------------------------------------------------------------------

def pick_few_shot_examples(train_file, k, max_chars=300):
    """First k short training rows -> (source_text, target_text) pairs."""
    shots = []
    for row in load_jsonl(train_file):
        if len(row["source_text"]) <= max_chars:
            shots.append((row["source_text"], row["target_text"]))
        if len(shots) == k:
            break
    return shots


def build_messages(source_text, shots):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for shot_source, shot_target in shots:
        messages.append({"role": "user", "content": shot_source})
        messages.append({"role": "assistant", "content": shot_target})
    messages.append({"role": "user", "content": source_text})
    return messages


def build_qwen_sanitize(base_model, adapter_path=None, shots=(), quant="none", margin=32):
    """adapter_path=None -> prompted base model; otherwise the fine-tuned sanitizer.
    quant="4bit" loads the base the way QLoRA (A6) was trained."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # so every prompt in a batch ends at the same position

    use_cuda = torch.cuda.is_available()
    dtype = torch.float16 if use_cuda else torch.float32
    quant_config = None
    if quant == "4bit":
        # Same settings as peft_sweep.py used for A6.
        quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=dtype, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        dtype=dtype,
        quantization_config=quant_config,
        device_map="auto" if use_cuda else None,
    )
    if adapter_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    device = next(model.parameters()).device

    @torch.no_grad()
    def qwen_sanitize(texts):
        prompts = [tokenizer.apply_chat_template(build_messages(t, shots), tokenize=False, add_generation_prompt=True) for t in texts]
        enc = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)

        # Output is about as long as the input, so budget generation from the longest input.
        longest = max(len(tokenizer(t, add_special_tokens=False)["input_ids"]) for t in texts)
        out = model.generate(**enc, max_new_tokens=longest + margin, do_sample=False, pad_token_id=tokenizer.pad_token_id)

        generated = out[:, enc["input_ids"].shape[1]:]  # drop the prompt tokens
        predicted_texts = tokenizer.batch_decode(generated, skip_special_tokens=True)

        # B.1: the model only gives text, so recover spans by aligning it against the source.
        return [(p, recover_predicted_spans(t, p)) for t, p in zip(texts, predicted_texts)]

    return qwen_sanitize
