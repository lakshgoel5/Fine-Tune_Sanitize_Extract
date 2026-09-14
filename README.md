# Fine-Tune, Sanitize, Extract

Fine-tuning LLMs on unvetted data often leads to the unintended inclusion of private information in the model's weights.

### Authors
- Laksh Goel (2023CS10848)

## Setup

```bash
# install torch for your CUDA version first, then:
pip install -r requirements.txt
python -m spacy download en_core_web_lg   # needed by the Presidio baseline
```

To use a local copy of the model, pass its folder instead of the Hub name (`--base_model /path/to/qwen-2.5-1.5b-instruct`, or `MODEL=/path bash run_all.sh`).

## Reproduce everything

```bash
bash run_all.sh
```

## Code layout

| File | Purpose |
|---|---|
| `src/peft_sweep.py` | Part A: one LoRA/QLoRA run, appends a row to `results/partA_sweep.csv` |
| `src/parta_utils.py` | Data loading, shared system prompt, Part A plot |
| `src/metrics.py` | Span recovery (alignment) and all entity / leak / over-masking metrics |
| `src/partb_utils.py` | Part B sanitizers: regex, Presidio, prompted / fine-tuned Qwen |
| `src/sanitize_eval.py` | Part B: run one system on a test file, write predictions + metrics |
| `src/sanitize_corpus.py` | Part C: pass a corpus through the Part B sanitizer (keeps all metadata) |
| `src/train_victim.py` | Part C: train a victim (Qwen2.5-0.5B base + LoRA), report held-out perplexity |
| `src/attack_extract.py` | Part C: discoverable, untargeted, canary and targeted attacks (+ held-out perplexity) |
| `src/partc_utils.py` | Part C: perplexity helpers, exposure plot, merging per-attack reports |

## Part B commands

```bash
# Fine-tuned sanitizer (A6 = QLoRA, so the base is loaded in 4-bit by default)
python src/sanitize_eval.py --base_model Qwen/Qwen2.5-1.5B-Instruct --adapter_path adapters/best_sanitizer \
  --test_file data/partB_test.jsonl --system finetuned \
  --output_predictions results/partB_predictions.jsonl --report_file results/partB_metrics.json
```

`--system` is one of `finetuned`, `regex`, `presidio`, `zeroshot`. Extra options:

| Option | Meaning |
|---|---|
| `--num_shots 5` | `zeroshot` system with 5 in-context examples from `--train_file` |
| `--quant {none,4bit}` | how the base model is loaded for `finetuned` (default `4bit`, matching A6) |
| `--batch_size N` | generation batch size (default 16) |
| `--limit N` | only evaluate the first N rows (quick checks) |
| `--label_map ood` | for `partB_ood_test.jsonl`: rename its labels to the training schema; F1 on matched labels, leak rate on all PII |
| `--rescore_from FILE` | re-score saved model outputs (re-runs span recovery and metrics, no generation) |

## Part C commands

```bash
# Sanitized corpus (derived locally, not shipped)
python src/sanitize_corpus.py --adapter_path adapters/best_sanitizer \
  --input data/partC_corpus_raw.jsonl --output data/partC_corpus_sanitized.jsonl

# Victims: same settings, only the corpus differs
python src/train_victim.py --model Qwen/Qwen2.5-0.5B --corpus data/partC_corpus_raw.jsonl --heldout data/partC_heldout.jsonl \
  --target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
  --lora_r 32 --lora_alpha 64 --epochs 4 --seed 42 --output_dir adapters/victim_raw

# Attacks (omit --adapter_path to attack the un-fine-tuned control model)
python src/attack_extract.py --victim_model Qwen/Qwen2.5-0.5B --adapter_path adapters/victim_raw --ref_model Qwen/Qwen2.5-0.5B \
  --corpus data/partC_corpus_raw.jsonl --heldout data/partC_heldout.jsonl --canaries data/partC_canaries.jsonl \
  --targeted_file data/partC_targeted.jsonl --attack all --num_samples 5000 --seed 42 \
  --report_file results/partC/raw_all.json

# Combine reports and plot exposure
python src/partc_utils.py --mode merge --report_dir results/partC --output results/partC_attacks.json
python src/partc_utils.py --mode plot --reports results/partC/raw_all.json results/partC/san_all.json results/partC/base_all.json \
  --names V-raw V-san base --output results/partC_exposure.png
```

`--attack` can also be a single attack (`discoverable`, `untargeted`, `canary`, `targeted`) or `utility` (held-out perplexity only), so the attacks can run as parallel jobs; `--mode merge` combines files named `<model>_<attack>.json`. `results/partC_attacks.json` has the required single-object shape for V-raw, with V-san and the un-fine-tuned control under the extra keys `victim_san` and `control`.

## Report

```bash
pdflatex report.tex
```
