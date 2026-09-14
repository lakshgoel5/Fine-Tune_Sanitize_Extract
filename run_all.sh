#!/usr/bin/env bash
# Reproduces every reported number. Run from the repo root.
# MODEL can point at a local copy of Qwen2.5-1.5B-Instruct, e.g.
#   MODEL=/path/to/qwen-2.5-1.5b-instruct bash run_all.sh
set -euo pipefail

MODEL=${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}
TEST=data/partB_test.jsonl
mkdir -p results/partB logs

# ============================================================================
# Part A: PEFT sweep (6 runs, each appends one row to results/partA_sweep.csv)
# ============================================================================
sweep() {  # run_id target_modules r alpha quant
  python src/peft_sweep.py --model "$MODEL" \
    --train_file data/partA_train.jsonl --val_file data/partA_validation.jsonl \
    --target_modules "$2" --lora_r "$3" --lora_alpha "$4" --lora_dropout 0.05 \
    --quant "$5" --seed 42 --epochs 2 \
    --run_id "$1" --output_dir "adapters/$1" --results_csv results/partA_sweep.csv
}
ATTN=q_proj,k_proj,v_proj,o_proj
ALL=q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj

sweep A1 q_proj,v_proj 8  16 none
sweep A2 $ATTN         8  16 none
sweep A3 $ALL          8  16 none
sweep A4 $ALL          32 16 none
sweep A5 $ALL          32 64 none
sweep A6 $ALL          32 64 4bit

python src/parta_utils.py --mode plot --results_csv results/partA_sweep.csv --output_dir results

# ============================================================================
# Part B: sanitizer evaluation (B.1-B.3)
# Best sanitizer = A6 (QLoRA), evaluated on a 4-bit base as trained.
# ============================================================================
rm -rf adapters/best_sanitizer && cp -r adapters/A6 adapters/best_sanitizer

python src/sanitize_eval.py --base_model "$MODEL" --adapter_path adapters/best_sanitizer --quant 4bit \
  --test_file $TEST --system finetuned --batch_size 64 \
  --output_predictions results/partB_predictions.jsonl --report_file results/partB_metrics.json

python src/sanitize_eval.py --base_model "$MODEL" --adapter_path adapters/A5 --quant none \
  --test_file $TEST --system finetuned --batch_size 64 \
  --output_predictions results/partB/A5_predictions.jsonl --report_file results/partB/A5_metrics.json

python src/sanitize_eval.py --test_file $TEST --system regex \
  --output_predictions results/partB/regex_predictions.jsonl --report_file results/partB/regex_metrics.json

python src/sanitize_eval.py --test_file $TEST --system presidio \
  --output_predictions results/partB/presidio_predictions.jsonl --report_file results/partB/presidio_metrics.json

python src/sanitize_eval.py --base_model "$MODEL" --test_file $TEST --system zeroshot --num_shots 0 --batch_size 64 \
  --output_predictions results/partB/zeroshot_predictions.jsonl --report_file results/partB/zeroshot_metrics.json

python src/sanitize_eval.py --base_model "$MODEL" --test_file $TEST --system zeroshot --num_shots 5 --batch_size 64 \
  --output_predictions results/partB/fewshot5_predictions.jsonl --report_file results/partB/fewshot5_metrics.json

# TODO B.4: near-OOD evaluation (partB_ood_test.jsonl) once the label mapping is added.
# ============================================================================
# Part C
# ============================================================================
# C.2 prep: sanitized corpus from the Part B sanitizer (canaries NOT exempted)
python src/sanitize_corpus.py --base_model "$MODEL" --adapter_path adapters/best_sanitizer --quant 4bit \
  --input data/partC_corpus_raw.jsonl --output data/partC_corpus_sanitized.jsonl --batch_size 64
# sanitized copy of the held-out set, only for the extra utility measurement
python src/sanitize_corpus.py --base_model "$MODEL" --adapter_path adapters/best_sanitizer --quant 4bit \
  --input data/partC_heldout.jsonl --output data/partC_heldout_sanitized.jsonl --batch_size 64

# C.2 victims: identical PEFT config and hyperparameters, only the corpus differs
VICTIM=${VICTIM:-Qwen/Qwen2.5-0.5B}
victim() {  # corpus output_dir
  python src/train_victim.py --model "$VICTIM" --corpus "$1" --heldout data/partC_heldout.jsonl \
    --target_modules $ALL --lora_r 32 --lora_alpha 64 --epochs 4 --seed 42 --batch_size 16 --output_dir "$2"
}
victim data/partC_corpus_raw.jsonl       adapters/victim_raw
victim data/partC_corpus_sanitized.jsonl adapters/victim_san

# C.3 attacks (all four + held-out perplexity) on V-raw, V-san and the un-fine-tuned control
attack() {  # name [adapter args]
  python src/attack_extract.py --victim_model "$VICTIM" --ref_model "$VICTIM" "${@:2}" \
    --corpus data/partC_corpus_raw.jsonl --sanitized_corpus data/partC_corpus_sanitized.jsonl \
    --heldout data/partC_heldout.jsonl --sanitized_heldout data/partC_heldout_sanitized.jsonl \
    --canaries data/partC_canaries.jsonl --targeted_file data/partC_targeted.jsonl \
    --attack all --num_samples 5000 --seed 42 --report_file "results/partC/$1_all.json"
}
attack raw  --adapter_path adapters/victim_raw
attack san  --adapter_path adapters/victim_san
attack base

python src/partc_utils.py --mode merge --report_dir results/partC --output results/partC_attacks.json
python src/partc_utils.py --mode plot --reports results/partC/raw_all.json results/partC/san_all.json results/partC/base_all.json --names V-raw V-san base --output results/partC_exposure.png
