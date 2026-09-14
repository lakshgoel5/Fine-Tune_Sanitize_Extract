#!/usr/bin/env bash
# Builds the submission ZIP. Run from the repo root: bash submission.sh
# Ships code, docs, report and results only; no data, adapters, corpus text or generation dumps.
set -euo pipefail

ENTRY=2023CS10848
STAGE=$(mktemp -d)/$ENTRY
mkdir -p "$STAGE/src" "$STAGE/results/partB" "$STAGE/results/partC"

# Report PDF (build it if missing)
if [ ! -f report.pdf ]; then
  pdflatex -interaction=nonstopmode report.tex > /dev/null && pdflatex -interaction=nonstopmode report.tex > /dev/null
fi

# Every file the grader-facing layout requires; stop if any is missing
required=(
  README.md requirements.txt run_all.sh report.pdf
  src/peft_sweep.py src/sanitize_eval.py src/train_victim.py src/attack_extract.py
  results/partA_sweep.csv results/partA_fig.png
  results/partB_metrics.json results/partB_predictions.jsonl
  results/partC_attacks.json results/partC_exposure.png
)
missing=0
for f in "${required[@]}"; do
  [ -f "$f" ] || { echo "MISSING: $f"; missing=1; }
done
[ $missing -eq 0 ] || { echo "Aborting: required files missing."; exit 1; }

# Docs and report
cp README.md requirements.txt run_all.sh report.pdf report.tex "$STAGE/"
cp -r assets "$STAGE/" 2>/dev/null || true   # figures used by report.tex

# All source files: the four required scripts import the helper modules
cp src/*.py "$STAGE/src/"

# Results: required files + per-system metrics and per-attack reports (no other prediction dumps)
cp results/partA_sweep.csv results/partA_fig.png results/partB_metrics.json results/partB_predictions.jsonl \
   results/partC_attacks.json results/partC_exposure.png "$STAGE/results/"
cp results/partB/*_metrics.json "$STAGE/results/partB/" 2>/dev/null || true
cp results/partC/*.json "$STAGE/results/partC/" 2>/dev/null || true

# Zip and check the size limit
rm -f "$ENTRY.zip"
(cd "$(dirname "$STAGE")" && zip -rq "$OLDPWD/$ENTRY.zip" "$ENTRY")
rm -rf "$(dirname "$STAGE")"

size_mb=$(( $(wc -c < "$ENTRY.zip") / 1024 / 1024 ))
unzip -l "$ENTRY.zip"
echo "Created $ENTRY.zip (${size_mb} MB)"
[ "$size_mb" -le 500 ] || echo "WARNING: ZIP is larger than 500 MB"
