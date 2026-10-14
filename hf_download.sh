python -c "from huggingface_hub import snapshot_download; snapshot_download('lakshgoel/col873-A1-adapters', allow_patterns=['PartA/*','PartB/*'], local_dir='hf_adapters')"

mkdir -p adapters
cp -r hf_adapters/PartA/* adapters/
cp -r hf_adapters/PartB/best_sanitizer adapters/best_sanitizer
rm -rf hf_adapters
ls adapters/best_sanitizer    # expect adapter_config.json, adapter_model.safetensors (README.md too)