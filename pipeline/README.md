# Inference pipelines

Install `requirements.txt` from the repository root. Use a running local SGLang/vLLM
server; defaults are port 8000 and model `SeerRay-Lab/Xiaomi-OCR-0`.
Input JSONL records have the form `{"images": ["/path/to/page.png"]}`.

```bash
python -m pipeline.e2e_img2md --input-jsonl images.jsonl --output_dir outputs/page
```

For region parsing, first export PaddleX layout (install a compatible PaddlePaddle
wheel and `requirements-paddlex.txt`; PP-DocLayoutV3 weights may download on first use):

```bash
python -m pipeline.layout_detect_paddlex --input_jsonl images.jsonl --output_jsonl layout.jsonl --gpu_ids 0
python -m pipeline.twostage_img2md --input-jsonl images.jsonl --layout-jsonl layout.jsonl --output_dir outputs/region
```

The alternative `--layout_model_dir` path uses Transformers-format PP-DocLayoutV3
weights and requires compatible Transformers and PyTorch packages. It does not accept
PaddleX-format weights. Use the exported-layout route above for PaddleX.

Both inference paths request OTSL by default and convert tables in postprocessing.
The `--retry-repetitive` option is opt-in; retain its report when comparing scores.
