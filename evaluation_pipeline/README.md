# Document Parsing Evaluation Pipeline

This directory provides two inference approaches for document parsing with Xiaomi-OCR-0:

1. **End-to-end (e2e)**: Convert a whole-page image directly into Markdown.
2. **Two-stage**: Use PaddleX to detect layout regions, recognize each region, and assemble the results into page-level Markdown.

Both approaches use the OmniDocBench v1.6 evaluation code and the same configuration. Our evaluation configuration is available at [`configs/end2end.yaml`](configs/end2end.yaml). For Chinese instructions, see [README_zh.md](README_zh.md).

## Environment Versions

The evaluation environment follows the [Environment Setup and Running instructions in OmniDocBench](https://github.com/opendatalab/OmniDocBench/blob/main/README.md#environment-setup-and-running). The main components are:

| Component | Version |
| --- | --- |
| Python | 3.10.x |
| TeX Live | 2025 |
| pdflatex | 3.141592653-2.6-1.40.28 |
| ImageMagick | 7.1.1-47 |
| Ghostscript | 9.55.0 |

The Docker image used for evaluation is:

```text
ghcr.io/zeng-weijun/omnidocbench-eval:repro-ubuntu2204
```

This image is used to compute evaluation metrics. For GPU inference, prepare a separate vLLM environment with Qwen3.5 multimodal model support and install `requirements.txt` from the repository root. Two-stage layout detection also requires a compatible PaddlePaddle installation and `requirements-paddlex.txt`. See [INSTALL.md](../INSTALL.md) for installation details.

## Input Data Format

Run all commands below from the repository root. The input is a JSONL file with one page image path per line:

```json
{"images": ["/absolute/path/to/page.png"]}
```

Image paths must be accessible to the inference process. Outputs are named `<image_stem>.md`, so input image filenames must be unique and correspond to the page names in the evaluation annotations. Evaluation annotations must use a JSON format supported by OmniDocBench.

Set the input file and model paths for your environment:

```bash
MODEL=SeerRay-Lab/Xiaomi-OCR-0
INPUT_JSONL=/absolute/path/to/images.jsonl
OUTPUT_DIR="$PWD/outputs"
GPUS=0,1,2,3,4,5,6,7
mkdir -p "$OUTPUT_DIR"
```

## vLLM Inference Hyperparameters

Primary inference for both approaches uses the following generation parameters:

| Parameter | Value |
| --- | --- |
| `temperature` | `0.0` |
| `top_p` | `0.8` |
| `presence_penalty` | `0.0` |
| `repetition_penalty` | `1.0` |
| `max_tokens` | `16384` |
| `chat_template_kwargs.enable_thinking` | `false` |
| Request timeout | 600 seconds |

## End-to-End Inference

Use [`e2e_img2md.py`](e2e_img2md.py) to generate per-page Markdown:

```bash
python -m evaluation_pipeline.e2e_img2md \
  --input-jsonl "$INPUT_JSONL" \
  --output_dir "$OUTPUT_DIR/e2e" \
  --model_path "$MODEL" --served_model_name Xiaomi-OCR-0 \
  --task document --gpu_devices "$GPUS" --port 26000 \
  --gpu_mem_util 0.8 --workers 8 \
  --presence_penalty 0.0 --repetition_penalty 1.0 \
  --max_tokens 16384 --request_timeout 600 \
  --retry-repetitive --retry-presence-penalty 1.5 \
  --start_server --stop_server
```

Postprocessing converts OTSL into HTML tables in the final Markdown. Evaluation uses the final per-page `.md` files.

## Two-Stage Inference

### 1. Generate Layout with PaddleX

Use [`layout_detect_paddlex.py`](layout_detect_paddlex.py) to generate layout detection results with **the PaddleX implementation of PP-DocLayoutV3 at threshold 0.25**:

```bash
LAYOUT_JSONL="$OUTPUT_DIR/layout_paddlex-th0.25.jsonl"
python -m evaluation_pipeline.layout_detect_paddlex \
  --input_jsonl "$INPUT_JSONL" \
  --output_jsonl "$LAYOUT_JSONL" \
  --gpu_ids "$GPUS" --batch_size 8 --threshold 0.25
```

PaddleX downloads PP-DocLayoutV3 on the first run. You can also use `--model_dir` to specify local weights in **PaddleX format**.

### 2. Recognize Regions and Assemble Pages

Run [`twostage_img2md.py`](twostage_img2md.py) in the vLLM environment, passing the file generated in the previous stage through `--layout-jsonl`:

```bash
python -m evaluation_pipeline.twostage_img2md \
  --input-jsonl "$INPUT_JSONL" \
  --layout-jsonl "$LAYOUT_JSONL" \
  --output_dir "$OUTPUT_DIR/stage2" \
  --model_path "$MODEL" --served_model_name Xiaomi-OCR-0 \
  --vlm_gpus "$GPUS" --port 26000 --gpu_mem_util 0.8 --workers 8 \
  --temperature 0.0 --top_p 0.8 \
  --presence_penalty 0.0 --repetition_penalty 1.0 \
  --max_tokens 16384 --request_timeout 600 \
  --retry-repetitive --retry-presence-penalty 1.5 \
  --start_server --stop_server
```

Keep the following settings when reproducing two-stage results:

- **Pass the PaddleX-generated layout through `--layout-jsonl`.** If you use `--layout_model_dir` instead, the script loads the layout model through HF Transformers, with a default threshold of `0.3` for that path.
- Image path strings in the input JSONL and layout JSONL must match exactly. The script uses these paths to associate detection results.
- **Run PaddleX independently for each threshold.** Native postprocessing means higher-threshold results are not necessarily a subset obtained by filtering lower-threshold results by score. Offline filtering cannot replace regeneration. Use separate layout files when comparing thresholds.
- Table regions also use OTSL prompts and are converted to HTML during page assembly. Evaluation uses the final page-level Markdown.

## Evaluation

Use OmniDocBench v1.6's `pdf_validation.py` from the Docker image with this repository's [`configs/end2end.yaml`](configs/end2end.yaml). The configuration includes the following metrics:

| Category | Metrics |
| --- | --- |
| Text | Edit Distance |
| Formulas | Edit Distance, CDM |
| Tables | TEDS, Edit Distance |
| Reading order | Edit Distance |

The matching method is `quick_match`; matching, CDM, and TEDS each use `16` workers; `quick_match_truncated_timeout_sec` and `match_timeout_sec` are both `1200`; `timeout_fallback_max_chunk_span=200` and `timeout_fallback_order_penalty=0.05`. Retain these settings for reproduction. The default data paths in the configuration are examples; the command below replaces them with the actual mounted paths in a temporary copy.

Set `GT` to the annotation JSON and `PRED_DIR` to the Markdown directory to evaluate.

```bash
GT=/absolute/path/to/ground_truth.json
PRED_DIR="$OUTPUT_DIR/e2e"  # For two-stage outputs, use "$OUTPUT_DIR/stage2"
RESULT_DIR="$PWD/evaluation_results"
mkdir -p "$RESULT_DIR"
docker pull ghcr.io/zeng-weijun/omnidocbench-eval:repro-ubuntu2204

docker run --rm --entrypoint bash \
  -v "$PWD/evaluation_pipeline/configs/end2end.yaml:/repro/end2end.yaml:ro" \
  -v "$GT:/repro/gt.json:ro" \
  -v "$PRED_DIR:/repro/predictions:ro" \
  -v "$RESULT_DIR:/workspace/result" \
  -w /workspace \
  ghcr.io/zeng-weijun/omnidocbench-eval:repro-ubuntu2204 -c '
set -euo pipefail
python - <<"PY_CONFIG"
import yaml
with open("/repro/end2end.yaml") as f:
    config = yaml.safe_load(f)
dataset = config["end2end_eval"]["dataset"]
dataset["ground_truth"]["data_path"] = "/repro/gt.json"
dataset["prediction"]["data_path"] = "/repro/predictions"
with open("/tmp/xiaomi-ocr-eval.yaml", "w") as f:
    yaml.safe_dump(config, f, sort_keys=False)
PY_CONFIG
python pdf_validation.py --config /tmp/xiaomi-ocr-eval.yaml
'
```

## Real5-OmniDocBench: e2e and two-stage results

| Method | Size | Overall ↑ | Scanning ↑ | Warping ↑ | Screen-Photo ↑ | Illumination ↑ | Skew ↑ |
|:--|--:|--:|--:|--:|--:|--:|--:|
| e2e | 0.8B | 93.45 | 94.524 | 93.401 | 93.279 | 93.858 | 92.170 |
| 2stage | 0.8B | 95.24 | 96.41 | 95.46 | 94.47 | 95.39 | 94.46 |

Overall is the arithmetic mean of the five subset scores. Two-stage inference uses PaddleX PP-DocLayoutV3 with threshold 0.25.

## OmniDocBench v1.6: reproduction reference results

| Method | overall↑ | text_block_Edit_dist↓ | display_formula_CDM↑ | display_formula_Edit_dist↓ | table_TEDS↑ | table_TEDS_structure_only↑ | table_Edit_dist↓ | reading_order_Edit_dist↓ |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| e2e | 94.629 | 0.0395 | 96.325 | 0.0988 | 91.5119 | 94.0208 | 0.0795 | 0.1364 |
| 2stage | 96.4739 | 0.0332 | 98.3444 | 0.079 | 94.3973 | 96.6835 | 0.0544 | 0.1221 |

## Wild-OmniDocBench: reproduction reference results

| Method | overall↑ | text_block_Edit_dist↓ | display_formula_CDM↑ | display_formula_Edit_dist↓ | table_TEDS↑ | table_TEDS_structure_only↑ | table_Edit_dist↓ | reading_order_Edit_dist↓ |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| e2e | 83.9104 | 0.1379 | 87.7774 | 0.1691 | 77.7438 | 81.8345 | 0.1928 | 0.2164 |
| 2stage | 83.9477 | 0.1262 | 78.983 | 0.1628 | 85.48 | 89.8564 | 0.1209 | 0.1967 |
