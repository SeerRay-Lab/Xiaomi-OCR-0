# Local installation

Xiaomi-OCR-0 runs on the user's machine. Download the `SeerRay-Lab/Xiaomi-OCR-0`
Hugging Face checkpoint and serve it with a compatible local SGLang or vLLM installation.
The MCP service connects only to the loopback listener on that machine. There is no
hosted OCR service in this workflow.

Python 3.10 or newer is required. The installer uses a compatible `python3` when available
and otherwise checks `python3.13` through `python3.10`; set `PYTHON_BIN` to choose one explicitly.

## 1. Check for existing local components

Before installing anything, check whether a compatible SGLang/vLLM server, model checkpoint,
MCP environment, or agent configuration already exists. Reuse compatible installations.
The setup does not choose a runtime on the user's behalf. Do not install SGLang/vLLM or
download model/layout weights without the user's choice and authorization.

## 2. Start the model locally

If the user chooses to install a runtime, use a build with Qwen3.5 multimodal support for
their GPU and driver. Explain which environment and model files will be downloaded first.
Runtime compatibility changes over time; use the current upstream install and model guides:

- [SGLang Qwen3.5 deployment guide](https://github.com/sgl-project/sglang/blob/main/docs_new/cookbook/autoregressive/Qwen/Qwen3.5.mdx)
- [vLLM Qwen3.5 implementation](https://docs.vllm.ai/en/latest/api/vllm/model_executor/models/qwen3_5/)
- [vLLM supported models](https://docs.vllm.ai/en/latest/models/supported_models/)

Basic local examples (change memory and parallelism options to suit the host):

```bash
python -m sglang.launch_server \
  --model-path SeerRay-Lab/Xiaomi-OCR-0 \
  --host 127.0.0.1 --port 8000 --context-length 16384
```

```bash
vllm serve SeerRay-Lab/Xiaomi-OCR-0 \
  --host 127.0.0.1 --port 8000 \
  --trust-remote-code --max-model-len 16384
```

The checkpoint is a multimodal `Qwen3_5ForConditionalGeneration` model. Confirm that
startup completes and the runtime accepts image inputs before configuring MCP. Keep the
listener bound to `127.0.0.1`.

## 3. Install the MCP package

```bash
cd skills/xiaomi-ocr
bash install.sh --agent=codex
```

After the user authorizes this step, the installer creates a private Python environment,
installs the packages in `skills/xiaomi-ocr/requirements.txt`, and writes `mcp-config.json`. With `--agent=codex` it also registers the
MCP server while preserving other entries. For other agents, omit `--agent=codex`; merge this entry into their
own MCP configuration, then restarted. If the local model runs on another port, set
`XIAOMI_OCR_LOCAL_URL` before running the installer, for example
`http://127.0.0.1:8001/v1`.

## 4. Optional region mode

Whole-page mode, KIE, and VQA need only the dependencies installed by `install.sh`. For
printed-document region parsing, with the user's authorization install PaddlePaddle and
PaddleX/PP-DocLayoutV3 using versions and device wheels appropriate for the host. This may
also download layout weights. The MCP process loads the layout model,
detects/crops regions, runs OCR requests concurrently, and restores reading order. No
separate layout service is started or configured.

Use region mode for regular printed documents, especially multi-page PDFs where region OCR
can run concurrently. Use page mode for scene text, handwriting, calligraphy, and historical
or irregular layouts: an incorrect region split can lose context and compound errors.

## MCP tasks

- `ocr_image` and `ocr_pdf`: document parsing in `page` or `region` mode.
- `kie_image`: key-value extraction as JSON; optionally supply desired field names.
- `vqa_image`: answer a question about the document image.

Image/PDF preprocessing, prompts, layout handling, concurrency, and output formatting stay
inside the MCP tools. Model prompts request OTSL; the tools convert tables into HTML within Markdown.
Set `XIAOMI_OCR_LAYOUT_MODEL_DIR` to reuse PaddleX-format layout weights and
`XIAOMI_OCR_LAYOUT_DEVICE` to select the layout device. These variables belong in the MCP environment
(or the Demo process environment). See [`skills/xiaomi-ocr/SKILL.md`](skills/xiaomi-ocr/SKILL.md) for the
agent workflow and [`skills/xiaomi-ocr/scripts/otsl_convert.py`](skills/xiaomi-ocr/scripts/otsl_convert.py)
for optional offline conversion; it is not a required agent step.

## Batch evaluation

The restored [pipeline commands](pipeline/README.md) and [postprocessing modules](postprocess/README.md) share the document assembly used by MCP and the Demo. Install `requirements.txt` for batch inference.
