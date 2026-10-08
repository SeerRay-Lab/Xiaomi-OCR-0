# Local installation

Xiaomi-OCR-0 runs on the user's machine. Choose a local inference runtime:
SGLang or vLLM with the `SeerRay-Lab/Xiaomi-OCR-0` Hugging Face checkpoint,
or Ollama with the experimental `longwayxu/xiaomi-ocr-0:bf16` community package.
The MCP service connects only to the loopback listener on that machine. There is no
hosted OCR service in this workflow.

Python 3.10 or newer is required. The installer uses a compatible `python3` when available
and otherwise checks `python3.13` through `python3.10`; set `PYTHON_BIN` to choose one explicitly.

## 1. Check for existing local components

Before installing anything, check whether a compatible SGLang/vLLM/Ollama server, model checkpoint,
MCP environment, or agent configuration already exists. Reuse compatible installations.
The setup does not choose a runtime on the user's behalf. Do not install SGLang/vLLM/Ollama or
download model/layout weights without the user's choice and authorization.

## 2. Start the model locally

Choose **one** of the following runtime paths.

### SGLang / vLLM

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

### Ollama (experimental)

Install [Ollama](https://ollama.com/download) and keep its local service running
through the desktop app or `ollama serve`. With model-download authorization, pull
the community BF16 package, which includes the language model and vision projector
(about 1.8 GB):

```bash
ollama pull longwayxu/xiaomi-ocr-0:bf16
```

Use `http://127.0.0.1:11434/v1` as the MCP inference URL and
`longwayxu/xiaomi-ocr-0:bf16` as the model name. Keep the service on loopback.
See [OLLAMA.md](OLLAMA.md) for conversion details and experimental validation notes.

## 3. Install the MCP package

Run the command for your selected runtime from the repository root.

For **SGLang / vLLM**:

```bash
XIAOMI_OCR_LOCAL_URL=http://127.0.0.1:8000/v1 \
XIAOMI_OCR_MODEL=SeerRay-Lab/Xiaomi-OCR-0 \
bash skills/xiaomi-ocr/install.sh --agent=codex
```

For **Ollama**:

```bash
XIAOMI_OCR_LOCAL_URL=http://127.0.0.1:11434/v1 \
XIAOMI_OCR_MODEL=longwayxu/xiaomi-ocr-0:bf16 \
bash skills/xiaomi-ocr/install.sh --agent=codex
```

The installer saves the URL and model in the MCP configuration. Pass both explicitly:
the current installer does not use `XIAOMI_OCR_BACKEND` to select the URL or model.

After the user authorizes this step, the installer creates a private Python environment,
installs the packages in `skills/xiaomi-ocr/requirements.txt`, and writes `mcp-config.json`. With `--agent=codex` it also registers the
MCP server while preserving other entries. For other agents, omit `--agent=codex`; merge this entry into their
own MCP configuration, then restart the agent. If the local model runs on another port, set
`XIAOMI_OCR_LOCAL_URL` before running the installer, for example
`http://127.0.0.1:8001/v1`.

## 4. Region mode (included in complete agent-assisted setup)

A complete agent-assisted installation includes region dependencies and a real region-mode
check when the platform supports them and the user's setup authorization covers the downloads.
Users can choose page-only setup. `install.sh` itself installs only the base MCP dependencies;
the agent selects and installs the platform-specific layout runtime as the next step.

For manual installation, use the official
[PaddlePaddle installation guide](https://paddlepaddle.github.io/PaddleX/latest/en/installation/paddlepaddle_install.html)
and [PaddleX installation guide](https://paddlepaddle.github.io/PaddleX/latest/en/installation/installation.html).
After installing a compatible PaddlePaddle wheel, run this from the repository root with
the MCP environment's Python:

```bash
skills/xiaomi-ocr/.venv/bin/python -m pip install -r requirements-paddlex.txt
```

For a standalone Demo, run the same pip command with the Demo environment's Python instead.
Use `XIAOMI_OCR_LAYOUT_DEVICE=cpu` for CPU layout; verify a real region request after installation.

Whole-page mode, KIE, and VQA need only the dependencies installed by `install.sh`. For
printed-document region parsing, with the user's authorization install PaddlePaddle and
PaddleX/PP-DocLayoutV3 using versions and device wheels appropriate for the host. Install the
PaddleX extras that PP-DocLayoutV3 needs (`requirements-paddlex.txt` includes `paddlex[cv]` and
`shapely`); a plain `paddlex` install is not sufficient and fails at layout time. This may
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

### Output budget and existing Codex entries

The MCP default output budget is 4096 tokens, leaving room for image/prompt tokens
inside the documented 16384-token context. Override `XIAOMI_OCR_MAX_TOKENS` only
when the model context can accommodate both the input and requested output.
Re-registering a previous HTTP `xiaomi-ocr` entry replaces its transport with stdio;
other MCP servers, custom environment variables, and tool preferences are preserved.
