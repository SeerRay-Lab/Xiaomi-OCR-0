# Xiaomi-OCR-0 · Code

This repository contains the local agent Skill, MCP service, browser demo, and evaluation helper for Xiaomi-OCR-0. The model checkpoint and model card are hosted separately on [Hugging Face](https://huggingface.co/SeerRay-Lab/Xiaomi-OCR-0).

## Get the code

```bash
git clone https://github.com/SeerRay-Lab/Xiaomi-OCR-0.git
cd Xiaomi-OCR-0
```

## Use it with an agent

> ⚠️ **Notice:** Running this Skill may lead your agent to create a Python environment, download model or layout weights, install and configure SGLang or vLLM, and register an MCP server. If you do not agree to this setup, do not run the command. Review the setup details on GitHub first.

```text
Read and execute https://raw.githubusercontent.com/SeerRay-Lab/Xiaomi-OCR-0/main/SKILL.md
```

The Skill checks for compatible local runtimes and cached model files before setup. It asks before installing a runtime or downloading weights. See [INSTALL.md](INSTALL.md) for the setup steps and [SKILL.md](SKILL.md) for the agent-facing entry point.

## Run the browser demo

The demo uses a local SGLang or vLLM server. If you already have a compatible server and the Xiaomi-OCR-0 checkpoint, skip to step 2.

### 1. Start a local model server

Choose one runtime and follow its installation guide for your operating system, GPU, and driver. Use a version that supports `Qwen3_5ForConditionalGeneration` and image inputs.

- [SGLang installation](https://docs.sglang.ai/get_started/install.html) · [SGLang supported models](https://docs.sglang.io/docs/supported-models/multimodal_language_models.md)
- [vLLM installation](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/) · [Qwen3.5 support](https://docs.vllm.ai/en/latest/api/vllm/model_executor/models/qwen3_5/)

Start the model on loopback at `127.0.0.1:8000`:

```bash
sglang serve --model-path SeerRay-Lab/Xiaomi-OCR-0 \
  --host 127.0.0.1 --port 8000 --context-length 16384
```

or

```bash
vllm serve SeerRay-Lab/Xiaomi-OCR-0 \
  --host 127.0.0.1 --port 8000 --max-model-len 16384
```

The first launch downloads the checkpoint from Hugging Face if it is not already cached. Keep the server running while using the demo.

### 2. Start the demo

In another terminal, from the repository directory:

```bash
python3 -m pip install Pillow
python3 demo/server.py --port 8787
```

Open <http://127.0.0.1:8787>. For PDF input, install `pypdfium2` as well:

```bash
python3 -m pip install pypdfium2
```

The browser demo supports document parsing, KIE, VQA, image uploads, and PDF parsing. Region mode additionally uses PaddlePaddle/PaddleX and PP-DocLayoutV3; see [demo/README.md](demo/README.md) for installation details. For regular printed documents, region mode can process detected regions concurrently. For scene text, handwriting, calligraphy, historical documents, and irregular layouts, use whole-page mode because incorrect region boundaries can lose context.

## Repository contents

| Path | Purpose |
|:--|:--|
| [`SKILL.md`](SKILL.md) | Agent setup entry point |
| [`skills/xiaomi-ocr/`](skills/xiaomi-ocr/) | OCR, PDF, KIE, and VQA MCP tools and their shared preprocessing/postprocessing |
| [`demo/`](demo/) | Local browser demo and its replay samples |
| [`example_pics/`](example_pics/) | Example input images |
| [`workflows/evaluate_cases.py`](workflows/evaluate_cases.py) | Run a local evaluation case set against a local inference server |
| [`INSTALL.md`](INSTALL.md) | Local runtime, MCP, and optional layout setup |

The MCP service owns image/PDF preparation, prompts, optional layout detection, concurrent region inference, and output assembly. The browser demo reuses that pipeline. Inference stays on the user's machine; this repository does not provide a hosted OCR API.
