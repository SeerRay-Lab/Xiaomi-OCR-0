# Xiaomi-OCR-0

<p>
  <a href="https://huggingface.co/SeerRay-Lab/Xiaomi-OCR-0">Hugging Face Model</a> ·
  <a href="https://huggingface.co/spaces/SeerRay-Lab/Xiaomi-OCR-0">Project Page</a> ·
  <a href="README_zh.md">简体中文</a>
</p>

Xiaomi-OCR-0 is a unified 0.8B vision-language model for document parsing and OCR-centric understanding. Starting from Qwen3.5-0.8B, it is trained on an approximately 170M-sample OCR-centric corpus with Q-Mask text anchoring, continued pretraining (CPT), and mixed-task reinforcement learning (Mix-RL).

## Get the repository

```bash
git clone https://github.com/SeerRay-Lab/Xiaomi-OCR-0.git
cd Xiaomi-OCR-0
```

## What it does

- **Document parsing:** reads document pages and produces structured Markdown, with tables in OTSL and formulas in LaTeX.
- **Key information extraction (KIE):** extracts requested fields as JSON.
- **OCR-centric visual question answering (VQA):** answers questions about text and content in document images.
- **Local agent tools:** an MCP service handles prompts, PDF/image preparation, optional region detection, parallel crop inference, and output assembly.
- **Browser demo:** try page and region parsing through a local web interface.

For printed, regular documents, region parsing can detect and process regions concurrently. For scene text, handwriting, calligraphy, historical books, and irregular layouts, use whole-page parsing because incorrect region boundaries can lose context.

## Try it with an agent

> ⚠️ **Notice:** With your authorization, this Skill may create a Python environment, install dependencies, download model weights, install and configure SGLang or vLLM, and register an MCP server. Model and layout downloads can be large. If you do not agree to this setup flow, do not execute the command below.

```text
Read and execute https://raw.githubusercontent.com/SeerRay-Lab/Xiaomi-OCR-0/main/SKILL.md
```

The model runs on your machine; no hosted Xiaomi-OCR API is required.

## Run the browser demo

You can run the browser demo without installing the Agent Skill or MCP server. The demo sends requests to a local SGLang or vLLM inference server at `http://127.0.0.1:8000/v1`.

### 1. Install an inference runtime

If you already have a compatible local server, skip to step 2. Otherwise, choose **one** runtime and follow its installation guide for your operating system, GPU, and driver. These runtimes have hardware-specific requirements; the linked guides list supported platforms and installation options:

- [SGLang installation guide](https://docs.sglang.ai/get_started/install.html) · [supported models](https://docs.sglang.io/docs/supported-models/multimodal_language_models.md)
- [vLLM GPU installation guide](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/) · [Qwen3.5 model support](https://docs.vllm.ai/en/latest/api/vllm/model_executor/models/qwen3_5/)

Use a release that supports the checkpoint architecture `Qwen3_5ForConditionalGeneration`. For macOS and non-NVIDIA hardware, check the runtime's platform-specific support before installing; the default commands below are for a compatible GPU installation.

### 2. Start Xiaomi-OCR-0

Run **one** of the following commands in a terminal. On first launch, the runtime downloads the checkpoint from Hugging Face if it is not already cached. Keep this terminal open while using the demo.

SGLang:

```bash
sglang serve --model-path SeerRay-Lab/Xiaomi-OCR-0 \
  --host 127.0.0.1 --port 8000 --context-length 16384
```

vLLM:

```bash
vllm serve SeerRay-Lab/Xiaomi-OCR-0 \
  --host 127.0.0.1 --port 8000 --max-model-len 16384
```

Wait for the server to finish loading the model before continuing.

### 3. Start the browser demo

Open a second terminal in the repository directory and run:

```bash
python3 -m pip install Pillow
python3 demo/server.py --port 8787
```

Open <http://127.0.0.1:8787>. The same browser page supports whole-page/region document parsing, KIE, VQA, and PDF parsing. The demo needs Pillow; PDF input additionally needs `pypdfium2`. Region mode optionally needs PaddlePaddle/PaddleX and PP-DocLayoutV3 weights. See [demo/README.md](demo/README.md) for dependencies and [INSTALL.md](INSTALL.md) for MCP setup.

## Repository contents

| Path | Contents |
|:--|:--|
| [`skills/xiaomi-ocr/`](skills/xiaomi-ocr/) | Agent Skill and MCP service for OCR, PDF parsing, KIE, and VQA |
| [`demo/`](demo/) | Local browser demo and sample cases |
| [`workflows/`](workflows/) | Evaluation utilities |
| [Hugging Face model](https://huggingface.co/SeerRay-Lab/Xiaomi-OCR-0) | Model card, examples, and benchmark assets |
| [Hugging Face Space](https://huggingface.co/spaces/SeerRay-Lab/Xiaomi-OCR-0/tree/main) | Space assets and project-page source |
| [`examples/`](examples/) | MCP configuration example |

## Key Performance

The comparisons below summarize selected results. Arrows indicate the preferred direction; bold marks Xiaomi-OCR-0 and does not necessarily indicate the best result in a column. See the [Hugging Face model card](https://huggingface.co/SeerRay-Lab/Xiaomi-OCR-0) for benchmark notes and additional results.

![Parameter efficiency on OmniDocBench v1.6](https://huggingface.co/SeerRay-Lab/Xiaomi-OCR-0/resolve/main/assets/omni-parameters.png)

![Parameter efficiency on OCR-oriented VQA](https://huggingface.co/SeerRay-Lab/Xiaomi-OCR-0/resolve/main/assets/vqa-parameters.png)

![Parameter efficiency on Real5-OmniDocBench](https://huggingface.co/SeerRay-Lab/Xiaomi-OCR-0/resolve/main/assets/real5-parameters.png)

![Parameter efficiency on Wild-OmniDocBench](https://huggingface.co/SeerRay-Lab/Xiaomi-OCR-0/resolve/main/assets/wild-parameters.png)

### Document parsing: selected comparisons

| Model | Size | OmniDocBench v1.6 ↑ | Real5 ↑ | Wild ↑ |
|:--|--:|--:|--:|--:|
| **Xiaomi-OCR-0** | **0.8B** | **96.83** | **95.24** | **87.94** |
| TeleOCR | 1.2B | 96.87 | — | 88.53 |
| OvisOCR2 | 0.8B | 96.58 | 92.29 | 87.91 |
| PaddleOCR-VL-1.6 | 0.9B | 96.33 | 93.19 | 87.36 |
| MinerU2.5-Pro | 1.2B | 95.75 | 88.94 | 87.33 |
| GLM-OCR | 0.9B | 95.22 | 90.32 | 85.08 |

All three columns report Overall scores; “—” means the source table does not report a value.

### OCR-centric visual question answering

| Model | Size | DocVQA | InfoVQA | ChartQA | OCRBench | TextVQA | Mean |
|:--|--:|--:|--:|--:|--:|--:|--:|
| **Xiaomi-OCR-0** | **0.8B** | **93.1** | **75.1** | **84.6** | **84.6** | **78.6** | **83.2** |
| Qwen3.5-0.8B | 0.8B | 88.5 | 60.3 | 69.5 | 77.9 | 68.3 | 72.9 |
| Qwen3.5-2B | 2B | 92.4 | 72.4 | 77.0 | 85.9 | 76.9 | 80.9 |
| Qwen3.5-4B | 4B | 94.4 | 80.4 | 82.4 | 86.6 | 80.8 | 84.9 |
| MiniCPM-V-4.5 | 8B | 84.9 | 69.6 | 87.4 | 89.0 | 82.2 | 82.6 |

Mean is the arithmetic average of the five benchmarks on a 0–100 scale.
