<div align="center">
<h1 align="center">Xiaomi-OCR-0</h1>
<p><b>A unified 0.8B model for document parsing and OCR-related understanding.</b></p>
<p><b>English</b> · <a href="README_zh.md">简体中文</a></p>
<p>
  <a href="https://huggingface.co/SeerRay-Lab/Xiaomi-OCR-0"><img src="https://img.shields.io/badge/Hugging%20Face-Model-FFD21E?logo=huggingface&amp;logoColor=FFD21E" alt="Hugging Face Model"></a>
  <a href="https://arxiv.org/abs/2609.36136"><img src="https://img.shields.io/badge/arXiv-2609.36136-b31b1b.svg" alt="arXiv:2609.36136"></a>
  <a href="https://huggingface.co/spaces/SeerRay-Lab/Xiaomi-OCR-0"><img src="https://img.shields.io/badge/Project-Page-FF6900?logo=huggingface&amp;logoColor=FFD21E" alt="Project Page"></a>
  <a href="https://github.com/SeerRay-Lab/Xiaomi-OCR-0"><img src="https://img.shields.io/badge/GitHub-SeerRay--Lab%2FXiaomi--OCR--0-181717?logo=github" alt="GitHub"></a>
</p>
</div>

Xiaomi-OCR-0 is a unified 0.8B vision-language model for document parsing and OCR-centric understanding. Starting from Qwen3.5-0.8B, it is trained on an approximately 170M-sample OCR-centric corpus with Q-Mask text anchoring, continued pretraining (CPT), and mixed-task reinforcement learning (Mix-RL).

## Get the repository

```bash
git clone https://github.com/SeerRay-Lab/Xiaomi-OCR-0.git
cd Xiaomi-OCR-0
```

## What it does

- **Document parsing:** reads document pages and produces structured Markdown, with OTSL model outputs converted to HTML tables in Markdown and formulas in LaTeX.
- **Key information extraction (KIE):** extracts requested fields as JSON.
- **OCR-centric visual question answering (VQA):** answers questions about text and content in document images.
- **Local agent tools:** an MCP service handles prompts, PDF/image preparation, optional region detection, parallel crop inference, and output assembly.
- **Browser demo:** try page and region parsing through a local web interface.

For printed, regular documents, region parsing can detect and process regions concurrently. For scene text, handwriting, calligraphy, historical books, and irregular layouts, use whole-page parsing because incorrect region boundaries can lose context.

## Try it with an agent

> ⚠️ **Notice:** With your authorization, this Skill may create a Python environment, install dependencies, download model weights, install and configure SGLang or vLLM, and register an MCP server. Model and layout downloads can be large. If you do not agree to this setup flow, do not execute the command below.

copy and send to your agent:

```text
Read and execute https://raw.githubusercontent.com/SeerRay-Lab/Xiaomi-OCR-0/main/SKILL.md
```

The model will run on your machine. For a complete setup, the agent also installs the
region-mode dependencies (PaddlePaddle, PaddleX and PP-DocLayoutV3) under your setup
authorization and verifies region parsing. You can request page-only setup. If your
platform does not support the layout runtime, the agent will explain the limitation.

## Run the browser demo

You can run the browser demo without installing the Agent Skill or MCP server. The demo sends requests to a local SGLang or vLLM inference server at `http://127.0.0.1:8000/v1`.
Python 3.10 or newer is required.

### 1. Install an inference runtime

If you already have a compatible local server, skip to step 2. Otherwise, choose **one** runtime and follow its installation guide for your operating system, GPU, and driver. These runtimes have hardware-specific requirements; the linked guides list supported platforms and installation options:

- [SGLang installation guide](https://docs.sglang.ai/get_started/install.html) · [supported models](https://docs.sglang.io/docs/supported-models/multimodal_language_models.md)
- [vLLM GPU installation guide](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/) · [Qwen3.5 model support](https://docs.vllm.ai/en/latest/api/vllm/model_executor/models/qwen3_5/)

Use a release that supports the checkpoint architecture `Qwen3_5ForConditionalGeneration`. For macOS and non-NVIDIA hardware, check the runtime's platform-specific support before installing; the default commands below are for a compatible GPU installation.

### 2. Start Xiaomi-OCR-0

Run **one** of the following commands in a terminal. On first launch, the runtime downloads the checkpoint from Hugging Face if it is not already cached. Keep this terminal open while using the demo.

SGLang:

```bash
python -m sglang.launch_server --model-path SeerRay-Lab/Xiaomi-OCR-0 \
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
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python demo/server.py --port 8787
```

Open <http://127.0.0.1:8787>. The same browser page supports whole-page/region document parsing, KIE, VQA, and PDF parsing. Install the shared requirements above, including Pillow and `pypdfium2` for images and PDFs. Region mode optionally needs PaddlePaddle/PaddleX and PP-DocLayoutV3 weights. See [demo/README.md](demo/README.md) for dependencies and [INSTALL.md](INSTALL.md) for MCP setup.

### 4. Region-mode dependencies (manual installation)

Region mode additionally needs PaddlePaddle, PaddleX and PP-DocLayoutV3. In the Python
environment that runs the Demo or MCP server, follow the official
[PaddlePaddle installation guide](https://paddlepaddle.github.io/PaddleX/latest/en/installation/paddlepaddle_install.html)
to select a compatible CPU/GPU build, then the
[PaddleX installation guide](https://paddlepaddle.github.io/PaddleX/latest/en/installation/installation.html).
From the repository root, install the layout dependencies:

```bash
python -m pip install -r requirements-paddlex.txt
```

This includes `paddlex[cv]` and `shapely`; PaddlePaddle is installed separately using the
official guide. PP-DocLayoutV3 weights download on first use if not cached. For CPU layout,
set `XIAOMI_OCR_LAYOUT_DEVICE=cpu` in the Demo or MCP environment. Whole-page OCR, KIE and
VQA work without these extra dependencies.

## Repository contents

| Path | Contents |
|:--|:--|
| [`skills/xiaomi-ocr/`](skills/xiaomi-ocr/) | Agent Skill and MCP service for OCR, PDF parsing, KIE, and VQA |
| [`demo/`](demo/) | Local browser demo and sample cases |
| [`example_pics/`](example_pics/) | Example inputs, reference Markdown, and demo animations mirrored from the Hugging Face model repository |
| [`pipeline/`](pipeline/) | Whole-page and region batch inference |
| [`postprocess/`](postprocess/) | Shared table conversion, formula/text assembly and repetition handling |
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

All three columns report Overall scores.

### OCR-centric visual question answering

| Model | Size | DocVQA | InfoVQA | ChartQA | OCRBench | TextVQA | Mean |
|:--|--:|--:|--:|--:|--:|--:|--:|
| **Xiaomi-OCR-0** | **0.8B** | **93.1** | **75.1** | **84.6** | **84.6** | **78.6** | **83.2** |
| Qwen3.5-0.8B | 0.8B | 88.5 | 60.3 | 69.5 | 77.9 | 68.3 | 72.9 |
| Qwen3.5-2B | 2B | 92.4 | 72.4 | 77.0 | 85.9 | 76.9 | 80.9 |
| Qwen3.5-4B | 4B | 94.4 | 80.4 | 82.4 | 86.6 | 80.8 | 84.9 |
| MiniCPM-V-4.5 | 8B | 84.9 | 69.6 | 87.4 | 89.0 | 82.2 | 82.6 |

Mean is the arithmetic average of the five benchmarks on a 0–100 scale.

## Citation

If you follow our work, please cite the following:

```bibtex
@misc{chen2026xiaomiocr0technicalreport,
      title={Xiaomi-OCR-0 Technical Report},
      author={Xin Chen and Anan Du and Feng Feng and Pei Fu and Jian Luan and Longwei Xu and Shaojie Zhang and Hang Li and Heng Qu and Cheng Tan},
      year={2026},
      eprint={2609.36136},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2609.36136},
}
```
