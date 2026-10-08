# Ollama backend (experimental)

The Demo and MCP support `XIAOMI_OCR_BACKEND=ollama`. The shared transport uses
the checkpoint's single-image, non-thinking prompt and raw generation, preserving
Markdown line-break spaces. PDF, layout detection, cropping and postprocessing
remain in the existing pipeline. Standalone batch scripts retain their existing
vLLM server-management flags.

**Accuracy parity is not established.** On Apple Silicon / Ollama 0.35.0, a newly
converted BF16 GGUF matched HF BF16/MPS on all 7 input token counts and 5 of 7
complete responses. A dense-equation response omitted a transpose present in the
image; another formula response used different LaTeX parentheses. This is not
a benchmark accuracy percentage or proof of parity with vLLM/SGLang.
A full F32 GGUF control run retained the same two formula differences.

A Linux / NVIDIA RTX 4090 smoke run with Ollama 0.35.0 imported the same text and
vision GGUF blobs and passed the MCP stdio handshake plus live OCR, KIE, VQA, and
single-page PDF calls. Other inference processes occupied most of the GPU, so
Ollama offloaded most layers to CPU; this confirms functionality, not performance.

Native Safetensors/MLX import also differed on a calligraphy character. GGUF
corrected that specific example. These are different execution paths; BF16 alone
does not establish identical computation.

## Demo

Install a Qwen3.5-compatible Ollama (tested with 0.35.0) and keep its local service
running through the desktop app or `ollama serve`. From the repository root,
prepare the Demo dependencies with Python 3.10+:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

With model-download authorization:

```bash
ollama pull longwayxu/xiaomi-ocr-0:bf16
export XIAOMI_OCR_BACKEND=ollama
export XIAOMI_OCR_MODEL=longwayxu/xiaomi-ocr-0:bf16
export XIAOMI_OCR_LOCAL_URL=http://127.0.0.1:11434/v1
python demo/server.py --port 8787
```

This community GGUF package is approximately 1.8 GB (1.6 GiB) and includes a vision
projector. Keep Ollama on loopback. The Demo needs the repository's image/PDF
dependencies; the optional region runtime still needs PaddleX. The Ollama
transport itself does not require Transformers or PyTorch.

Set `XIAOMI_OCR_MODEL` to a local candidate when testing a rebuilt model. The
conversion trial used `xiaomi-ocr-0:gguf-bf16-aligned`; it is a local tag, not a
published model that can be pulled.

## Agent

```bash
XIAOMI_OCR_BACKEND=ollama \
XIAOMI_OCR_LOCAL_URL=http://127.0.0.1:11434/v1 \
XIAOMI_OCR_MODEL=longwayxu/xiaomi-ocr-0:bf16 \
bash skills/xiaomi-ocr/install.sh --agent=codex
```

The installer creates/reuses the MCP environment and persists the URL and model
in generated MCP JSON and Codex configuration. Set the URL and model explicitly;
the current installer does not persist or interpret `XIAOMI_OCR_BACKEND`. For other agents omit
`--agent=codex` and merge that JSON into the current agent. Existing authorization
requirements for installations/downloads/configuration still apply. Restart/rescan
the agent and verify a real request. The four tool schemas are unchanged; region
mode requires a separate live test with its layout dependencies.

## Conversion and acceptance

Use llama.cpp's `Qwen3_5ForConditionalGeneration` converter with `--outtype bf16`
for text and with `--mmproj --outtype bf16` for vision. Keep revisions pinned.
The local trial used HF `e4d1c4a6804bd9ef342b93d705a73af003e2ef4e` and llama.cpp
`d89651a7b205c03c4a0b13cd0646d400dc929f79`. The converter retains normalization
and selected vision tensors as F32; do not force those down to BF16.

The transport follows this checkpoint's 16px patches, 2x merge, and pixel bounds
65536–16777216, after the existing repository 2048px long-edge bound. Compare
the same prepared PNG, task prompt, tokenizer, thinking mode, output budget and
greedy decoding. Compare raw responses before postprocessing and inspect formula
symbol differences against the image. Matching token counts alone does not prove
matching embeddings, logits or recognition accuracy.

Before promoting this option, test representative ground-truth data, formula and
table metrics, and the actual published package on supported platforms. Do not
infer general accuracy parity from a successful import or a small smoke test.
