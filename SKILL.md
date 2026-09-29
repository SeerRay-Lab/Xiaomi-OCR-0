---
name: xiaomi-ocr-setup
description: Install and configure the Xiaomi-OCR-0 Skill and local MCP service for document OCR, KIE, and VQA.
---

# Xiaomi-OCR-0 Skill

The only user-facing installation prompt is:

```text
Read and execute [https://raw.githubusercontent.com/SeerRay-Lab/Xiaomi-OCR-0/main/SKILL.md](https://raw.githubusercontent.com/SeerRay-Lab/Xiaomi-OCR-0/main/SKILL.md)
```

## Install this Skill into the current agent

If this file was read from the Xiaomi-OCR-0 repository and the user asked to install it,
install only the `xiaomi-ocr` Skill globally for the current agent:

```bash
npx skills add SeerRay-Lab/Xiaomi-OCR-0 --skill xiaomi-ocr --full-depth --global --yes
```

Use the Skills CLI's detected current agent; do not install to every agent or modify a
project-local skill directory. Then inspect the current agent's MCP configuration and local
machine for an existing compatible MCP environment, SGLang/vLLM process, and downloaded
checkpoint. Reuse working installations and ask which supported runtime the user prefers
if neither SGLang nor vLLM is already available.

Before running `install.sh`, explain that it creates a private `.venv`, downloads the
lightweight MCP dependencies (Pillow and pypdfium2), and registers one MCP server entry in
Codex config when applicable. Ask the user to authorize that step if they have not already
authorized environment creation and dependency installation. Preserve all other MCP entries.
Do not install SGLang/vLLM, create an inference environment, or download model weights
without the user's explicit choice and authorization. State clearly if an engine or checkpoint
would need to be downloaded; reuse existing compatible installations whenever possible.
Keep any inference listener bound to loopback and do not claim inference is ready until an
image request succeeds.

For the runtime workflow, task selection, and output handling, follow
[`skills/xiaomi-ocr/SKILL.md`](skills/xiaomi-ocr/SKILL.md). The model is served locally
through SGLang or vLLM; the MCP service performs task-specific prompting, page/region
preprocessing, and result assembly.

Choose `ocr_image`/`ocr_pdf` for document parsing, `kie_image` for JSON key fields, and
`vqa_image` for questions about a document image. Prefer region mode for printed documents;
use whole-page mode for scene text, handwriting, calligraphy, historical documents, and
other nonstandard layouts where region cuts may damage context.
