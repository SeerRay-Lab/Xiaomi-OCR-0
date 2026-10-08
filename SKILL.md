---
name: xiaomi-ocr-setup
description: Install and configure the Xiaomi-OCR-0 Skill and local MCP service for document OCR, KIE, and VQA.
---

# Xiaomi-OCR-0 Skill

The only user-facing installation prompt is:

```text
Read and execute https://raw.githubusercontent.com/SeerRay-Lab/Xiaomi-OCR-0/main/SKILL.md
```

## Install this Skill into the current agent

If this file was read from the Xiaomi-OCR-0 repository and the user asked to install it,
install only the `xiaomi-ocr` Skill globally for the current agent:

```bash
npx skills add SeerRay-Lab/Xiaomi-OCR-0 --skill xiaomi-ocr --full-depth --global --agent <current-agent> --yes
```

Replace `<current-agent>` with the current agent's Skills CLI identifier; do not install to every agent or modify a
project-local skill directory. Then inspect the current agent's MCP configuration and local
machine, and follow [backend selection in the runtime Skill](skills/xiaomi-ocr/SKILL.md#choose-the-inference-backend).
Choose the service from the existing setup, hardware, and accuracy requirements; the user
does not need to name a backend or model in the installation prompt. Apply the selected
backend's environment variables before running the installer.

The runtime imports shared `pipeline/` and `postprocess/` modules from the full repository.
Reuse a complete checkout or clone `https://github.com/SeerRay-Lab/Xiaomi-OCR-0.git`
into a user-owned code directory. Run `skills/xiaomi-ocr/install.sh` from that checkout;
do not run a detached Skill copy without setting `XIAOMI_OCR_REPO` to the checkout.
For Codex use `bash install.sh --agent=codex`; for other agents use `bash install.sh`
and merge its generated configuration into only the current agent.

Before running `install.sh`, explain that it creates a private `.venv`, downloads the
MCP dependencies listed in `requirements.txt`, and registers one MCP server entry in
Codex config when applicable. Ask the user to authorize that step if they have not already
authorized environment creation and dependency installation. Preserve all other MCP entries.
Do not install SGLang/vLLM, create an inference environment, or download model weights
unless the user's setup authorization covers those changes. State clearly if an engine or checkpoint
would need to be downloaded; reuse existing compatible installations whenever possible.
Keep any inference listener bound to loopback and do not claim inference is ready until an
image request succeeds.

For a complete agent-assisted installation, include region mode in the setup plan: explain
that PaddlePaddle, PaddleX and PP-DocLayoutV3 weights are needed, then install them in the
MCP environment under the user's existing setup authorization. Do not ask again when
that authorization already covers these dependencies and downloads. Follow step 5 of the
runtime Skill to choose a compatible CPU/GPU build, install `requirements-paddlex.txt`,
and verify a real `mode="region"` request. Respect requests for page-only setup. If the
platform is unsupported or installation fails, report region mode as unavailable and
explain which base functions passed verification; do not claim full setup is complete.

For the runtime workflow, task selection, and output handling, follow
[`skills/xiaomi-ocr/SKILL.md`](skills/xiaomi-ocr/SKILL.md). The model is served locally
through SGLang, vLLM, or the optional Ollama backend; the MCP service performs task-specific prompting, page/region
preprocessing, and result assembly.

Choose `ocr_image`/`ocr_pdf` for document parsing, `kie_image` for JSON key fields, and
`vqa_image` for questions about a document image. Prefer region mode for printed documents;
use whole-page mode for scene text, handwriting, calligraphy, historical documents, and
other nonstandard layouts where region cuts may damage context.
