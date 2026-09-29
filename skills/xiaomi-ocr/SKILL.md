---
name: xiaomi-ocr
description: Run Xiaomi-OCR-0 locally through MCP to parse document images and PDFs, extract key fields (KIE), or answer questions about document images (VQA).
---

# Xiaomi-OCR-0

Use this Skill when a task needs OCR, document parsing, KIE, or OCR-VQA. Inference runs
on the user's machine with the Hugging Face checkpoint `SeerRay-Lab/Xiaomi-OCR-0` and
SGLang or vLLM. The MCP service owns prompts, image/PDF preparation, optional layout
segmentation, crop processing, concurrency, and output formatting. Call the task tool;
do not recreate these steps with separate tools or direct model requests.

## Set up local inference

1. Inspect the host for a running compatible endpoint, an installed SGLang/vLLM runtime, and
   a cached `SeerRay-Lab/Xiaomi-OCR-0` checkpoint. Reuse compatible installations and
   weights. Do not install a runtime or download model weights without the user's choice and
   authorization. If setup would create an environment or download files, state what and how
   much will be downloaded before asking to proceed.
2. If the user chooses a runtime that is not installed, check the machine's GPU, driver,
   CUDA/PyTorch environment, and available memory. Follow that runtime's current official
   installation guide; Qwen3.5 model support and image handling require a build that supports
   `Qwen3_5ForConditionalGeneration`. Do not replace another installed runtime by default.
3. Once the user chooses to serve the model, bind it only to loopback. Example commands
   (adjust tensor parallelism and memory limits to the user's hardware):

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

   If the selected engine cannot load the checkpoint or accept image inputs, consult its
   current Qwen3.5 compatibility notes and ask before upgrading. Do not switch to a hosted
   inference endpoint.
4. Reuse or clone the complete GitHub repository. The MCP imports its shared `pipeline/`
   and `postprocess/` modules. Run the installer in `skills/xiaomi-ocr/` in that checkout
   (`bash install.sh --agent=codex` for Codex, `bash install.sh` for other agents).
   If using a detached Skill, set `XIAOMI_OCR_REPO` to the complete checkout.
   Before running `bash install.sh`, explain that it creates a private `.venv`, downloads
   the packages in `requirements.txt`, and registers one MCP entry in Codex config when applicable. Reuse an
   existing compatible MCP environment when available. Get the user's authorization before
   creating the environment or downloading dependencies if that authorization has not already
   been given. For other agents, merge the generated `mcp-config.json` entry into the current
   agent's MCP settings without replacing unrelated entries, then restart/rescan it. The MCP
   service connects only to the model's loopback listener.
5. For printed-document region mode, first explain that PP-DocLayoutV3 requires additional
   PaddlePaddle/PaddleX packages and may download layout weights. Install only after the user
   chooses region mode and authorizes those downloads. Keep the packages in this Skill's
   environment and ensure the layout model can load locally.
   This is optional; whole-page OCR, KIE, and VQA do not need the layout runtime. Region
   detection, cropping, and OCR remain inside this MCP server; no separate layout service
   or layout tool is exposed to the agent.

   Use the official [PaddlePaddle installation guide](https://paddlepaddle.github.io/PaddleX/latest/en/installation/paddlepaddle_install.html)
   to select a compatible CPU/GPU package, then follow the [PaddleX installation guide](https://paddlepaddle.github.io/PaddleX/latest/en/installation/installation.html).
   `pdx.create_model("PP-DocLayoutV3")` downloads the layout checkpoint on first use if it is
   not already cached. The [PP-DocLayoutV3 model card](https://huggingface.co/PaddlePaddle/PP-DocLayoutV3)
   describes the weights. Compatibility depends on the machine's OS, Python, and CUDA setup.

Official runtime references:
[SGLang Qwen3.5 guide](https://github.com/sgl-project/sglang/blob/main/docs_new/cookbook/autoregressive/Qwen/Qwen3.5.mdx) ·
[vLLM Qwen3.5 model implementation](https://docs.vllm.ai/en/latest/api/vllm/model_executor/models/qwen3_5/) ·
[vLLM supported models](https://docs.vllm.ai/en/latest/models/supported_models/).

## Choose the right task and document mode

- **Printed, regular documents**: use `ocr_image` or `ocr_pdf` with `mode="region"`. The
  MCP runs PP-DocLayoutV3 locally, crops regions, sends independent requests concurrently,
  then restores reading order and combines output. Model prompts request OTSL; MCP converts tables to HTML inside the returned Markdown and formats LaTeX formulas.
- **Scene text, handwritten notes, calligraphy, historical books, or irregular layouts**:
  use `mode="page"`. Region detection may split a line, column, or connected handwriting
  incorrectly; that can lose context and compound recognition errors. Whole-page mode
  preserves those relationships.
- **Key information extraction**: call `kie_image`. Pass `fields` when the output keys are
  known; the MCP builds the schema prompt and returns parsed JSON. Leave fields empty for
  general key-value extraction.
- **Question answering about a document image**: call `vqa_image` with the image and
  question. The MCP applies the concise answer format.
- **PDF**: call `ocr_pdf`, choosing page/region by the same document guidance. Prefer local
  file paths over base64 to avoid payload truncation.

Available MCP tools are `ocr_image`, `ocr_pdf`, `kie_image`, and `vqa_image`. Page/region
preprocessing and postprocessing are internal to these calls.

## Output handling

Keep the model's reading order and Markdown. Do not fill in unreadable content; retain the
model's uncertainty. Table conversion and document assembly happen inside MCP. Do not ask the agent to invoke a separate postprocessor. For KIE, preserve
JSON keys and values; for VQA, return the concise answer without adding unsupported detail.

If MCP reports that the local model is unavailable, ask which runtime/model setup the user
wants before installing or downloading anything. If region mode reports missing
PaddleX/PaddlePaddle, explain the extra packages and possible layout-weight download, then
get authorization before installing; whole-page mode remains available in the meantime.
