# Xiaomi-OCR-0 Browser Demo

A compact local web UI for document parsing, key information extraction (KIE), OCR-VQA, and PDF parsing. It calls a Xiaomi-OCR-0 inference server running on the same machine.

## Start

Start Xiaomi-OCR-0 with SGLang or vLLM on `127.0.0.1:8000`, then from the repository root run:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python demo/server.py --port 8787
```

Open <http://127.0.0.1:8787>. Set `XIAOMI_OCR_LOCAL_URL` if the inference server uses another local port. The demo only accepts loopback inference URLs.

`requirements.txt` includes the PDF renderer.

## Interface language

Use the header's **English / 中文** button to switch languages. The initial language follows
the browser preference (Chinese for `zh`, English otherwise); your choice is remembered.
Open `/?lang=en` or `/?lang=zh` to share a language-specific view. Switching languages keeps
the selected document, entered prompts/questions, current task and recognition results.
Only interface text changes; OCR output retains the document's language.

## Use

- **Document parsing:** choose whole-page or region mode. Whole-page mode has an editable prompt and suits handwriting, calligraphy, scene text, historical documents, and irregular layouts. Region mode suits regular printed documents and runs layout detection, region OCR, and output assembly in the local server.
- **KIE:** choose fields separated by commas or newlines; leave the field box empty to extract general key information. The result is parsed and displayed as JSON.
- **VQA:** enter a question about text or content in the selected image. The demo uses the same concise-answer prompt as the MCP service.
- **PDF:** upload a PDF to preview it and parse its pages in whole-page or region mode. Page rendering and OCR progress are shown in the existing status bar. PDF supports document parsing; use image inputs for KIE and VQA.

The demo imports the Skill's `mcp_ocr_server.py` pipeline for image normalization, PP-DocLayoutV3 detection, confidence filtering, cropping, region prompts, and KIE prompt construction. Region and PDF-region output use the same reading-order assembly and formula formatting as the MCP implementation. Whole-page, KIE, and VQA inference keep the demo's token streaming. Final document results come from the server-side shared postprocessor.

## Region-mode dependencies

Whole-page mode does **not** need PaddleX or PP-DocLayoutV3. Region mode uses PP-DocLayoutV3 for page layout detection. Install a PaddlePaddle build compatible with your operating system, Python, and (if used) CUDA version, then install PaddleX. PP-DocLayoutV3 needs PaddleX's `cv` extras and Shapely, so use the repository's `requirements-paddlex.txt` (`paddlex[cv]` + `shapely`) rather than plain `paddlex`. On first use, `paddlex.create_model("PP-DocLayoutV3")` may download the layout model weights; later runs reuse PaddleX's local cache.

- [PaddlePaddle installation guide](https://paddlepaddle.github.io/PaddleX/latest/en/installation/paddlepaddle_install.html)
- [PaddleX installation guide](https://paddlepaddle.github.io/PaddleX/latest/en/installation/installation.html)
- [PP-DocLayoutV3 model card and weights](https://huggingface.co/PaddlePaddle/PP-DocLayoutV3)

Choose the CPU/GPU package from the official compatibility guidance for the machine that will run the demo. This optional layout runtime is separate from the Xiaomi-OCR-0 model runtime.

## Model configuration and troubleshooting

If your inference runtime uses a custom served model name, set `XIAOMI_OCR_MODEL`
to the exact ID returned by its `/v1/models` endpoint. `XIAOMI_OCR_LOCAL_URL`
defaults to `http://127.0.0.1:8000/v1`. Both variables must be set in the terminal
that launches the demo. The connection indicator checks the configured model via
`GET /api/health`; it does not imply that optional layout dependencies are installed.

The default output budget is 4096 tokens. Input image tokens and the prompt also
consume context, so setting the output budget equal to a 16384-token context will
fail. Increase the budget only if the model has enough remaining context.

The UI includes explicit run/upload actions, rendered/raw/JSON views, copy and
result downloads, and a mobile layout. Samples are the repository's real receipt,
dense-equations and Qianziwen images. The handwritten-formula example is excluded.
Live failures remain errors; the demo does not silently replace them with cached results.
