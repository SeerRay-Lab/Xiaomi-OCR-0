#!/usr/bin/env python3
"""Local Xiaomi-OCR-0 MCP service.

Inference is served by a local SGLang or vLLM process. This MCP process owns image
normalization, optional PP-DocLayoutV3 region detection/cropping, concurrent OCR,
task prompts, and result formatting. No hosted OCR or layout service is used.
"""
from __future__ import annotations

import base64
import io
import json
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import closing
from pathlib import Path

# A standalone Skill installation must point at a complete repository checkout.
REPO_ROOT = Path(os.environ.get("XIAOMI_OCR_REPO", Path(__file__).resolve().parents[2])).expanduser()
sys.path.insert(0, str(REPO_ROOT))
from postprocess.document import finalize_document
from postprocess.region import merge_page_to_markdown
from postprocess.otsl import convert_otsl_to_html

OCR_BASE = os.environ.get("XIAOMI_OCR_LOCAL_URL", "http://127.0.0.1:8000/v1").strip().rstrip("/")
OCR_MODEL = os.environ.get("XIAOMI_OCR_MODEL", "SeerRay-Lab/Xiaomi-OCR-0").strip()
MAX_TOKENS = int(os.environ.get("XIAOMI_OCR_MAX_TOKENS", "4096"))
CONCURRENCY = max(1, min(int(os.environ.get("XIAOMI_OCR_CONCURRENCY", "8")), 32))
LAYOUT_THRESHOLD = float(os.environ.get("XIAOMI_OCR_LAYOUT_THRESHOLD", "0.3"))

PROMPTS = {
    "document": (
        "Extract all information from the main body of the document image and represent it "
        "in markdown format, ignoring headers and footers. Tables should be expressed in OTSL "
        "format, formulas in the document should be represented using LATEX format, and the "
        "parsing should be organized according to the reading order."
    ),
    "text": "Extract the text in the image.",
    "table": "Parse the table in the image into OTSL.",
    "formula": "Identify the formula in the image and represent it using LATEX format.",
}
LABEL_MAP = {
    "text": "text", "paragraph": "text", "plain text": "text", "paragraph_title": "title",
    "title": "title", "header": "title", "section_header": "title", "table": "table",
    "table_caption": "text", "formula": "formula", "equation": "formula",
    "isolate_formula": "formula", "figure": "figure", "image": "figure", "chart": "figure",
    "figure_caption": "text", "figure_title": "text", "chart_title": "text",
    "list": "text", "list_item": "text", "caption": "text", "abstract": "text",
    "number": "text", "reference": "text", "footnote": "text",
}
REGION_MAX_TOKENS = {"title": 256, "text": 1024, "table": 2048, "formula": 512, "figure": 256}


def _local_url(base: str | None = None) -> str:
    """Refuse non-loopback inference endpoints: model requests must stay on this machine."""
    base = (base or OCR_BASE).rstrip("/")
    parsed = urllib.parse.urlparse(base)
    if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise RuntimeError("XIAOMI_OCR_LOCAL_URL must point to a local http://127.0.0.1 inference server")
    return base


def _http_json(url: str, payload: dict, timeout: int = 300) -> dict:
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        if e.code == 400 and "maximum context length" in body.lower():
            raise RuntimeError(
                "The image/prompt plus requested output exceeds the model context. "
                "Reduce max_tokens and retry; 16384 is the output ceiling, not a guaranteed "
                "completion size for every input."
            ) from e
        raise RuntimeError(f"Local inference returned HTTP {e.code}: {body[:500]}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"Local model is unavailable at {_local_url()}: {e.reason}. Start SGLang or vLLM first.") from e


def _chat(image: bytes, prompt: str, max_tokens: int, *, model_url: str | None = None, model: str | None = None) -> dict:
    url = _local_url(model_url) + "/chat/completions"
    image = _image_png(image)
    encoded = base64.b64encode(image).decode()
    payload = {"model": model or OCR_MODEL, "messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded}"}},
        {"type": "text", "text": prompt},
    ]}], "max_tokens": max_tokens, "temperature": 0, "top_p": 0.8, "presence_penalty": 0.0, "repetition_penalty": 1.0,
        "chat_template_kwargs": {"enable_thinking": False}}
    data = _http_json(url, payload)
    message = data["choices"][0]["message"]["content"]
    if isinstance(message, list):
        message = "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in message)
    text = str(message or "").strip()
    if not text:
        raise RuntimeError("Local model returned an empty response")
    if data["choices"][0].get("finish_reason") == "length":
        raise RuntimeError("Model output was truncated; increase the output/context budget or use smaller inputs")
    return {"text": text, "usage": data.get("usage") or {}}


def _read_bytes(path: str | None, url: str | None, b64: str | None, kind: str) -> bytes:
    if path:
        p = Path(path).expanduser()
        if not p.is_file():
            raise RuntimeError(f"{kind} path not found: {path}")
        return p.read_bytes()
    if url:
        with urllib.request.urlopen(url, timeout=30) as response:
            return response.read()
    if b64:
        try:
            raw = base64.b64decode(b64.split(",", 1)[-1], validate=True)
        except Exception as exc:
            raise RuntimeError(f"Invalid or truncated {kind} base64; pass a local file path instead") from exc
        return raw
    raise RuntimeError(f"Provide {kind}_path or {kind}_base64/image_url")


def _image_png(raw: bytes) -> bytes:
    try:
        from PIL import Image
        from pipeline.cropping import fit_image
        image = Image.open(io.BytesIO(raw)).convert("RGB")
        fit_image(image)
        out = io.BytesIO()
        image.save(out, format="PNG")
        return out.getvalue()
    except Exception as exc:
        raise RuntimeError(f"Cannot decode image: {exc}") from exc


def _layout_regions(png: bytes) -> list[dict]:
    """Run layout detection in-process; never asks the caller to invoke a layout tool."""
    try:
        import numpy as np
        import paddlex as pdx
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Region mode needs PaddleX/PaddlePaddle installed in this environment; install the PP-DocLayoutV3 runtime described in the Skill.") from exc
    from pipeline.layout_detect_paddlex import serialize_regions
    img = Image.open(io.BytesIO(png)).convert("RGB")
    global _LAYOUT_MODEL
    with _LAYOUT_LOCK:
        if _LAYOUT_MODEL is None:
            options = {"threshold": LAYOUT_THRESHOLD}
            if os.environ.get("XIAOMI_OCR_LAYOUT_MODEL_DIR"):
                options["model_dir"] = os.environ["XIAOMI_OCR_LAYOUT_MODEL_DIR"]
            if os.environ.get("XIAOMI_OCR_LAYOUT_DEVICE"):
                options["device"] = os.environ["XIAOMI_OCR_LAYOUT_DEVICE"]
            _LAYOUT_MODEL = pdx.create_model("PP-DocLayoutV3", **options)
        result = list(_LAYOUT_MODEL.predict(np.asarray(img)))[0]
    # Preserve the detector's reading order; do not sort columns by y/x.
    return serialize_regions(result, *img.size)


_LAYOUT_MODEL = None
_LAYOUT_LOCK = threading.Lock()


def _crop(image, box: list[float], polygon=None) -> bytes:
    from pipeline.cropping import crop_region
    crop = crop_region(image, box, polygon=polygon)
    out = io.BytesIO()
    crop.save(out, format="PNG")
    return out.getvalue()


def assemble_regions(regions: list[dict], texts: list[str]) -> str:
    rows = []
    for region, text in zip(regions, texts):
        if region.get("task_type") == "table":
            text = convert_otsl_to_html(text)
        rows.append({**region, "merged": text})
    return merge_page_to_markdown(rows)


def ocr_image(image_path: str | None = None, image_url: str | None = None,
              image_base64: str | None = None, mode: str = "page", max_tokens: int = MAX_TOKENS) -> dict:
    if mode not in {"page", "region"}:
        raise RuntimeError("mode must be 'page' or 'region'")
    png = _image_png(_read_bytes(image_path, image_url, image_base64, "image"))
    if mode == "page":
        result = _chat(png, PROMPTS["document"], max_tokens)
        return {"mode": "page", "markdown": finalize_document(result["text"]), "usage": result["usage"]}

    from PIL import Image
    page = Image.open(io.BytesIO(png)).convert("RGB")
    regions = _layout_regions(png)
    regions = [r for r in regions if r.get("task_type") not in {"skip", "abandon"}]
    if not regions:
        result = _chat(png, PROMPTS["document"], max_tokens)
        return {"mode": "page_fallback", "reason": "layout returned no usable regions", "markdown": finalize_document(result["text"])}
    results = [""] * len(regions)
    def run(i: int):
        region = regions[i]
        task_type = region.get("task_type", "text")
        region_max_tokens = min(max_tokens, REGION_MAX_TOKENS.get(task_type, REGION_MAX_TOKENS["text"]))
        crop = _crop(page, region["bbox_2d"])
        return i, _chat(crop, PROMPTS.get(task_type, PROMPTS["text"]), region_max_tokens)["text"]
    with ThreadPoolExecutor(max_workers=min(CONCURRENCY, max(1, len(regions)))) as pool:
        for future in as_completed([pool.submit(run, i) for i in range(len(regions))]):
            i, results[i] = future.result()
    return {"mode": "region", "regions": len(regions), "markdown": assemble_regions(regions, results)}


_PDF_LOCK = threading.Lock()


def _render_pdf(raw: bytes, max_pages: int | None, dpi: int) -> list[bytes]:
    if dpi <= 0 or (max_pages is not None and max_pages < 1):
        raise ValueError("dpi and max_pages must be positive")
    if not raw.startswith(b"%PDF"):
        raise RuntimeError("Input is not a PDF")
    try:
        import pypdfium2 as pdfium
    except ImportError as exc:
        raise RuntimeError("PDF support needs pypdfium2; install requirements.txt") from exc
    # PDFium is not thread-safe, even for separate documents. Serialize rendering,
    # then release the lock before the much slower concurrent model requests.
    with _PDF_LOCK, pdfium.PdfDocument(raw) as doc:
        count = len(doc) if max_pages is None else min(len(doc), max_pages)
        pages = []
        for i in range(count):
            with closing(doc[i]) as page:
                bitmap = page.render(scale=dpi / 72)
                try:
                    image = bitmap.to_pil().convert("RGB")
                finally:
                    bitmap.close()
            if image.width > 2200:
                scale = 2200 / image.width
                image = image.resize((2200, int(image.height * scale)), __import__("PIL.Image", fromlist=["Image"]).Resampling.LANCZOS)
            out = io.BytesIO()
            image.save(out, "PNG")
            pages.append(out.getvalue())
        return pages



def ocr_pdf(pdf_path: str | None = None, pdf_base64: str | None = None,
            mode: str = "page", max_pages: int | None = None, dpi: int = 150,
            max_tokens: int = MAX_TOKENS) -> dict:
    raw = _read_bytes(pdf_path, None, pdf_base64, "pdf")
    pages = _render_pdf(raw, max_pages, dpi)
    if mode == "region":
        # Use the same end-to-end in-process pipeline per page; page-level jobs run concurrently.
        png_pages = pages
        def run(i: int):
            return i, ocr_image(image_base64=base64.b64encode(png_pages[i]).decode(), mode="region", max_tokens=max_tokens)["markdown"]
    elif mode == "page":
        def run(i: int):
            return i, finalize_document(_chat(pages[i], PROMPTS["document"], max_tokens)["text"])
    else:
        raise RuntimeError("mode must be 'page' or 'region'")
    outputs = [""] * len(pages)
    with ThreadPoolExecutor(max_workers=min(CONCURRENCY, max(1, len(pages)))) as pool:
        for future in as_completed([pool.submit(run, i) for i in range(len(pages))]):
            i, text = future.result()
            outputs[i] = text
    body = "\n\n---\n\n".join(f"<!-- page {i+1} -->\n\n{text}" for i, text in enumerate(outputs))
    return {"pages": len(pages), "mode": mode, "markdown": body}


def _schema_prompt(fields: list[str] | None) -> str:
    if not fields:
        return "Extract the key information from the document image as a JSON object. Use concise field names. Return JSON only; use an empty string for missing values."
    schema = {str(field): "" for field in fields}
    return ("Extract key information in the image\n\nPlease output the key information in JSON format according to the following schema:\n"
            + json.dumps(schema, ensure_ascii=False, indent=4))


def kie_image(image_path: str | None = None, image_url: str | None = None,
              image_base64: str | None = None, fields: list[str] | None = None,
              max_tokens: int = MAX_TOKENS) -> dict:
    png = _image_png(_read_bytes(image_path, image_url, image_base64, "image"))
    raw = _chat(png, _schema_prompt(fields), max_tokens)["text"]
    match = re.search(r"\{[\s\S]*\}", raw)
    if not match:
        return {"json": None, "raw": raw, "warning": "Model output did not contain a JSON object"}
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {"json": None, "raw": raw, "warning": "Model output contained invalid JSON"}
    if fields:
        data = {field: data.get(field, "") for field in fields}
    return {"json": data}


def vqa_image(question: str, image_path: str | None = None, image_url: str | None = None,
              image_base64: str | None = None) -> dict:
    if not question.strip():
        raise RuntimeError("question must not be empty")
    png = _image_png(_read_bytes(image_path, image_url, image_base64, "image"))
    prompt = question.strip() + "\nAnswer the question using a single word or phrase."
    return {"answer": _chat(png, prompt, 1024)["text"]}


def _props_image():
    return {"image_path": {"type": "string", "description": "Local file path (preferred)"},
            "image_url": {"type": "string"},
            "image_base64": {"type": "string", "description": "Base64/data URL; may be truncated by a harness"}}


TOOLS = [
    {"name": "ocr_image", "description": "Parse one image. Use mode='region' for standard printed documents (MCP detects layout, crops and OCRs regions concurrently); use mode='page' for scene text, handwriting, calligraphy, or historic/irregular documents where segmentation could split content incorrectly. The default is page.", "inputSchema": {"type": "object", "properties": {**_props_image(), "mode": {"type": "string", "enum": ["page", "region"], "default": "page"}, "max_tokens": {"type": "integer", "default": MAX_TOKENS}}}},
    {"name": "ocr_pdf", "description": "Parse a PDF to Markdown. Standard printed documents may use mode='region' for internal layout detection and concurrent region OCR; scene text, handwriting, calligraphy and historical documents should use mode='page'.", "inputSchema": {"type": "object", "properties": {"pdf_path": {"type": "string"}, "pdf_base64": {"type": "string"}, "mode": {"type": "string", "enum": ["page", "region"], "default": "page"}, "max_pages": {"type": "integer"}, "dpi": {"type": "integer", "default": 150}, "max_tokens": {"type": "integer", "default": MAX_TOKENS}}}},
    {"name": "kie_image", "description": "Extract key information from an image as JSON. Optionally provide field names as a schema. Image reading, prompting and JSON cleanup are handled by this MCP service.", "inputSchema": {"type": "object", "properties": {**_props_image(), "fields": {"type": "array", "items": {"type": "string"}, "description": "Optional output field names"}, "max_tokens": {"type": "integer", "default": MAX_TOKENS}}}},
    {"name": "vqa_image", "description": "Answer a question about text or information in a document image. The MCP applies the concise OCR-VQA answer format; provide the image and question only.", "inputSchema": {"type": "object", "properties": {**_props_image(), "question": {"type": "string"}}, "required": ["question"]}},
]


def _tool_call(name: str, args: dict) -> dict:
    try:
        if name in {"ocr_image", "kie_image", "vqa_image"}:
            params = {k: args.get(k) for k in ("image_path", "image_url", "image_base64")}
            if name == "ocr_image": result = ocr_image(**params, mode=args.get("mode", "page"), max_tokens=int(args.get("max_tokens", MAX_TOKENS)))
            elif name == "kie_image": result = kie_image(**params, fields=args.get("fields"), max_tokens=int(args.get("max_tokens", MAX_TOKENS)))
            else: result = vqa_image(**params, question=args.get("question", ""))
        elif name == "ocr_pdf":
            result = ocr_pdf(pdf_path=args.get("pdf_path"), pdf_base64=args.get("pdf_base64"), mode=args.get("mode", "page"), max_pages=args.get("max_pages"), dpi=int(args.get("dpi", 150)), max_tokens=int(args.get("max_tokens", MAX_TOKENS)))
        else:
            return {"content": [{"type": "text", "text": f"unknown tool: {name}"}], "isError": True}
        return {"content": [{"type": "text", "text": json.dumps(result, ensure_ascii=False, indent=2) if isinstance(result, dict) and name in {"kie_image", "vqa_image"} else result.get("markdown", json.dumps(result, ensure_ascii=False))}]}
    except Exception as exc:
        return {"content": [{"type": "text", "text": f"ERROR: {exc}"}], "isError": True}


def main() -> None:
    for line in sys.stdin:
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        method, params, msg_id = msg.get("method"), msg.get("params") or {}, msg.get("id")
        if method == "notifications/initialized":
            continue
        if method == "initialize":
            result = {"protocolVersion": params.get("protocolVersion", "2024-11-05"), "capabilities": {"tools": {"listChanged": False}}, "serverInfo": {"name": "xiaomi-ocr", "version": "0.2.0"}}
        elif method == "ping": result = {}
        elif method == "tools/list": result = {"tools": TOOLS}
        elif method == "tools/call": result = _tool_call(params.get("name", ""), params.get("arguments") or {})
        else:
            if msg_id is not None:
                print(json.dumps({"jsonrpc": "2.0", "id": msg_id, "error": {"code": -32601, "message": f"method not found: {method}"}}), flush=True)
            continue
        if msg_id is not None:
            print(json.dumps({"jsonrpc": "2.0", "id": msg_id, "result": result}), flush=True)


if __name__ == "__main__":
    main()
