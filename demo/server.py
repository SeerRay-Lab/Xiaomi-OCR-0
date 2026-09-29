#!/usr/bin/env python3
"""Xiaomi-OCR-0 Demo — local model-backed web app and static server.

Zero third-party deps (stdlib only) except pypdfium2 for PDF page rendering.
Run:
    python3 server.py --port 8787
"""

from __future__ import annotations

import argparse
import base64
import importlib.util
import json
import mimetypes
import os
import sys
import threading
import time
import uuid
import urllib.error
import urllib.request
import urllib.parse
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STATIC = ROOT / "static"
SAMPLES = ROOT / "samples"
EXAMPLE_PICS = ROOT.parent / "example_pics"
BOOK_TMP = ROOT / "out" / "books"

MAX_BODY = 80 * 1024 * 1024  # 80 MB (PDF books)

DOC_PROMPT = (
    "Extract all information from the main body of the document image and represent it in markdown format, "
    "ignoring headers and footers. Tables should be expressed in OTSL format, formulas in the document should "
    "be represented using LATEX format, and the parsing should be organized according to the reading order."
)


def build_task_prompt(task: str, prompt: str = "", fields: str = "", question: str = "") -> str:
    """Build constrained prompts for the compact browser demo tasks."""
    task = (task or "document").strip().lower()
    if task == "document":
        return (prompt or DOC_PROMPT).strip()
    if task == "kie":
        names = [x.strip() for x in fields.replace("，", "\n").replace(",", "\n").splitlines() if x.strip()]
        return mcp_pipeline()._schema_prompt(names or None)
    if task == "vqa":
        q = question.strip()
        if not q:
            raise ValueError("请输入关于图像内容的问题")
        return q + "\nAnswer the question using a single word or phrase."
    raise ValueError("task must be document, kie, or vqa")


def normalize_image_b64(image_b64: str) -> str:
    """Apply the MCP's RGB PNG normalization before every page/image inference."""
    try:
        normalized = mcp_pipeline()._image_png(base64.b64decode(image_b64, validate=True))
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Cannot decode image: {exc}") from exc
    return base64.b64encode(normalized).decode()

# ── region-mode prompts (per README contract) ───────────────────────────
REGION_PROMPTS = {
    "text": "Extract the text in the image.",
    "table": "Parse the table in the image into OTSL.",
    "formula": "Identify the formula in the image and represent it using LATEX format.",
    "title": "Extract the text in the image.",
    "figure": "Extract all information from the main body of the document image and represent it in markdown format.",
}
# labels we might get from layout → bucket
LABEL_MAP = {
    "text": "text", "paragraph": "text", "plain text": "text", "text_line": "text",
    "title": "title", "header": "title", "section_header": "title",
    "table": "table", "table_caption": "text",
    "formula": "formula", "equation": "formula", "isolate_formula": "formula",
    "figure": "figure", "image": "figure", "figure_caption": "text",
    "list": "text", "list_item": "text", "caption": "text",
}

REGION_CONCURRENCY = int(os.environ.get("REGION_CONCURRENCY", "16"))
LOCAL_MODEL_URL = os.environ.get("XIAOMI_OCR_LOCAL_URL", "http://127.0.0.1:8000/v1").strip()
LOCAL_MODEL = os.environ.get("XIAOMI_OCR_MODEL", "SeerRay-Lab/Xiaomi-OCR-0").strip()

_MCP_MODULE = None


def mcp_pipeline(model_url: str | None = None, model: str | None = None):
    """Load the shared Skill implementation so Demo and MCP use the same transforms."""
    global _MCP_MODULE
    if _MCP_MODULE is None:
        source = ROOT.parent / "skills" / "xiaomi-ocr" / "mcp_ocr_server.py"
        spec = importlib.util.spec_from_file_location("xiaomi_ocr_mcp_pipeline", source)
        if spec is None or spec.loader is None:
            raise RuntimeError("Could not load the shared Xiaomi OCR MCP pipeline")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _MCP_MODULE = module
    _MCP_MODULE.OCR_BASE = (model_url or LOCAL_MODEL_URL).rstrip("/")
    _MCP_MODULE.OCR_MODEL = model or LOCAL_MODEL
    return _MCP_MODULE


_LAYOUT_MODEL = None
_LAYOUT_LOCK = threading.Lock()


def call_layout(image_b64: str, mime: str = "image/png") -> list[dict]:
    """Use the Skill's confidence filter, resize policy, layout model, and reading order."""
    mcp = mcp_pipeline()
    raw = base64.b64decode(image_b64)
    regions = []
    for r in mcp._layout_regions(raw):
        cls = r["cls"]
        bucket = "formula" if cls in {"formula", "equation", "isolate_formula"} else (
            "table" if cls == "table" else "text"
        )
        prompt = mcp.PROMPTS["formula" if bucket == "formula" else "table" if bucket == "table" else "text"]
        regions.append({"bbox": r["box"], "label": cls, "bucket": bucket,
                        "score": r["score"], "prompt": prompt})
    return regions


def crop_region(png_bytes: bytes, bbox: list[float], pad: int = 4) -> bytes:
    import io
    from PIL import Image
    return mcp_pipeline()._crop(Image.open(io.BytesIO(png_bytes)), bbox, pad)


def ocr_one_region(
    idx: int, png: bytes, bucket: str, prompt: str,
    model_url: str, model: str, max_tokens: int, label: str = "text",
) -> dict:
    t0 = time.perf_counter()
    try:
        mcp = mcp_pipeline(model_url, model)
        token_cap = max_tokens if label == "page_fallback" else mcp.REGION_MAX_TOKENS.get(label, mcp.REGION_MAX_TOKENS.get(bucket, 1024))
        result = mcp._chat(png, prompt or mcp.PROMPTS.get(bucket, mcp.PROMPTS["text"]),
                           min(max_tokens, token_cap))
        text = (result.get("text") or "").strip()
        return {
            "idx": idx,
            "ok": True,
            "bucket": bucket,
            "text": text,
            "ms": int((time.perf_counter() - t0) * 1000),
            "usage": result.get("usage") or {},
        }
    except Exception as e:  # noqa: BLE001
        return {
            "idx": idx,
            "ok": False,
            "bucket": bucket,
            "text": f"<!-- region {idx} failed: {e} -->",
            "ms": int((time.perf_counter() - t0) * 1000),
            "usage": {},
            "error": str(e),
        }


def assemble_regions(regions: list[dict], results: dict[int, dict]) -> str:
    """Join region texts in reading order; tables/formulas wrapped."""
    parts: list[str] = []
    for i, reg in enumerate(regions):
        r = results.get(i) or {"ok": False, "text": "", "bucket": reg.get("bucket", "text")}
        text = (r.get("text") or "").strip()
        if not text:
            continue
        bucket = r.get("bucket") or reg.get("bucket") or "text"
        if bucket == "table":
            # keep OTSL/HTML as-is for downstream renderer
            parts.append(text)
        elif bucket == "formula":
            # ensure display math
            if not (text.startswith("$") or text.startswith("\\[")):
                parts.append(f"\\[{text}\\]")
            else:
                parts.append(text)
        else:
            parts.append(text)
    return "\n\n".join(parts).strip()


def parse_image_regions(
    png_bytes: bytes,
    model_url: str,
    model: str,
    max_tokens: int,
    concurrency: int = REGION_CONCURRENCY,
) -> dict:
    """Layout → crop → concurrent region OCR → assemble."""
    import base64 as _b64
    t0 = time.perf_counter()
    image_b64 = _b64.b64encode(png_bytes).decode()

    # 1) layout
    regions = call_layout(image_b64, "image/png")
    if not regions:
        regions = [{"bbox": [0, 0, 0, 0], "label": "page_fallback", "bucket": "text", "score": 1.0, "prompt": DOC_PROMPT}]
    t_layout = int((time.perf_counter() - t0) * 1000)

    # 2) crop
    crops: list[bytes] = []
    for reg in regions:
        b = reg["bbox"]
        if b == [0, 0, 0, 0]:
            crops.append(png_bytes)
        else:
            crops.append(crop_region(png_bytes, b))

    # 3) concurrent OCR
    from concurrent.futures import ThreadPoolExecutor, as_completed
    results: dict[int, dict] = {}
    max_conc = max(1, min(int(concurrency), 32))
    with ThreadPoolExecutor(max_workers=max_conc) as pool:
        futs = {
            pool.submit(
                ocr_one_region, i, crops[i], regions[i]["bucket"], regions[i].get("prompt", ""),
                model_url, model, max_tokens, regions[i].get("label", "text"),
            ): i
            for i in range(len(regions))
        }
        for fut in as_completed(futs):
            r = fut.result()
            results[r["idx"]] = r

    md = assemble_regions(regions, results)
    per = [results[i] for i in range(len(regions))]
    return {
        "content": md,
        "regions": [
            {
                "bbox": regions[i]["bbox"],
                "label": regions[i]["label"],
                "bucket": regions[i].get("bucket"),
                "ok": per[i]["ok"] if i < len(per) else False,
                "ms": per[i].get("ms", 0) if i < len(per) else 0,
                "chars": len(per[i].get("text", "")) if i < len(per) else 0,
            }
            for i in range(len(regions))
        ],
        "layout_ms": t_layout,
        "latency_ms": int((time.perf_counter() - t0) * 1000),
        "concurrency": max_conc,
        "usage": {
            "prompt_tokens": sum((per[i].get("usage") or {}).get("prompt_tokens") or 0 for i in range(len(per))),
            "completion_tokens": sum((per[i].get("usage") or {}).get("completion_tokens") or 0 for i in range(len(per))),
        },
        "model": model,
        "mode": "region",
    }


def stream_image_regions(
    png_bytes: bytes,
    model_url: str,
    model: str,
    max_tokens: int,
    concurrency: int = REGION_CONCURRENCY,
) -> dict:
    """Layout → crop → concurrent OCR, yielding events as each region completes.

    Yields dicts:
      {"type":"layout", "count":N, "regions":[{index,bucket,label,...}], "layout_ms":ms}
      {"type":"region", "index":i, "bucket":.., "text":.., "ok":.., "ms":..}
      {"type":"done", "latency_ms":.., "usage":{..}, "concurrency":..}
    """
    import base64 as _b64
    from concurrent.futures import ThreadPoolExecutor, as_completed

    t0 = time.perf_counter()
    image_b64 = _b64.b64encode(png_bytes).decode()
    regions = call_layout(image_b64, "image/png")
    if not regions:
        regions = [{"bbox": [0, 0, 0, 0], "label": "page_fallback", "bucket": "text", "score": 1.0, "prompt": DOC_PROMPT}]
    t_layout = int((time.perf_counter() - t0) * 1000)

    yield {
        "type": "layout",
        "count": len(regions),
        "regions": [
            {"index": i, "bucket": r.get("bucket", "text"),
             "label": r.get("label", ""), "bbox": r["bbox"]}
            for i, r in enumerate(regions)
        ],
        "layout_ms": t_layout,
    }

    crops = [
        png_bytes if r["bbox"] == [0, 0, 0, 0] else crop_region(png_bytes, r["bbox"])
        for r in regions
    ]
    results: dict[int, dict] = {}
    max_conc = max(1, min(int(concurrency), 32))
    with ThreadPoolExecutor(max_workers=max_conc) as pool:
        futs = {
            pool.submit(
                ocr_one_region, i, crops[i], regions[i]["bucket"], regions[i].get("prompt", ""),
                model_url, model, max_tokens, regions[i].get("label", "text"),
            ): i
            for i in range(len(regions))
        }
        for fut in as_completed(futs):
            r = fut.result()
            results[r["idx"]] = r
            yield {
                "type": "region",
                "index": r["idx"],
                "bucket": r.get("bucket", "text"),
                "ok": r.get("ok", False),
                "ms": r.get("ms", 0),
                "text": r.get("text", ""),
            }

    per = [results[i] for i in range(len(regions))]
    yield {
        "type": "done",
        "latency_ms": int((time.perf_counter() - t0) * 1000),
        "layout_ms": t_layout,
        "concurrency": max_conc,
        "usage": {
            "prompt_tokens": sum((per[i].get("usage") or {}).get("prompt_tokens") or 0 for i in range(len(per))),
            "completion_tokens": sum((per[i].get("usage") or {}).get("completion_tokens") or 0 for i in range(len(per))),
        },
    }

# ── in-memory book jobs ─────────────────────────────────────────────────
JOBS: dict[str, dict] = {}
JOBS_LOCK = threading.Lock()


def job_get(jid: str) -> dict | None:
    with JOBS_LOCK:
        return JOBS.get(jid)


def job_update(jid: str, **kw) -> None:
    with JOBS_LOCK:
        JOBS.setdefault(jid, {}).update(kw)


def build_user_content(image_b64: str, mime: str, prompt: str) -> list:
    return [
        {
            "type": "image_url",
            "image_url": {"url": f"data:{mime};base64,{image_b64}"},
        },
        {"type": "text", "text": prompt},
    ]


def call_local_model(
    model_url: str,
    model: str,
    image_b64: str,
    mime: str,
    prompt: str,
    temperature: float,
    max_tokens: int,
) -> dict:
    base = model_url.rstrip("/")
    parsed_base = urllib.parse.urlparse(base)
    if parsed_base.scheme != "http" or parsed_base.hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise RuntimeError("Inference must run locally; configure XIAOMI_OCR_LOCAL_URL with a loopback address.")
    if not base.endswith("/chat/completions"):
        if base.endswith("/v1"):
            url = base + "/chat/completions"
        else:
            url = base + "/v1/chat/completions"
    else:
        url = base

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": build_user_content(image_b64, mime, prompt),
            }
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    data = json.dumps(payload).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            raw = resp.read()
            status = resp.status
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {e.code}: {body[:2000]}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"Connection failed: {e.reason}") from e

    latency_ms = int((time.perf_counter() - t0) * 1000)
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Invalid response from local model (HTTP {status}): {raw[:500]!r}") from e

    usage = parsed.get("usage") or {}
    choices = parsed.get("choices") or []
    content = ""
    if choices:
        msg = choices[0].get("message") or {}
        content = msg.get("content") or ""
        if isinstance(content, list):
            content = "".join(
                p.get("text", "") if isinstance(p, dict) else str(p) for p in content
            )

    return {
        "content": content,
        "model": parsed.get("model", model),
        "usage": usage,
        "latency_ms": latency_ms,
        "raw": parsed,
    }


def ocr_png_bytes(png: bytes, model_url: str, model: str, max_tokens: int) -> tuple[str, int, dict]:
    b64 = base64.b64encode(png).decode()
    return (
        call_local_model(
            model_url=model_url,
            model=model,
            image_b64=b64,
            mime="image/png",
            prompt=DOC_PROMPT,
            temperature=0.0,
            max_tokens=max_tokens,
        )["content"],
        0,
        {},
    )


def run_book_job(jid: str, pdf_path: Path, model_url: str, model: str,
                 max_tokens: int, max_pages: int | None, dpi: int,
                 title: str | None = None, concurrency: int = 4,
                 prompt: str = DOC_PROMPT, mode: str = "page") -> None:
    try:
        import pypdfium2 as pdfium
    except ImportError:
        job_update(jid, status="error", error="pypdfium2 not installed on server")
        return
    try:
        from PIL import Image
    except ImportError:
        job_update(jid, status="error", error="Pillow not installed on server; install it with python3 -m pip install Pillow")
        return

    try:
        doc = pdfium.PdfDocument(str(pdf_path))
    except Exception as e:  # noqa: BLE001
        job_update(jid, status="error", error=f"PDF open failed: {e}")
        return

    total = len(doc) if max_pages is None else min(len(doc), max_pages)
    pages_dir = BOOK_TMP / jid / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)
    job_update(
        jid,
        status="running",
        total=total,
        total_pdf=len(doc),
        done=0,
        current=0,
        pages_ok=0,
        page_log=[],
        pages_dir=str(pages_dir),
        concurrency=max(1, concurrency),
    )

    scale = dpi / 72.0
    t_all = time.perf_counter()
    import io

    # ── Phase 1: render all pages locally (fast) ──
    page_pngs: list[bytes] = []
    try:
        for i in range(total):
            job = job_get(jid) or {}
            if job.get("cancel"):
                job_update(jid, status="cancelled")
                return
            job_update(jid, current=i + 1, phase="render",
                       phase_label=f"渲染 {i + 1}/{total}")
            page = doc[i]
            bitmap = page.render(scale=scale)
            pil = bitmap.to_pil().convert("RGB")
            if pil.width > 2200:
                ratio = 2200 / pil.width
                pil = pil.resize((2200, int(pil.height * ratio)), Image.Resampling.LANCZOS)
            page_png = pages_dir / f"p{i + 1:04d}.png"
            pil.save(page_png, format="PNG", optimize=True)
            buf = io.BytesIO()
            pil.save(buf, format="PNG", optimize=True)
            page_pngs.append(buf.getvalue())
    finally:
        try:
            doc.close()
        except Exception:  # noqa: BLE001
            pass

    if job_get(jid, ).get("cancel"):
        job_update(jid, status="cancelled")
        return

    # ── Phase 2: parallel OCR (bounded by service concurrency limit) ──
    from concurrent.futures import ThreadPoolExecutor, as_completed

    results: dict[int, dict] = {}
    results_lock = threading.Lock()
    done_count = 0

    def ocr_one(idx: int) -> tuple[int, dict]:
        if (job_get(jid) or {}).get("cancel"):
            return idx, {"ok": False, "text": "", "ms": 0, "usage": {}, "error": "cancelled"}
        t0 = time.perf_counter()
        try:
            if mode == "region":
                result = parse_image_regions(
                    png_bytes=page_pngs[idx], model_url=model_url, model=model,
                    max_tokens=max_tokens, concurrency=min(REGION_CONCURRENCY, 8),
                )
            else:
                result = call_local_model(
                    model_url=model_url,
                    model=model,
                    image_b64=base64.b64encode(page_pngs[idx]).decode(),
                    mime="image/png",
                    prompt=prompt,
                    temperature=0.0,
                    max_tokens=max_tokens,
                )
            text = (result.get("content") or "").strip()
            lat = result.get("latency_ms") or int((time.perf_counter() - t0) * 1000)
            return idx, {
                "ok": True,
                "text": text,
                "ms": lat,
                "usage": result.get("usage") or {},
                "error": "",
            }
        except Exception as e:  # noqa: BLE001
            return idx, {
                "ok": False,
                "text": f"<!-- page {idx + 1} OCR failed: {e} -->",
                "ms": int((time.perf_counter() - t0) * 1000),
                "usage": {},
                "error": str(e),
            }

    job_update(jid, phase="ocr", current=0, phase_label=f"OCR 并发 {concurrency}")

    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        futures = [pool.submit(ocr_one, i) for i in range(total)]
        for fut in as_completed(futures):
            idx, r = fut.result()
            with results_lock:
                results[idx] = r
                done_count += 1
                cur = job_get(jid) or {}
                page_log = list(cur.get("page_log") or [])
                old_usage = cur.get("usage") or {}
                new_usage = r.get("usage") or {}
                usage_totals = {
                    key: (old_usage.get(key) or 0) + (new_usage.get(key) or 0)
                    for key in set(old_usage) | set(new_usage)
                }
                page_log.append({
                    "page": idx + 1,
                    "ok": r["ok"],
                    "ms": r["ms"],
                    "chars": len(r["text"]),
                    "error": r["error"],
                    "img": f"/api/book/{jid}/page/{idx + 1}",
                })
                # keep log sorted by page for stable UI
                page_log.sort(key=lambda x: x["page"])
                job_update(
                    jid,
                    done=done_count,
                    current=idx + 1,
                    phase="ocr",
                    phase_label=f"OCR {done_count}/{total}",
                    pages_ok=(cur.get("pages_ok") or 0) + (1 if r["ok"] else 0),
                    last_ms=r["ms"],
                    usage=usage_totals,
                    page_log=page_log,
                    elapsed_ms=int((time.perf_counter() - t_all) * 1000),
                )

    if (job_get(jid) or {}).get("cancel"):
        job_update(jid, status="cancelled")
        return

    # ── Assemble in page order ──
    parts: list[str] = []
    for i in range(total):
        r = results.get(i) or {"ok": False, "text": f"<!-- page {i + 1} missing -->", "ms": 0, "usage": {}, "error": "missing"}
        parts.append(f"<!-- page {i + 1} -->\n\n{r['text']}")

    md = "\n\n---\n\n".join(parts)
    out_path = BOOK_TMP / f"{jid}.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md, encoding="utf-8")
    job_update(
        jid,
        status="done",
        markdown=md,
        result_path=str(out_path),
        elapsed_ms=int((time.perf_counter() - t_all) * 1000),
        phase_label="完成",
    )


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(STATIC), **kwargs)

    def log_message(self, fmt, *args):
        sys.stderr.write("[demo] %s - %s\n" % (self.address_string(), fmt % args))

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path in ("/", "/index.html"):
            self.path = "/index.html"
            return super().do_GET()
        if path.startswith("/samples/"):
            return self._serve_samples()
        if path.startswith("/example_pics/"):
            return self._serve_example_pics()
        if path.startswith("/api/book/"):
            # /api/book/<id> | /api/book/<id>/raw | /api/book/<id>/page/<n>
            parts = path.strip("/").split("/")
            # ["api","book", id, ...]
            if len(parts) >= 4 and parts[3] == "raw":
                return self._book_raw(parts[2])
            if len(parts) >= 5 and parts[3] == "page":
                return self._book_page(parts[2], parts[4])
            return self._book_status(path)
        return super().do_GET()

    def _serve_samples(self):
        rel = self.path.split("/samples/", 1)[1]
        return self._serve_file_from(SAMPLES, rel, "sample")

    def _serve_example_pics(self):
        rel = self.path.split("/example_pics/", 1)[1]
        return self._serve_file_from(EXAMPLE_PICS, rel, "example")

    def _serve_file_from(self, root: Path, rel: str, label: str):
        rel = rel.split("?", 1)[0]
        base = root.resolve()
        fp = (base / rel).resolve()
        if not str(fp).startswith(str(base) + os.sep) and fp != base:
            self.send_error(403, "forbidden")
            return
        if not fp.is_file():
            self.send_error(404, f"{label} not found")
            return
        ctype = mimetypes.guess_type(str(fp))[0] or "application/octet-stream"
        data = fp.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _book_status(self, path: str):
        # /api/book/<id>
        jid = path.rsplit("/", 1)[-1]
        job = job_get(jid)
        if not job:
            self._json(404, {"error": "job not found"})
            return
        # do not ship full markdown on every poll
        public = {k: v for k, v in job.items() if k != "markdown"}
        public["has_markdown"] = bool(job.get("markdown"))
        self._json(200, public)

    def _book_raw(self, jid: str):
        job = job_get(jid)
        if not job:
            self._json(404, {"error": "job not found"})
            return
        md = job.get("markdown") or ""
        self._json(200, {
            "id": jid,
            "status": job.get("status"),
            "markdown": md,
            "chars": len(md),
            "elapsed_ms": job.get("elapsed_ms"),
        })

    def _book_page(self, jid: str, page_s: str):
        job = job_get(jid)
        if not job:
            self.send_error(404, "job not found")
            return
        try:
            n = int(page_s)
        except ValueError:
            self.send_error(400, "bad page")
            return
        pages_dir = job.get("pages_dir")
        if not pages_dir:
            self.send_error(404, "no pages yet")
            return
        fp = Path(pages_dir) / f"p{n:04d}.png"
        if not fp.is_file():
            self.send_error(404, "page not found")
            return
        data = fp.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "image/png")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "max-age=3600")
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        path = self.path.split("?", 1)[0]
        if path == "/api/ocr":
            return self._handle_ocr()
        if path == "/api/ocr/stream":
            return self._handle_ocr_stream()
        if path == "/api/parse/regions":
            return self._handle_parse_regions()
        if path == "/api/parse/regions/stream":
            return self._handle_parse_regions_stream()
        if path == "/api/book":
            return self._handle_book_start()
        if path.startswith("/api/book/") and path.endswith("/cancel"):
            jid = path.split("/")[-2]
            job_update(jid, cancel=True)
            self._json(200, {"ok": True})
            return
        self.send_error(404, "not found")

    def _read_json(self) -> dict | None:
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            length = 0
        if length <= 0 or length > MAX_BODY:
            self._json(400, {"error": "invalid or missing Content-Length"})
            return None
        body = self.rfile.read(length)
        try:
            return json.loads(body)
        except json.JSONDecodeError:
            self._json(400, {"error": "request body must be JSON"})
            return None

    def _handle_ocr(self):
        req = self._read_json()
        if req is None:
            return

        model_url = LOCAL_MODEL_URL
        model = LOCAL_MODEL
        try:
            prompt = build_task_prompt(req.get("task", "document"), req.get("prompt", ""),
                                       req.get("fields", ""), req.get("question", ""))
        except ValueError as exc:
            self._json(400, {"error": str(exc)})
            return
        image_b64 = req.get("image_base64") or ""
        mime = req.get("image_mime") or "image/png"
        temperature = float(req.get("temperature") or 0.0)
        max_tokens = int(req.get("max_tokens") or 4096)

        if image_b64.startswith("data:"):
            try:
                header, b64 = image_b64.split(",", 1)
                mime = header.split(":", 1)[1].split(";", 1)[0] or mime
                image_b64 = b64
            except ValueError:
                self._json(400, {"error": "malformed data URL"})
                return

        if not image_b64:
            self._json(400, {"error": "image is required"})
            return
        if not prompt:
            self._json(400, {"error": "prompt is required"})
            return
        try:
            image_b64 = normalize_image_b64(image_b64)
            mime = "image/png"
        except ValueError as exc:
            self._json(400, {"error": str(exc)})
            return

        try:
            result = call_local_model(
                model_url=model_url,
                    model=model,
                image_b64=image_b64,
                mime=mime,
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except Exception as e:  # noqa: BLE001
            self._json(502, {"error": str(e)})
            return

        self._json(200, result)

    def _handle_ocr_stream(self):
        """Relay token output from the local model process.

        Request JSON same as /api/ocr. Response: text/event-stream with
        event: delta   data: {"t": "<token>"}
        event: done    data: {"latency_ms":..., "usage":{...}, "model":...}
        event: error   data: {"error":"..."}
        """
        req = self._read_json()
        if req is None:
            return

        model_url = LOCAL_MODEL_URL
        model = LOCAL_MODEL
        try:
            prompt = build_task_prompt(req.get("task", "document"), req.get("prompt", ""),
                                       req.get("fields", ""), req.get("question", ""))
        except ValueError as exc:
            self._json(400, {"error": str(exc)})
            return
        image_b64 = req.get("image_base64") or ""
        mime = req.get("image_mime") or "image/png"
        temperature = float(req.get("temperature") or 0.0)
        max_tokens = int(req.get("max_tokens") or 4096)

        if image_b64.startswith("data:"):
            try:
                header, b64 = image_b64.split(",", 1)
                mime = header.split(":", 1)[1].split(";", 1)[0] or mime
                image_b64 = b64
            except ValueError:
                self._json(400, {"error": "malformed data URL"})
                return
        if not image_b64 or not prompt:
            self._json(400, {"error": "image / prompt required"})
            return
        try:
            image_b64 = normalize_image_b64(image_b64)
            mime = "image/png"
        except ValueError as exc:
            self._json(400, {"error": str(exc)})
            return

        base = model_url.rstrip("/")
        parsed_base = urllib.parse.urlparse(base)
        if parsed_base.scheme != "http" or parsed_base.hostname not in {"127.0.0.1", "localhost", "::1"}:
            self._json(400, {"error": "The demo only accepts a local model listener."})
            return
        if not base.endswith("/chat/completions"):
            url = base + ("/chat/completions" if base.endswith("/v1") else "/v1/chat/completions")
        else:
            url = base

        payload = {
            "model": model,
            "messages": [{
                "role": "user",
                "content": build_user_content(image_b64, mime, prompt),
            }],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
        t0 = time.perf_counter()
        try:
            upstream_req = urllib.request.Request(
                url, data=json.dumps(payload).encode("utf-8"),
                headers=headers, method="POST",
            )
            upstream = urllib.request.urlopen(upstream_req, timeout=300)
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            self._json(502, {"error": f"HTTP {e.code}: {body[:800]}"})
            return
        except Exception as e:  # noqa: BLE001
            self._json(502, {"error": str(e)})
            return

        # start SSE response
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

        def sse(event: str, data: dict) -> None:
            chunk = f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
            try:
                self.wfile.write(chunk.encode("utf-8"))
                self.wfile.flush()
            except BrokenPipeError:
                raise

        usage: dict = {}
        try:
            for raw_line in upstream:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line or not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if obj.get("usage"):
                    usage = obj["usage"]
                choices = obj.get("choices") or []
                if not choices:
                    continue
                delta = (choices[0].get("delta") or {}).get("content") or ""
                if delta:
                    sse("delta", {"t": delta})
            sse("done", {
                "latency_ms": int((time.perf_counter() - t0) * 1000),
                "usage": usage,
                "model": model,
            })
        except BrokenPipeError:
            pass
        except Exception as e:  # noqa: BLE001
            try:
                sse("error", {"error": str(e)})
            except Exception:  # noqa: BLE001
                pass
        finally:
            try:
                upstream.close()
            except Exception:  # noqa: BLE001
                pass

    def _handle_parse_regions(self):
        """POST /api/parse/regions — layout → crop → concurrent region OCR.

        Body: {image_base64, max_tokens?, concurrency?}
        """
        req = self._read_json()
        if req is None:
            return
        model_url = LOCAL_MODEL_URL
        model = LOCAL_MODEL
        max_tokens = int(req.get("max_tokens") or 2048)
        concurrency = int(req.get("concurrency") or REGION_CONCURRENCY)
        image_b64 = req.get("image_base64") or ""
        mime = req.get("image_mime") or "image/png"

        if image_b64.startswith("data:"):
            try:
                header, b64 = image_b64.split(",", 1)
                mime = header.split(":", 1)[1].split(";", 1)[0] or mime
                image_b64 = b64
            except ValueError:
                self._json(400, {"error": "malformed data URL"})
                return
        if not image_b64:
            self._json(400, {"error": "image required"})
            return

        try:
            png_bytes = base64.b64decode(normalize_image_b64(image_b64))
            result = parse_image_regions(
                png_bytes=png_bytes,
                model_url=model_url,
                    model=model,
                max_tokens=max_tokens,
                concurrency=concurrency,
            )
        except Exception as e:  # noqa: BLE001
            self._json(502, {"error": str(e)})
            return
        self._json(200, result)

    def _handle_parse_regions_stream(self):
        """POST /api/parse/regions/stream — SSE: layout event, then one event per region as it finishes.

        Same body as /api/parse/regions. Events:
          event: layout  data: {"count":N,"regions":[...]}
          event: region  data: {"index":i,"bucket":..,"ok":..,"ms":..,"text":..}
          event: done    data: {"latency_ms":..,"usage":{..},"concurrency":..}
        """
        req = self._read_json()
        if req is None:
            return
        model_url = LOCAL_MODEL_URL
        model = LOCAL_MODEL
        max_tokens = int(req.get("max_tokens") or 2048)
        concurrency = int(req.get("concurrency") or REGION_CONCURRENCY)
        image_b64 = req.get("image_base64") or ""
        mime = req.get("image_mime") or "image/png"

        if image_b64.startswith("data:"):
            try:
                header, b64 = image_b64.split(",", 1)
                mime = header.split(":", 1)[1].split(";", 1)[0] or mime
                image_b64 = b64
            except ValueError:
                self._json(400, {"error": "malformed data URL"})
                return
        if not image_b64:
            self._json(400, {"error": "image required"})
            return

        try:
            png_bytes = base64.b64decode(normalize_image_b64(image_b64))
        except Exception as e:  # noqa: BLE001
            self._json(502, {"error": str(e)})
            return

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

        def sse(event: str, data: dict) -> None:
            chunk = f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
            try:
                self.wfile.write(chunk.encode("utf-8"))
                self.wfile.flush()
            except BrokenPipeError:
                raise

        try:
            for ev in stream_image_regions(
                png_bytes=png_bytes,
                model_url=model_url,
                    model=model,
                max_tokens=max_tokens,
                concurrency=concurrency,
            ):
                sse(ev["type"], {k: v for k, v in ev.items() if k != "type"})
        except BrokenPipeError:
            return
        except Exception as e:  # noqa: BLE001
            try:
                sse("error", {"error": str(e)})
            except Exception:  # noqa: BLE001
                pass

    def _handle_book_start(self):
        req = self._read_json()
        if req is None:
            return

        model_url = LOCAL_MODEL_URL
        model = LOCAL_MODEL
        max_tokens = int(req.get("max_tokens") or 4096)
        max_pages = req.get("max_pages")
        max_pages = int(max_pages) if max_pages else None
        dpi = int(req.get("dpi") or 150)
        mode = (req.get("mode") or "page").strip().lower()
        if mode not in {"page", "region"}:
            self._json(400, {"error": "PDF mode must be page or region"})
            return
        try:
            prompt = build_task_prompt("document", req.get("prompt", ""))
        except ValueError as exc:
            self._json(400, {"error": str(exc)})
            return
        # Bound nested region OCR fan-out as well as page-level concurrency.
        concurrency = int(req.get("concurrency") or (2 if mode == "region" else 4))
        concurrency = max(1, min(8 if mode == "region" else 4, concurrency))
        sample = (req.get("sample") or "").strip()
        pdf_b64 = req.get("pdf_base64") or ""
        pdf_name = (req.get("pdf_name") or "book.pdf").strip()

        # resolve PDF source
        try:
            if sample:
                fp = (SAMPLES / os.path.basename(sample)).resolve()
                if not str(fp).startswith(str(SAMPLES.resolve()) + os.sep) or not fp.is_file():
                    self._json(404, {"error": f"sample not found: {sample}"})
                    return
                pdf_path = fp
                pdf_name = fp.name
            elif pdf_b64.startswith("data:"):
                try:
                    _, b64 = pdf_b64.split(",", 1)
                    pdf_b64 = b64
                except ValueError:
                    self._json(400, {"error": "malformed data URL"})
                    return
                pdf_path = self._save_upload(pdf_b64, pdf_name)
            elif pdf_b64:
                pdf_path = self._save_upload(pdf_b64, pdf_name)
            else:
                self._json(400, {"error": "provide sample or pdf_base64"})
                return
        except Exception as e:  # noqa: BLE001
            self._json(400, {"error": f"PDF save failed: {e}"})
            return

        sys.stderr.write(f"[book] start source={pdf_name} size={pdf_path.stat().st_size} model={model} max_pages={max_pages} conc={concurrency}\n")

        jid = uuid.uuid4().hex[:12]
        job_update(
            jid,
            status="queued",
            total=0,
            done=0,
            current=0,
            model=model,
            source=pdf_name,
            created=int(time.time()),
            cancel=False,
            concurrency=concurrency,
        )
        th = threading.Thread(
            target=run_book_job,
            args=(jid, pdf_path, model_url, model, max_tokens, max_pages, dpi, None, concurrency, prompt, mode),
            daemon=True,
        )
        th.start()
        self._json(200, {"id": jid, "source": pdf_name, "concurrency": concurrency, "mode": mode})

    def _save_upload(self, b64: str, name: str) -> Path:
        try:
            data = base64.b64decode(b64)
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(f"invalid base64: {e}") from e
        BOOK_TMP.mkdir(parents=True, exist_ok=True)
        safe = os.path.basename(name) or "book.pdf"
        fp = BOOK_TMP / f"upload_{uuid.uuid4().hex[:8]}_{safe}"
        fp.write_bytes(data)
        return fp

    def _json(self, status: int, obj: dict):
        data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(data)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.end_headers()


def main():
    ap = argparse.ArgumentParser(description="Xiaomi-OCR-0 demo server")
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8787)))
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()

    gen = ROOT / "generate_samples.py"
    if gen.is_file() and (not SAMPLES.exists() or not any(SAMPLES.iterdir())):
        try:
            import runpy

            runpy.run_path(str(gen), run_name="__main__")
        except Exception as e:  # noqa: BLE001
            print(f"[demo] sample generation skipped: {e}", file=sys.stderr)

    BOOK_TMP.mkdir(parents=True, exist_ok=True)
    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    url = f"http://{args.host}:{args.port}/"
    print(f"Xiaomi-OCR-0 Demo  →  {url}")
    print("Ctrl+C to stop.")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")


if __name__ == "__main__":
    main()
