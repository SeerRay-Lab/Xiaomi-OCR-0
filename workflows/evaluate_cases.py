#!/usr/bin/env python3
"""使用本机 Xiaomi-OCR-0 运行评测案例集并记录结果。

用法:
  python3 evaluate_cases.py [CASES_JSONL] [IMAGES_DIR]

环境:
  XIAOMI_OCR_LOCAL_URL 默认 http://127.0.0.1:8000/v1
  XIAOMI_OCR_MODEL     默认 SeerRay-Lab/Xiaomi-OCR-0
  OCR_MAX_TOKENS 默认 4096

判定:
  exact      原文完全相同
  normalized 归一化后相同（去首尾空白、折叠连续空白、统一换行）
  wrong      其余
输出: <cases目录>/evaluate_report.json，控制台打印汇总与"全对"清单。
"""
from __future__ import annotations

import base64
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request

OCR_BASE = os.environ.get("XIAOMI_OCR_LOCAL_URL", "http://127.0.0.1:8000/v1").rstrip("/")
OCR_MODEL = os.environ.get("XIAOMI_OCR_MODEL", "SeerRay-Lab/Xiaomi-OCR-0")
MAX_TOKENS = int(os.environ.get("OCR_MAX_TOKENS", "4096"))
TIMEOUT = int(os.environ.get("OCR_TIMEOUT", "300"))

DOC_PROMPT = (
    "Extract all information from the main body of the document image and represent it "
    "in markdown format, ignoring headers and footers. Tables should be expressed in OTSL "
    "format, formulas in the document should be represented using LATEX format, and the "
    "parsing should be organized according to the reading order."
)
PROMPTS = {
    "text": "Extract the text in the image.",
    "table": "Parse the table in the image into OTSL.",
    "formula": "Identify the formula in the image and represent it using LATEX format.",
}
CAT_PROMPT = {
    "KIE": "Extract the key-value information in the image into JSON.",
    "general-VQA": "Extract the text in the image.",
    "表格": PROMPTS["table"],
    "公式": PROMPTS["formula"],
    "图表": "Identify the figure in the image and represent it using matplotlib code in Python format.",
    "OCR-VQA": DOC_PROMPT,
    "spotting": "Detect and recognize the text in the image with bounding boxes.",
    "文本": PROMPTS["text"],
    "文档": DOC_PROMPT,
}


def normalize(s: str) -> str:
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    return re.sub(r"\s+", " ", s).strip()


def call(prompt: str, img_path: str) -> tuple[str, float]:
    from urllib.parse import urlparse
    parsed = urlparse(OCR_BASE)
    if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise RuntimeError("Evaluation only connects to a local model listener")
    ext = img_path.rsplit(".", 1)[-1].lower()
    mime = "image/jpeg" if ext in ("jpg", "jpeg") else "image/png"
    b64 = base64.b64encode(open(img_path, "rb").read()).decode()
    payload = {
        "model": OCR_MODEL,
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
            {"type": "text", "text": prompt},
        ]}],
        "max_tokens": MAX_TOKENS,
        "temperature": 0.0,
    }
    req = urllib.request.Request(
        OCR_BASE + "/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
        data = json.load(resp)
    dt = time.perf_counter() - t0
    text = data["choices"][0]["message"]["content"]
    if isinstance(text, list):
        text = "".join(p.get("text", "") for p in text if isinstance(p, dict))
    return str(text or "").strip(), dt


def main() -> int:
    cases_path = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/xiaomi-ocr-cases/cases.jsonl")
    images_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(cases_path), "images")
    cases = [json.loads(l) for l in open(cases_path, encoding="utf-8")]
    print(f"Local model {OCR_BASE}  model={OCR_MODEL}")
    print(f"案例 {len(cases)} 个  图片目录 {images_dir}\n")

    results = []
    tally = {"exact": 0, "normalized": 0, "wrong": 0, "failed": 0}
    for i, c in enumerate(cases, 1):
        img = os.path.join(images_dir, os.path.basename(c["image"]))
        prompt = c.get("prompt") or CAT_PROMPT.get(c.get("category", ""), DOC_PROMPT)
        answer = c.get("answer", "")
        try:
            pred, dt = call(prompt, img)
        except Exception as exc:  # noqa: BLE001
            print(f"[{i:>2}/{len(cases)}] {c['id']:<22} FAILED {type(exc).__name__}: {str(exc)[:70]}")
            tally["failed"] += 1
            results.append({"id": c["id"], "category": c["category"], "status": "failed",
                            "error": str(exc)[:200], "prompt": prompt, "answer": answer})
            continue

        if pred == answer:
            status = "exact"
        elif normalize(pred) == normalize(answer):
            status = "normalized"
        else:
            status = "wrong"
        tally[status] += 1
        mark = {"exact": "✓", "normalized": "≈", "wrong": "✗"}[status]
        print(f"[{i:>2}/{len(cases)}] {c['id']:<22} {mark} {status:<10} {dt:5.1f}s "
              f"pred={len(pred)} ans={len(answer)}")
        results.append({"id": c["id"], "category": c["category"], "status": status,
                        "prompt": prompt, "answer": answer, "pred": pred,
                        "pred_len": len(pred), "ans_len": len(answer), "seconds": round(dt, 2)})

    print("\n=== 汇总 ===")
    print(f"  完全一致(exact)      {tally['exact']}")
    print(f"  归一化一致(normalized) {tally['normalized']}")
    print(f"  不一致(wrong)        {tally['wrong']}")
    print(f"  调用失败(failed)     {tally['failed']}")

    perfect = [r for r in results if r["status"] in ("exact", "normalized")]
    print(f"\n=== 全对的 case（{len(perfect)} 个）===")
    for r in perfect:
        print(f"  {r['id']:<22} [{r['category']}] {r['status']}")

    out = os.path.join(os.path.dirname(cases_path), "evaluate_report.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({"model": OCR_MODEL, "local_url": OCR_BASE, "tally": tally,
                   "perfect_ids": [r["id"] for r in perfect], "results": results},
                  fh, ensure_ascii=False, indent=1)
    print(f"\n明细: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
