#!/usr/bin/env python3
"""
Document image → markdown inference script.

Input: JSONL with {"images": ["/path/to/img.png"]} per line.
Output: one .md file per image.

Usage:
    python pipeline/e2e_img2md.py \
        --input-jsonl /path/to/images.jsonl \
        --output_dir /path/to/output_md \
        --start_server --stop_server
"""

import argparse
import base64
import html
import itertools
import json
import os
import re
import signal
import subprocess
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI
from PIL import Image

from pipeline.cropping import fit_image
from postprocess.repetition_guard import analyze_repetition

try:
    from postprocess.otsl import convert_otsl_to_html
except Exception:
    convert_otsl_to_html = None


OTSL_NL = "<nl>"
OTSL_FCEL = "<fcel>"
OTSL_ECEL = "<ecel>"
OTSL_LCEL = "<lcel>"
OTSL_UCEL = "<ucel>"
OTSL_XCEL = "<xcel>"
OTSL_FIND_PATTERN = re.compile(r"(?:<fcel>|<ecel>|<nl>|<lcel>|<ucel>|<xcel>).*?(?=(?:<fcel>|<ecel>|<nl>|<lcel>|<ucel>|<xcel>)|$)", flags=re.DOTALL)


def _fallback_otsl_pad_to_sqr(otsl_str: str) -> str:
    otsl_str = otsl_str.strip()
    if OTSL_NL not in otsl_str:
        return otsl_str + OTSL_NL
    rows = []
    for line in otsl_str.split(OTSL_NL):
        if not line:
            continue
        cells = OTSL_FIND_PATTERN.findall(line)
        if not cells:
            continue
        min_len = 0
        for idx, cell in enumerate(cells):
            if cell.startswith(OTSL_FCEL):
                min_len = idx + 1
        rows.append({"cells": cells, "min_len": min_len})
    if not rows:
        return OTSL_NL
    width = max(max(row["min_len"] for row in rows), max(len(row["cells"]) for row in rows))
    repaired = []
    for row in rows:
        cells = row["cells"][:width]
        cells += [OTSL_ECEL] * (width - len(cells))
        repaired.append("".join(cells))
    return OTSL_NL.join(repaired) + OTSL_NL


def _fallback_otsl_extract_tokens_and_text(s: str):
    pattern = r"(" + r"|".join([OTSL_NL, OTSL_FCEL, OTSL_ECEL, OTSL_LCEL, OTSL_UCEL, OTSL_XCEL]) + r")"
    tokens = re.findall(pattern, s)
    texts = [token for token in re.split(pattern, s) if token.strip()]
    return tokens, texts


def _fallback_otsl_to_html(otsl_content: str) -> str:
    otsl_content = _fallback_otsl_pad_to_sqr(otsl_content)
    row_items = []
    for line in otsl_content.split(OTSL_NL):
        if not line.strip():
            continue
        items = []
        for cell in OTSL_FIND_PATTERN.findall(line):
            match = re.match(r"(<fcel>|<ecel>|<lcel>|<ucel>|<xcel>)(.*)", cell, flags=re.DOTALL)
            if match:
                items.append((match.group(1), match.group(2)))
        if items:
            row_items.append(items)
    if not row_items:
        return ""

    max_cols = max(len(row) for row in row_items)
    for row in row_items:
        row.extend([(OTSL_ECEL, "")] * (max_cols - len(row)))

    used = set()
    body = ""
    for row_idx, row in enumerate(row_items):
        body += "<tr>"
        for col_idx, (token, text) in enumerate(row):
            if (row_idx, col_idx) in used or token not in {OTSL_FCEL, OTSL_ECEL}:
                continue
            colspan = 1
            while col_idx + colspan < max_cols and row[col_idx + colspan][0] in {OTSL_LCEL, OTSL_XCEL}:
                used.add((row_idx, col_idx + colspan))
                colspan += 1
            rowspan = 1
            while row_idx + rowspan < len(row_items) and row_items[row_idx + rowspan][col_idx][0] in {OTSL_UCEL, OTSL_XCEL}:
                used.add((row_idx + rowspan, col_idx))
                rowspan += 1
            attrs = ""
            if rowspan > 1:
                attrs += f' rowspan="{rowspan}"'
            if colspan > 1:
                attrs += f' colspan="{colspan}"'
            body += f"<td{attrs}>{html.escape(text.strip())}</td>"
        body += "</tr>"
    return f"<table>{body}</table>"


# Prompts
PROMPTS = {
    "text":     "Extract the text in the image.",
    "table":    "Parse the table in the image into OTSL.",
    "formula":  "Identify the formula in the image and represent it using LATEX format.",
    "document": (
        "Extract all information from the main body of the document image and represent it "
        "in markdown format, ignoring headers and footers. Tables should be expressed in OTSL "
        "format, formulas in the document should be represented using LATEX format, and the "
        "parsing should be organized according to the reading order."
    ),
}

TASK_EXT = {
    "document": ".md",
    "text": ".txt",
    "formula": ".tex",
    "table": ".html",
}

MODEL_PATH = "SeerRay-Lab/Xiaomi-OCR-0"
SERVED_MODEL_NAME = "SeerRay-Lab/Xiaomi-OCR-0"
BASE_PORT = 8000
EXTRA_ARGS = ["--trust-remote-code"]


def encode_image_base64(image_path):
    """Encode an RGB PNG within the model's 2048 x 2048 input bound."""
    from io import BytesIO

    with Image.open(image_path) as source:
        image = fit_image(source.convert("RGB"))
        buffer = BytesIO()
        image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


# ======================================================================
# vLLM server management
# ======================================================================

def wait_for_service(port, proc, timeout=1200, interval=5):
    """Poll /v1/models until VLLM service is ready."""
    url = f"http://127.0.0.1:{port}/v1/models"
    no_proxy = urllib.request.ProxyHandler({})
    opener = urllib.request.build_opener(no_proxy)
    start = time.time()
    while time.time() - start < timeout:
        if proc.poll() is not None:
            return False, int(time.time() - start)
        try:
            req = urllib.request.Request(url, method="GET")
            with opener.open(req, timeout=5) as resp:
                if resp.status == 200:
                    return True, int(time.time() - start)
        except Exception:
            pass
        time.sleep(interval)
    return False, int(time.time() - start)


def start_vllm_replicas(model_cfg, gpu_ids, gpu_mem_util, log_dir):
    """Start N VLLM replicas (one per GPU) for a single model.

    Returns (processes, log_handles, ports).
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    model_name = model_cfg["name"]
    model_path = model_cfg["model_path"]
    base_port = model_cfg["base_port"]
    extra_args = model_cfg["extra_args"]
    gpu_mem_str = str(gpu_mem_util)

    processes = []
    log_handles = []
    ports = []

    print(f"  Starting {len(gpu_ids)} VLLM replica(s) for {model_name}...")
    for i, gpu_id in enumerate(gpu_ids):
        port = base_port + i
        ports.append(port)

        cmd = [
            "vllm", "serve", model_path,
            "--served-model-name", model_name,
            "--port", str(port),
            "--tensor-parallel-size", "1",
            "--gpu-memory-utilization", gpu_mem_str,
            "--uvicorn-log-level", "warning",
        ] + extra_args

        log_path = log_dir / f"vllm_{model_name}_gpu{gpu_id}_port{port}.log"
        log_fh = open(log_path, "w")
        log_handles.append(log_fh)

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        env["OMP_NUM_THREADS"] = "8"
        env["MKL_NUM_THREADS"] = "8"
        env["OPENBLAS_NUM_THREADS"] = "8"
        env["VECLIB_MAXIMUM_THREADS"] = "8"
        env["NUMEXPR_NUM_THREADS"] = "8"

        print(f"    [{i+1}/{len(gpu_ids)}] GPU {gpu_id} | port {port} | log: {log_path}")
        proc = subprocess.Popen(
            cmd, env=env, stdout=log_fh, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        processes.append(proc)
        time.sleep(6)

    # Wait for all replicas in parallel
    print(f"  Waiting for {model_name} replicas to be ready...")
    results = {}

    def _wait(idx, port, proc):
        ok, elapsed = wait_for_service(port, proc)
        results[idx] = (ok, elapsed)

    threads = []
    for idx, (port, proc) in enumerate(zip(ports, processes)):
        t = threading.Thread(target=_wait, args=(idx, port, proc))
        t.start()
        threads.append(t)
    for t in threads:
        t.join()

    failed = []
    for idx in sorted(results):
        ok, elapsed = results[idx]
        status = "ready" if ok else "FAILED"
        print(f"    GPU {gpu_ids[idx]} | port {ports[idx]} | {status} ({elapsed}s)")
        if not ok:
            failed.append(f"GPU {gpu_ids[idx]} port {ports[idx]}")

    if failed:
        # Show log tails for failed replicas
        for idx in sorted(results):
            ok, _ = results[idx]
            if ok:
                continue
            log_path = log_dir / f"vllm_{model_name}_gpu{gpu_ids[idx]}_port{ports[idx]}.log"
            try:
                log_handles[idx].flush()
            except Exception:
                pass
            try:
                with open(log_path, "r") as f:
                    lines = f.readlines()
                error_lines = [l for l in lines if "ERROR" in l or "ValueError" in l
                               or "RuntimeError" in l or "OOM" in l.upper()]
                if error_lines:
                    print(f"\n  === Log errors for GPU {gpu_ids[idx]} port {ports[idx]} ===")
                    for l in error_lines[-5:]:
                        print(f"  {l.rstrip()}")
                else:
                    print(f"\n  === Last 10 lines of log for GPU {gpu_ids[idx]} port {ports[idx]} ===")
                    for l in lines[-10:]:
                        print(f"  {l.rstrip()}")
            except Exception:
                pass

        stop_vllm_replicas(processes, log_handles)
        raise RuntimeError(
            f"{model_name} replicas failed:\n"
            + "\n".join(f"  - {f}" for f in failed)
            + "\n\nCheck logs above for details. Common causes:\n"
            + "  - GPU OOM: lower --gpu_memory_utilization (e.g. 0.3)\n"
            + "  - GPU in use: free GPU memory or use different --gpu_ids"
        )

    print(f"  All {model_name} replicas ready!")
    return processes, log_handles, ports


def stop_vllm_replicas(processes, log_handles):
    """Two-stage shutdown: SIGTERM -> 15s timeout -> SIGKILL."""
    for proc in processes:
        if proc.poll() is not None:
            continue
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except (ProcessLookupError, OSError, PermissionError):
            pass
    for proc in processes:
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                proc.wait(timeout=5)
            except (ProcessLookupError, OSError, PermissionError,
                    subprocess.TimeoutExpired):
                pass
    for fh in log_handles:
        try:
            fh.close()
        except Exception:
            pass


# ======================================================================
# Inference
# ======================================================================

def build_model_cfg(model_path, served_name, port, extra_args):
    """Build a model config dict compatible with start_vllm_replicas."""
    return {
        "name": served_name,
        "key": served_name,
        "model_path": model_path,
        "base_port": port,
        "extra_args": extra_args,
    }


from postprocess.document import postprocess_e2e_markdown, _clean_truncated_repeats


def infer_one(client, model_name, image_path, prompt, *,
              temperature=0.0, top_p=0.8, presence_penalty=0.0,
              repetition_penalty=1.0, max_tokens=16384, request_timeout=600,
              no_think=True):
    """Run inference on a single image; return (ok, result_text_or_error, metadata).

    后处理 (math delimiters / OTSL) 不在此做, 由写盘方决定, 以便重试机制比较原始输出。
    """
    try:
        b64 = encode_image_base64(image_path)
        data_url = f"data:image/png;base64,{b64}"

        kwargs = dict(
            model=model_name,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }],
            temperature=temperature,
            top_p=top_p,
            presence_penalty=presence_penalty,
            max_tokens=max_tokens,
            timeout=request_timeout,
        )
        extra_body = {"repetition_penalty": repetition_penalty}
        if no_think:
            extra_body["chat_template_kwargs"] = {"enable_thinking": False}
        kwargs["extra_body"] = extra_body
        response = client.chat.completions.create(**kwargs)

        choice = response.choices[0]
        result = choice.message.content or ""
        usage = getattr(response, "usage", None)
        metadata = {
            "finish_reason": getattr(choice, "finish_reason", None),
            "completion_tokens": getattr(usage, "completion_tokens", None),
        }
        if not result.strip():
            return False, "empty response", metadata
        return True, result.strip(), metadata
    except Exception as e:
        return False, str(e), {}


def load_image_list(jsonl_path):
    """Load image paths from JSONL: {"images": ["/path/to/img.png"]}"""
    images = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            for p in obj.get("images", []):
                if os.path.exists(p):
                    images.append(p)
    return images




# ======================================================================
# CLI
# ======================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Image → markdown/text/formula/table inference"
    )
    # Input JSONL: auto-detects document mode {"images": [...]} or region mode
    parser.add_argument("--input-jsonl", "--input_jsonl", type=str, required=True,
                        help="Path to JSONL: {\"images\": [...]} or pipeline region JSONL")
    parser.add_argument("--output_dir", required=True,
                        help="Directory to write output files")
    parser.add_argument("--task", choices=["document", "text", "formula", "table", "all"],
                        default="document",
                        help="Task type (default: document). 'all' processes all types from JSONL")
    parser.add_argument("--model_path", default=MODEL_PATH,
                        help=f"Model path (default: {MODEL_PATH})")
    parser.add_argument("--served_model_name", default=SERVED_MODEL_NAME,
                        help=f"Served model name (default: {SERVED_MODEL_NAME})")
    parser.add_argument("--port", type=int, default=BASE_PORT,
                        help=f"vLLM server port (default: {BASE_PORT})")
    parser.add_argument("--gpu_devices", default="0",
                        help="CUDA_VISIBLE_DEVICES (default: 0)")
    parser.add_argument("--gpu_mem_util", type=float, default=0.8,
                        help="GPU memory utilization (default: 0.9)")
    parser.add_argument("--workers", type=int, default=8,
                        help="Max concurrent requests per port (default: 8)")
    parser.add_argument("--start_server", action="store_true",
                        help="Auto-start vLLM server before inference")
    parser.add_argument("--stop_server", action="store_true",
                        help="Stop vLLM server after inference")
    parser.add_argument("--log_dir", default="logs/vllm_e2e",
                        help="Directory for vLLM server logs")
    parser.add_argument("--think", action="store_true",
                        help="Enable thinking mode (default: OFF; opt-in to allow <think>...</think>)")
    parser.add_argument("--presence_penalty", type=float, default=0.0,
                        help="Presence penalty for primary inference (default: 0.0, pure greedy)")
    parser.add_argument("--repetition_penalty", type=float, default=1.0,
                        help="vLLM repetition penalty passed via extra_body (default: 1.0)")
    parser.add_argument("--max_tokens", type=int, default=16384,
                        help="Max output tokens per image (default: 16384; lower it if input plus output exceeds context)")
    parser.add_argument("--request_timeout", type=int, default=600,
                        help="Per-request timeout in seconds (default: 600)")
    parser.add_argument("--retry-repetitive", action="store_true",
                        help="Retry suspicious repetitive outputs with a different presence penalty")
    parser.add_argument("--retry-presence-penalty", type=float, default=1.5,
                        help="Presence penalty used only for repetition retries (default: 1.5)")
    parser.add_argument("--retry-same-character-run", type=int, default=128,
                        help="Minimum consecutive non-whitespace character run for retry (default: 128)")
    parser.add_argument("--retry-min-repeats", type=int, default=8,
                        help="Minimum repeated suffix or line count for retry (default: 8)")
    parser.add_argument("--retry-min-repeated-chars", type=int, default=512,
                        help="Minimum characters covered by a repetition pattern (default: 512)")
    parser.add_argument("--retry-min-ratio", type=float, default=0.5,
                        help="Minimum output fraction covered by repetition (default: 0.5)")
    parser.add_argument("--retry-max-unit-chars", type=int, default=256,
                        help="Maximum repeated suffix unit length to inspect (default: 256)")
    args = parser.parse_args()
    if args.retry_same_character_run < 1:
        raise ValueError("--retry-same-character-run must be >= 1")
    if args.retry_min_repeats < 2:
        raise ValueError("--retry-min-repeats must be >= 2")
    if args.retry_min_repeated_chars < 1:
        raise ValueError("--retry-min-repeated-chars must be >= 1")
    if not 0.0 < args.retry_min_ratio <= 1.0:
        raise ValueError("--retry-min-ratio must be in (0, 1]")
    if args.retry_max_unit_chars < 1:
        raise ValueError("--retry-max-unit-chars must be >= 1")
    return args


def run_file_mode(args, clients):
    """Document mode: whole page → markdown. Write individual output files.

    与分区版 (twostage_img2md.py) 一致的重复崩塌重试机制:
    首选推理用 args.presence_penalty (默认 0.0 纯贪心), 命中 repetition_guard 的
    可疑输出用 args.retry_presence_penalty (默认 1.5) 重试, 重试不再可疑才采用,
    否则保留首选结果。全程写 repetition_retry_report.json。
    """
    from tqdm import tqdm

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    images = load_image_list(args.input_jsonl)
    print(f"Total images: {len(images)}")
    total = len(images)
    ext = TASK_EXT.get(args.task, ".txt")
    prompt = PROMPTS[args.task]

    todo = []
    for img_path in images:
        stem = Path(img_path).stem
        out_path = output_dir / f"{stem}{ext}"
        if out_path.exists() and out_path.stat().st_size > 0:
            continue
        todo.append((img_path, str(out_path), prompt))

    print(f"Already done: {total - len(todo)}, remaining: {len(todo)}")
    print(f"Task: {args.task}")

    def repetition_analysis(result, metadata):
        return analyze_repetition(
            result,
            finish_reason=metadata.get("finish_reason"),
            same_character_run=args.retry_same_character_run,
            min_repeats=args.retry_min_repeats,
            min_repeated_chars=args.retry_min_repeated_chars,
            min_ratio=args.retry_min_ratio,
            max_unit_chars=args.retry_max_unit_chars,
        )

    def write_result(out_path, raw_text):
        # 先截断尾部周期重复 (对初始/重试采用版统一生效), 再做 math/OTSL 后处理
        text = postprocess_e2e_markdown(_clean_truncated_repeats(raw_text),
                                        convert_otsl=True)
        Path(out_path).write_text(text, encoding="utf-8")

    retry_report = {
        "enabled": args.retry_repetitive,
        "config": {
            "primary_presence_penalty": args.presence_penalty,
            "retry_presence_penalty": args.retry_presence_penalty,
            "repetition_penalty": args.repetition_penalty,
            "same_character_run": args.retry_same_character_run,
            "min_repeats": args.retry_min_repeats,
            "min_repeated_chars": args.retry_min_repeated_chars,
            "min_ratio": args.retry_min_ratio,
            "max_unit_chars": args.retry_max_unit_chars,
        },
        "images": [],
    }

    retry_items = []      # [(img_path, out_path)] 待重试
    primary_results = {}  # img_path -> {"result", "metadata", "analysis"}
    retry_success = 0
    retry_adopted = 0
    retry_still_suspicious = 0
    retry_failed = 0

    if not todo:
        print("All images already processed!")
    else:
        done_count = 0
        fail_count = 0
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {}
            for i, (img_path, out_path, prompt) in enumerate(todo):
                client = clients[i % len(clients)]
                f = pool.submit(infer_one, client, args.served_model_name,
                                img_path, prompt,
                                presence_penalty=args.presence_penalty,
                                repetition_penalty=args.repetition_penalty,
                                max_tokens=args.max_tokens,
                                request_timeout=args.request_timeout,
                                no_think=not args.think)
                futures[f] = (img_path, out_path)

            pbar = tqdm(total=len(todo), desc="Inference", unit="img")
            for f, (img_path, out_path) in futures.items():
                try:
                    ok, result, metadata = f.result()
                except Exception as e:
                    ok, result, metadata = False, str(e), {}
                if ok:
                    done_count += 1
                    if args.retry_repetitive:
                        analysis = repetition_analysis(result, metadata)
                        if analysis["suspicious"]:
                            # 可疑: 先不写盘, 等重试决定采用哪版
                            primary_results[img_path] = {
                                "result": result,
                                "metadata": metadata,
                                "analysis": analysis,
                            }
                            retry_items.append((img_path, out_path))
                        else:
                            write_result(out_path, result)
                    else:
                        write_result(out_path, result)
                else:
                    fail_count += 1
                    tqdm.write(f"FAIL: {img_path} -> {result}")
                pbar.update(1)
                pbar.set_postfix(done=done_count, fail=fail_count)
            pbar.close()

        print(f"\nInference complete: done={done_count}, failed={fail_count}")

        # ── Repetition retry ──
        if args.retry_repetitive and retry_items:
            print(
                "  Repetition retry: "
                f"{len(retry_items)} image(s) triggered; "
                f"presence_penalty {args.presence_penalty} -> {args.retry_presence_penalty}"
            )
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                futures = {}
                for i, (img_path, out_path) in enumerate(retry_items):
                    client = clients[i % len(clients)]
                    f = pool.submit(infer_one, client, args.served_model_name,
                                    img_path, prompt,
                                    presence_penalty=args.retry_presence_penalty,
                                    repetition_penalty=args.repetition_penalty,
                                    max_tokens=args.max_tokens,
                                    request_timeout=args.request_timeout,
                                    no_think=not args.think)
                    futures[f] = (img_path, out_path)

                pbar = tqdm(total=len(retry_items), desc="Repetition retry", unit="img")
                for f in as_completed(futures):
                    img_path, out_path = futures[f]
                    primary = primary_results[img_path]
                    try:
                        retry_ok, retry_result, retry_metadata = f.result()
                    except Exception as e:
                        retry_ok, retry_result, retry_metadata = False, str(e), {}

                    retry_analysis = (
                        repetition_analysis(retry_result, retry_metadata)
                        if retry_ok
                        else None
                    )
                    adopted = bool(
                        retry_ok
                        and retry_analysis is not None
                        and not retry_analysis["suspicious"]
                    )
                    if retry_ok:
                        retry_success += 1
                    else:
                        retry_failed += 1
                    if adopted:
                        retry_adopted += 1
                        write_result(out_path, retry_result)
                    else:
                        if retry_ok:
                            retry_still_suspicious += 1
                        write_result(out_path, primary["result"])

                    retry_report["images"].append({
                        "image_path": img_path,
                        "adopted": adopted,
                        "primary": {
                            "metadata": primary["metadata"],
                            "analysis": primary["analysis"],
                        },
                        "retry": {
                            "success": retry_ok,
                            "error": None if retry_ok else retry_result,
                            "metadata": retry_metadata,
                            "analysis": retry_analysis,
                        },
                    })
                    outcome = "ADOPT" if adopted else "KEEP_PRIMARY"
                    retry_reasons = (
                        retry_analysis["reasons"] if retry_analysis is not None else [retry_result]
                    )
                    tqdm.write(
                        f"[RETRY:{outcome}] {img_path} "
                        f"primary={primary['analysis']['reasons']} retry={retry_reasons}"
                    )
                    pbar.update(1)
                pbar.close()

        retry_report["summary"] = {
            "triggered": len(retry_items),
            "retry_success": retry_success,
            "adopted": retry_adopted,
            "retry_still_suspicious": retry_still_suspicious,
            "retry_failed": retry_failed,
        }
        retry_report["images"].sort(key=lambda record: record["image_path"])

    retry_report_path = output_dir / "repetition_retry_report.json"
    retry_report_path.write_text(
        json.dumps(retry_report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if args.retry_repetitive:
        summary = retry_report["summary"]
        print(
            "  Repetition retry done: "
            f"triggered={summary['triggered']}, success={summary['retry_success']}, "
            f"adopted={summary['adopted']}, still_suspicious={summary['retry_still_suspicious']}, "
            f"failed={summary['retry_failed']}"
        )
    print(f"  Repetition retry report: {retry_report_path}")


def main():
    args = parse_args()


    # Start vLLM server if requested
    server_procs = None
    server_logs = None
    ports = [args.port]
    if args.start_server:
        gpu_ids = [int(x) for x in args.gpu_devices.split(",")]
        cfg = build_model_cfg(
            args.model_path, args.served_model_name,
            args.port, EXTRA_ARGS,
        )
        print(f"Starting vLLM server on GPUs {gpu_ids}, port {args.port}...")
        server_procs, server_logs, ports = start_vllm_replicas(
            cfg, gpu_ids, args.gpu_mem_util, args.log_dir,
        )

    try:
        # Create OpenAI clients (one per port for load balancing)
        clients = [
            OpenAI(base_url=f"http://127.0.0.1:{p}/v1", api_key="not-needed",
                   max_retries=2, timeout=300)
            for p in ports
        ]

        run_file_mode(args, clients)

    finally:
        if args.stop_server and server_procs:
            print("Stopping vLLM server...")
            stop_vllm_replicas(server_procs, server_logs)
            print("Server stopped.")


if __name__ == "__main__":
    main()
