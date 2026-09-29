#!/usr/bin/env python3
"""
Image → layout detect → VLM recognize → markdown.

Pipeline:
  Stage 1: PP-DocLayoutV3 layout detection (single-GPU batch inference)
  Stage 2: per-region OCR via vLLM (crop regions → task-specific prompt)
  Stage 3: merge_page_to_markdown → per-page .md files

Usage:
    python pipeline/twostage_img2md.py \
        --input-jsonl /path/to/images.jsonl \
        --output_dir /path/to/output \
        --start_server --stop_server

    python pipeline/twostage_img2md.py \
        --input-jsonl /path/to/images.jsonl \
        --layout-jsonl /path/to/layout.jsonl \
        --output_dir /path/to/output \
        --vlm_gpus 0,1,2,3 \
        --start_server --stop_server
"""

import argparse
import base64
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm

try:
    from wordfreq import zipf_frequency
except Exception:
    zipf_frequency = None

from openai import OpenAI
from postprocess.repetition_guard import analyze_repetition
from postprocess.otsl import convert_otsl_to_html

# ═══════════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════════

LAYOUT_MODEL_DIR = None
LAYOUT_THRESHOLD = 0.3
LAYOUT_NMS = True
LAYOUT_UNCLIP_RATIO = [1.0, 1.0]
LAYOUT_MERGE_BBOXES_MODE = {
    0: "large", 1: "large", 2: "large", 3: "large", 4: "large",
    5: "large", 6: "large", 7: "large", 8: "large", 9: "large",
    10: "large", 11: "large", 12: "large", 13: "large", 14: "large",
    15: "large", 16: "large", 17: "large", 18: "small", 19: "large",
    20: "large", 21: "large", 22: "large", 23: "large", 24: "large",
}

ID2LABEL = {
    0: "abstract", 1: "algorithm", 2: "aside_text", 3: "chart",
    4: "content", 5: "display_formula", 6: "doc_title", 7: "figure_title",
    8: "footer", 9: "footer_image", 10: "footnote", 11: "formula_number",
    12: "header", 13: "header_image", 14: "image", 15: "inline_formula",
    16: "number", 17: "paragraph_title", 18: "reference",
    19: "reference_content", 20: "seal", 21: "table", 22: "text",
    23: "vertical_text", 24: "vision_footnote",
}

LABEL_TASK_MAPPING = {
    "text": [
        "abstract", "algorithm", "content", "doc_title", "figure_title",
        "paragraph_title", "reference_content", "text", "vertical_text",
        "vision_footnote", "seal", "formula_number", "header", "footer",
        "number", "footnote", "aside_text", "reference",
    ],
    "table": ["table"],
    "formula": ["display_formula", "inline_formula"],
    "skip": ["chart", "image", "footer_image", "header_image"],
    "abandon": [],
}

PROMPTS = {
    "text": "Extract the text in the image.",
    "table": "Parse the table in the image into OTSL.",
    "formula": "Identify the formula in the image and represent it using LATEX format.",
    "document": (
        "Extract all information from the main body of the document image and represent it "
        "in markdown format, ignoring headers and footers. Tables should be expressed in OTSL "
        "format, formulas in the document should be represented using LATEX format, and the "
        "parsing should be organized according to the reading order."
    ),
}

VLM_MODEL_PATH = "SeerRay-Lab/Xiaomi-OCR-0"
VLM_SERVED_NAME = "SeerRay-Lab/Xiaomi-OCR-0"
VLM_BASE_PORT = 8000
VLM_EXTRA_ARGS = ["--trust-remote-code"]


# ═══════════════════════════════════════════════════════════════════════════════
# Layout post-processing
# ═══════════════════════════════════════════════════════════════════════════════

def _iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1_p, y1_p, x2_p, y2_p = box2
    x1_i = max(x1, x1_p)
    y1_i = max(y1, y1_p)
    x2_i = min(x2, x2_p)
    y2_i = min(y2, y2_p)
    inter_area = max(0, x2_i - x1_i + 1) * max(0, y2_i - y1_i + 1)
    box1_area = (x2 - x1 + 1) * (y2 - y1 + 1)
    box2_area = (x2_p - x1_p + 1) * (y2_p - y1_p + 1)
    return inter_area / float(box1_area + box2_area - inter_area)


def _nms(boxes, iou_same=0.6, iou_diff=0.95):
    scores = boxes[:, 1]
    indices = np.argsort(scores)[::-1]
    selected_boxes = []
    while len(indices) > 0:
        current = indices[0]
        current_box = boxes[current]
        current_class = current_box[0]
        current_coords = current_box[2:]
        selected_boxes.append(current)
        indices = indices[1:]
        filtered_indices = []
        for i in indices:
            box = boxes[i]
            box_class = box[0]
            box_coords = box[2:]
            iou_value = _iou(current_coords, box_coords)
            threshold = iou_same if current_class == box_class else iou_diff
            if iou_value < threshold:
                filtered_indices.append(i)
        indices = filtered_indices
    return selected_boxes


def _is_contained(box1, box2):
    _, _, x1, y1, x2, y2 = box1
    _, _, x1_p, y1_p, x2_p, y2_p = box2
    box1_area = (x2 - x1) * (y2 - y1)
    xi1 = max(x1, x1_p)
    yi1 = max(y1, y1_p)
    xi2 = min(x2, x2_p)
    yi2 = min(y2, y2_p)
    inter_width = max(0, xi2 - xi1)
    inter_height = max(0, yi2 - yi1)
    intersect_area = inter_width * inter_height
    iou_val = intersect_area / box1_area if box1_area > 0 else 0
    return iou_val >= 0.8


def _check_containment(boxes, preserve_indices=None, category_index=None, mode=None):
    n = len(boxes)
    contains_other = np.zeros(n, dtype=int)
    contained_by_other = np.zeros(n, dtype=int)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if preserve_indices is not None and boxes[i][0] in preserve_indices:
                continue
            if category_index is not None and mode is not None:
                if mode == "large" and boxes[j][0] == category_index:
                    if _is_contained(boxes[i], boxes[j]):
                        contained_by_other[i] = 1
                        contains_other[j] = 1
                if mode == "small" and boxes[i][0] == category_index:
                    if _is_contained(boxes[i], boxes[j]):
                        contained_by_other[i] = 1
                        contains_other[j] = 1
            else:
                if _is_contained(boxes[i], boxes[j]):
                    contained_by_other[i] = 1
                    contains_other[j] = 1
    return contains_other, contained_by_other


def _unclip_boxes(boxes, unclip_ratio=None):
    if unclip_ratio is None:
        return boxes
    if isinstance(unclip_ratio, dict):
        expanded_boxes = []
        for box in boxes:
            class_id, score, x1, y1, x2, y2 = box[:6]
            if class_id in unclip_ratio:
                width_ratio, height_ratio = unclip_ratio[class_id]
                width = x2 - x1
                height = y2 - y1
                new_w = width * width_ratio
                new_h = height * height_ratio
                center_x = x1 + width / 2
                center_y = y1 + height / 2
                new_x1 = center_x - new_w / 2
                new_y1 = center_y - new_h / 2
                new_x2 = center_x + new_w / 2
                new_y2 = center_y + new_h / 2
                expanded_box = [class_id, score, new_x1, new_y1, new_x2, new_y2]
                if len(box) > 6:
                    expanded_box.extend(box[6:])
                expanded_boxes.append(expanded_box)
            else:
                expanded_boxes.append(box)
        return np.array(expanded_boxes)
    else:
        widths = boxes[:, 4] - boxes[:, 2]
        heights = boxes[:, 5] - boxes[:, 3]
        new_w = widths * unclip_ratio[0]
        new_h = heights * unclip_ratio[1]
        center_x = boxes[:, 2] + widths / 2
        center_y = boxes[:, 3] + heights / 2
        new_x1 = center_x - new_w / 2
        new_y1 = center_y - new_h / 2
        new_x2 = center_x + new_w / 2
        new_y2 = center_y + new_h / 2
        expanded_boxes = np.column_stack(
            (boxes[:, 0], boxes[:, 1], new_x1, new_y1, new_x2, new_y2)
        )
        if boxes.shape[1] > 6:
            expanded_boxes = np.column_stack((expanded_boxes, boxes[:, 6:]))
        return expanded_boxes


def apply_layout_postprocess(
    raw_results, id2label, img_sizes,
    layout_nms=True, layout_unclip_ratio=None, layout_merge_bboxes_mode=None,
):
    all_labels = list(id2label.values())
    paddle_format_results = []
    for img_idx, result in enumerate(raw_results):
        scores = result["scores"].cpu().numpy()
        labels = result["labels"].cpu().numpy()
        boxes = result["boxes"].cpu().numpy()
        order_seq = result["order_seq"].cpu().numpy()
        polygon_points = result.get("polygon_points", [])
        img_size = img_sizes[img_idx]
        boxes_with_order = []
        for i in range(len(scores)):
            cls_id = int(labels[i])
            score = float(scores[i])
            x1, y1, x2, y2 = boxes[i]
            order = int(order_seq[i])
            boxes_with_order.append([cls_id, score, x1, y1, x2, y2, order])
        if len(boxes_with_order) == 0:
            paddle_format_results.append([])
            continue
        boxes_array = np.array(boxes_with_order)
        if layout_nms:
            selected_indices = _nms(boxes_array[:, :6], iou_same=0.6, iou_diff=0.98)
            boxes_array = boxes_array[selected_indices]
        filter_large_image = True
        if filter_large_image and len(boxes_array) > 1:
            area_thres = 0.82 if img_size[0] > img_size[1] else 0.93
            image_index = all_labels.index("image") if "image" in all_labels else None
            img_area = img_size[0] * img_size[1]
            filtered_boxes = []
            for box in boxes_array:
                label_index, score_val, xmin, ymin, xmax, ymax = box[:6]
                if label_index == image_index:
                    xmin = max(0, xmin)
                    ymin = max(0, ymin)
                    xmax = min(img_size[0], xmax)
                    ymax = min(img_size[1], ymax)
                    box_area = (xmax - xmin) * (ymax - ymin)
                    if box_area <= area_thres * img_area:
                        filtered_boxes.append(box)
                else:
                    filtered_boxes.append(box)
            if len(filtered_boxes) > 0:
                boxes_array = np.array(filtered_boxes)
        if layout_merge_bboxes_mode:
            preserve_labels = ["image", "seal", "chart"]
            preserve_indices = set()
            for label in preserve_labels:
                if label in all_labels:
                    preserve_indices.add(all_labels.index(label))
            if isinstance(layout_merge_bboxes_mode, str):
                if layout_merge_bboxes_mode == "union":
                    pass
                else:
                    contains_other, contained_by_other = _check_containment(
                        boxes_array[:, :6], preserve_indices
                    )
                    if layout_merge_bboxes_mode == "large":
                        boxes_array = boxes_array[contained_by_other == 0]
                    elif layout_merge_bboxes_mode == "small":
                        boxes_array = boxes_array[
                            (contains_other == 0) | (contained_by_other == 1)
                        ]
            elif isinstance(layout_merge_bboxes_mode, dict):
                keep_mask = np.ones(len(boxes_array), dtype=bool)
                for category_index, layout_mode in layout_merge_bboxes_mode.items():
                    if layout_mode == "union":
                        pass
                    else:
                        if layout_mode == "large":
                            co, cbo = _check_containment(
                                boxes_array[:, :6], preserve_indices,
                                category_index, mode=layout_mode,
                            )
                            keep_mask &= cbo == 0
                        elif layout_mode == "small":
                            co, cbo = _check_containment(
                                boxes_array[:, :6], preserve_indices,
                                category_index, mode=layout_mode,
                            )
                            keep_mask &= (co == 0) | (cbo == 1)
                boxes_array = boxes_array[keep_mask]
        if len(boxes_array) == 0:
            paddle_format_results.append([])
            continue
        sorted_idx = np.argsort(boxes_array[:, 6])
        boxes_array = boxes_array[sorted_idx]
        if layout_unclip_ratio:
            if isinstance(layout_unclip_ratio, float):
                layout_unclip_ratio = (layout_unclip_ratio, layout_unclip_ratio)
            boxes_array = _unclip_boxes(boxes_array, layout_unclip_ratio)
        img_width, img_height = img_size
        image_results = []
        for i, box_data in enumerate(boxes_array):
            cls_id = int(box_data[0])
            score_val = float(box_data[1])
            x1, y1, x2, y2 = box_data[2:6]
            order = int(box_data[6]) if box_data[6] > 0 else None
            label_name = id2label.get(cls_id, f"class_{cls_id}")
            x1 = max(0, min(float(x1), img_width))
            y1 = max(0, min(float(y1), img_height))
            x2 = max(0, min(float(x2), img_width))
            y2 = max(0, min(float(y2), img_height))
            if x1 >= x2 or y1 >= y2:
                continue
            image_results.append({
                "cls_id": cls_id, "label": label_name, "score": score_val,
                "coordinate": [int(x1), int(y1), int(x2), int(y2)],
                "order": order,
            })
        paddle_format_results.append(image_results)
    return paddle_format_results


# ═══════════════════════════════════════════════════════════════════════════════
# Merge to markdown
# ═══════════════════════════════════════════════════════════════════════════════

from postprocess.region import merge_page_to_markdown


def load_layout_model(device="cuda:0", model_dir=LAYOUT_MODEL_DIR):
    if not model_dir:
        raise ValueError("Pass --layout_model_dir with a local Transformers PP-DocLayoutV3 directory, or use --layout-jsonl from PaddleX.")
    import torch
    from transformers import PPDocLayoutV3ForObjectDetection, PPDocLayoutV3ImageProcessor
    print(f"  Loading layout model on {device}...")
    processor = PPDocLayoutV3ImageProcessor.from_pretrained(model_dir)
    model = PPDocLayoutV3ForObjectDetection.from_pretrained(model_dir)
    model.eval()
    model = model.to(device)
    print(f"  Layout model ready on {device}")
    return processor, model


def run_layout_detection(images, processor, model, device="cuda:0", threshold=LAYOUT_THRESHOLD):
    import torch
    if not images:
        return []
    inputs = processor(images=images, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    target_sizes = torch.tensor([img.size[::-1] for img in images], device=device)
    raw_results = processor.post_process_object_detection(
        outputs, threshold=threshold, target_sizes=target_sizes,
    )
    img_sizes = [img.size for img in images]
    paddle_results = apply_layout_postprocess(
        raw_results=raw_results, id2label=ID2LABEL, img_sizes=img_sizes,
        layout_nms=LAYOUT_NMS, layout_unclip_ratio=LAYOUT_UNCLIP_RATIO,
        layout_merge_bboxes_mode=LAYOUT_MERGE_BBOXES_MODE,
    )
    all_layouts = []
    for img_idx, paddle_res in enumerate(paddle_results):
        image_width, image_height = images[img_idx].size
        results = []
        valid_index = 0
        for det in paddle_res:
            label = det["label"]
            score_val = det["score"]
            box = det["coordinate"]
            task_type = None
            for task_item, labels in LABEL_TASK_MAPPING.items():
                if isinstance(labels, list) and label in labels:
                    task_type = task_item
                    break
            if task_type is None or task_type == "abandon":
                continue
            x1, y1, x2, y2 = box
            x1_norm = int(float(x1) / image_width * 1000)
            y1_norm = int(float(y1) / image_height * 1000)
            x2_norm = int(float(x2) / image_width * 1000)
            y2_norm = int(float(y2) / image_height * 1000)
            results.append({
                "index": valid_index, "label": label,
                "score": float(score_val),
                "bbox_2d": [x1_norm, y1_norm, x2_norm, y2_norm],
                "task_type": task_type,
            })
            valid_index += 1
        all_layouts.append(results)
    del inputs, outputs, raw_results
    torch.cuda.empty_cache()
    return all_layouts


from pipeline.cropping import crop_region


def encode_image_base64(image, fmt="PNG"):
    buf = BytesIO()
    image.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def wait_for_service(port, proc, timeout=1200, interval=5):
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


def start_vllm_server(model_path, served_name, port, gpu_ids, gpu_mem_util=0.8,
                      extra_args=None, log_dir="logs/vllm_twostage"):
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    extra_args = extra_args or []
    processes, log_handles, ports = [], [], []
    print(f"  Starting {len(gpu_ids)} vLLM replica(s) for {served_name}...")
    for i, gpu_id in enumerate(gpu_ids):
        port_i = port + i
        ports.append(port_i)
        cmd = [
            "vllm", "serve", model_path,
            "--served-model-name", served_name,
            "--port", str(port_i),
            "--tensor-parallel-size", "1",
            "--gpu-memory-utilization", str(gpu_mem_util),
            "--uvicorn-log-level", "warning",
        ] + extra_args
        log_path = log_dir / f"vllm_{served_name}_gpu{gpu_id}_port{port_i}.log"
        log_fh = open(log_path, "w")
        log_handles.append(log_fh)
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        env["OMP_NUM_THREADS"] = "8"
        env["MKL_NUM_THREADS"] = "8"
        print(f"    [{i+1}/{len(gpu_ids)}] GPU {gpu_id} | port {port_i}")
        proc = subprocess.Popen(
            cmd, env=env, stdout=log_fh, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        processes.append(proc)
        time.sleep(6)
    print(f"  Waiting for replicas to be ready...")
    results = {}

    def _wait(idx, port, proc):
        ok, elapsed = wait_for_service(port, proc)
        results[idx] = (ok, elapsed)

    threads = []
    for idx, (port_i, proc) in enumerate(zip(ports, processes)):
        t = threading.Thread(target=_wait, args=(idx, port_i, proc))
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
        stop_vllm_replicas(processes, log_handles)
        raise RuntimeError(f"vLLM replicas failed:\n" + "\n".join(f"  - {f}" for f in failed))
    print(f"  All replicas ready!")
    return processes, log_handles, ports


def stop_vllm_replicas(processes, log_handles):
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
            except (ProcessLookupError, OSError, PermissionError, subprocess.TimeoutExpired):
                pass
    for fh in log_handles:
        try:
            fh.close()
        except Exception:
            pass


def infer_region(client, model_name, image_path, bbox_2d, prompt,
                 temperature=0.0, top_p=0.8, presence_penalty=0.0,
                 repetition_penalty=1.0, max_tokens=16384,
                 request_timeout=600, no_think=True, polygon=None):
    try:
        img = Image.open(image_path).convert("RGB")
        region_img = crop_region(img, bbox_2d, polygon=polygon)
        img.close()
        b64 = encode_image_base64(region_img)
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


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Two-stage: image → layout → VLM → markdown"
    )
    parser.add_argument("--input-jsonl", "--input_jsonl", type=str, required=True,
                        help="Path to JSONL file: {\"images\": [\"/path/to/img.png\"]}")
    parser.add_argument("--output_dir", required=True,
                        help="Directory to write .md output files")
    parser.add_argument("--model_path", default=VLM_MODEL_PATH,
                        help=f"VLM model path (default: {VLM_MODEL_PATH})")
    parser.add_argument("--served_model_name", default=VLM_SERVED_NAME,
                        help=f"Served model name (default: {VLM_SERVED_NAME})")
    parser.add_argument("--port", type=int, default=VLM_BASE_PORT,
                        help=f"vLLM server base port (default: {VLM_BASE_PORT})")
    parser.add_argument("--layout_gpus", default="0",
                        help="GPU IDs for layout detection (comma-separated)")
    parser.add_argument("--vlm_gpus", default="0",
                        help="GPU IDs for vLLM server (comma-separated)")
    parser.add_argument("--gpu_mem_util", type=float, default=0.8,
                        help="GPU memory utilization for vLLM (default: 0.9)")
    parser.add_argument("--workers", type=int, default=8,
                        help="Max concurrent VLM requests per port (default: 8)")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Layout detection batch size (default: 16)")
    parser.add_argument("--start_server", action="store_true",
                        help="Auto-start vLLM server before inference")
    parser.add_argument("--stop_server", action="store_true",
                        help="Stop vLLM server after inference")
    parser.add_argument("--log_dir", default="logs/vllm_twostage",
                        help="Directory for vLLM server logs")
    parser.add_argument("--layout_model_dir", default=LAYOUT_MODEL_DIR,
                        help="PP-DocLayoutV3 model directory")
    parser.add_argument("--layout_threshold", type=float, default=LAYOUT_THRESHOLD,
                        help="Layout detection confidence threshold")
    parser.add_argument("--layout-jsonl", default=None,
                        help="Pre-computed layout JSONL (skip layout detection)")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature (default: 0.0, deterministic)")
    parser.add_argument("--top_p", type=float, default=0.8,
                        help="Top-p nucleus sampling (default: 0.8)")
    parser.add_argument("--presence_penalty", type=float, default=0.0,
                        help="Presence penalty (default: 0.0, pure greedy)")
    parser.add_argument("--repetition_penalty", type=float, default=1.0,
                        help="vLLM repetition penalty passed via extra_body (default: 1.0)")
    parser.add_argument("--retry-repetitive", action="store_true",
                        help="Retry suspicious repetitive regions with a different presence penalty")
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
    parser.add_argument("--max_tokens", type=int, default=16384,
                        help="Max output tokens per region (default: 16384)")
    parser.add_argument("--request_timeout", type=int, default=600,
                        help="Per-request timeout in seconds (default: 600)")
    parser.add_argument("--think", action="store_true",
                        help="Enable thinking mode (default: OFF; opt-in to allow <think>...</think>)")
    parser.add_argument("--polygon", action="store_true", default=False,
                        help="多边形裁切: 用 region['polygon'] 做 mask, 灰色(128)填充外框, "
                             "按 polygon 外接矩形裁切。polygon 为空或 <=4 顶点时退化为矩形裁切")
    return parser.parse_args()


def main():
    args = parse_args()
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
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load images ──
    image_paths = load_image_list(args.input_jsonl)
    print(f"Found {len(image_paths)} images in {args.input_jsonl}")
    if not image_paths:
        print("No images found!")
        return

    # ── Stage 1: Layout Detection ──
    print(f"\n{'='*60}")
    print(f"  Stage 1: Layout Detection")
    print(f"{'='*60}")

    all_data = []
    if args.layout_jsonl:
        # Load pre-computed layout results from JSONL
        print(f"  Loading pre-computed layout from: {args.layout_jsonl}")
        layout_by_path = {}
        with open(args.layout_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                for img_path in rec.get("images", []):
                    layout_by_path[img_path] = rec.get("layout_result", [])
        for path in image_paths:
            if path in layout_by_path:
                all_data.append({"image_path": path, "layout_result": layout_by_path[path]})
        total_regions = sum(len(d["layout_result"]) for d in all_data)
        print(f"  Layout loaded: {len(all_data)} images, {total_regions} regions")
    else:
        layout_gpu_ids = [int(x.strip()) for x in args.layout_gpus.split(",")]
        layout_device = f"cuda:{layout_gpu_ids[0]}"
        processor, layout_model = load_layout_model(
            device=layout_device, model_dir=args.layout_model_dir,
        )

        for i in tqdm(range(0, len(image_paths), args.batch_size),
                      desc="Layout detection", unit="batch"):
            batch_paths = image_paths[i:i + args.batch_size]
            batch_images, valid_paths = [], []
            for p in batch_paths:
                try:
                    batch_images.append(Image.open(p).convert("RGB"))
                    valid_paths.append(p)
                except Exception as e:
                    print(f"  WARN: Failed to load {p}: {e}")
            if not batch_images:
                continue
            layouts = run_layout_detection(
                batch_images, processor, layout_model,
                device=layout_device, threshold=args.layout_threshold,
            )
            for path, layout in zip(valid_paths, layouts):
                all_data.append({"image_path": path, "layout_result": layout})

        del processor, layout_model
        import torch
        torch.cuda.empty_cache()
        total_regions = sum(len(d["layout_result"]) for d in all_data)
        print(f"  Layout done: {len(all_data)} images, {total_regions} regions")

    # ── Stage 2: VLM Inference ──
    print(f"\n{'='*60}")
    print(f"  Stage 2: VLM Inference ({args.served_model_name})")
    print(f"{'='*60}")

    server_procs, server_logs, ports = None, None, [args.port]
    if args.start_server:
        vlm_gpu_ids = [int(x.strip()) for x in args.vlm_gpus.split(",")]
        server_procs, server_logs, ports = start_vllm_server(
            model_path=args.model_path, served_name=args.served_model_name,
            port=args.port, gpu_ids=vlm_gpu_ids, gpu_mem_util=args.gpu_mem_util,
            extra_args=VLM_EXTRA_ARGS, log_dir=args.log_dir,
        )

    try:
        clients = [
            OpenAI(base_url=f"http://127.0.0.1:{p}/v1", api_key="not-needed",
                   max_retries=2, timeout=300)
            for p in ports
        ]

        region_tasks = []
        for data_idx, data in enumerate(all_data):
            for region_idx, region in enumerate(data["layout_result"]):
                task_type = region.get("task_type", "")
                if task_type == "skip":
                    continue
                prompt = PROMPTS.get(task_type, PROMPTS["text"])
                poly = region.get("polygon") if args.polygon else None
                region_tasks.append(
                    (data_idx, region_idx, data["image_path"], region["bbox_2d"], prompt, poly)
                )

        print(f"  Total regions to recognize: {len(region_tasks)}")

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

        def store_result(data_idx, region_idx, result):
            region = all_data[data_idx]["layout_result"][region_idx]
            if region.get("task_type") == "table":
                try:
                    result = convert_otsl_to_html(result)
                except Exception:
                    pass
            region["merged"] = result

        done_count, fail_count = 0, 0
        retry_tasks = []
        primary_results = {}
        with ThreadPoolExecutor(max_workers=args.workers * len(ports)) as pool:
            futures = {}
            for task_index, task in enumerate(region_tasks):
                didx, ridx, img_path, bbox, prompt, poly = task
                client = clients[task_index % len(clients)]
                f = pool.submit(
                    infer_region, client, args.served_model_name, img_path, bbox, prompt,
                    temperature=args.temperature, top_p=args.top_p,
                    presence_penalty=args.presence_penalty,
                    repetition_penalty=args.repetition_penalty,
                    max_tokens=args.max_tokens,
                    request_timeout=args.request_timeout,
                    no_think=not args.think,
                    polygon=poly,
                )
                futures[f] = task

            pbar = tqdm(total=len(region_tasks), desc="VLM inference", unit="region")
            for f in as_completed(futures):
                task = futures[f]
                didx, ridx, img_path, bbox, prompt, poly = task
                try:
                    ok, result, metadata = f.result()
                except Exception as e:
                    ok, result, metadata = False, str(e), {}
                if ok:
                    done_count += 1
                    if args.retry_repetitive:
                        analysis = repetition_analysis(result, metadata)
                        if analysis["suspicious"]:
                            primary_results[(didx, ridx)] = {
                                "result": result,
                                "metadata": metadata,
                                "analysis": analysis,
                            }
                            retry_tasks.append(task)
                        store_result(didx, ridx, result)
                    else:
                        store_result(didx, ridx, result)
                else:
                    all_data[didx]["layout_result"][ridx]["merged"] = ""
                    fail_count += 1
                    tqdm.write(f"FAIL: {img_path} r{ridx} -> {result}")
                pbar.update(1)
                pbar.set_postfix(done=done_count, fail=fail_count)
            pbar.close()

        primary_all_data = deepcopy(all_data) if args.retry_repetitive else None
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
            "regions": [],
        }
        retry_success = 0
        retry_adopted = 0
        retry_still_suspicious = 0
        retry_failed = 0

        if args.retry_repetitive:
            print(
                "  Repetition retry: "
                f"{len(retry_tasks)} region(s) triggered; "
                f"presence_penalty {args.presence_penalty} -> {args.retry_presence_penalty}"
            )
            with ThreadPoolExecutor(max_workers=args.workers * len(ports)) as pool:
                futures = {}
                for task_index, task in enumerate(retry_tasks):
                    didx, ridx, img_path, bbox, prompt, poly = task
                    client = clients[task_index % len(clients)]
                    future = pool.submit(
                        infer_region,
                        client,
                        args.served_model_name,
                        img_path,
                        bbox,
                        prompt,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        presence_penalty=args.retry_presence_penalty,
                        repetition_penalty=args.repetition_penalty,
                        max_tokens=args.max_tokens,
                        request_timeout=args.request_timeout,
                        no_think=not args.think,
                        polygon=poly,
                    )
                    futures[future] = task

                pbar = tqdm(total=len(retry_tasks), desc="Repetition retry", unit="region")
                for future in as_completed(futures):
                    task = futures[future]
                    didx, ridx, img_path, bbox, prompt, poly = task
                    primary = primary_results[(didx, ridx)]
                    try:
                        retry_ok, retry_result, retry_metadata = future.result()
                    except Exception as error:
                        retry_ok, retry_result, retry_metadata = False, str(error), {}

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
                        store_result(didx, ridx, retry_result)
                    else:
                        if retry_ok:
                            retry_still_suspicious += 1
                        store_result(didx, ridx, primary["result"])

                    region = all_data[didx]["layout_result"][ridx]
                    retry_record = {
                        "image_path": img_path,
                        "region_index": ridx,
                        "task_type": region.get("task_type"),
                        "bbox_2d": bbox,
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
                    }
                    region["vlm_repetition_retry"] = {
                        "adopted": adopted,
                        "primary_reasons": primary["analysis"]["reasons"],
                        "primary_finish_reason": primary["metadata"].get("finish_reason"),
                        "retry_success": retry_ok,
                        "retry_reasons": (
                            retry_analysis["reasons"] if retry_analysis is not None else []
                        ),
                        "retry_finish_reason": retry_metadata.get("finish_reason"),
                    }
                    retry_report["regions"].append(retry_record)
                    outcome = "ADOPT" if adopted else "KEEP_PRIMARY"
                    retry_reasons = (
                        retry_analysis["reasons"] if retry_analysis is not None else [retry_result]
                    )
                    tqdm.write(
                        f"[RETRY:{outcome}] {img_path} r{ridx} "
                        f"primary={primary['analysis']['reasons']} retry={retry_reasons}"
                    )
                    pbar.update(1)
                pbar.close()

        retry_report["summary"] = {
            "triggered": len(retry_tasks),
            "retry_success": retry_success,
            "adopted": retry_adopted,
            "retry_still_suspicious": retry_still_suspicious,
            "retry_failed": retry_failed,
        }
        retry_report["regions"].sort(
            key=lambda record: (record["image_path"], record["region_index"])
        )
        retry_report_path = output_dir / "repetition_retry_report.json"
        retry_report_path.write_text(
            json.dumps(retry_report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(
            "  Repetition retry done: "
            f"triggered={len(retry_tasks)}, success={retry_success}, "
            f"adopted={retry_adopted}, still_suspicious={retry_still_suspicious}, "
            f"failed={retry_failed}"
        )
        print(f"  Repetition retry report: {retry_report_path}")
        print(f"  VLM done: {done_count} success, {fail_count} failures")

    finally:
        if args.stop_server and server_procs:
            print("  Stopping vLLM server...")
            stop_vllm_replicas(server_procs, server_logs)
            print("  Server stopped.")

    # ── Stage 3: Merge to Markdown ──
    print(f"\n{'='*60}")
    print(f"  Stage 3: Merge to Markdown")
    print(f"{'='*60}")

    md_count = 0
    for data in tqdm(all_data, desc="Merging to markdown", unit="page"):
        md = merge_page_to_markdown(data["layout_result"]) or ""
        stem = Path(data["image_path"]).stem
        (output_dir / f"{stem}.md").write_text(md, encoding="utf-8")
        md_count += 1

    print(f"\n  Done: {md_count} markdown files, 0 skipped")
    print(f"  Output: {output_dir}")

    jsonl_path = output_dir / "results.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for data in all_data:
            md = merge_page_to_markdown(data["layout_result"])
            out = {
                "image_path": data["image_path"],
                "layout_result": data["layout_result"],
                "markdown_result": md,
            }
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
    print(f"  Debug JSONL: {jsonl_path}")

    if primary_all_data is not None:
        primary_jsonl_path = output_dir / "primary_results.jsonl"
        with open(primary_jsonl_path, "w", encoding="utf-8") as f:
            for data in primary_all_data:
                md = merge_page_to_markdown(data["layout_result"])
                out = {
                    "image_path": data["image_path"],
                    "layout_result": data["layout_result"],
                    "markdown_result": md,
                }
                f.write(json.dumps(out, ensure_ascii=False) + "\n")
        print(f"  Primary control JSONL: {primary_jsonl_path}")


if __name__ == "__main__":
    main()
