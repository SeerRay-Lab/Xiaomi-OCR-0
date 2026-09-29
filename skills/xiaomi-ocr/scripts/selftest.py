#!/usr/bin/env python3
"""Check the MCP stdio handshake and advertised task tools without running inference."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

SERVER = Path(__file__).resolve().parents[1] / "mcp_ocr_server.py"
messages = [
    {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2024-11-05"}},
    {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
]
proc = subprocess.run([sys.executable, str(SERVER)], input="\n".join(map(json.dumps, messages)) + "\n",
                      text=True, capture_output=True, timeout=15, check=True)
responses = [json.loads(line) for line in proc.stdout.splitlines() if line.strip()]
if len(responses) != 2:
    raise SystemExit(f"Expected two MCP responses, got {len(responses)}; stderr={proc.stderr}")
tools = {tool["name"] for tool in responses[1]["result"]["tools"]}
expected = {"ocr_image", "ocr_pdf", "kie_image", "vqa_image"}
if tools != expected:
    raise SystemExit(f"Unexpected MCP tool list: {sorted(tools)}")
print("MCP handshake OK; OCR, PDF, KIE, and VQA tools are registered.")
print("Start local SGLang/vLLM to run an inference smoke test.")
