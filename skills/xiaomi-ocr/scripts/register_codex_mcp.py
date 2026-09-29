#!/usr/bin/env python3
"""Register the installed Xiaomi-OCR MCP server in the current user's Codex config."""
from __future__ import annotations

import json
import os
import re
import sys
import tempfile
from pathlib import Path


def main() -> int:
    skill_dir = Path(__file__).resolve().parents[1]
    python = skill_dir / ".venv" / "bin" / "python"
    if not python.exists():
        python = skill_dir / ".venv" / "Scripts" / "python.exe"
    if not python.exists():
        print("MCP environment is missing; run install.sh first.", file=sys.stderr)
        return 1

    codex_home = Path(os.environ.get("CODEX_HOME", Path.home() / ".codex")).expanduser()
    config = codex_home / "config.toml"
    # Do not create or modify a Codex config on a machine where Codex is not installed.
    if "CODEX_HOME" not in os.environ and not codex_home.exists():
        print("Codex config not detected; use mcp-config.json with this agent's MCP settings.")
        return 0

    endpoint = os.environ.get("XIAOMI_OCR_LOCAL_URL", "http://127.0.0.1:8000/v1")
    model = os.environ.get("XIAOMI_OCR_MODEL", "SeerRay-Lab/Xiaomi-OCR-0")
    if not endpoint.startswith(("http://127.0.0.1:", "http://localhost:", "http://[::1]:")):
        print("Refusing non-loopback OCR endpoint.", file=sys.stderr)
        return 1

    block = "\n".join(
        [
            "[mcp_servers.xiaomi-ocr]",
            f"command = {json.dumps(str(python), ensure_ascii=False)}",
            f"args = [{json.dumps(str(skill_dir / 'mcp_ocr_server.py'), ensure_ascii=False)}]",
            "env = { "
            + f"XIAOMI_OCR_LOCAL_URL = {json.dumps(endpoint)}, "
            + f"XIAOMI_OCR_MODEL = {json.dumps(model, ensure_ascii=False)} "
            + "}",
        ]
    )
    config.parent.mkdir(parents=True, exist_ok=True)
    original = config.read_text(encoding="utf-8") if config.exists() else ""
    marker = re.compile(r"(?m)^\[mcp_servers\.xiaomi-ocr\]\s*$")
    match = marker.search(original)
    if match:
        next_table = re.search(r"(?m)^\[", original[match.end() :])
        end = match.end() + next_table.start() if next_table else len(original)
        updated = original[: match.start()] + block + "\n" + original[end:].lstrip("\n")
    else:
        updated = original.rstrip() + ("\n\n" if original.strip() else "") + block + "\n"

    config.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix="config.toml.", dir=config.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(updated)
        if config.exists():
            os.chmod(temp_name, config.stat().st_mode & 0o777)
        os.replace(temp_name, config)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)

    print(f"Registered Xiaomi-OCR MCP server in {config}.")
    print("Restart Codex to load the four OCR, PDF, KIE, and VQA tools.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
