#!/usr/bin/env bash
# Install the local MCP client environment. Model weights/runtime are installed separately.
set -euo pipefail

SKILL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_URL="${XIAOMI_OCR_LOCAL_URL:-http://127.0.0.1:8000/v1}"
MODEL="${XIAOMI_OCR_MODEL:-SeerRay-Lab/Xiaomi-OCR-0}"
VENV="$SKILL_DIR/.venv"

if [[ -n "${PYTHON_BIN:-}" ]]; then
  PYTHON_CANDIDATES=("$PYTHON_BIN")
else
  PYTHON_CANDIDATES=(python3 python3.13 python3.12 python3.11 python3.10)
fi
PYTHON_BIN=""
for candidate in "${PYTHON_CANDIDATES[@]}"; do
  if command -v "$candidate" >/dev/null \
      && "$candidate" -c 'import sys; raise SystemExit(sys.version_info < (3, 10))' >/dev/null 2>&1; then
    PYTHON_BIN="$candidate"
    break
  fi
done
if [[ -z "$PYTHON_BIN" ]]; then
  echo "Python 3.10 or newer is required. Set PYTHON_BIN to a compatible interpreter." >&2
  exit 1
fi

if [[ "$LOCAL_URL" != http://127.0.0.1:* && "$LOCAL_URL" != http://localhost:* && "$LOCAL_URL" != http://\[::1\]:* ]]; then
  echo "XIAOMI_OCR_LOCAL_URL must use a loopback address (127.0.0.1, localhost, or ::1)." >&2
  exit 1
fi
if [[ ! -x "$VENV/bin/python" && ! -x "$VENV/Scripts/python.exe" ]]; then
  "$PYTHON_BIN" -m venv "$VENV"
fi
PY="$VENV/bin/python"
[[ -x "$PY" ]] || PY="$VENV/Scripts/python.exe"
if ! "$PY" -c 'import sys; raise SystemExit(sys.version_info < (3, 10))' >/dev/null 2>&1 \
    || ! "$PY" -m pip --version >/dev/null 2>&1; then
  echo "Existing environment at $VENV is incomplete or uses Python older than 3.10; remove it and rerun." >&2
  exit 1
fi
if ! "$PY" -c 'import PIL, pypdfium2, numpy, tqdm, wordfreq, tomlkit' >/dev/null 2>&1; then
  "$PY" -m pip install -r "$SKILL_DIR/requirements.txt"
else
  echo "Reusing existing MCP dependencies in $VENV"
fi

REPO_DIR="${XIAOMI_OCR_REPO:-$(cd "$SKILL_DIR/../.." && pwd)}"
if [[ ! -f "$REPO_DIR/postprocess/document.py" ]]; then
  echo "Use install.sh from a complete Xiaomi-OCR-0 checkout, or set XIAOMI_OCR_REPO to that checkout." >&2
  exit 1
fi
export XIAOMI_OCR_REPO="$REPO_DIR"
"$PY" - "$SKILL_DIR/mcp-config.json" "$PY" "$SKILL_DIR/mcp_ocr_server.py" "$LOCAL_URL" "$MODEL" "$REPO_DIR" <<'PYCONFIG'
import json, sys
from pathlib import Path
output, python, server, url, model, repo = sys.argv[1:]
Path(output).write_text(json.dumps({"mcpServers": {"xiaomi-ocr": {
    "command": python, "args": [server], "env": {
        "XIAOMI_OCR_LOCAL_URL": url, "XIAOMI_OCR_MODEL": model, "XIAOMI_OCR_REPO": repo
    }}}}, indent=2) + "\n")
PYCONFIG

echo "MCP environment ready: $VENV"
echo "Config written to: $SKILL_DIR/mcp-config.json"
if [[ "${1:-}" == "--agent=codex" ]]; then
  "$PY" "$SKILL_DIR/scripts/register_codex_mcp.py"
else
  echo "Merge mcp-config.json into your current agent settings; Codex auto-registration requires --agent=codex."
fi
echo "Start SGLang or vLLM locally before testing OCR. Region mode also needs local PaddleX/PaddlePaddle."
