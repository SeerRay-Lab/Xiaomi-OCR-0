#!/usr/bin/env bash
# Install the local MCP client environment. Model weights/runtime are installed separately.
set -euo pipefail

SKILL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
LOCAL_URL="${XIAOMI_OCR_LOCAL_URL:-http://127.0.0.1:8000/v1}"
MODEL="${XIAOMI_OCR_MODEL:-SeerRay-Lab/Xiaomi-OCR-0}"
VENV="$SKILL_DIR/.venv"

if [[ "$LOCAL_URL" != http://127.0.0.1:* && "$LOCAL_URL" != http://localhost:* && "$LOCAL_URL" != http://\[::1\]:* ]]; then
  echo "XIAOMI_OCR_LOCAL_URL must use a loopback address (127.0.0.1, localhost, or ::1)." >&2
  exit 1
fi
command -v "$PYTHON_BIN" >/dev/null || { echo "Python 3 is required." >&2; exit 1; }
if [[ ! -x "$VENV/bin/python" && ! -x "$VENV/Scripts/python.exe" ]]; then
  "$PYTHON_BIN" -m venv "$VENV"
fi
PY="$VENV/bin/python"
[[ -x "$PY" ]] || PY="$VENV/Scripts/python.exe"
if ! "$PY" -c 'import PIL, pypdfium2' >/dev/null 2>&1; then
  "$PY" -m pip install -r "$SKILL_DIR/requirements.txt"
else
  echo "Reusing existing MCP dependencies in $VENV"
fi

cat > "$SKILL_DIR/mcp-config.json" <<JSON
{
  "mcpServers": {
    "xiaomi-ocr": {
      "command": "$PY",
      "args": ["$SKILL_DIR/mcp_ocr_server.py"],
      "env": {
        "XIAOMI_OCR_LOCAL_URL": "$LOCAL_URL",
        "XIAOMI_OCR_MODEL": "$MODEL"
      }
    }
  }
}
JSON

echo "MCP environment ready: $VENV"
echo "Config written to: $SKILL_DIR/mcp-config.json"
"$PY" "$SKILL_DIR/scripts/register_codex_mcp.py"
echo "Start SGLang or vLLM locally before testing OCR. Region mode also needs local PaddleX/PaddlePaddle."
