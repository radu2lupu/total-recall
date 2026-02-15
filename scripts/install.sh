#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLI="$SCRIPT_DIR/total-recall"

# Client-only installs don't need qmd — skip the check
IS_CLIENT=0
for arg in "$@"; do
  [[ "$arg" == "--client" ]] && IS_CLIENT=1
done

if [[ "$IS_CLIENT" == "0" ]]; then
  if ! command -v qmd >/dev/null 2>&1; then
    if ! command -v bun >/dev/null 2>&1; then
      echo "qmd is required and bun is not installed."
      echo "Install bun first: curl -fsSL https://bun.sh/install | bash"
      exit 1
    fi
    echo "Installing qmd..."
    bun install -g github:tobi/qmd
  fi
fi

chmod +x "$CLI"
chmod +x "$SCRIPT_DIR/ingest_sessions.py"
chmod +x "$SCRIPT_DIR/total-recall-server"

echo "Running automated install..."
"$CLI" install "$@"

echo "Done."
