#!/bin/bash
# qmd-memory plugin installer for Claude Code
#
# Usage:
#   bash <(curl -fsSL https://raw.githubusercontent.com/radulupu/qmd-memory/main/scripts/install.sh)
#
# This script is a convenience wrapper. You can also install directly in Claude Code:
#   /plugin marketplace add radulupu/qmd-memory
#   /plugin install qmd-memory@qmd-memory

set -e

echo "=== qmd-memory installer ==="
echo ""

# Check prerequisites
if ! command -v bun &>/dev/null; then
  echo "bun is required but not installed."
  echo "Install it with: curl -fsSL https://bun.sh/install | bash"
  exit 1
fi

if ! command -v qmd &>/dev/null; then
  echo "Installing qmd..."
  bun install -g qmd
fi

echo ""
echo "qmd is installed. To set up the Claude Code plugin:"
echo ""
echo "  1. Open Claude Code"
echo "  2. Run: /plugin marketplace add radulupu/qmd-memory"
echo "  3. Run: /plugin install qmd-memory@qmd-memory"
echo "  4. Run: /memory-setup  (in any project to initialize memory)"
echo ""
echo "Commands available after install:"
echo "  /memory-setup              Initialize memory for current project"
echo "  /memory-write [desc]       Write a session summary"
echo "  /memory-rebuild [flags]    Rebuild memory from git history"
