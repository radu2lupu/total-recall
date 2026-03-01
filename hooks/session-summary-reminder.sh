#!/bin/bash
# Stop hook — blocks session end until Claude writes a memory summary.
# Activates for any git repo automatically. Project name is derived from git remote
# origin (deterministic across machines) or falls back to directory name.
INPUT=$(cat)

# Prevent infinite loop — if we already triggered a stop hook continuation, let it end
if [ "$(echo "$INPUT" | jq -r '.stop_hook_active // false')" = "true" ]; then
  exit 0
fi

# Get cwd from hook input (more reliable than $PWD in hook context)
CWD=$(echo "$INPUT" | jq -r '.cwd // empty')
CWD="${CWD:-$PWD}"

# Ensure bun-installed globals (like qmd) are on PATH
for p in "$HOME/.bun/bin" "/opt/homebrew/bin" "/usr/local/bin"; do
  [ -d "$p" ] && export PATH="$p:$PATH"
done

CLI="${CLAUDE_PLUGIN_ROOT}/scripts/total-recall"

# Must have the CLI available
if [ ! -x "$CLI" ] && ! command -v total-recall >/dev/null 2>&1; then
  exit 0
fi

# Must have qmd installed (for local mode) or a remote client configured
CLIENT_JSON="$HOME/.ai-memory/client.json"
if ! command -v qmd >/dev/null 2>&1 && [ ! -f "$CLIENT_JSON" ]; then
  exit 0
fi

jq -n --arg cli "$CLI" '{
  "decision": "block",
  "reason": ("BEFORE STOPPING: You must persist what you learned this session. Run:\n`" + $cli + " write \"<concise summary: what changed, why, and what was learned>\"`\nThen you may stop. If this session was truly trivial (only reading files, no changes, no decisions), you may skip this — but explain why.")
}'
