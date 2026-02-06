#!/bin/bash
INPUT=$(cat)

# Prevent infinite loop
if [ "$(echo "$INPUT" | jq -r '.stop_hook_active // false')" = "true" ]; then
  exit 0
fi

PROJECT=$(basename "$(git -C "$PWD" remote get-url origin 2>/dev/null | sed 's/.*\///' | sed 's/\.git$//')" 2>/dev/null || basename "$PWD")
KNOWLEDGE_DIR="$HOME/.claude/knowledge/$PROJECT/sessions"

if [ -d "$KNOWLEDGE_DIR" ]; then
  jq -n --arg dir "$KNOWLEDGE_DIR" '{
    "additionalContext": ("Before ending: Run /memory-write if this session involved non-trivial work. Knowledge dir: " + $dir)
  }'
fi

exit 0
