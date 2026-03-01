#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLI="$SCRIPT_DIR/total-recall"

chmod +x "$CLI"
chmod +x "$SCRIPT_DIR/ingest_sessions.py"
chmod +x "$SCRIPT_DIR/total-recall-server"

detect_project_name() {
  local remote repo
  if remote="$(git -C "$PWD" remote get-url origin 2>/dev/null)"; then
    repo="$(basename "$remote")"
    repo="${repo%.git}"
    printf '%s' "$repo"
    return 0
  fi
  basename "$PWD"
}

is_tty() {
  [[ -t 0 && -t 1 ]]
}

prompt_default() {
  local prompt="$1"
  local default_value="$2"
  local reply
  read -r -p "$prompt [$default_value]: " reply
  printf '%s' "${reply:-$default_value}"
}

prompt_required() {
  local prompt="$1"
  local reply=""
  while [[ -z "$reply" ]]; do
    read -r -p "$prompt: " reply
  done
  printf '%s' "$reply"
}

prompt_yes_no() {
  local prompt="$1"
  local default_choice="$2" # y or n
  local suffix reply normalized
  if [[ "$default_choice" == "y" ]]; then
    suffix="[Y/n]"
  else
    suffix="[y/N]"
  fi
  while true; do
    read -r -p "$prompt $suffix: " reply
    normalized="$(printf '%s' "${reply:-$default_choice}" | tr '[:upper:]' '[:lower:]')"
    case "$normalized" in
      y|yes) return 0 ;;
      n|no) return 1 ;;
      *) echo "Please answer y or n." ;;
    esac
  done
}

quote_cmd() {
  local out=""
  local arg
  for arg in "$@"; do
    if [[ -z "$out" ]]; then
      out="$(printf '%q' "$arg")"
    else
      out="$out $(printf '%q' "$arg")"
    fi
  done
  printf '%s' "$out"
}

run_interactive_installer() {
  local project_default project_name mode_choice mode
  local install_cmd=("$CLI" install)
  local on_macos has_icloud_dir
  local use_icloud=0 use_launch_agent=0 interval_minutes="15"
  local server_url="" api_key="" server_port="7899"

  project_default="$(detect_project_name)"

  echo "Total Recall interactive installer"
  echo ""
  project_name="$(prompt_default "Project name" "$project_default")"

  echo ""
  echo "Install mode:"
  echo "  1) Standalone (local memory on this machine)"
  echo "  2) Server (host memory for other machines)"
  echo "  3) Client (connect to an existing server)"
  mode_choice="$(prompt_default "Choose mode" "1")"
  case "$mode_choice" in
    1) mode="standalone" ;;
    2) mode="server" ;;
    3) mode="client" ;;
    *) echo "Unknown mode: $mode_choice"; exit 1 ;;
  esac

  on_macos=0
  has_icloud_dir=0
  [[ "$(uname -s)" == "Darwin" ]] && on_macos=1
  [[ -d "$HOME/Library/Mobile Documents/com~apple~CloudDocs" ]] && has_icloud_dir=1

  if [[ "$mode" == "client" ]]; then
    echo ""
    server_url="$(prompt_required "Server URL (for example http://server:7899)")"
    api_key="$(prompt_required "API key (starts with tr_sk_)")"
  else
    if [[ "$on_macos" -eq 1 && "$has_icloud_dir" -eq 1 ]]; then
      if prompt_yes_no "Enable iCloud backup?" "y"; then
        use_icloud=1
      fi
    fi

    if [[ "$on_macos" -eq 1 ]]; then
      if prompt_yes_no "Enable background sync launch agent?" "y"; then
        use_launch_agent=1
        interval_minutes="$(prompt_default "Sync interval in minutes" "15")"
        while ! [[ "$interval_minutes" =~ ^[0-9]+$ ]] || (( interval_minutes < 1 )); do
          interval_minutes="$(prompt_default "Please enter a valid interval >= 1" "15")"
        done
      fi
    fi

    if [[ "$mode" == "server" ]]; then
      server_port="$(prompt_default "Server port" "7899")"
      while ! [[ "$server_port" =~ ^[0-9]+$ ]] || (( server_port < 1 )) || (( server_port > 65535 )); do
        server_port="$(prompt_default "Please enter a valid port 1-65535" "7899")"
      done
    fi
  fi

  install_cmd+=(--project "$project_name")

  case "$mode" in
    standalone)
      ;;
    server)
      install_cmd+=(--server --port "$server_port")
      ;;
    client)
      install_cmd+=(--client --server-url "$server_url" --api-key "$api_key")
      ;;
  esac

  if [[ "$mode" != "client" ]]; then
    if [[ "$use_icloud" -ne 1 ]]; then
      install_cmd+=(--no-icloud)
    fi
    if [[ "$use_launch_agent" -ne 1 ]]; then
      install_cmd+=(--no-launch-agent)
    else
      install_cmd+=(--interval-minutes "$interval_minutes")
    fi
  fi

  echo ""
  echo "Install command:"
  echo "  $(quote_cmd "${install_cmd[@]}")"
  echo ""
  if ! prompt_yes_no "Proceed with install?" "y"; then
    echo "Install cancelled."
    exit 0
  fi

  "${install_cmd[@]}"
}

interactive_requested=0
declare -a passthrough_args=()
for arg in "$@"; do
  case "$arg" in
    --interactive)
      interactive_requested=1
      ;;
    *)
      passthrough_args+=("$arg")
      ;;
  esac
done

if (( interactive_requested == 0 )) && (( ${#passthrough_args[@]} > 0 )); then
  echo "Running install with provided options..."
  "$CLI" install "${passthrough_args[@]}"
  echo "Done."
  exit 0
fi

if ! is_tty; then
  echo "No interactive terminal detected; running default install."
  "$CLI" install "${passthrough_args[@]}"
  echo "Done."
  exit 0
fi

run_interactive_installer
echo "Done."
