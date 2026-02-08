#!/usr/bin/env python3
import argparse
import datetime as dt
import glob
import json
import os
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "memory"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_json(path: Path, default):
    if not path.exists():
        return default
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def save_json(path: Path, payload) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def clean_text(value: str) -> str:
    text = (value or "").replace("\r", " ").replace("\n", " ").strip()
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return ""
    if text.startswith("<user_instructions>") or text.startswith("<environment_context>"):
        return ""
    if len(text) > 900:
        text = text[:897] + "..."
    return text


def parse_iso_date(value: str) -> str:
    try:
        if value.endswith("Z"):
            value = value[:-1] + "+00:00"
        return dt.datetime.fromisoformat(value).date().isoformat()
    except Exception:
        return dt.date.today().isoformat()


def parse_epoch_millis_date(value) -> str:
    try:
        ms = int(value)
        return dt.datetime.utcfromtimestamp(ms / 1000.0).date().isoformat()
    except Exception:
        return dt.date.today().isoformat()


def now_utc_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


class ProjectSlugResolver:
    def __init__(self):
        self.cache = {}

    def from_path(self, maybe_path: str) -> str:
        if not maybe_path:
            return ""
        path = os.path.expanduser(maybe_path)
        if path in self.cache:
            return self.cache[path]

        slug = ""
        try:
            remote = subprocess.check_output(
                ["git", "-C", path, "remote", "get-url", "origin"],
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
            if remote:
                repo = os.path.basename(remote)
                if repo.endswith(".git"):
                    repo = repo[:-4]
                slug = slugify(repo)
        except Exception:
            pass

        if not slug:
            slug = slugify(os.path.basename(path.rstrip(os.sep)))
        self.cache[path] = slug
        return slug


def extract_user_texts_from_content(content):
    texts = []
    if not isinstance(content, list):
        return texts
    for item in content:
        if not isinstance(item, dict):
            continue
        text = item.get("text") or item.get("value") or ""
        text = clean_text(text)
        if text:
            texts.append(text)
    return texts


def extract_codex_session(session_path: Path):
    session_id = session_path.stem
    first_date = ""
    cwd = ""
    messages = []

    with session_path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except Exception:
                continue

            ts = obj.get("timestamp", "")
            if not first_date and isinstance(ts, str) and ts:
                first_date = parse_iso_date(ts)

            if obj.get("type") == "session_meta":
                payload = obj.get("payload", {})
                session_id = payload.get("id") or session_id
                cwd = payload.get("cwd") or cwd
                ts2 = payload.get("timestamp")
                if not first_date and isinstance(ts2, str) and ts2:
                    first_date = parse_iso_date(ts2)
                continue

            role = obj.get("role")
            if role == "user":
                for text in extract_user_texts_from_content(obj.get("content")):
                    messages.append((ts, text))
                continue

            if obj.get("type") == "response_item":
                payload = obj.get("payload", {})
                if payload.get("type") == "message" and payload.get("role") == "user":
                    for text in extract_user_texts_from_content(payload.get("content")):
                        messages.append((ts, text))
                continue

    if not first_date:
        first_date = dt.datetime.fromtimestamp(session_path.stat().st_mtime).date().isoformat()
    return {"session_id": session_id, "date": first_date, "cwd": cwd, "messages": messages}


def write_codex_markdown(output_dir: Path, source_file: Path, session_info: dict) -> bool:
    session_id = slugify(session_info["session_id"])
    date = session_info["date"]
    out_path = output_dir / f"{date}-codex-{session_id}.md"
    if out_path.exists():
        return False

    ensure_dir(output_dir)
    lines = []
    lines.append(f"# Imported Session: Codex {session_info['session_id']}")
    lines.append("")
    lines.append(f"**Date:** {date}")
    lines.append(f"**Source:** `{source_file}`")
    if session_info["cwd"]:
        lines.append(f"**CWD:** `{session_info['cwd']}`")
    lines.append(f"**Imported At:** {now_utc_iso()}")
    lines.append("")
    lines.append("## User Messages")

    for timestamp, text in session_info["messages"][:200]:
        prefix = ""
        if isinstance(timestamp, str) and timestamp:
            prefix = f"[{timestamp}] "
        lines.append(f"- {prefix}{text}")

    lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return True


def ingest_codex(args, state, resolver, imported_dir: Path):
    processed = state.setdefault("processed_codex_files", {})
    patterns = [
        os.path.expanduser(os.path.join(args.codex_home, "sessions", "**", "*.jsonl")),
        os.path.expanduser(os.path.join(args.codex_home, "archived_sessions", "*.jsonl")),
    ]

    files = []
    for pattern in patterns:
        files.extend(glob.glob(pattern, recursive=True))
    files = sorted(set(files))

    wrote = 0
    scanned = 0
    for file_path in files:
        scanned += 1
        abs_path = os.path.realpath(file_path)
        mtime = os.path.getmtime(abs_path)
        previous = processed.get(abs_path)
        if previous and float(previous) == float(mtime):
            continue

        info = extract_codex_session(Path(abs_path))
        slug = resolver.from_path(info.get("cwd", ""))
        if slug != args.project:
            processed[abs_path] = mtime
            continue
        if not info["messages"]:
            processed[abs_path] = mtime
            continue

        if write_codex_markdown(imported_dir / "codex", Path(abs_path), info):
            wrote += 1
        processed[abs_path] = mtime

    state["processed_codex_files"] = processed
    return {"scanned": scanned, "written": wrote}


def maybe_skip_claude_display(display: str) -> bool:
    text = (display or "").strip()
    if not text:
        return True
    if text.startswith("/") and " " not in text:
        return True
    return False


def ingest_claude(args, state, resolver, imported_dir: Path):
    history_path = Path(os.path.expanduser(os.path.join(args.claude_home, "history.jsonl")))
    if not history_path.exists():
        return {"written": 0, "messages": 0, "sessions": 0}

    offset_key = "claude_history_offset"
    old_offset = int(state.get(offset_key, 0))
    file_size = history_path.stat().st_size
    if old_offset > file_size:
        old_offset = 0

    sessions = defaultdict(list)
    new_messages = 0
    with history_path.open("rb") as f:
        f.seek(old_offset)
        while True:
            line = f.readline()
            if not line:
                break
            try:
                obj = json.loads(line.decode("utf-8", errors="ignore"))
            except Exception:
                continue

            project_path = obj.get("project", "")
            if resolver.from_path(project_path) != args.project:
                continue

            display = clean_text(obj.get("display", ""))
            if not display or maybe_skip_claude_display(display):
                continue

            session_id = obj.get("sessionId") or "unknown"
            date = parse_epoch_millis_date(obj.get("timestamp"))
            ts = obj.get("timestamp")
            sessions[session_id].append((date, ts, display, project_path))
            new_messages += 1

        state[offset_key] = f.tell()

    wrote = 0
    for session_id, items in sessions.items():
        items.sort(key=lambda x: int(x[1]) if x[1] else 0)
        date = items[0][0]
        out_path = imported_dir / "claude" / f"{date}-claude-{slugify(session_id)}.md"
        ensure_dir(out_path.parent)
        is_new = not out_path.exists()

        with out_path.open("a", encoding="utf-8") as out:
            if is_new:
                out.write(f"# Imported Session: Claude {session_id}\n\n")
                out.write(f"**Date:** {date}\n")
                out.write(f"**Project Path:** `{items[0][3]}`\n")
                out.write(f"**Imported At:** {now_utc_iso()}\n\n")
                out.write("## User Messages\n")

            for _, ts, display, _ in items[:400]:
                stamp = ""
                if ts:
                    try:
                        stamp = dt.datetime.utcfromtimestamp(int(ts) / 1000.0).isoformat() + "Z"
                    except Exception:
                        stamp = str(ts)
                if stamp:
                    out.write(f"- [{stamp}] {display}\n")
                else:
                    out.write(f"- {display}\n")
        wrote += 1

    return {"written": wrote, "messages": new_messages, "sessions": len(sessions)}


def main():
    parser = argparse.ArgumentParser(description="Ingest Codex + Claude sessions into shared qmd memory")
    parser.add_argument("--project", required=True, help="Project slug (kebab-case)")
    parser.add_argument("--shared-root", required=True, help="Shared root directory")
    parser.add_argument("--state-file", required=True, help="State file path")
    parser.add_argument("--codex-home", default=os.path.expanduser("~/.codex"))
    parser.add_argument("--claude-home", default=os.path.expanduser("~/.claude"))
    args = parser.parse_args()

    shared_root = Path(os.path.expanduser(args.shared_root)).resolve()
    state_file = Path(os.path.expanduser(args.state_file)).resolve()
    project_dir = shared_root / args.project
    imported_dir = project_dir / "sessions" / "imported"
    ensure_dir(imported_dir)

    state = load_json(state_file, {})
    resolver = ProjectSlugResolver()

    codex_stats = ingest_codex(args, state, resolver, imported_dir)
    claude_stats = ingest_claude(args, state, resolver, imported_dir)
    save_json(state_file, state)

    result = {
        "project": args.project,
        "codex": codex_stats,
        "claude": claude_stats,
        "state_file": str(state_file),
        "project_dir": str(project_dir),
        "timestamp": now_utc_iso(),
    }
    json.dump(result, sys.stdout, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
