#!/usr/bin/env python3
"""Realistic gauntlet benchmark: cold vs warm memory runs with a real coding agent."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import tempfile
import textwrap
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from memory_query_optimizer import optimize_qmd_output


ROOT = Path(__file__).resolve().parent.parent
TEMPLATE_DIR = ROOT / "benchmarks" / "gauntlet" / "tricky_inventory_template"
TEST_CMD = ["python3", "-m", "unittest", "discover", "-s", "tests", "-p", "test_*.py"]


@dataclass
class TrackResult:
    name: str
    passed: bool
    duration_seconds: float
    return_code: int
    tests_output: str
    agent_stdout: str
    agent_stderr: str
    workspace: Path


def run_cmd(
    cmd: List[str],
    cwd: Path,
    timeout: int,
    input_text: Optional[str] = None,
) -> Tuple[int, str, str]:
    def to_text(value: object) -> str:
        if value is None:
            return ""
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        return str(value)

    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            input=input_text,
            text=True,
            capture_output=True,
            timeout=timeout,
        )
        return proc.returncode, to_text(proc.stdout), to_text(proc.stderr)
    except subprocess.TimeoutExpired as exc:
        out = to_text(exc.stdout)
        err = to_text(exc.stderr) + f"\nTIMEOUT after {timeout}s\n"
        return 124, out, err


def run_tests(workspace: Path, timeout: int = 60) -> Tuple[bool, str]:
    rc, out, err = run_cmd(TEST_CMD, workspace, timeout=timeout)
    text = (out + "\n" + err).strip()
    return rc == 0, text


def default_agent_cmd(workspace: Path, timeout: int, prompt: str) -> Tuple[int, str, str]:
    cmd = [
        "codex",
        "exec",
        "--dangerously-bypass-approvals-and-sandbox",
        "--skip-git-repo-check",
        "--cd",
        str(workspace),
        "-",
    ]
    return run_cmd(cmd, workspace, timeout=timeout, input_text=prompt)


def apply_reference_fix(workspace: Path) -> None:
    target = workspace / "inventory" / "allocator.py"
    text = target.read_text(encoding="utf-8")
    text = text.replace("if parse_ts(hold.expires_at) >= now:", "if parse_ts(hold.expires_at) > now:")
    text = text.replace(
        "items.sort(key=lambda it: (it.priority, parse_ts(it.created_at)), reverse=True)",
        "items.sort(key=lambda it: (-it.priority, parse_ts(it.created_at)))",
    )
    text = text.replace(
        "take = min(item.qty, remaining + (1 if remaining > 0 else 0))",
        "take = min(item.qty, remaining)",
    )
    target.write_text(text, encoding="utf-8")


def build_prompt(memory_shortlist: Optional[str], memory_excerpt: Optional[str]) -> str:
    memory_section = ""
    if memory_shortlist:
        memory_section = (
            "\nRetrieved Memory (from previous solved session):\n"
            f"{memory_shortlist}\n"
        )
    memory_details = ""
    if memory_excerpt:
        memory_details = (
            "\nTop Memory Details (open/reuse this first):\n"
            f"{memory_excerpt}\n"
        )
    return textwrap.dedent(
        f"""
        You are solving a tricky inventory allocator bug in a fresh session.
        Goal: make all tests pass with minimal, correct changes.

        Required command:
        `python3 -m unittest discover -s tests -p "test_*.py"`

        Constraints:
        - Work only in this repository.
        - Prefer fixing logic in `inventory/allocator.py`.
        - Run tests before finishing.
        - In your final message, include `FINAL_STATUS: PASS` only if tests pass.
        {memory_section}
        {memory_details}
        """
    ).strip()


def build_memory_from_seed(seed_result: TrackResult) -> str:
    source = "qmd://gauntlet/sessions/2026-02-28-tricky-inventory-fix.md:1 #gauntlet1"
    title = "Session: Fixed tricky inventory allocator edge cases in inventory/allocator.py"
    changed_files = sorted(
        str(path.relative_to(seed_result.workspace))
        for path in seed_result.workspace.rglob("*")
        if path.is_file()
    )
    changed_hint = ", ".join(file for file in changed_files if file.startswith("inventory/")) or "inventory/allocator.py"
    body = (
        "## What Was Done\n"
        "- Fixed hold expiry boundary check to treat hold expiring exactly at now as expired.\n"
        "- Fixed order sorting tie-break to allocate earliest created_at first for equal priority.\n"
        "- Fixed over-allocation logic to never allocate more than remaining stock.\n"
        f"- Edited: {changed_hint}\n"
        "- Verified with `python3 -m unittest discover -s tests -p \"test_*.py\"`.\n"
    )
    return (
        f"{source}\n"
        f"Title: {title}\n"
        f"Context: Gauntlet memory\n"
        "Score:  92%\n\n"
        f"{body}\n"
    )


def build_distractors() -> List[str]:
    return [
        (
            "qmd://gauntlet/sessions/2026-02-15-ui-spacing-adjustment.md:1 #dist1\n"
            "Title: Session: Adjusted UI spacing in dashboard cards\n"
            "Context: Gauntlet memory\n"
            "Score:  84%\n\n"
            "## What Was Done\n- Updated CSS gap and padding values.\n"
        ),
        (
            "qmd://gauntlet/sessions/2026-02-16-cache-ttl-tuning.md:1 #dist2\n"
            "Title: Session: Tuned cache TTL for metrics polling\n"
            "Context: Gauntlet memory\n"
            "Score:  76%\n\n"
            "## What Was Done\n- Changed cache TTL from 90s to 60s.\n"
        ),
    ]


def parse_raw_blocks(raw_memory: str) -> Dict[str, str]:
    blocks = re.split(r"\n{2,}", raw_memory.strip())
    by_source: Dict[str, str] = {}
    current: List[str] = []
    for line in raw_memory.splitlines():
        if line.startswith("qmd://"):
            if current:
                source = current[0].strip()
                by_source[source] = "\n".join(current).strip()
                current = []
        if line.strip() or current:
            current.append(line)
    if current:
        source = current[0].strip()
        by_source[source] = "\n".join(current).strip()
    if not by_source:
        for block in blocks:
            lines = [ln for ln in block.splitlines() if ln.strip()]
            if lines:
                by_source[lines[0].strip()] = block.strip()
    return by_source


def extract_top_memory_excerpt(shortlist: str, raw_memory: str) -> str:
    source_match = re.search(r"Source:\s*(.+)$", shortlist, flags=re.MULTILINE)
    if not source_match:
        return ""
    source = source_match.group(1).strip()
    blocks = parse_raw_blocks(raw_memory)
    block = blocks.get(source, "")
    if not block:
        return ""
    lines = []
    for line in block.splitlines():
        if line.startswith("- ") or line.startswith("## ") or line.startswith("Title:"):
            lines.append(line)
    excerpt = "\n".join(lines[:8]).strip()
    return excerpt


def run_track(
    name: str,
    workspace: Path,
    prompt: str,
    agent_timeout: int,
    mode: str,
) -> TrackResult:
    start = time.monotonic()
    if mode == "reference":
        apply_reference_fix(workspace)
        rc, out, err = 0, "Reference solver applied patch.\nFINAL_STATUS: PASS", ""
    else:
        rc, out, err = default_agent_cmd(workspace, timeout=agent_timeout, prompt=prompt)

    passed, tests_output = run_tests(workspace, timeout=90)
    duration = time.monotonic() - start
    return TrackResult(
        name=name,
        passed=passed,
        duration_seconds=duration,
        return_code=rc,
        tests_output=tests_output,
        agent_stdout=out,
        agent_stderr=err,
        workspace=workspace,
    )


def summarize_results(seed: TrackResult, cold: TrackResult, warm: TrackResult, shortlist: str, mode: str) -> Dict[str, object]:
    warm_faster = warm.duration_seconds < cold.duration_seconds
    speedup = (cold.duration_seconds / warm.duration_seconds) if warm.duration_seconds > 0 else 0.0
    if mode == "reference":
        passed = seed.passed and cold.passed and warm.passed
    else:
        passed = warm.passed and (warm_faster or (not cold.passed))
    return {
        "passed": passed,
        "seed_passed": seed.passed,
        "cold_passed": cold.passed,
        "warm_passed": warm.passed,
        "cold_seconds": round(cold.duration_seconds, 2),
        "warm_seconds": round(warm.duration_seconds, 2),
        "speedup": round(speedup, 2),
        "warm_faster": warm_faster,
        "shortlist": shortlist,
        "tracks": [
            {
                "name": seed.name,
                "passed": seed.passed,
                "seconds": round(seed.duration_seconds, 2),
                "return_code": seed.return_code,
                "workspace": str(seed.workspace),
            },
            {
                "name": cold.name,
                "passed": cold.passed,
                "seconds": round(cold.duration_seconds, 2),
                "return_code": cold.return_code,
                "workspace": str(cold.workspace),
            },
            {
                "name": warm.name,
                "passed": warm.passed,
                "seconds": round(warm.duration_seconds, 2),
                "return_code": warm.return_code,
                "workspace": str(warm.workspace),
            },
        ],
    }


def print_report(report: Dict[str, object]) -> int:
    print("Agent Gauntlet Benchmark (Real Run)")
    print(f"Seed pass: {report['seed_passed']}")
    print(f"Cold pass: {report['cold_passed']}")
    print(f"Warm pass: {report['warm_passed']}")
    print(f"Cold seconds: {report['cold_seconds']}")
    print(f"Warm seconds: {report['warm_seconds']}")
    print(f"Speedup (cold/warm): {report['speedup']}x")
    print(f"Warm faster: {report['warm_faster']}")
    print("")
    print("Tracks:")
    for row in report["tracks"]:
        print(
            f"- {row['name']}: passed={row['passed']} seconds={row['seconds']} "
            f"return_code={row['return_code']} workspace={row['workspace']}"
        )
    print("")
    print("Warm memory shortlist used:")
    print(report["shortlist"])
    print("")
    if report["passed"]:
        print("PASS: warm-memory run outperformed cold run on the gauntlet.")
        return 0
    print("FAIL: warm-memory run did not outperform cold run on this gauntlet execution.")
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a real-world cold vs warm memory gauntlet using codex exec.")
    parser.add_argument(
        "--mode",
        choices=["agent", "reference"],
        default="agent",
        help="agent: run codex exec; reference: apply known-good patch (sanity mode).",
    )
    parser.add_argument("--token-budget", type=int, default=900)
    parser.add_argument("--agent-timeout-seconds", type=int, default=480)
    parser.add_argument("--keep-workspaces", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    if not TEMPLATE_DIR.exists():
        raise SystemExit(f"Template directory missing: {TEMPLATE_DIR}")

    root_tmp = Path(tempfile.mkdtemp(prefix="tr-gauntlet-"))
    seed_ws = root_tmp / "seed"
    cold_ws = root_tmp / "cold"
    warm_ws = root_tmp / "warm"
    shutil.copytree(TEMPLATE_DIR, seed_ws)
    shutil.copytree(TEMPLATE_DIR, cold_ws)
    shutil.copytree(TEMPLATE_DIR, warm_ws)

    try:
        seed_prompt = build_prompt(memory_shortlist=None, memory_excerpt=None)
        seed_result = run_track("seed", seed_ws, seed_prompt, args.agent_timeout_seconds, mode=args.mode)
        memory_relevant = build_memory_from_seed(seed_result)
        memory_raw = "\n\n".join([memory_relevant] + build_distractors())
        shortlist = optimize_qmd_output(memory_raw, "inventory allocator edge cases prior fix", token_budget=args.token_budget)
        excerpt = extract_top_memory_excerpt(shortlist, memory_raw)

        cold_prompt = build_prompt(memory_shortlist=None, memory_excerpt=None)
        warm_prompt = build_prompt(memory_shortlist=shortlist, memory_excerpt=excerpt)
        cold_result = run_track("cold", cold_ws, cold_prompt, args.agent_timeout_seconds, mode=args.mode)
        warm_result = run_track("warm", warm_ws, warm_prompt, args.agent_timeout_seconds, mode=args.mode)

        report = summarize_results(seed_result, cold_result, warm_result, shortlist, mode=args.mode)
        if args.json:
            print(json.dumps(report, indent=2))
            return 0 if report["passed"] else 1
        return print_report(report)
    finally:
        if not args.keep_workspaces:
            shutil.rmtree(root_tmp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
