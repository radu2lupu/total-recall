#!/usr/bin/env python3
"""Evaluate structured memory write behavior for cue quality and durability."""

from __future__ import annotations

import argparse
import re
import tempfile
from pathlib import Path

from memory_note_builder import write_memory_files


def estimate_tokens(text: str) -> int:
    words = len(re.findall(r"\S+", text))
    return max(1, int(words * 1.3))


def assert_case(summary: str, expect_dir: str, max_tokens: int) -> tuple[bool, str]:
    with tempfile.TemporaryDirectory(prefix="tr-memory-eval-") as tmp:
        project_dir = Path(tmp) / "total-recall"
        result = write_memory_files(
            project_dir=project_dir,
            project="total-recall",
            summary=summary,
            machine="test-machine",
            date_str="2026-02-28",
        )
        session_file = Path(result["session_file"])
        session_text = session_file.read_text(encoding="utf-8")
        promoted = [Path(path) for path in result.get("promoted_files", [])]

        checks = {
            "retrieval_cues_section": "## Retrieval Cues" in session_text,
            "intent_metadata": any(line.startswith("intent:") for line in session_text.splitlines()[:12]),
            "cue_problem_metadata": any(line.startswith("cue_problem:") for line in session_text.splitlines()[:16]),
            "has_keywords_line": "- Keywords:" in session_text,
            "has_problem_action_verify_lines": (
                "- Problem cue:" in session_text and "- Action cue:" in session_text and "- Verify cue:" in session_text
            ),
            "token_budget_note": estimate_tokens(session_text) <= max_tokens,
            "promoted_expected_dir": any(f"/{expect_dir}/" in str(path) for path in promoted),
        }
        failed = [name for name, ok in checks.items() if not ok]
        if failed:
            return False, f"FAIL ({expect_dir}): {', '.join(failed)}"
        return True, f"PASS ({expect_dir}): promoted={len(promoted)} tokens={estimate_tokens(session_text)}"


def assert_synthesis_case() -> tuple[bool, str]:
    with tempfile.TemporaryDirectory(prefix="tr-memory-synth-") as tmp:
        project_dir = Path(tmp) / "total-recall"
        summary = "Implemented memory ranking loop for token budget optimization and query precision."
        first = write_memory_files(project_dir, "total-recall", summary, "test-machine", date_str="2026-02-28")
        second = write_memory_files(project_dir, "total-recall", summary, "test-machine", date_str="2026-02-28")
        promoted = [Path(path) for path in second.get("promoted_files", [])]
        synthesis_files = [path for path in promoted if path.name.endswith("-synthesized.md")]
        if not synthesis_files:
            return False, "FAIL (synthesis): no synthesized pattern file created for repeated topic"
        content = synthesis_files[0].read_text(encoding="utf-8")
        checks = {
            "pattern_header": "# Pattern:" in content,
            "source_sessions": "## Source Sessions" in content,
            "source_count": "**Source Count:** 2" in content,
        }
        failed = [name for name, ok in checks.items() if not ok]
        if failed:
            return False, f"FAIL (synthesis): {', '.join(failed)}"
        return True, f"PASS (synthesis): {synthesis_files[0].name}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate memory write cue/promotion behavior.")
    parser.add_argument("--max-session-tokens", type=int, default=420)
    args = parser.parse_args()

    cases = [
        (
            "Fixed regression in scripts/total-recall by handling empty qmd query output and verified with bash -n and evaluator scripts.",
            "bugs",
        ),
        (
            "Decided to prioritize durable decision notes by boosting /decisions/ retrieval score after reviewing token budget tradeoffs.",
            "decisions",
        ),
        (
            "Created reusable workflow pattern for memory optimization iterations with evaluator checks and repeatable PASS/FAIL gates.",
            "patterns",
        ),
    ]

    failures = 0
    for summary, expect_dir in cases:
        ok, message = assert_case(summary, expect_dir, args.max_session_tokens)
        print(message)
        if not ok:
            failures += 1

    synth_ok, synth_message = assert_synthesis_case()
    print(synth_message)
    if not synth_ok:
        failures += 1

    if failures:
        print(f"FAIL: {failures} write-evaluation case(s) failed")
        return 1

    print("PASS: write-path memory structuring meets cue and durability criteria")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
