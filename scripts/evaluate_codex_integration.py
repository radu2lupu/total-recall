#!/usr/bin/env python3
"""Verify Codex integration contract for total-recall."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple


ROOT = Path(__file__).resolve().parent.parent
CLI = ROOT / "scripts" / "total-recall"


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str
    required: bool = True


def run(cmd: List[str], env: dict | None = None, timeout: int = 60) -> Tuple[int, str, str]:
    proc = subprocess.run(
        cmd,
        text=True,
        capture_output=True,
        env=env,
        timeout=timeout,
    )
    return proc.returncode, proc.stdout or "", proc.stderr or ""


def check_agents_file(path: Path) -> List[CheckResult]:
    checks: List[CheckResult] = []
    if not path.exists():
        return [CheckResult("codex_agents_exists", False, f"missing: {path}")]

    text = path.read_text(encoding="utf-8")
    checks.append(CheckResult("codex_agents_markers", ("<!-- total-recall:auto:start -->" in text and "<!-- total-recall:auto:end -->" in text), str(path)))

    required_phrases = [
        "MANDATORY — your FIRST action every session",
        "total-recall query",
        "total-recall write",
        "Do NOT tell the user to run these commands manually",
    ]
    for phrase in required_phrases:
        checks.append(
            CheckResult(
                f"codex_agents_contains:{phrase[:28]}",
                phrase in text,
                phrase,
            )
        )
    has_new_write_rule = "MANDATORY — before your final response in non-trivial sessions" in text
    has_old_write_rule = "MANDATORY — after completing non-trivial work" in text
    checks.append(
        CheckResult(
            "codex_agents_write_trigger_present",
            has_new_write_rule or has_old_write_rule,
            "new-rule or old-rule write trigger text",
        )
    )
    return checks


def check_codex_config(path: Path, required: bool) -> List[CheckResult]:
    checks: List[CheckResult] = []
    if not path.exists():
        return [CheckResult("codex_config_exists", False, f"missing: {path}", required=required)]

    text = path.read_text(encoding="utf-8")
    checks.append(CheckResult("codex_qmd_section", bool(re.search(r"(?m)^\[mcp_servers\.qmd\]\s*$", text)), str(path), required=required))
    checks.append(CheckResult("codex_qmd_command", bool(re.search(r'(?m)^\s*command\s*=\s*["\']qmd["\']\s*$', text)), "command = qmd", required=required))
    checks.append(CheckResult("codex_qmd_args_mcp", bool(re.search(r'(?m)^\s*args\s*=\s*\[[^\]]*["\']mcp["\'][^\]]*\]\s*$', text)), 'args include "mcp"', required=required))
    return checks


def evaluate_integration(home: Path, project: str, run_query_smoke: bool, require_mcp: bool) -> List[CheckResult]:
    codex_home = Path(os.getenv("CODEX_HOME", str(home / ".codex")))
    codex_agents = codex_home / "AGENTS.md"
    codex_config = codex_home / "config.toml"
    checks = []
    checks.extend(check_agents_file(codex_agents))
    checks.extend(check_codex_config(codex_config, required=require_mcp))

    cli_link = home / ".local" / "bin" / "total-recall"
    checks.append(CheckResult("cli_symlink_exists", cli_link.exists(), str(cli_link)))
    checks.append(CheckResult("cli_symlink_executable", os.access(cli_link, os.X_OK), str(cli_link)))

    if run_query_smoke:
        rc, out, err = run(
            [str(cli_link), "query", "--project", project, "codex integration smoke query"],
            timeout=90,
        )
        detail = (out + "\n" + err).strip()[:300]
        checks.append(CheckResult("query_smoke_exit_zero", rc == 0, detail or "no output"))
    return checks


def sandbox_install_and_verify(project: str) -> List[CheckResult]:
    checks: List[CheckResult] = []
    with tempfile.TemporaryDirectory(prefix="tr-codex-verify-") as tmp:
        home = Path(tmp)
        env = os.environ.copy()
        env["HOME"] = str(home)
        env["CODEX_HOME"] = str(home / ".codex")
        env["CLAUDE_HOME"] = str(home / ".claude")
        env["TOTAL_RECALL_SHARED_ROOT"] = str(home / ".ai-memory" / "knowledge")
        env["TOTAL_RECALL_ICLOUD_ROOT"] = str(home / ".icloud")

        rc, out, err = run(
            [
                str(CLI),
                "install",
                "--project",
                project,
                "--client",
                "--server-url",
                "http://127.0.0.1:9",
                "--api-key",
                "tr_test_key",
            ],
            env=env,
            timeout=120,
        )
        checks.append(CheckResult("sandbox_install_client_exit_zero", rc == 0, (out + "\n" + err).strip()[:300]))

        old_codex_home = os.environ.get("CODEX_HOME")
        try:
            os.environ["CODEX_HOME"] = str(home / ".codex")
            checks.extend(evaluate_integration(home=home, project=project, run_query_smoke=True, require_mcp=False))
        finally:
            if old_codex_home is None:
                os.environ.pop("CODEX_HOME", None)
            else:
                os.environ["CODEX_HOME"] = old_codex_home

    return checks


def local_verify(project: str, run_query_smoke: bool) -> List[CheckResult]:
    home = Path.home()
    require_mcp = True
    return evaluate_integration(home=home, project=project, run_query_smoke=run_query_smoke, require_mcp=require_mcp)


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify Codex integration contract for total-recall.")
    parser.add_argument("--project", default="codex-integration-check")
    parser.add_argument("--mode", choices=["sandbox", "local"], default="sandbox")
    parser.add_argument("--run-query-smoke", action="store_true", help="In local mode, run `total-recall query` smoke test.")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    if args.mode == "sandbox":
        checks = sandbox_install_and_verify(project=args.project)
    else:
        checks = local_verify(project=args.project, run_query_smoke=args.run_query_smoke)

    passed = all(c.ok for c in checks if c.required)
    report = {
        "mode": args.mode,
        "project": args.project,
        "passed": passed,
        "checks": [
            {"name": c.name, "ok": c.ok, "detail": c.detail, "required": c.required}
            for c in checks
        ],
    }

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"Codex integration verification ({args.mode})")
        for c in checks:
            status = "PASS" if c.ok else ("WARN" if not c.required else "FAIL")
            print(f"- [{status}] {c.name}: {c.detail}")
        print(f"\nRESULT: {'PASS' if passed else 'FAIL'}")

    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
