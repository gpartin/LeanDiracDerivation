"""Activate and verify this repository's versioned Git hooks."""

from __future__ import annotations

from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    subprocess.run(
        ["git", "config", "core.hooksPath", ".githooks"],
        cwd=ROOT,
        check=True,
    )
    configured = subprocess.run(
        ["git", "config", "--get", "core.hooksPath"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if configured != ".githooks":
        raise SystemExit(f"HOOK_INSTALL_FAIL: core.hooksPath={configured!r}")

    attribute = subprocess.run(
        ["git", "check-attr", "eol", "--", ".githooks/pre-commit"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if not attribute.endswith(": lf"):
        raise SystemExit(f"HOOK_INSTALL_FAIL: expected LF attribute: {attribute}")

    print("HOOK_INSTALL_PASS: core.hooksPath=.githooks; hook eol=lf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
