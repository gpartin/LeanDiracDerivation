"""Run every production proof check against the exact staged tree.

This is both a manually runnable release gate and the repository's Git
pre-commit hook target.  It refuses to certify a tree containing unstaged or
untracked files, ensuring the code tested here is exactly the code committed.
"""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def require_exact_staged_tree() -> None:
    unstaged = subprocess.run(
        ["git", "diff", "--quiet", "--exit-code"], cwd=ROOT, check=False
    )
    if unstaged.returncode != 0:
        raise SystemExit(
            "PRECOMMIT_AUDIT_FAIL: tracked unstaged changes exist; stage or "
            "revert them so the tested tree equals the committed tree"
        )

    untracked = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if untracked:
        raise SystemExit(
            "PRECOMMIT_AUDIT_FAIL: untracked files are outside the audit:\n"
            + untracked
        )


def main() -> int:
    require_exact_staged_tree()
    run(["git", "diff", "--cached", "--check"])
    run(["lake", "build"])
    run(["lake", "build", "LeanDiracAudit"])
    run([sys.executable, "scripts/production_guard.py"])
    run([sys.executable, "scripts/adversarial_guard_test.py"])
    require_exact_staged_tree()
    print("PRECOMMIT_AUDIT_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
