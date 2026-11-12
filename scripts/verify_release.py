"""Single entry point for the formal, adversarial and contract audits."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run(*command: str) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> int:
    run(sys.executable, "scripts/production_guard.py")
    run(sys.executable, "scripts/adversarial_guard_test.py", "--skip-production-build")
    run(sys.executable, "scripts/derivation_contract_audit.py", "--strict")
    run(sys.executable, "scripts/final_chain_audit.py")
    print("RELEASE_VERIFICATION_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
