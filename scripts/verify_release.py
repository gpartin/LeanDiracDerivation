"""Single entry point for the V2 first-principles Dirac proof audit."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run(*command: str) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> int:
    run(sys.executable, "experiments/v2_first_principles_free_dirac/verify.py")
    print("V2_RELEASE_VERIFICATION_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
