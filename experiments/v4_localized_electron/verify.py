"""Verify the scoped V4 exact-localized-rest result."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ALLOWED = {"propext", "Classical.choice", "Quot.sound"}
THEOREMS = (
    "LeanDirac.V4.exact_localized_rest_seed",
    "LeanDirac.V4.prior_free_lorentz_spin_certificate",
)


def run(command: list[str]) -> str:
    result = subprocess.run(
        command,
        cwd=ROOT,
        check=False,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
    )
    output = result.stdout + result.stderr
    if result.returncode:
        raise SystemExit(f"command failed: {' '.join(command)}\n{output}")
    return output


def main() -> None:
    build = run(["lake", "build", "LeanDirac.V4.LocalizedRestSeed"])
    if "Build completed successfully" not in build:
        raise SystemExit("Lean build success marker is missing")

    audit = run(["lake", "env", "lean", str(HERE / "AxiomAudit.lean")])
    if "LEAN_DIRAC_V4_LOCALIZED_REST_AXIOM_AUDIT_PASS" not in audit:
        raise SystemExit("V4 compiled axiom-audit marker is missing")
    for theorem in THEOREMS:
        match = re.search(
            rf"'{re.escape(theorem)}' depends on axioms:\s*\[([^\]]*)\]",
            audit,
        )
        if not match:
            raise SystemExit(f"missing axiom report for {theorem}")
        actual = {part.strip() for part in match.group(1).split(",") if part.strip()}
        if actual - ALLOWED:
            raise SystemExit(f"nonstandard axioms for {theorem}: {actual - ALLOWED}")

    print("V4_LOCALIZED_REST_VERIFICATION_PASS")


if __name__ == "__main__":
    main()
