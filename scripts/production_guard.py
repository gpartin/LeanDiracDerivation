"""Fail closed if the DIRAC-001/002 production boundary drifts."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "project" / "action_lock.json"
STATUS = ROOT / "project" / "status.json"
REVIEW = ROOT / "project" / "no_bolton_review.json"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate() -> list[str]:
    errors: list[str] = []
    lock = load(LOCK)
    status = load(STATUS)
    review = load(REVIEW)

    if lock.get("immutable") is not True or lock.get("canonical") is not False:
        errors.append("action lock must be immutable and noncanonical")
    if lock.get("field_families") != 2:
        errors.append("action does not declare exactly two field families")
    if status.get("highest_passed_gate") != "DIRAC-002":
        errors.append("status does not stop at DIRAC-002")
    if status.get("next_gate") != "DIRAC-003":
        errors.append("DIRAC-003 is not the next gate")
    if review.get("status") != "PASS_ZERO_CRITICAL_HIGH":
        errors.append("no-bolt-on review is not cleared")
    if review.get("critical_findings") or review.get("high_findings"):
        errors.append("review contains critical/high findings")

    for relative, expected in lock.get("files", {}).items():
        path = ROOT / relative
        if not path.exists():
            errors.append(f"locked file missing: {relative}")
        elif sha256(path) != expected:
            errors.append(f"locked file changed: {relative}")

    production_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted((ROOT / "LeanDirac").glob("*.lean"))
    )
    for forbidden in ("sorry", "admit", "axiom"):
        if re.search(rf"\b{forbidden}\b", production_text):
            errors.append(f"forbidden proof token present: {forbidden}")

    imports = (ROOT / "LeanDirac.lean").read_text(encoding="utf-8")
    if imports.strip() != "import LeanDirac.Publication\n\n/-! Sole production facade for the LFM-to-Dirac derivation. -/":
        errors.append("public facade has more than the one approved import")

    return errors


def main() -> int:
    errors = validate()
    if errors:
        print("PRODUCTION_GUARD_FAIL")
        for error in errors:
            print(f"- {error}")
        return 1
    print("PRODUCTION_GUARD_PASS")
    print("highest_passed_gate=DIRAC-002")
    print("next_gate=DIRAC-003")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

