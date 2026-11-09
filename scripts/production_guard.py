"""Fail closed if the admitted DIRAC-001/002/003 proof boundary drifts."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "project" / "action_lock.json"
STATUS = ROOT / "project" / "status.json"
REVIEW = ROOT / "project" / "no_bolton_review.json"

EXPECTED_GATE_MODULES = {
    "DIRAC-001": "LeanDirac/Dirac001.lean",
    "DIRAC-002": "LeanDirac/Dirac002.lean",
    "DIRAC-003": "LeanDirac/Dirac003.lean",
}

EXPECTED_GATES = {
    **{f"DIRAC-{index:03d}": "PASS" for index in range(1, 4)},
    "DIRAC-004": "NEXT",
    **{f"DIRAC-{index:03d}": "OPEN" for index in range(5, 16)},
}

EXPECTED_LOCKED_FILES = {
    "LeanDirac/Foundation.lean",
    "LeanDirac/C19Geometry.lean",
    "LeanDirac/CubeGeometry.lean",
    "LeanDirac/ExteriorArea.lean",
    "LeanDirac/NativeAction.lean",
    "LeanDirac/PreparedBackground.lean",
    "LeanDirac/Dirac001.lean",
    "LeanDirac/Dirac002.lean",
    "LeanDirac/Dirac003.lean",
    "LeanDirac/Publication.lean",
    "LeanDirac.lean",
    "LeanDiracAudit.lean",
}

AUDITED_THEOREMS = {
    "LeanDirac.dirac001_native_framework",
    "LeanDirac.dirac002_positive_c19_two_field_action",
    "LeanDirac.dirac002_prepared_geometry",
    "LeanDirac.cubic_unit_area_forces_wedge",
    "LeanDirac.dirac003_c19_two_reciprocal_action_variations",
    "LeanDirac.current_highest_gate_is_dirac003",
}

ALLOWED_AXIOMS = {"propext", "Classical.choice", "Quot.sound"}


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def import_lines(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip().startswith("import ")
    ]


def validate_axioms() -> list[str]:
    errors: list[str] = []
    completed = subprocess.run(
        ["lake", "env", "lean", "LeanDiracAudit.lean"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        return ["Lean axiom audit failed to execute:\n" + completed.stdout + completed.stderr]

    output = completed.stdout + completed.stderr
    seen: set[str] = set()
    for theorem in AUDITED_THEOREMS:
        escaped = re.escape(theorem)
        match = re.search(
            rf"'{escaped}' depends on axioms:\s*\[([^\]]*)\]", output
        )
        empty = re.search(rf"'{escaped}' does not depend on any axioms", output)
        if not match and not empty:
            errors.append(f"axiom audit did not report theorem: {theorem}")
            continue
        seen.add(theorem)
        dependencies = (
            {entry.strip() for entry in match.group(1).split(",") if entry.strip()}
            if match
            else set()
        )
        unexpected = dependencies - ALLOWED_AXIOMS
        if unexpected:
            errors.append(
                f"unapproved axiom dependency in {theorem}: {sorted(unexpected)}"
            )
    if seen != AUDITED_THEOREMS:
        errors.append("axiom audit theorem set is incomplete")
    return errors


def validate() -> list[str]:
    errors: list[str] = []
    lock = load(LOCK)
    status = load(STATUS)
    review = load(REVIEW)

    if lock.get("immutable") is not True or lock.get("canonical") is not False:
        errors.append("action lock must be immutable and noncanonical")
    if lock.get("field_families") != 2:
        errors.append("action does not declare exactly two field families")
    if status.get("highest_passed_gate") != "DIRAC-003":
        errors.append("status does not stop at DIRAC-003")
    if status.get("next_gate") != "DIRAC-004":
        errors.append("DIRAC-004 is not the next gate")
    if status.get("completed") != 3 or status.get("total") != 15:
        errors.append("gate completion counts are incorrect")
    if status.get("gates") != EXPECTED_GATES:
        errors.append("complete fifteen-gate status map is incorrect")
    if status.get("gate_modules") != EXPECTED_GATE_MODULES:
        errors.append("gate-to-module map is missing or incorrect")
    if review.get("status") != "PASS_ZERO_CRITICAL_HIGH":
        errors.append("no-bolt-on review is not cleared")
    if review.get("critical_findings") or review.get("high_findings"):
        errors.append("review contains critical/high findings")

    locked_files = lock.get("files", {})
    if set(locked_files) != EXPECTED_LOCKED_FILES:
        errors.append("action lock does not contain exactly the approved proof files")
    if lock.get("publication_theorem") != "LeanDirac.current_highest_gate_is_dirac003":
        errors.append("action lock does not name the DIRAC-003 publication theorem")

    for relative, expected in locked_files.items():
        path = ROOT / relative
        if not path.exists():
            errors.append(f"locked file missing: {relative}")
        elif sha256(path) != expected:
            errors.append(f"locked file changed: {relative}")

    production_paths = sorted((ROOT / "LeanDirac").rglob("*.lean"))
    production_paths.extend([ROOT / "LeanDirac.lean", ROOT / "LeanDiracAudit.lean"])
    production_text = "\n".join(
        path.read_text(encoding="utf-8") for path in production_paths
    )
    for forbidden in ("sorry", "admit", "axiom"):
        if re.search(rf"\b{forbidden}\b", production_text):
            errors.append(f"forbidden proof token present: {forbidden}")

    if import_lines(ROOT / "LeanDirac.lean") != ["import LeanDirac.Publication"]:
        errors.append("public facade has more than the one approved import")

    if import_lines(ROOT / "LeanDirac" / "Dirac001.lean") != [
        "import LeanDirac.C19Geometry"
    ]:
        errors.append("DIRAC-001 crosses its foundation/C19 dependency boundary")

    if import_lines(ROOT / "LeanDirac" / "Dirac002.lean") != [
        "import LeanDirac.Dirac001",
        "import LeanDirac.PreparedBackground",
    ]:
        errors.append("DIRAC-002 crosses its approved dependency boundary")

    if import_lines(ROOT / "LeanDirac" / "Dirac003.lean") != [
        "import LeanDirac.Dirac002"
    ]:
        errors.append("DIRAC-003 crosses its approved dependency boundary")

    if import_lines(ROOT / "LeanDirac" / "Publication.lean") != [
        "import LeanDirac.Dirac003"
    ]:
        errors.append("publication root does not stop at DIRAC-003")

    errors.extend(validate_axioms())

    return errors


def main() -> int:
    errors = validate()
    if errors:
        print("PRODUCTION_GUARD_FAIL")
        for error in errors:
            print(f"- {error}")
        return 1
    print("PRODUCTION_GUARD_PASS")
    print("highest_passed_gate=DIRAC-003")
    print("next_gate=DIRAC-004")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

