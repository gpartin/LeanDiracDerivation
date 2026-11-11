"""Manual clean-rebuild audit for the admitted DIRAC-001--009 chain.

This script is deliberately not a Git hook and never changes the index.  It
checks the repository tree that the caller invokes, runs Lean, audits axioms,
verifies the locked sources and checks the nondegenerate DIRAC-009 evidence.
"""

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
STABILITY = (
    ROOT
    / "experiments"
    / "dirac009_native_coupling"
    / "candidate_007_l4_nondegenerate_stability.json"
)
ALLOWED_AXIOMS = {"propext", "Classical.choice", "Quot.sound"}


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def production_sources() -> list[Path]:
    return sorted((ROOT / "LeanDirac").glob("*.lean")) + [
        ROOT / "LeanDirac.lean",
        ROOT / "LeanDiracAudit.lean",
    ]


def forbidden_tokens(source: str) -> list[str]:
    patterns = {
        "sorry": r"\bsorry\b",
        "admit": r"\badmit\b",
        "axiom declaration": r"(?m)^\s*axiom\b",
    }
    return [name for name, pattern in patterns.items() if re.search(pattern, source)]


def parse_axioms(output: str) -> set[str]:
    found: set[str] = set()
    for block in re.findall(r"depends on axioms:\s*\[([^\]]*)\]", output, re.S):
        found.update(re.findall(r"[A-Za-z_][A-Za-z0-9_.]*", block))
    return found


def validate_stability(payload: dict) -> list[str]:
    errors: list[str] = []
    if payload.get("verdict") != "PASS":
        errors.append("nondegenerate DIRAC-009 stability verdict is not PASS")
    if payload.get("lattice") != "4x4x4 periodic C19":
        errors.append("DIRAC-009 stability did not use the locked L=4 lattice")
    if payload.get("cycles") != 100:
        errors.append("DIRAC-009 stability did not run 100 cycles")
    runs = payload.get("runs", [])
    if [row.get("updates_per_cycle") for row in runs] != [64, 128, 256]:
        errors.append("DIRAC-009 timestep ensemble is incomplete")
    for row in runs:
        if not row.get("passes"):
            errors.append(f"stability row failed: {row.get('updates_per_cycle')}")
        if row.get("initial_incidence_energy", 0.0) <= 1.0e-14:
            errors.append("incidence energy was inactive in a stability row")
        if row.get("initial_incidence_residual_norm", 0.0) <= 1.0e-10:
            errors.append("incidence residual was inactive in a stability row")
        if row.get("relative_energy_drift", 1.0) >= 1.0e-4:
            errors.append("energy drift exceeded 1e-4")
        if max(row.get("maximums", {"missing": 1.0}).values()) >= 0.01:
            errors.append("pattern defect exceeded 1%")
    return errors


def run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
    )


def validate(run_builds: bool = True) -> list[str]:
    errors: list[str] = []
    lock = load(LOCK)
    status = load(STATUS)
    review = load(REVIEW)

    if status.get("highest_passed_gate") != "DIRAC-009":
        errors.append("status does not stop at DIRAC-009")
    if status.get("next_gate") != "DIRAC-010":
        errors.append("DIRAC-010 is not the next gate")
    if status.get("completed") != 9 or status.get("total") != 15:
        errors.append("gate counts are incorrect")
    if review.get("status") != "PASS_ZERO_CRITICAL_HIGH":
        errors.append("critical/high red-team findings remain")
    if review.get("critical_findings") or review.get("high_findings"):
        errors.append("red-team finding lists are not empty")
    if lock.get("publication_theorem") != "LeanDirac.current_highest_gate_is_dirac009":
        errors.append("action lock names the wrong publication theorem")

    actual = {path.relative_to(ROOT).as_posix() for path in production_sources()}
    locked = set(lock.get("files", {}))
    if actual != locked:
        errors.append(
            f"locked source set differs: extra={sorted(actual - locked)}, "
            f"missing={sorted(locked - actual)}"
        )
    for relative, expected in lock.get("files", {}).items():
        path = ROOT / relative
        if not path.exists():
            errors.append(f"locked source missing: {relative}")
        elif sha256(path) != expected:
            errors.append(f"locked source changed: {relative}")

    combined = "\n".join(path.read_text(encoding="utf-8") for path in production_sources())
    errors.extend(f"forbidden proof token present: {item}" for item in forbidden_tokens(combined))
    errors.extend(validate_stability(load(STABILITY)))

    if run_builds:
        build = run(["lake", "build"])
        if build.returncode:
            errors.append("lake build failed\n" + build.stdout + build.stderr)
        audit = run(["lake", "build", "LeanDiracAudit"])
        if audit.returncode:
            errors.append("Lean axiom audit build failed\n" + audit.stdout + audit.stderr)
        else:
            axioms = parse_axioms(audit.stdout + audit.stderr)
            unexpected = axioms - ALLOWED_AXIOMS
            if unexpected:
                errors.append(f"unexpected axioms: {sorted(unexpected)}")
    return errors


def main() -> int:
    errors = validate(run_builds=True)
    if errors:
        print("PRODUCTION_AUDIT_FAIL")
        for error in errors:
            print(f"- {error}")
        return 1
    print("PRODUCTION_AUDIT_PASS")
    print("highest_passed_gate=DIRAC-009")
    print("next_gate=DIRAC-010 status=OPEN")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
