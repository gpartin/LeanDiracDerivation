"""One-command release audit for the admitted DIRAC-001--010 chain.

This is a read-only verifier, not a Git hook.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FRAMEWORK_LOCK = ROOT / "project" / "framework_lock.json"
PROOF_MANIFEST = ROOT / "project" / "proof_manifest.json"
EVIDENCE_MANIFEST = ROOT / "project" / "evidence_manifest.json"
RELEASE_MANIFEST = ROOT / "project" / "release_manifest.json"
STATUS = ROOT / "project" / "status.json"
REVIEW = ROOT / "project" / "no_bolton_review.json"
EVIDENCE = ROOT / "experiments" / "dirac010_standard_map" / "result.json"
ALLOWED_AXIOMS = {"propext", "Classical.choice", "Quot.sound"}


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def validate_manifest(path: Path) -> tuple[dict, list[str]]:
    payload = load(path)
    errors: list[str] = []
    for relative, expected in payload.get("files", {}).items():
        source = ROOT / relative
        if not source.is_file():
            errors.append(f"{path.name}: missing file: {relative}")
        elif sha256(source) != expected:
            errors.append(f"{path.name}: changed file: {relative}")
    return payload, errors


def production_sources() -> list[Path]:
    return sorted((ROOT / "LeanDirac").rglob("*.lean")) + [
        ROOT / "LeanDirac.lean", ROOT / "LeanDiracAudit.lean"
    ]


def lean_module_name(path: Path) -> str:
    return ".".join(path.relative_to(ROOT).with_suffix("").parts)


def production_modules() -> list[str]:
    return sorted({lean_module_name(path) for path in production_sources()})


def compiled_audit_source() -> str:
    imports = "\n".join(f"import {module}" for module in production_modules())
    reports = "\n".join(
        line.strip()
        for line in (ROOT / "LeanDiracAudit.lean").read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip().startswith("#print axioms ")
    )
    return f"{imports}\n\n#audit_lean_dirac_environment\n\n{reports}\n"


def run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command, cwd=ROOT, text=True, encoding="utf-8", errors="replace",
        capture_output=True, check=False
    )


def run_compiled_audit() -> subprocess.CompletedProcess[str]:
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", suffix=".lean",
        prefix="LeanDiracProductionAudit_", dir=ROOT, delete=False
    ) as driver:
        driver.write(compiled_audit_source())
        path = Path(driver.name)
    try:
        return run(["lake", "env", "lean", path.name])
    finally:
        path.unlink(missing_ok=True)


def parse_axiom_reports(output: str) -> dict[str, set[str]]:
    reports: dict[str, set[str]] = {}
    pattern = re.compile(
        r"'([A-Za-z_][A-Za-z0-9_.]*)' depends on axioms:\s*\[([^\]]*)\]",
        re.S,
    )
    for name, block in pattern.findall(output):
        reports.setdefault(name, set()).update(
            re.findall(r"[A-Za-z_][A-Za-z0-9_.]*", block)
        )
    for name in re.findall(
        r"'([A-Za-z_][A-Za-z0-9_.]*)' does not depend on any axioms", output
    ):
        reports.setdefault(name, set())
    return reports


def expected_axiom_reports() -> set[str]:
    source = (ROOT / "LeanDiracAudit.lean").read_text(encoding="utf-8")
    return set(re.findall(r"#print\s+axioms\s+([A-Za-z_][A-Za-z0-9_.]*)", source))


def validate_dirac010_evidence(payload: dict) -> list[str]:
    errors: list[str] = []
    if payload.get("status") != "PASS":
        errors.append("DIRAC-010 independent matrix verdict is not PASS")
    value = payload.get("max_error")
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        errors.append("DIRAC-010 maximum error is missing or non-finite")
    elif value > 1.0e-12:
        errors.append("DIRAC-010 maximum error exceeds 1e-12")
    errors_map = payload.get("errors")
    if not isinstance(errors_map, dict) or not errors_map:
        errors.append("DIRAC-010 component error table is missing")
    elif any(not isinstance(v, (int, float)) or not math.isfinite(v)
             for v in errors_map.values()):
        errors.append("DIRAC-010 component errors are incomplete or non-finite")
    return errors


def validate(run_builds: bool = True) -> list[str]:
    errors: list[str] = []
    framework, e1 = validate_manifest(FRAMEWORK_LOCK)
    proof, e2 = validate_manifest(PROOF_MANIFEST)
    evidence, e3 = validate_manifest(EVIDENCE_MANIFEST)
    release, e4 = validate_manifest(RELEASE_MANIFEST)
    errors.extend(e1 + e2 + e3 + e4)
    status = load(STATUS)
    review = load(REVIEW)
    if status.get("highest_passed_gate") != "DIRAC-010":
        errors.append("status does not stop at DIRAC-010")
    if status.get("next_gate") != "DIRAC-011":
        errors.append("DIRAC-011 is not the next gate")
    if status.get("completed") != 10 or status.get("total") != 15:
        errors.append("gate counts are incorrect")
    if review.get("status") != "PASS_ZERO_CRITICAL_HIGH":
        errors.append("critical/high red-team findings remain")
    if review.get("critical_findings") or review.get("high_findings"):
        errors.append("red-team finding lists are not empty")
    publication = "LeanDirac.current_highest_gate_is_dirac010"
    if proof.get("publication_theorem") != publication:
        errors.append("proof manifest names the wrong publication theorem")
    action_ids = {m.get("action_id") for m in (framework, proof, evidence, release)}
    if len(action_ids) != 1 or None in action_ids:
        errors.append(f"role manifests disagree on action id: {sorted(map(str, action_ids))}")
    actual = {path.relative_to(ROOT).as_posix() for path in production_sources()}
    locked = set(proof.get("files", {}))
    if actual != locked:
        errors.append(
            f"proof source set differs: extra={sorted(actual - locked)}, "
            f"missing={sorted(locked - actual)}"
        )
    framework_files = set(framework.get("files", {}))
    if not framework_files or not framework_files < locked:
        errors.append("framework lock is not a nonempty strict subset of proof files")
    if evidence.get("numerical_evidence") != EVIDENCE.relative_to(ROOT).as_posix():
        errors.append("evidence manifest names the wrong result")
    errors.extend(validate_dirac010_evidence(load(EVIDENCE)))
    if run_builds:
        build = run(["lake", "build", *production_modules()])
        if build.returncode:
            errors.append("production build failed\n" + build.stdout + build.stderr)
        audit = run_compiled_audit()
        if audit.returncode:
            errors.append("Lean axiom audit failed\n" + audit.stdout + audit.stderr)
        else:
            output = audit.stdout + audit.stderr
            if "LEAN_DIRAC_ENVIRONMENT_AUDIT_PASS" not in output:
                errors.append("compiled-environment audit marker is missing")
            audited = set(re.findall(
                r"LEAN_DIRAC_AUDITED_MODULE\s+([A-Za-z_][A-Za-z0-9_.]*)", output
            ))
            expected_modules = set(production_modules())
            if audited != expected_modules:
                errors.append(
                    "compiled module coverage differs: "
                    f"extra={sorted(audited - expected_modules)}, "
                    f"missing={sorted(expected_modules - audited)}"
                )
            reports = parse_axiom_reports(output)
            missing = expected_axiom_reports() - set(reports)
            if missing:
                errors.append(f"missing axiom reports: {sorted(missing)}")
            if publication not in reports:
                errors.append("publication theorem axiom report is missing")
            unexpected = set().union(*reports.values()) - ALLOWED_AXIOMS if reports else set()
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
    print("highest_passed_gate=DIRAC-010")
    print("next_gate=DIRAC-011 status=OPEN")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
