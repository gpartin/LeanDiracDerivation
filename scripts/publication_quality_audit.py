"""Fail-closed audit of publication metadata, workflows and claim wording."""

from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def text(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def payload(relative: str) -> dict:
    return json.loads(text(relative))


def main() -> int:
    errors: list[str] = []
    readme = text("README.md")
    reproducibility = text("docs/REPRODUCIBILITY.md")
    framework = text("docs/FRAMEWORK.md")
    lean_workflow = text(".github/workflows/lean.yml")
    numerical_workflow = text(".github/workflows/numerical.yml")
    status = payload("project/status.json")
    readiness = payload("project/dirac019_release_readiness_status.json")

    if re.search(r"(?m)^\s*lake update\s*$", readme) or \
            re.search(r"(?m)^\s*lake update\s*$", reproducibility):
        errors.append("reproduction instructions mutate the locked dependency graph")
    if re.search(r"(?m)^\[$|^\]$", framework):
        errors.append("FRAMEWORK.md contains non-rendering display-math delimiters")
    if text("requirements.txt").splitlines() != ["numpy==2.3.5"]:
        errors.append("independent evidence declares unused or unlocked dependencies")
    if "import Mathlib\n" in text("LeanDirac/Foundation.lean"):
        errors.append("Foundation imports the entire Mathlib umbrella")
    if "maxRecDepth" in "".join(
            path.read_text(encoding="utf-8")
            for path in (ROOT / "LeanDirac").glob("*.lean")):
        errors.append("production proofs retain a recursion-depth override")
    heartbeat_values = [
        int(value)
        for path in (ROOT / "LeanDirac").glob("*.lean")
        for value in re.findall(r"set_option maxHeartbeats\s+(\d+)",
                                path.read_text(encoding="utf-8"))
    ]
    if any(value > 800_000 for value in heartbeat_values):
        errors.append("production proof heartbeat budget exceeds 800000")
    if status.get("gates", {}).get("DIRAC-017") != \
            "PASS_CONDITIONAL_CUBICAL_SPIN_HALF_REPRESENTATION":
        errors.append("machine-readable DIRAC-017 status erases conditionality")
    if status.get("gates", {}).get("DIRAC-018") != \
            "PASS_SCOPED_NATIVE_CUBICAL_FACTORIZATION_CLASSIFICATION":
        errors.append("machine-readable DIRAC-018 status erases classification scope")
    if status.get("gates", {}).get("DIRAC-019") != \
            "PASS_SCOPED_FINITE_POSITIVE_FREQUENCY_MODE_TRANSPORT":
        errors.append("machine-readable DIRAC-019 status erases transport scope")
    if readiness.get("package_artifacts") != [
            "LeanDiracDerivation-DIRAC-019.zip",
            "LeanDiracDerivation-DIRAC-019.zip.sha256"]:
        errors.append("current release-readiness artifact names are stale")
    for workflow_name, workflow in (
            ("lean", lean_workflow), ("numerical", numerical_workflow)):
        if re.search(r"uses:\s*[^\s]+@v\d", workflow):
            errors.append(f"{workflow_name} workflow uses a mutable action tag")
    if "if-no-files-found: error" not in lean_workflow:
        errors.append("publication artifact upload does not fail closed")
    for gate in (
            "dirac016_copy_translation",
            "dirac017_spin_half",
            "dirac018_factorization_classification",
            "dirac019_wavepacket_transport"):
        if not (ROOT / "experiments" / gate / "result.json").is_file():
            errors.append(f"missing independent evidence for {gate}")
    if "continuous-angle Hamiltonian covariance" not in \
            text("project/status.json"):
        errors.append("status omits the exact all-angle covariance scope")
    if "arbitrary nonlocal or flavor-mixing roots" not in \
            text("project/status.json"):
        errors.append("status omits DIRAC-018's restricted uniqueness scope")
    if "coherent real-space packet theorem in Lean" not in \
            text("project/status.json"):
        errors.append("status omits DIRAC-019's spectral-only boundary")
    claims = text("docs/CLAIMS.md")
    if "localized dynamically stable carrier" not in claims or \
            "A physical electron" not in claims:
        errors.append("claim boundary omits the localized-electron non-claim")

    if errors:
        print("PUBLICATION_QUALITY_AUDIT_FAIL")
        for error in errors:
            print(f"- {error}")
        return 1
    print("PUBLICATION_QUALITY_AUDIT_PASS")
    print(f"heartbeat_overrides={heartbeat_values}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
