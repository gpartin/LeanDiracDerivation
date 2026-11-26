"""Reproduce the cumulative V3 Lorentz/spin release audit."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


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
    summary = json.loads((HERE / "summary.json").read_text(encoding="utf-8"))
    audit = json.loads((HERE / "artifact_audit.json").read_text(encoding="utf-8"))
    if summary["artifact_version"] != audit["audit_version"]:
        raise SystemExit("summary/audit version mismatch")
    if summary["artifact_status"] != audit["status"]:
        raise SystemExit("summary/audit status mismatch")
    if summary["proof_source_commit"] != audit["proof_source_commit"]:
        raise SystemExit("summary/audit proof commit mismatch")
    for name in audit["required_files"]:
        if not (HERE / name).is_file():
            raise SystemExit(f"missing V3 artifact file: {name}")

    if run(["git", "status", "--porcelain"]).strip():
        raise SystemExit("V3 verification requires a clean checkout")
    proof = audit["proof_source_commit"]
    run(["git", "cat-file", "-e", f"{proof}^{{commit}}"])
    run(["git", "merge-base", "--is-ancestor", proof, "HEAD"])
    changed = run([
        "git", "diff", "--name-only", proof, "HEAD", "--",
        "LeanDirac", "LeanDirac.lean", "LeanDiracAudit.lean",
        "README.md", "docs", "lean-toolchain", "lake-manifest.json",
        "lakefile.toml", "scripts/verify_release.py",
        "experiments/v3_lorentz_spin",
    ]).splitlines()
    allowed_extensions = {
        "LeanDirac.lean",
        "README.md",
        "experiments/v3_lorentz_spin/artifact_audit.json",
        "experiments/v3_lorentz_spin/summary.json",
        "experiments/v3_lorentz_spin/verify.py",
    }
    forbidden_changes = []
    for name in changed:
        normalized = name.replace("\\", "/")
        is_v32_module = (
            normalized.startswith("LeanDirac/V2/V32")
            and normalized.endswith(".lean")
        )
        is_v32_doc = normalized.startswith("docs/V32")
        if (
            normalized not in allowed_extensions
            and not is_v32_module
            and not is_v32_doc
        ):
            forbidden_changes.append(normalized)
    if forbidden_changes:
        raise SystemExit(
            "sealed V3.1 source changed outside the V3.2 extension boundary:\n"
            + "\n".join(forbidden_changes)
        )

    build = run(["lake", "build", *audit["lean_targets"]])
    if "Build completed successfully" not in build:
        raise SystemExit("Lean build success marker is missing")
    axioms = run(["lake", "env", "lean", str(HERE / "AxiomAudit.lean")])
    if "LEAN_DIRAC_ENVIRONMENT_AUDIT_PASS" not in axioms:
        raise SystemExit("compiled environment audit marker is missing")
    if "LEAN_DIRAC_V3_AXIOM_AUDIT_PASS" not in axioms:
        raise SystemExit("V3 axiom audit marker is missing")
    regression = run([
        "lake", "env", "lean",
        str(HERE / "PauliLubanskiRegression.lean"),
    ])
    if "LEAN_DIRAC_V3_PAULI_LUBANSKI_REGRESSION_PASS" not in regression:
        raise SystemExit("Pauli--Lubanski regression marker is missing")
    allowed = set(audit["allowed_axioms"])
    for theorem in audit["axiom_theorems"]:
        match = re.search(
            rf"'{re.escape(theorem)}' depends on axioms:\s*\[([^\]]*)\]",
            axioms,
        )
        if not match:
            raise SystemExit(f"missing axiom report for {theorem}")
        actual = {item.strip() for item in match.group(1).split(",") if item.strip()}
        if actual - allowed:
            raise SystemExit(f"nonstandard axioms for {theorem}: {actual - allowed}")
    for theorem in audit["regression_axiom_theorems"]:
        match = re.search(
            rf"'{re.escape(theorem)}' depends on axioms:\s*\[([^\]]*)\]",
            regression,
        )
        if not match:
            raise SystemExit(f"missing regression axiom report for {theorem}")
        actual = {item.strip() for item in match.group(1).split(",") if item.strip()}
        if actual - allowed:
            raise SystemExit(
                f"nonstandard regression axioms for {theorem}: {actual - allowed}"
            )
    print("V3_LORENTZ_SPIN_RELEASE_VERIFICATION_PASS")


if __name__ == "__main__":
    main()
