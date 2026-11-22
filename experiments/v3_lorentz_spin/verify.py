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
    run([
        "git", "diff", "--quiet", proof, "HEAD", "--",
        "LeanDirac", "LeanDirac.lean", "LeanDiracAudit.lean",
        "README.md", "docs/LORENTZ_SPIN_CHAIN.md",
        "docs/LORENTZ_SPIN_RED_TEAM.md",
    ])

    build = run(["lake", "build", *audit["lean_targets"]])
    if "Build completed successfully" not in build:
        raise SystemExit("Lean build success marker is missing")
    axioms = run(["lake", "env", "lean", str(HERE / "AxiomAudit.lean")])
    if "LEAN_DIRAC_ENVIRONMENT_AUDIT_PASS" not in axioms:
        raise SystemExit("compiled environment audit marker is missing")
    if "LEAN_DIRAC_V3_AXIOM_AUDIT_PASS" not in axioms:
        raise SystemExit("V3 axiom audit marker is missing")
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
    print("V3_LORENTZ_SPIN_RELEASE_VERIFICATION_PASS")


if __name__ == "__main__":
    main()
