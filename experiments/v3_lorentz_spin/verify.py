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


def git_text(commit: str, path: str) -> str:
    return run(["git", "show", f"{commit}:{path}"]).replace("\r\n", "\n")


def normalized_namespace(text: str) -> str:
    return text.replace("LeanDirac/V2", "LeanDirac").replace(
        "LeanDirac.V2", "LeanDirac"
    )


def exact_namespace_migration(proof: str, path: str) -> bool:
    destination = ROOT / path
    if not destination.is_file():
        return False
    try:
        old = git_text(proof, path)
    except SystemExit:
        return False
    current = destination.read_text(encoding="utf-8").replace("\r\n", "\n")
    return normalized_namespace(old) == current


def migrated_module_paths(proof: str) -> set[str]:
    listing = run([
        "git", "ls-tree", "-r", "--name-only", proof, "LeanDirac/V2"
    ]).splitlines()
    migrated: set[str] = set()
    failures: list[str] = []
    for old_path in listing:
        if old_path == "LeanDirac/V2.lean" or not old_path.endswith(".lean"):
            continue
        new_path = "LeanDirac/" + Path(old_path).name
        migrated.add(new_path)
        destination = ROOT / new_path
        if not destination.is_file():
            failures.append(f"missing migrated module: {new_path}")
            continue
        expected = normalized_namespace(git_text(proof, old_path))
        current = destination.read_text(encoding="utf-8").replace("\r\n", "\n")
        if expected != current:
            failures.append(f"non-mechanical source change: {new_path}")
    if failures:
        raise SystemExit("V3.1 namespace migration audit failed:\n" + "\n".join(failures))
    if (ROOT / "LeanDirac" / "V2").exists():
        raise SystemExit("obsolete LeanDirac/V2 production directory still exists")
    return migrated


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
    migrated_modules = migrated_module_paths(proof)
    changed = run([
        "git", "diff", "--name-only", proof, "HEAD", "--",
        "LeanDirac", "LeanDirac.lean", "LeanDiracAudit.lean",
        "README.md", "docs", "lean-toolchain", "lake-manifest.json",
        "lakefile.toml", "scripts/verify_release.py",
        "experiments/v3_lorentz_spin",
    ]).splitlines()
    allowed_extensions = {
        "LeanDirac.lean",
        "LeanDiracAudit.lean",
        "README.md",
        "docs/CLAIMS.md",
        "docs/REPRODUCIBILITY.md",
        "experiments/v3_lorentz_spin/AxiomAudit.lean",
        "experiments/v3_lorentz_spin/artifact_audit.json",
        "experiments/v3_lorentz_spin/summary.json",
        "experiments/v3_lorentz_spin/verify.py",
    }
    forbidden_changes = []
    for name in changed:
        normalized = name.replace("\\", "/")
        is_cumulative_extension_module = (
            (normalized.startswith("LeanDirac/V32") or
             normalized.startswith("LeanDirac/V33") or
             normalized.startswith("LeanDirac/V34") or
             normalized.startswith("LeanDirac/V35"))
            and normalized.endswith(".lean")
        )
        is_cumulative_extension_doc = (
            normalized.startswith("docs/V32") or
            normalized.startswith("docs/V33") or
            normalized.startswith("docs/V34") or
            normalized.startswith("docs/V35")
        )
        is_namespace_only_change = exact_namespace_migration(proof, normalized)
        if (
            normalized not in allowed_extensions
            and not is_cumulative_extension_module
            and not is_cumulative_extension_doc
            and normalized not in migrated_modules
            and normalized != "LeanDirac/V2.lean"
            and not is_namespace_only_change
        ):
            forbidden_changes.append(normalized)
    if forbidden_changes:
        raise SystemExit(
            "sealed V3.1 source changed outside the cumulative extension boundary:\n"
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
