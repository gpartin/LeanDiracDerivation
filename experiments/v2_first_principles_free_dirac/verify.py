"""Reproduce the scoped V2 first-principles free-Dirac artifact checks.

Proof trust comes from Lean compilation plus `LeanDirac.EnvironmentAudit`,
which inspects compiled declarations and their transitive axiom dependencies.
The source checks below verify the publication manifest; they are not used as
a substitute for Lean's parser or environment.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import tempfile
from datetime import date
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def sha256(path: Path) -> str:
    # Git may materialize text files with CRLF on Windows and LF on Linux.
    # Seal their repository content rather than the platform-specific checkout.
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content.replace(b"\r\n", b"\n")).hexdigest()


def run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=ROOT,
        check=False,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
    )


def require_success(result: subprocess.CompletedProcess[str], label: str) -> str:
    output = result.stdout + result.stderr
    if result.returncode:
        raise SystemExit(f"{label} failed\n{output}")
    return output


def require_equal(actual: object, expected: object, label: str) -> None:
    if actual != expected:
        raise SystemExit(f"{label}: {actual!r} != {expected!r}")


def historical_source_path(relative: str) -> str:
    base_modules = {
        "LeanDirac/Foundation.lean",
        "LeanDirac/ExteriorArea.lean",
        "LeanDirac/C19Geometry.lean",
        "LeanDirac/NativeAction.lean",
    }
    if relative in base_modules:
        return relative
    if relative == "LeanDirac.lean":
        return "LeanDirac/V2.lean"
    if relative.startswith("LeanDirac/"):
        return "LeanDirac/V2/" + Path(relative).name
    return relative


def git_blob(commit: str, relative: str) -> bytes:
    result = subprocess.run(
        ["git", "show", f"{commit}:{relative}"],
        cwd=ROOT,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode:
        raise SystemExit(
            f"sealed Git object is missing: {commit}:{relative}\n"
            + result.stderr.decode("utf-8", errors="replace")
        )
    return result.stdout


def main() -> None:
    summary = json.loads((HERE / "summary.json").read_text(encoding="utf-8"))
    audit = json.loads((HERE / "artifact_audit.json").read_text(encoding="utf-8"))
    report = (HERE / "REPORT.md").read_text(encoding="utf-8")

    final_status = "PASS_WITH_SCOPED_OPEN_GATES_AND_COMPILED_AUDIT"
    require_equal(summary["artifact_version"], audit["audit_version"], "audit version")
    require_equal(summary["artifact_status"], final_status, "summary final status")
    require_equal(audit["status"], final_status, "audit final status")
    if not audit.get("verified_date"):
        raise SystemExit("verified_date is missing")
    try:
        date.fromisoformat(audit["verified_date"])
    except ValueError as error:
        raise SystemExit(f"verified_date is not ISO-8601: {error}") from error

    for name in audit["required_files"]:
        if not (HERE / name).is_file():
            raise SystemExit(f"missing artifact: {name}")

    required_report_phrases = (
        "LOSSLESS_FOUR_COMPONENT_FIRST_ORDER_BRIDGE    = PASS",
        "SAME_READOUT_DIRAC_DENSITY_CURRENT_CONTINUITY  = PASS",
        "ARBITRARY_NOETHER_EQUALS_POSITIVE_DIRAC_DENSITY = PROVED_FALSE",
        "LOCALIZED_NONLINEAR_MOVING_ELECTRON            = SEPARATE_OPEN_GATE",
        "free linearized Dirac theorem",
        "does not dynamically align",
    )
    missing = [item for item in required_report_phrases if item not in report]
    if missing:
        raise SystemExit(f"REPORT.md misses scoped conclusions: {missing}")
    if "UniformVacuumDiracBridge" in report:
        raise SystemExit("REPORT.md still credits the retired auxiliary bridge")

    sources = [ROOT / relative for relative in summary["lean_modules"]]
    expected_source_set = set(audit["source_hashes_sha256"])
    actual_source_set = set(summary["lean_modules"])
    if actual_source_set != expected_source_set:
        raise SystemExit(
            "source manifest/hash coverage differs: "
            f"extra={sorted(actual_source_set - expected_source_set)}, "
            f"missing={sorted(expected_source_set - actual_source_set)}"
        )
    for path in sources:
        if not path.is_file():
            raise SystemExit(f"missing Lean source: {path.relative_to(ROOT)}")

    proof_commit = audit["proof_source_commit"]
    for relative, expected in audit["source_hashes_sha256"].items():
        sealed_path = historical_source_path(relative)
        actual = sha256_bytes(git_blob(proof_commit, sealed_path))
        if actual != expected:
            raise SystemExit(
                f"sealed Lean source mismatch: {sealed_path}: {actual} != {expected}"
            )

    support_files = audit["audit_support_files"]
    support_hashes = audit["audit_support_hashes_sha256"]
    require_equal(set(support_files), set(support_hashes), "audit-support hash coverage")
    for relative in support_files:
        path = ROOT / relative
        if not path.is_file():
            raise SystemExit(f"missing audit-support file: {relative}")
        actual = sha256_bytes(git_blob(proof_commit, relative))
        if actual != support_hashes[relative]:
            raise SystemExit(
                f"sealed audit-support object changed: {relative}: "
                f"{actual} != {support_hashes[relative]}"
            )

    status_output = require_success(
        run(["git", "status", "--porcelain"]), "working-tree cleanliness check"
    )
    if status_output.strip():
        raise SystemExit("final verification requires a clean checkout")
    require_success(
        run(["git", "cat-file", "-e", f"{proof_commit}^{{commit}}"]),
        "proof-source commit lookup",
    )
    require_success(
        run(["git", "merge-base", "--is-ancestor", proof_commit, "HEAD"]),
        "proof-source ancestry check",
    )
    build_output = require_success(
        run(["lake", "build", *audit["lean_targets"]]),
        "Lean target build",
    )

    axiom_output = require_success(
        run(["lake", "env", "lean", str(HERE / "AxiomAudit.lean")]),
        "compiled-environment axiom audit",
    )
    if "LEAN_DIRAC_ENVIRONMENT_AUDIT_PASS" not in axiom_output:
        raise SystemExit("compiled-environment audit pass marker is missing")

    audited_modules = set(
        re.findall(r"LEAN_DIRAC_AUDITED_MODULE\s+([^\s]+)", axiom_output)
    )
    require_equal(
        audited_modules,
        set(audit["compiled_audited_modules"]) - {"LeanDirac"},
        "compiled LeanDirac module closure",
    )
    forbidden_modules = set(audit["forbidden_active_imports"])
    if audited_modules & forbidden_modules:
        raise SystemExit(
            "forbidden compiled modules are active: "
            f"{sorted(audited_modules & forbidden_modules)}"
        )

    declaration_source = [
        "import LeanDirac.CompleteFreeDirac",
        "",
        *[f"#check LeanDirac.{name}" for name in audit["required_theorems"]],
        "",
        '#eval IO.println "LEAN_DIRAC_DECLARATION_AUDIT_PASS"',
        "",
    ]
    with tempfile.TemporaryDirectory(prefix="lean-dirac-declaration-audit-") as temp_dir:
        declaration_audit = Path(temp_dir) / "DeclarationAudit.lean"
        declaration_audit.write_text("\n".join(declaration_source), encoding="utf-8")
        declaration_output = require_success(
            run(["lake", "env", "lean", str(declaration_audit)]),
            "compiled declaration audit",
        )
    if "LEAN_DIRAC_DECLARATION_AUDIT_PASS" not in declaration_output:
        raise SystemExit("compiled declaration audit pass marker is missing")

    allowed_axioms = set(audit["allowed_axioms"])
    for theorem_name in audit["axiom_theorems"]:
        match = re.search(
            rf"'{re.escape(theorem_name)}' depends on axioms:\s*\[([^\]]*)\]",
            axiom_output,
        )
        if not match:
            raise SystemExit(f"#print axioms report is missing: {theorem_name}")
        actual_axioms = {
            item.strip() for item in match.group(1).split(",") if item.strip()
        }
        if actual_axioms - allowed_axioms:
            raise SystemExit(
                f"{theorem_name} depends on nonstandard axioms: "
                f"{sorted(actual_axioms - allowed_axioms)}"
            )

    verdicts = summary["verdicts"]
    expected_verdicts = {
        "lossless_four_component_first_order_bridge": "PASS",
        "uniform_tangent_u1_continuity": "PASS",
        "canonical_spectral_readout_bijective": "PASS",
        "same_readout_exact_c19_verlet_position_chain": "PASS",
        "positive_mode_same_state_density_current": "PASS",
        "finite_c19_current_remainder_bound_and_limit": "PASS",
        "integrated_positive_packet_charge_bridge": "PASS",
        "integrated_positive_packet_current_bridge":
            "PASS_WITH_EXPLICIT_C19_REMAINDER",
        "pointwise_positive_packet_charge_bridge": "PROVED_FALSE",
        "arbitrary_noether_equals_positive_dirac_density": "PROVED_FALSE",
    }
    for key, expected in expected_verdicts.items():
        require_equal(verdicts.get(key), expected, f"verdict {key}")

    if not build_output.strip():
        raise SystemExit("Lean target build produced no audit output")
    print("Lean target build: PASS")
    print(
        "v2_first_principles_free_dirac: "
        "PASS_WITH_SCOPED_OPEN_GATES_AND_COMPILED_AXIOM_AUDIT"
    )


if __name__ == "__main__":
    main()
