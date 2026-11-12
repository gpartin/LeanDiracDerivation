"""Final read-only adversarial audit for DIRAC-001 through DIRAC-015."""

from __future__ import annotations

import json
from pathlib import Path

import production_guard as guard


ROOT = Path(__file__).resolve().parents[1]


def load(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def main() -> int:
    errors = guard.validate(run_builds=False)
    status = load("project/status.json")
    review = load("project/no_bolton_review.json")
    audit = load("project/dirac014_final_audit_status.json")
    reproduction = load("project/dirac015_clean_reproduction_status.json")
    proof = load("project/proof_manifest.json")
    boundary = status.get("claim_boundary", "").lower()

    checks = {
        "all_fifteen_gates_pass": (
            status.get("completed") == 15
            and status.get("highest_passed_gate") == "DIRAC-015"
            and all(value == "PASS" for value in status.get("gates", {}).values())
        ),
        "zero_critical_high": (
            review.get("status") == "PASS_ZERO_CRITICAL_HIGH"
            and not review.get("critical_findings")
            and not review.get("high_findings")
        ),
        "audit_gate_passes": audit.get("status") == "PASS",
        "clean_reproduction_gate_passes": reproduction.get("status") == "PASS",
        "final_theorem_is_action_rooted": (
            proof.get("highest_gate") == "DIRAC-015"
            and "LeanDirac/Dirac015.lean" in proof.get("files", {})
            and "LeanDirac/Publication.lean" in proof.get("files", {})
            and all(audit.get("checks", {}).get(name) is True for name in (
                "native_action_contains_no_dirac_matrix_spinor_or_projector",
                "finite_lattice_and_timestep_remainders_remain_explicit",
                "current_is_derived_from_action_and_hamiltonian_response",
            ))
        ),
        "publication_names_final_theorem": (
            proof.get("publication_theorem") ==
                "LeanDirac.current_highest_gate_is_dirac015"
            and reproduction.get("formal_theorem") ==
                "LeanDirac.lfm_free_dirac_continuum_limit"
            and reproduction.get("publication_theorem") ==
                proof.get("publication_theorem")
        ),
        "claim_boundary_is_retained": all(
            phrase in boundary for phrase in (
                "conditional free-dirac continuum derivation",
                "does not claim uniqueness",
                "electromagnetic coupling",
                "localized physical electron",
            )
        ),
        "clean_package_builder_present": (
            ROOT.joinpath("scripts/build_publication_package.py").is_file()
        ),
    }
    errors.extend(name for name, passed in checks.items() if not passed)

    print(json.dumps({
        "checks": checks,
        "verdict": "DIRAC_001_015_FINAL_AUDIT_PASS" if not errors
            else "DIRAC_001_015_FINAL_AUDIT_FAIL",
        "errors": errors,
    }, indent=2))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
