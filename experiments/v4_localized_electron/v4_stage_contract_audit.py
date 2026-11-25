"""Audit the current coherent V4 stage boundary without rerunning solves.

This does not promote an action.  It distinguishes the historical exact-orbit
endurance result from the current physically reachable perturbation failure
and verifies that V4-006 is not judged by a rigid sub-cell interpolation.
"""

from __future__ import annotations

import json
from pathlib import Path


HERE = Path(__file__).resolve().parent


def load(name: str) -> dict:
    return json.loads((HERE / name).read_text(encoding="utf-8"))


def main() -> None:
    stability = load("unit_geometric_nonlinear_summary.json")
    reachability = load("v4_002_reachable_sector_audit.json")
    hop = load("v4_006_stable_action_exact_hop_xplus_16_screen.json")
    lock = (HERE / "EXACT_TRANSLATION_MAP_LOCK.md").read_text(
        encoding="utf-8")

    rest = stability["zero_chi_charge_100_cycle"]
    rest_pass = (
        stability["status"] == "V4_UNIT_GEOMETRIC_EXACT_REST_ORBIT_PASS"
        and rest["updates"] >= 8500
        and rest["maximum_relative_shape_residual"] < 1e-10
        and rest["minimum_Psi_carrier_fidelity"] > 1 - 1e-12
        and rest["maximum_Psi_exterior_fraction"] < 1e-12
        and rest["maximum_interval_energy_relative_drift"] < 1e-10
    )
    endpoint_locked = all(text in lock for text in (
        "literal same-assignment translation",
        "both committed layers",
        "no receiver preload, projection, reset",
    ))
    hop_row = hop["rows"][0]
    hop_pass = (
        hop["verdict"] == "EXACT_ONE_CELL_RELATIVE_PERIODIC_ORBIT_FOUND"
        and bool(hop_row["converged"])
        and hop_row["unreset_second_cell_endpoint_residual"] < 1e-7
    )

    physical_growth = bool(
        reachability["physical_launches_exhibit_material_growth"])
    result = {
        "status": "CURRENT_COHERENT_STAGE_AUDIT",
        "release_boundary": "V4-001",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "stage_contract": {
            "V4-001": "PASS_EXACT_REST_EXISTENCE",
            "V4-002": (
                "FAIL_PHYSICAL_DIRAC_LAUNCH_GROWTH"
                if physical_growth else "OPEN_PHYSICAL_SECTOR_STABILITY"),
            "V4-003": "OPEN_FIXED_READOUT_WITH_CONTROLLED_REMAINDER",
            "V4-004": "OPEN_OPERATIONAL_SPIN",
            "V4-005": "PARTIAL_INTEGER_SHIFT_COVARIANCE_AND_CURRENT",
            "V4-006": (
                "PASS_EXACT_ONE_CELL_HANDOFF" if hop_pass
                else "OPEN_EXACT_ONE_CELL_HANDOFF"),
            "V4-007": "OPEN_REPEATABLE_SECOND_HANDOFF",
            "V4-008": "OPEN_COARSE_DIRAC_COMPARISON",
            "V4-009": "OPEN_PHYSICAL_ROBUSTNESS_AND_CONVERGENCE",
            "V4-010": "OPEN_ANALYTIC_THEOREM_AND_RELEASE_AUDIT",
        },
        "historical_exact_orbit_evidence": rest,
        "physical_sector_reachability": {
            "artifact": "v4_002_reachable_sector_audit.json",
            "case_count": reachability["case_count"],
            "one_step_dirac_sector_leakage": reachability[
                "one_step_dirac_sector_leakage"],
            "maximum_phase_space_norm_ratio": reachability[
                "maximum_phase_space_norm_ratio"],
            "maximum_phase_aligned_relative_error": reachability[
                "maximum_phase_aligned_relative_error"],
            "verdict": reachability["verdict"],
        },
        "finite_hop_screen": hop_row,
        "endpoint_locked": endpoint_locked,
        "neutral_translation_mode_required": False,
        "claim_boundary": (
            "The saved historical 100-cycle result proves only an exactly "
            "prepared orbit. The current complete-frame action materially "
            "amplifies all tested physical packet launches. The exact rigid "
            "hop remains an optional stronger control; V4-006 instead uses a "
            "solved crossing time and permits Dirac-predicted spreading."),
    }
    output = HERE / "v4_stage_contract_audit.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
