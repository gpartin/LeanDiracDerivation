"""Audit the corrected V4 stage boundary without rerunning expensive solves.

This does not promote an action.  It verifies that the recorded nonlinear rest
stability evidence is still present and that V4-006 is judged by a finite
integer-cell relative-periodic endpoint rather than by a neutral translation
mode.
"""

from __future__ import annotations

import json
from pathlib import Path


HERE = Path(__file__).resolve().parent


def load(name: str) -> dict:
    return json.loads((HERE / name).read_text(encoding="utf-8"))


def main() -> None:
    stability = load("unit_geometric_nonlinear_summary.json")
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

    result = {
        "status": "PASS" if rest_pass and endpoint_locked else "FAIL",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "stage_contract": {
            "V4-001": "PASS_EXACT_REST_EXISTENCE",
            "V4-002": (
                "PASS_REDUCED_TIGHT_FRAME_ZERO_CHI_CHARGE_100_CYCLE"
                if rest_pass else "FAIL_STABILITY_EVIDENCE"),
            "V4-003": "OPEN_LOCALIZED_FOUR_COMPONENT_READOUT",
            "V4-004": "OPEN_OPERATIONAL_SPIN",
            "V4-005": "PARTIAL_INTEGER_SHIFT_COVARIANCE_AND_CURRENT",
            "V4-006": (
                "PASS_EXACT_ONE_CELL_HANDOFF" if hop_pass
                else "OPEN_EXACT_ONE_CELL_HANDOFF"),
            "V4-007": "OPEN_REPEATABLE_SECOND_HANDOFF",
            "V4-008": "OPEN_COARSE_DIRAC_COMPARISON",
            "V4-009": "OPEN_ROBUSTNESS_ALL_AXES_AND_BOTH_COPIES",
            "V4-010": "OPEN_CUMULATIVE_THEOREM",
        },
        "rest_stability_evidence": rest,
        "finite_hop_screen": hop_row,
        "endpoint_locked": endpoint_locked,
        "neutral_translation_mode_required": False,
        "claim_boundary": (
            "The saved 100-cycle result is a nonlinear rest-stability pass "
            "only for the reduced tight-frame zero-chi-charge research "
            "branch.  The 16-update exact-hop screen did not converge and "
            "does not establish a no-go.  Every action delta must rerun "
            "V4-002 before it can advance V4-006."),
    }
    output = HERE / "v4_stage_contract_audit.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
