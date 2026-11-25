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
    reachability = load("v4_002_exact_1197/reach_100cycles_final.json")
    stage2 = load("v4_002_exact_1197/v4_002_stage_gate.json")
    stage3 = load("v4_003_stage_gate.json")
    stage4 = load("v4_004_stage_gate.json")
    stage5 = load("v4_005_stage_gate.json")
    stage5_red_team = load("v4_005_red_team.json")
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
    stage2_pass = stage2["status"] == "PASS" and not physical_growth
    stage3_pass = stage2_pass and stage3["status"] == "PASS"
    stage4_pass = stage3_pass and stage4["status"] == "PASS"
    stage5_pass = (
        stage4_pass
        and stage5["status"] == "PASS"
        and stage5_red_team["status"] == "PASS"
    )
    result = {
        "status": "CURRENT_COHERENT_STAGE_AUDIT",
        "release_boundary": (
            "V4-005" if stage5_pass else "V4-004" if stage4_pass
            else "V4-003" if stage3_pass
            else "V4-002" if stage2_pass
            else "V4-001"),
        "canonical_action_changed": False,
        "research_action_admitted": stage2_pass,
        "research_action_scope": (
            "frozen V4 coherent chain only; canonical V40 and public main "
            "remain unchanged until V4-010"),
        "stage_contract": {
            "V4-001": "PASS_EXACT_REST_EXISTENCE",
            "V4-002": (
                "PASS_PHYSICAL_SECTOR_STABILITY"
                if stage2_pass else
                "FAIL_PHYSICAL_DIRAC_LAUNCH_GROWTH"
                if physical_growth else "OPEN_PHYSICAL_SECTOR_STABILITY"),
            "V4-003": (
                "PASS_FIXED_NATIVE_READOUT_WITH_CONTROLLED_REMAINDER"
                if stage3_pass else
                "NEXT_FIXED_READOUT_WITH_CONTROLLED_REMAINDER"),
            "V4-004": (
                "PASS_LOCALIZED_READOUT_SPIN_HALF" if stage4_pass else
                "NEXT_OPERATIONAL_SPIN" if stage3_pass else
                "BLOCKED_BY_V4_003"),
            "V4-005": (
                "PASS_SHORT_TIME_DIRAC_TRANSPORT_AND_NATIVE_CURRENT"
                if stage5_pass else
                "NEXT_SHORT_TIME_DIRAC_TRANSPORT_AND_NATIVE_CURRENT"
                if stage4_pass else "BLOCKED_BY_V4_004"),
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
            "artifact": "v4_002_exact_1197/reach_100cycles_final.json",
            "case_count": reachability["case_count"],
            "one_step_dirac_sector_leakage": reachability[
                "one_step_dirac_sector_leakage"],
            "maximum_phase_space_norm_ratio": reachability[
                "maximum_phase_space_norm_ratio"],
            "maximum_phase_aligned_relative_error": reachability[
                "maximum_phase_aligned_relative_error"],
            "verdict": reachability["verdict"],
        },
        "fixed_phase_space_readout": {
            "artifact": "v4_003_stage_gate.json",
            "status": stage3["status"],
            "maximum_coarse_complementary_residual": max(
                row["coarse_complementary_residual"]
                for row in stage3["rows"]),
            "maximum_fine_complementary_residual": max(
                row["fine_complementary_residual"]
                for row in stage3["rows"]),
        },
        "localized_spin_readout": {
            "artifact": "v4_004_stage_gate.json",
            "status": stage4["status"],
            "Lean_theorem": stage4["Lean_theorem"],
            "graph_remainder_refinement_ratio": stage4[
                "graph_remainder_refinement_ratio"],
        },
        "short_time_transport_current": {
            "artifact": "v4_005_stage_gate.json",
            "status": stage5["status"],
            "red_team_status": stage5_red_team["status"],
            "Lean_theorem": stage5["Lean_theorem"],
            "metrics": stage5["metrics"],
        },
        "finite_hop_screen": hop_row,
        "endpoint_locked": endpoint_locked,
        "neutral_translation_mode_required": False,
        "claim_boundary": (
            "V4-002 passes bounded physical-sector stability, V4-003 passes "
            "a fixed native-action-derived four-component readout, and "
            "V4-004 proves its cubical spin-one-half rotation law, and "
            "V4-005 closes short-time signed transport and center-resolved "
            "Noether current on the frozen B-normalized complete-frame "
            "research action. A complete cell crossing remains V4-006. The "
            "exact rigid hop remains "
            "an optional stronger "
            "control; V4-006 uses a solved crossing time and permits "
            "Dirac-predicted spreading."),
    }
    output = HERE / "v4_stage_contract_audit.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
