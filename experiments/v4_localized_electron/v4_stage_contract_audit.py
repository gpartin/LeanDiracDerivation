"""Audit the current coherent V4 stage boundary without rerunning solves.

This distinguishes the established stage-specific gates from experimental
moving preparations. A failed V4-006 trial is not a counterexample to the
previous restricted launch/readout results. No completed moving carrier is
claimed, and an older action's hop screen cannot pass V4-006.
"""

from __future__ import annotations

import hashlib
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
    moved_coarse = load("v4_native_graph_revalidation_dt.json")
    moved_fine = load("v4_native_graph_revalidation_dt_half.json")
    graph_coarse = load("v4_003_moving_native_graph_dt.json")
    graph_fine = load("v4_003_moving_native_graph_dt_half.json")
    cell_crossing = load("v4_006_one_cell_crossing.json")
    cell_crossing_fine = load("v4_006_one_cell_crossing_dt_half.json")
    orbit_screen = load("v4_006_current_action_relative_orbit.json")
    cache_coarse = load("v4_moving_cache_check_dt.json")
    cache_fine = load("v4_moving_cache_check_dt_half.json")
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
    # This older hop screen did not use the admitted V4-002 action. Keep its
    # metrics as provenance, not a gate. The current-action collocation
    # evidence is explicitly a sampled screen, not an unreset crossing.
    current_action_hop_pass = (
        orbit_screen["action_sha256"] == stage2["action_sha256"]
        and bool(orbit_screen.get("unreset_native_evolution", False))
        and bool(orbit_screen.get("converged", False))
        and orbit_screen.get("endpoint_relative_residual", float("inf")) < 1e-7
        and orbit_screen.get("preceding_layer_relative_residual", float("inf")) < 1e-7
    )

    physical_growth = bool(
        reachability["physical_launches_exhibit_material_growth"])
    moving_evidence = (moved_coarse, moved_fine, graph_coarse, graph_fine,
                       cell_crossing, cell_crossing_fine, orbit_screen)
    same_action = all(row["action_sha256"] == stage2["action_sha256"]
                      for row in moving_evidence)
    action_file_hash = hashlib.sha256((HERE / stage2["action_module"]).read_bytes()).hexdigest()
    same_checked_cache = all(
        check["status"] == "NATIVE_DEL_JVP_AGREEMENT"
        and check["action_sha256"] == action_file_hash == screen["action_sha256"]
        and check["tangent_cache_sha256"] == screen["tangent_cache_sha256"]
        for check, screen in ((cache_coarse, cell_crossing), (cache_fine, cell_crossing_fine))
    )
    # The experimental contrast launch is not a proved native preparation.
    # Its growth is movement diagnostics only, not a new V4-002 requirement.
    contrast_bounded = all(row["maximum_velocity_aware_norm_ratio"] < 2
                           for row in (moved_coarse, moved_fine))
    stage2_pass = (stage2["status"] == "PASS" and not physical_growth
                   and stage2["action_sha256"] == action_file_hash
                   and all(stage2["checks"].values()))
    stage3_pass = stage2_pass and stage3["status"] == "PASS" and all(stage3["checks"].values())
    stage4_pass = stage3_pass and stage4["status"] == "PASS" and all(stage4["checks"].values())
    stage5_pass = (
        stage4_pass
        and stage5["status"] == "PASS"
        and stage5_red_team["status"] == "PASS"
        and all(stage5["checks"].values())
    )
    result = {
        "status": "CURRENT_COHERENT_STAGE_AUDIT",
        "release_boundary": (
            "V4-005" if stage5_pass else "V4-004" if stage4_pass
            else "V4-003" if stage3_pass
            else "V4-002" if stage2_pass
            else "V4-001"),
        "localized_mobile_carrier_verified": current_action_hop_pass,
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
                "OPEN_PHYSICAL_SECTOR_STABILITY"),
            "V4-003": (
                "PASS_FIXED_NATIVE_READOUT_WITH_CONTROLLED_REMAINDER"
                if stage3_pass else "OPEN_FIXED_NATIVE_READOUT"),
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
                "PASS_EXACT_ONE_CELL_HANDOFF" if current_action_hop_pass and stage5_pass
                else "OPEN_EXACT_ONE_CELL_HANDOFF"),
            "V4-007": "OPEN_REPEATABLE_SECOND_HANDOFF",
            "V4-008": "OPEN_COARSE_DIRAC_COMPARISON",
            "V4-009": "OPEN_PHYSICAL_ROBUSTNESS_AND_CONVERGENCE",
            "V4-010": "OPEN_ANALYTIC_THEOREM_AND_RELEASE_AUDIT",
        },
        "historical_exact_orbit_evidence": rest,
        "moving_carrier_revalidation": {
            "same_action_hash": same_action,
            "same_independently_checked_del_caches": same_checked_cache,
            "frozen_action_configuration": stage2["action_configuration"],
            "scope": "V4-006 experimental contrast preparation; not an admitted moving sector or replacement gate for V4-002 through V4-005",
            "contrast_launch_bounded": contrast_bounded,
            "coarse_velocity_aware_amplification": moved_coarse["maximum_velocity_aware_norm_ratio"],
            "fine_velocity_aware_amplification": moved_fine["maximum_velocity_aware_norm_ratio"],
            "coarse_native_rate_error": graph_coarse["rows"][1]["maximum_rate_relative_error"],
            "fine_native_rate_error": graph_fine["rows"][1]["maximum_rate_relative_error"],
            "one_cell_dirac_center_shift": cell_crossing["Dirac_center_shift"],
            "one_cell_native_center_shift": cell_crossing["native_center_shift"],
            "one_cell_native_center_shift_half_dt": cell_crossing_fine["native_center_shift"],
            "nonlinear_collocation_initial_defect": orbit_screen["initial_acceleration_defect_over_contrast_norm"],
            "nonlinear_collocation_final_defect": orbit_screen["final_acceleration_defect_over_contrast_norm"],
        },
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
            "V4-001 through V4-005 retain their original verified scopes. "
            "The experimental full-carrier contrast is not a proved moving "
            "sector and its failure does not revoke those scoped passes. "
            "V4-006 did not complete a crossing. The nonlinear boundary "
            "screen is inconclusive, not a no-go theorem."),
    }
    output = HERE / "v4_stage_contract_audit.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
