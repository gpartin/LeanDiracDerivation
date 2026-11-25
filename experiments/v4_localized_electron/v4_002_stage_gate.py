"""Aggregate only the coherent, exact-action evidence for V4-002."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import live_clifford_bond_action as action


HERE = Path(__file__).resolve().parent
ACTION = HERE/"live_clifford_bond_action.py"
EVIDENCE = HERE/"v4_002_exact_1197"
HALF = HERE/"v4_002_exact_1197_dt_half"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    rest = load(EVIDENCE/"rest_gate.json")
    consistency = load(EVIDENCE/"tangent_consistency_final.json")
    ten = load(EVIDENCE/"reach_10cycles_final.json")
    hundred = load(EVIDENCE/"reach_100cycles_final.json")
    half = load(HALF/"reach_10cycles_final.json")
    nonlinear = load(EVIDENCE/"nonlinear_all8_backward.json")
    charge = load(EVIDENCE/"noether_full_1e6.json")
    energy = load(EVIDENCE/"discrete_energy.json")
    box = load(EVIDENCE/"box_refinement_final.json")

    configuration = {
        "link_source": "joint_face_laplacian_plus_frame_oscillator",
        "scale": 1197.0,
        "spacetime_scale": 6/19**2,
        "parallel_matched_scale": 1.0,
        "parallel_potential_scale": 1.0,
    }
    config_rows = (
        consistency["action_configuration"],
        {
            "link_source": hundred["action_configuration"]["link_source"],
            "scale": hundred["action_configuration"]["action_scale"],
            "spacetime_scale": hundred["action_configuration"]["spacetime_scale"],
            "parallel_matched_scale": hundred["action_configuration"]["parallel_matched_scale"],
            "parallel_potential_scale": hundred["action_configuration"]["parallel_clock_potential_scale"],
        },
        energy["action_configuration"],
    )
    config_match = all(
        row["link_source"] == configuration["link_source"]
        and abs(row["scale"]-configuration["scale"]) < 1e-12
        and abs(row["spacetime_scale"]-configuration["spacetime_scale"]) < 1e-12
        and abs(row["parallel_matched_scale"]-
                configuration["parallel_matched_scale"]) < 1e-12
        and abs(row["parallel_potential_scale"]-
                configuration["parallel_potential_scale"]) < 1e-12
        for row in config_rows)
    default_action_locked = (
        action.LINK_SOURCE == configuration["link_source"]
        and action.SCALE == configuration["scale"]
        and abs(action.SPACETIME_PLAQUETTE_SCALE
                - configuration["spacetime_scale"]) < 1e-12
        and action.PARALLEL_MATCHED_SCALE == 1
        and action.PARALLEL_MATCHED_POTENTIAL_SCALE == 1
        and action.CHI_PAYLOAD == 1
        and action.PLAQUETTE_SCALE == 0
        and action.SYMPLECTIC_PLAQUETTE_SCALE == 0
        and action.PARALLEL_OVERLAP_SCALE == 0
        and action.PARALLEL_FACE_SCALE == 0
        and action.TRIANGLE_DELTA_SCALE == 0
        and action.MATERIAL_AREA_SCALE == 0)

    leakage_ratio = (half["one_step_dirac_sector_leakage"]
                     / ten["one_step_dirac_sector_leakage"])
    checks = {
        "one_action_configuration": config_match and default_action_locked,
        "exact_action_cache": (
            consistency["verdict"] == "TANGENT_CACHE_MATCHES_ACTION"
            and consistency["maximum_relative_error"] < 1e-12),
        "rest_orbit": rest["verdict"] == "LIVE_CLIFFORD_BOND_REST_NULL",
        "physical_launches_100_cycles": (
            hundred["case_count"] == 16 and hundred["cycles"] >= 100
            and not hundred["physical_launches_exhibit_material_growth"]
            and hundred["maximum_phase_space_norm_ratio"] < 2
            and hundred["final_minimum_phase_space_norm_ratio"] > 0.5),
        "second_order_timestep_refinement": (
            0.20 < leakage_ratio < 0.30
            and abs(half["maximum_phase_aligned_relative_error"]
                    - ten["maximum_phase_aligned_relative_error"]) < 0.002
            and abs(half["maximum_phase_space_norm_ratio"]
                    - ten["maximum_phase_space_norm_ratio"]) < 0.002),
        "unrestricted_nonlinear_short_run": (
            nonlinear["verdict"] == "LIVE_CLIFFORD_NONLINEAR_SHORT_RUN_PASS"
            and nonlinear["all_solvers_converged"]),
        "native_noether_charge": (
            charge["charge_mode"] == "full"
            and charge["verdict"] == "LIVE_FACE_NOETHER_CONTINUITY_PASS"),
        "discrete_energy": (
            energy["verdict"] == "DISCRETE_ENERGY_BOUNDED_PASS"),
        "box_stability": (
            box["verdict"] == "LIVE_FACE_GENERIC_BOX_STABILITY_PASS"),
    }
    passed = all(checks.values())
    result = {
        "stage": "V4-002",
        "status": "PASS" if passed else "OPEN",
        "action_module": ACTION.name,
        "action_sha256": sha256(ACTION),
        "action_configuration": configuration,
        "inactive_optional_action_scales": {
            "plaquette": action.PLAQUETTE_SCALE,
            "symplectic_plaquette": action.SYMPLECTIC_PLAQUETTE_SCALE,
            "parallel_overlap": action.PARALLEL_OVERLAP_SCALE,
            "parallel_face": action.PARALLEL_FACE_SCALE,
            "triangle_delta": action.TRIANGLE_DELTA_SCALE,
            "material_area": action.MATERIAL_AREA_SCALE,
        },
        "coefficient_provenance": {
            "scale": "existing native chi kinetic normalization B=chi0/kappa=1197",
            "spacetime_scale": "six face channels divided by chi0^2",
            "matched_frame_scales": "unit normalization of the parallel live frame",
            "fitted_to_Dirac": False,
        },
        "checks": checks,
        "metrics": {
            "maximum_exact_cache_relative_error": consistency["maximum_relative_error"],
            "rest_DEL_backward_error": rest["rest_DEL_backward_error"],
            "one_step_leakage_dt": ten["one_step_dirac_sector_leakage"],
            "one_step_leakage_dt_over_2": half["one_step_dirac_sector_leakage"],
            "leakage_refinement_ratio": leakage_ratio,
            "maximum_100_cycle_norm_ratio": hundred["maximum_phase_space_norm_ratio"],
            "minimum_final_100_cycle_norm_ratio": hundred["final_minimum_phase_space_norm_ratio"],
            "maximum_100_cycle_Dirac_comparator_error_diagnostic": hundred["maximum_phase_aligned_relative_error"],
            "maximum_global_charge_relative_drift": charge["maximum_global_charge_relative_drift"],
            "maximum_local_continuity_relative_defect": charge["maximum_local_continuity_relative_defect"],
            "energy_relative_drift": energy["rows"][-1]["maximum_relative_drift"],
        },
        "evidence": [
            "rest_gate.json", "tangent_consistency_final.json",
            "reach_10cycles_final.json", "reach_100cycles_final.json",
            "../v4_002_exact_1197_dt_half/reach_10cycles_final.json",
            "nonlinear_all8_backward.json", "noether_full_1e6.json",
            "discrete_energy.json", "box_refinement_final.json",
        ],
        "claim_boundary": (
            "V4-002 establishes bounded physical-sector stability, native "
            "charge/energy behavior, and timestep/box robustness for the "
            "declared research action. Its accumulated 100-cycle Dirac "
            "comparator error is diagnostic here and remains a V4-003/V4-005 "
            "mapping obligation. This does not prove cell translation."),
    }
    output = EVIDENCE/"v4_002_stage_gate.json"
    output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n",
                      encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
