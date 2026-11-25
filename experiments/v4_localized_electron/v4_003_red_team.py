"""Adversarial audit of V4-003's fixed native phase-space readout."""

from __future__ import annotations

import ast
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent


def load(name: str) -> dict:
    return json.loads((HERE / name).read_text(encoding="utf-8"))


def main() -> None:
    coarse = load("v4_003_dt_native.json")
    fine = load("v4_003_dt_half_native.json")
    gate = load("v4_003_stage_gate.json")
    stage2 = load("v4_002_exact_1197/v4_002_stage_gate.json")
    source_path = HERE / "v4_003_fixed_phase_space_readout.py"
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        node.module or ""
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    }
    forbidden_imports = sorted(name for name in imports if any(
        token in name.lower() for token in ("dirac", "comparator")))
    all_rows = coarse["rows"] + fine["rows"]
    checks = {
        "stage_gate_passes": gate["status"] == "PASS",
        "same_frozen_action_as_v4_002": (
            gate["action_sha256"] == stage2["action_sha256"]),
        "no_Dirac_or_comparator_import": not forbidden_imports,
        "native_graph_is_the_only_phase_source": (
            coarse["phase_source"] == fine["phase_source"] == "native_graph"
            and "--phase-source" not in source),
        "chi_scale_is_not_fitted": (
            coarse["chi_scale_source"]
            == fine["chi_scale_source"]
            == "native_kinetic_normalization_1_over_sqrt_B"
            and "--chi-scale" not in source),
        "both_copies_and_four_complex_components": (
            {row["copy"] for row in coarse["rows"]} == {"plus", "minus"}
            and all(row["real_component_count"] == 8 for row in all_rows)),
        "native_graph_is_well_conditioned": all(
            row["previous_layer_graph_condition_number"] < 1.01
            for row in all_rows),
        "pinv_recovery_is_not_the_only_evidence": all(
            row["direct_phase_relative_error"] > 0
            and row["complementary_sector_relative_residual"] > 0
            for row in all_rows),
        "full_native_update_remainder_is_controlled": all(
            row["direct_phase_refinement_ratio"] < 0.30
            and row["complementary_refinement_ratio"] < 0.30
            for row in gate["rows"]),
        "surrounding_residual_is_not_discarded": all(
            row["coarse_surrounding_fraction"] > 0.70
            and row["fine_surrounding_fraction"] > 0.70
            for row in gate["rows"]),
    }
    passed = all(checks.values())
    result = {
        "stage": "V4-003",
        "status": "PASS" if passed else "FAIL",
        "checks": checks,
        "forbidden_imports": forbidden_imports,
        "tautology_audit": (
            "The Moore-Penrose recovery identity is bookkeeping only. The "
            "nontrivial evidence is the unrestricted exact-action update, "
            "its nonzero complementary residual, and its measured second-"
            "order convergence under an independently rebuilt tangent."),
        "claim_boundary": gate["claim_boundary"],
    }
    output = HERE / "v4_003_red_team.json"
    output.write_text(json.dumps(result, indent=2) + "\n",
                      encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
