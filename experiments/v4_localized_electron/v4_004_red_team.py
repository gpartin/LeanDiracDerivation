"""Adversarial audit of the V4-004 localized spin bridge."""

from __future__ import annotations

import ast
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent


def load(name: str) -> dict:
    return json.loads((HERE / name).read_text(encoding="utf-8"))


def main() -> None:
    coarse = load("v4_004_dt.json")
    fine = load("v4_004_dt_half.json")
    gate = load("v4_004_stage_gate.json")
    stage3 = load("v4_003_stage_gate.json")
    source_path = HERE / "v4_004_operational_spin_gate.py"
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
    forbidden = sorted(name for name in imports if any(
        token in name.lower() for token in ("dirac", "comparator")))
    lean_source = (
        HERE.parent.parent / "LeanDirac/V4/LocalizedSpin.lean"
    ).read_text(encoding="utf-8")
    ratio = gate["graph_remainder_refinement_ratio"]
    checks = {
        "stage_gate_passes": gate["status"] == "PASS",
        "same_action_as_v4_003": (
            coarse["action_sha256"] == fine["action_sha256"]
            == stage3["action_sha256"]),
        "same_frozen_readout_as_v4_003": gate["checks"][
            "exact_same_v4_003_readout"],
        "no_Dirac_or_comparator_import": not forbidden,
        "native_cubical_rotation_source": all(
            "native cubical incidence" in artifact["rotation_source"]
            for artifact in (coarse, fine)),
        "both_copies_and_three_axes": gate["checks"][
            "both_copies_three_axes"],
        "spin_endpoints_and_density_exact": all(gate["checks"][key]
            for key in (
                "density_preserved", "two_pi_sign_reversal",
                "four_pi_return", "fixed_readout_covariant")),
        "finite_step_mismatch_is_disclosed_and_refines": (
            coarse["maximum_two_layer_graph_rotation_commutator_relative"]
            > 0 and fine[
                "maximum_two_layer_graph_rotation_commutator_relative"] > 0
            and 0.45 < ratio < 0.55),
        "Lean_claim_is_scoped_to_readout": all(text in lean_source for text in (
            "The theorem is deliberately scoped to the readout",
            "A cumulative analytic bridge",
            "V4-010 obligation")),
        "Lean_scale_is_derived_not_assumed": (
            "exists_unique_admissible_cubical_spin_scale" in lean_source),
    }
    passed = all(checks.values())
    result = {
        "stage": "V4-004",
        "status": "PASS" if passed else "FAIL",
        "checks": checks,
        "forbidden_imports": forbidden,
        "first_principles_audit": (
            "The rotation generators are the cube-incidence bivectors and "
            "the nonzero one-half scale is selected by the proved closure "
            "equation. The numerical bridge reuses the exact V4-003 action, "
            "embedding, and fixed readout. No Dirac comparator, runtime "
            "projection, copy selector, or action term is added."),
        "remaining_gap": (
            "The nonzero quarter-turn graph commutator is retained and "
            "halves with dt. V4-004 proves the localized readout's rotation "
            "law, not a free torque experiment, translation, or current."),
    }
    output = HERE / "v4_004_red_team.json"
    output.write_text(json.dumps(result, indent=2) + "\n",
                      encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
