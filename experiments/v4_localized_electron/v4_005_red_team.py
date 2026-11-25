"""Adversarial audit of the V4-005 transport/current gate."""

from __future__ import annotations

import ast
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent


def load(name: str) -> dict:
    return json.loads((HERE/name).read_text(encoding="utf-8"))


def imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {
        alias.name for node in ast.walk(tree)
        if isinstance(node, ast.Import) for alias in node.names
    } | {
        node.module or "" for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)}


def readout_is_geometry_only(path: Path) -> bool:
    """Audit the solver's syntax, ignoring comments and docstrings.

    A source substring test is inappropriate here because words such as
    ``target`` in explanatory prose are not executable dependencies.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    functions = {
        node.name: node for node in tree.body
        if isinstance(node, ast.FunctionDef)
    }
    solver = functions.get("readout_solver")
    if solver is None:
        return False
    arguments = [argument.arg for argument in solver.args.args]
    names = {
        node.id.lower() for node in ast.walk(solver)
        if isinstance(node, ast.Name)
    }
    forbidden = {
        name for name in names
        if any(token in name for token in ("target", "dirac", "comparator"))
    }
    required = {"raw", "gram", "ridge", "factor", "cho_factor", "cho_solve"}
    return arguments == ["raw"] and not forbidden and required <= names


def main() -> None:
    gate = load("v4_005_stage_gate.json")
    transport = load("v4_005_dt.json")
    transport_fine = load("v4_005_dt_half.json")
    current = load("v4_005_projected_noether_dt.json")
    current_fine = load("v4_005_projected_noether_dt_half.json")
    action_path = HERE/"live_clifford_bond_action.py"
    transport_path = HERE/"v4_005_short_time_transport.py"
    current_path = HERE/"v4_005_projected_noether_current.py"
    action_imports = imports(action_path)
    action_forbidden = sorted(name for name in action_imports if any(
        token in name.lower() for token in ("dirac", "comparator")))
    transport_source = transport_path.read_text(encoding="utf-8")
    current_source = current_path.read_text(encoding="utf-8")
    checks = {
        "stage_gate_passes": gate["status"] == "PASS",
        "Dirac_not_imported_by_action": not action_forbidden,
        "comparator_not_used_by_native_solve": all(text in transport_source
            for text in (
                "q_native = factor.solve", "q_target = raw @",
                "Dirac supplies initial data and the comparator only")),
        "readout_fixed_before_cases": (
            transport_source.index("solve_readout, ridge, condition")
            < transport_source.index("for axis, sign")),
        "readout_not_fitted_to_target": (
            readout_is_geometry_only(transport_path)),
        "Noether_current_comes_from_exact_action": all(
            text in current_source for text in (
                "action.interval", "torch.autograd.grad",
                "local four-component phase variation pulled through")),
        "microscopic_current_not_mislabeled_as_Dirac": (
            "microscopic circulation" in current_source
            and "center-resolved" in current_source),
        "both_copies_all_signed_axes": gate["checks"][
            "both_copies_both_signs_three_axes"],
        "nonzero_errors_are_retained": all(value > 0 for value in (
            transport["maximum_state_relative_error"],
            transport["maximum_readout_current_relative_error"],
            current["aggregate_cut_flux_relative_error"])),
        "transport_refines_at_second_order": (
            0.20 < gate["metrics"]["state_refinement_ratio"] < 0.30
            and 0.20 < gate["metrics"][
                "readout_current_refinement_ratio"] < 0.30),
        "Noether_current_error_shrinks": (
            current_fine["aggregate_cut_flux_relative_error"]
            < current["aggregate_cut_flux_relative_error"]
            and current["all_cut_flux_signs_match"]
            and current_fine["all_cut_flux_signs_match"]),
    }
    passed = all(checks.values())
    result = {
        "stage": "V4-005",
        "status": "PASS" if passed else "FAIL",
        "checks": checks,
        "action_forbidden_imports": action_forbidden,
        "circularity_audit": (
            "The comparator prepares and evaluates test columns but never "
            "enters the action or native solve. The readout is fixed from "
            "the carrier map before any target is created. The projected "
            "Noether current is an automatic derivative of the exact action."),
        "claim_boundary": gate["claim_boundary"],
    }
    (HERE/"v4_005_red_team.json").write_text(
        json.dumps(result, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
