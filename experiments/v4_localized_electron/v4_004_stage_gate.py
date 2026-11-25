"""Aggregate V4-004 endpoint, readout, refinement, and Lean evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--coarse", type=Path, required=True)
    parser.add_argument("--fine", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    coarse_path = (args.coarse if args.coarse.is_absolute()
                   else HERE / args.coarse).resolve()
    fine_path = (args.fine if args.fine.is_absolute()
                 else HERE / args.fine).resolve()
    coarse, fine = load(coarse_path), load(fine_path)
    stage3_coarse = load(HERE / "v4_003_dt_native.json")
    stage3_fine = load(HERE / "v4_003_dt_half_native.json")

    def hashes(artifact: dict) -> dict[str, tuple[str, str]]:
        return {row["copy"]: (
            row["phase_embedding_sha256"], row["fixed_readout_sha256"])
            for row in artifact["rows"]}

    graph_ratio = (
        fine["maximum_two_layer_graph_rotation_commutator_relative"]
        / coarse["maximum_two_layer_graph_rotation_commutator_relative"])
    checks = {
        "same_frozen_action": (
            coarse["action_sha256"] == fine["action_sha256"]
            == stage3_coarse["action_sha256"]
            == stage3_fine["action_sha256"]),
        "exact_same_v4_003_readout": (
            hashes(coarse) == hashes(stage3_coarse)
            and hashes(fine) == hashes(stage3_fine)),
        "both_copies_three_axes": all(
            {row["copy"] for row in artifact["rows"]} == {"plus", "minus"}
            and all(len(row["axes"]) == 3 for row in artifact["rows"])
            for artifact in (coarse, fine)),
        "density_preserved": max(
            coarse["maximum_density_unitarity_defect"],
            fine["maximum_density_unitarity_defect"]) < 1e-12,
        "two_pi_sign_reversal": max(
            coarse["maximum_two_pi_sign_reversal_defect"],
            fine["maximum_two_pi_sign_reversal_defect"]) < 1e-12,
        "four_pi_return": max(
            coarse["maximum_four_pi_return_defect"],
            fine["maximum_four_pi_return_defect"]) < 1e-12,
        "fixed_readout_covariant": max(
            coarse["maximum_fixed_readout_covariance_defect"],
            fine["maximum_fixed_readout_covariance_defect"]) < 1e-12,
        "finite_step_graph_remainder_shrinks": 0.45 < graph_ratio < 0.55,
        "Lean_theorem_declared": (
            (HERE.parent.parent / "LeanDirac/V4/LocalizedSpin.lean").is_file()),
    }
    passed = all(checks.values())
    result = {
        "stage": "V4-004",
        "status": "PASS" if passed else "FAIL",
        "checks": checks,
        "coarse_evidence": coarse_path.name,
        "fine_evidence": fine_path.name,
        "graph_remainder_refinement_ratio": graph_ratio,
        "Lean_theorem": "LeanDirac.V4.localized_readout_spin_half",
        "claim_boundary": (
            "The same V4-003 localized readout has the native cubical "
            "density-preserving 2pi sign reversal and 4pi return. The "
            "quarter-turn two-layer graph mismatch is retained as a finite-"
            "step diagnostic. This does not prove an externally torqued "
            "trajectory, translation, or current."),
    }
    output = args.output if args.output.is_absolute() else HERE / args.output
    output.write_text(json.dumps(result, indent=2) + "\n",
                      encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
