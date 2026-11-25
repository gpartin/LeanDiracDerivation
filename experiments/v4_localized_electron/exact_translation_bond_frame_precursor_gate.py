"""Combine the exact translation map with the native full-frame precursor.

This is the calculation that was missing between the successful bond-frame
response screen and the inconclusive finite-path solve.  The action is the
previously derived positive reciprocal chi kinetic metric with:

* the local Psi-density mold response;
* transverse scale 19 from chi0;
* a six-face native bond-frame activation;
* full chi/Psi/(chi wedge Psi) frame transport on every C19 bond.

The eight occupied carrier sites are held exactly equal to the proved rest
carrier.  Only ordinary surrounding Psi and chi values may change.  The target
is the literal same-assignment one-cell translation of the entire resulting
profile.  No receiver preload, copying force, projector, Dirac matrix, corner
label, or target direction occurs in the action.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

import edge_snap_action_rank as rank
import edge_snap_transfer_four_update_orbit as transfer
import exact_translation_precursor_linear_gate as precursor


HERE = Path(__file__).resolve().parent
SIDE = precursor.SIDE
GEOMETRIC_FACTOR = 19.0
BOND_FRAME_FACTOR = 6.0


def configure_action() -> None:
    transfer.GEOMETRIC_FACTOR = GEOMETRIC_FACTOR
    transfer.FILTER_FACTOR = 0.0
    transfer.COLLECTIVE_FACTOR = 0.0
    transfer.RADIAL_FACTOR = 0.0
    transfer.NORMAL_FACTOR = -1.0
    transfer.BOND_FRAME_FACTOR = BOND_FRAME_FACTOR
    transfer.TRANSFER_NEIGHBOR_NORMAL = 0.0
    transfer.TRANSFER_NEIGHBOR_RADIAL = 0.0


def del_three(previous: torch.Tensor, current: torch.Tensor,
              following: torch.Tensor, dt: float) -> torch.Tensor:
    def summed(left, middle, right):
        return (transfer.interval_action(left, middle, dt, 0.0)
                + transfer.interval_action(middle, right, dt, 0.0))
    # Differentiate only with respect to the middle layer.  torch.func keeps
    # this partial derivative differentiable with respect to all three layers,
    # which is required when the translated endpoints vary with the profile.
    return torch.func.grad(summed, argnums=1)(previous, current, following)


def profile_residual(raw: torch.Tensor, axis: int, sign: int, dt: float,
                     psi_phase: float, chi_phase: float) -> torch.Tensor:
    following = precursor.translate(
        raw, axis, sign, psi_phase, chi_phase)
    previous = precursor.translate(
        raw, axis, -sign, -psi_phase, -chi_phase)
    return del_three(previous, raw, following, dt)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--axis", type=int, default=0, choices=(0, 1, 2))
    parser.add_argument("--sign", type=int, default=1, choices=(-1, 1))
    parser.add_argument("--maxiter", type=int, default=180)
    args = parser.parse_args()
    rank.lock_inertia()
    configure_action()

    evidence = json.loads((HERE / "edge_snap_verlet_gate.json").read_text(
        encoding="utf-8"))
    row = next(item for item in evidence["rows"]
               if item["axis"] == args.axis and item["sign"] == args.sign)
    amplitude_sq = rank.R**2 / (2 + 54 / rank.R**2)
    psi, chi = rank.native.carrier_fields(SIDE, amplitude_sq)
    q0 = torch.tensor(rank.pack(psi, chi), dtype=torch.float64)
    core, shell = precursor.site_masks(psi)

    def function(value: torch.Tensor) -> torch.Tensor:
        return profile_residual(
            value, args.axis, args.sign, row["best_dt"],
            row["best_phase_psi"], row["best_phase_chi"])

    residual0 = function(q0).detach()
    case = precursor.solve_case(
        "C19_first_shell", shell, core, shell, q0, residual0,
        function, args.maxiter)
    result = {
        "status": "V4_EXACT_TRANSLATION_BOND_FRAME_PRECURSOR_GATE",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "axis": args.axis,
        "sign": args.sign,
        "target_pattern": (
            "same eight-site Psi-chi carrier values and entire native "
            "profile translated exactly one cell"),
        "action_candidate": (
            "positive reciprocal chi kinetic metric with density mold, "
            "chi0=19 transverse scaling, and six-face C19 live-frame "
            "transport of chi, Psi, and chi wedge Psi"),
        "coefficient_provenance": {
            "transverse_factor": "chi0=19",
            "bond_frame_factor": "six C19 face directions",
        },
        "core_values_held_exact_during_inverse_gate": True,
        "dt": row["best_dt"],
        "psi_phase": row["best_phase_psi"],
        "chi_phase": row["best_phase_chi"],
        "compact_profile_del_norm": float(
            torch.linalg.vector_norm(residual0)),
        "first_shell_case": case,
        "verdict": (
            "FULL_FRAME_PRECURSOR_ADVANCES_TO_NONLINEAR_PROFILE_SOLVE"
            if (case["best_nonlinear"]["relative_residual"] < 0.1
                and case["correction_norm_over_state_norm"] < 0.5)
            else "FULL_FRAME_PRECURSOR_DOES_NOT_CLOSE_EXACT_SHIFT"),
        "claim_boundary": (
            "One signed-axis linearized inverse-profile gate using the "
            "previously derived reciprocal action candidate. A favorable "
            "result is not an admitted action or a moving-electron pass; it "
            "requires a nonlinear exact-profile solve and all-axis tests."),
    }
    sign_label = "plus" if args.sign > 0 else "minus"
    output = HERE / (
        f"exact_translation_bond_frame_precursor_axis{args.axis}_"
        f"{sign_label}.json")
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n",
                      encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "compact_profile_del_norm": result["compact_profile_del_norm"],
        "linear_relative_residual": case["linear_relative_residual"],
        "correction_norm_over_state_norm": case[
            "correction_norm_over_state_norm"],
        "best_nonlinear": case["best_nonlinear"],
        "residual_diagnostics": case[
            "best_nonlinear_residual_diagnostics"],
        "verdict": result["verdict"],
    }, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
