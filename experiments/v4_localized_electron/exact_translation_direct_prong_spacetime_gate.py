"""Direct component-to-component spacetime C19 communication gate.

For each interval, every complex Psi and chi component communicates only with
the same component of its C19 neighbors.  No live-frame projection mixes x,
y, and z.  A density-transfer invariant activates the neighbor link, and the
forward/backward interval average makes the kinetic action time-reversal
symmetric and positive.  At the exact rest orbit rhoPsiDot=0, so the action
reduces identically to canonical local inertia.

This is a research action screen against the locked literal translation map.
It does not modify canonical V40 or install a copying rule.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

import density_mold_connection_probe as local
import edge_snap_action_rank as rank
import edge_snap_four_update_orbit as four
import exact_translation_paired_frame_precursor_gate as inverse
import exact_translation_precursor_linear_gate as precursor


HERE = Path(__file__).resolve().parent
SIDE = precursor.SIDE
FACE_FACTOR = 6.0


def neighbor_average(value: torch.Tensor) -> torch.Tensor:
    return sum(weight * torch.roll(value, shifts=offset, dims=(0, 1, 2))
               for offset, weight in rank.native.DIRECTIONS
               ) / rank.C19_WEIGHT_SUM


def squared_norm(value: torch.Tensor) -> torch.Tensor:
    return torch.sum(value.real**2 + value.imag**2, dim=-1)


def linked_velocity(left: torch.Tensor, right: torch.Tensor,
                    activation: torch.Tensor, dt: float) -> torch.Tensor:
    source_forward = ((1 - activation)[..., None] * left
                      + activation[..., None] * neighbor_average(left))
    source_backward = ((1 - activation)[..., None] * right
                       + activation[..., None] * neighbor_average(right))
    forward = (right - source_forward) / dt
    backward = (left - source_backward) / dt
    # Return both orientations; their squares are averaged in the action.
    return torch.stack((forward, backward), dim=0)


def interval(left: torch.Tensor, right: torch.Tensor, dt: float,
             mode: str) -> torch.Tensor:
    left_psi, left_chi = local.unpack_tensor(left, SIDE)
    right_psi, right_chi = local.unpack_tensor(right, SIDE)
    psi = (left_psi + right_psi) / 2
    psi_dot = (right_psi - left_psi) / dt
    rho_dot = 2 * torch.real(torch.sum(
        torch.conj(psi) * psi_dot, dim=-1))
    rho_psi = squared_norm(psi)
    transfer = rho_dot / (
        2 * rank.R * (rho_psi + rank.KAPPA * rank.R**2))
    activation = FACE_FACTOR * transfer**2 / (1 + FACE_FACTOR * transfer**2)

    if mode in ("psi", "both"):
        psi_velocities = linked_velocity(
            left_psi, right_psi, activation, dt)
        psi_kinetic = torch.sum(squared_norm(psi_velocities)) / 4
    else:
        psi_kinetic = torch.sum(squared_norm(psi_dot)) / 2

    chi_dot = (right_chi - left_chi) / dt
    if mode in ("chi", "both"):
        chi_velocities = linked_velocity(
            left_chi, right_chi, activation, dt)
        chi_kinetic = rank.B * torch.sum(
            squared_norm(chi_velocities)) / 4
    else:
        chi_kinetic = rank.B * torch.sum(squared_norm(chi_dot)) / 2

    return dt * (psi_kinetic + chi_kinetic
                 - (four.orbit.potential(left)
                    + four.orbit.potential(right)) / 2)


def del_three(previous: torch.Tensor, current: torch.Tensor,
              following: torch.Tensor, dt: float, mode: str) -> torch.Tensor:
    def summed(left, middle, right):
        return (interval(left, middle, dt, mode)
                + interval(middle, right, dt, mode))
    return torch.func.grad(summed, argnums=1)(previous, current, following)


def profile_residual(raw: torch.Tensor, axis: int, sign: int, dt: float,
                     psi_phase: float, chi_phase: float,
                     mode: str) -> torch.Tensor:
    following = precursor.translate(
        raw, axis, sign, psi_phase, chi_phase)
    previous = precursor.translate(
        raw, axis, -sign, -psi_phase, -chi_phase)
    return del_three(previous, raw, following, dt, mode)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("psi", "chi", "both"),
                        default="both")
    parser.add_argument("--axis", type=int, default=0, choices=(0, 1, 2))
    parser.add_argument("--sign", type=int, default=1, choices=(-1, 1))
    parser.add_argument("--maxiter", type=int, default=30)
    parser.add_argument("--next-maxiter", type=int, default=100)
    args = parser.parse_args()
    rank.lock_inertia()

    evidence = json.loads((HERE / "edge_snap_verlet_gate.json").read_text(
        encoding="utf-8"))
    row = next(item for item in evidence["rows"]
               if item["axis"] == args.axis and item["sign"] == args.sign)
    amplitude_sq = rank.R**2 / (2 + 54 / rank.R**2)
    psi, chi = rank.native.carrier_fields(SIDE, amplitude_sq)
    q0 = torch.tensor(rank.pack(psi, chi), dtype=torch.float64)
    core, shell = precursor.site_masks(psi)

    def selected_profile(raw, axis, sign, dt, psi_phase, chi_phase):
        return profile_residual(
            raw, axis, sign, dt, psi_phase, chi_phase, args.mode)

    def selected_del(previous, current, following, dt):
        return del_three(previous, current, following, dt, args.mode)

    rest_phase = float(torch.arccos(torch.tensor(
        1 - row["best_dt"]**2 * rank.R**2 / 2)))
    rest_previous = precursor.translate(q0, args.axis, 0, -rest_phase, 0.0)
    rest_following = precursor.translate(q0, args.axis, 0, rest_phase, 0.0)
    rest_residual = selected_del(
        rest_previous, q0, rest_following, row["best_dt"]).detach()

    case = inverse.solve_shell_and_clock(
        shell, core, q0, args.axis, args.sign, row["best_dt"],
        row["best_phase_psi"], row["best_phase_chi"], args.maxiter,
        selected_profile, selected_del, args.next_maxiter)
    update = case["full_nonlinear_solve"]["state_update_defect"]
    pass_update = (
        update["postsolve_DEL_relative_residual"] < 1e-8
        and update["correction_over_exact_translation_step"] < 1e-6)
    result = {
        "status": "V4_EXACT_TRANSLATION_DIRECT_PRONG_SPACETIME_GATE",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "mode": args.mode,
        "target_pattern": (
            "same-assignment exact one-cell translation of the complete "
            "joint Psi-chi profile with unchanged eight-site core values"),
        "action_candidate": (
            "positive reciprocal time-symmetric interval action linking "
            "each complex component only to the same component of every C19 "
            "neighbor; density-transfer activation vanishes at rest"),
        "component_routing": [
            "Psi_x to neighboring Psi_x", "Psi_y to neighboring Psi_y",
            "Psi_z to neighboring Psi_z", "chi_x to neighboring chi_x",
            "chi_y to neighboring chi_y", "chi_z to neighboring chi_z"
        ],
        "coefficient_provenance": (
            "factor 6 from the six C19 face directions; kappa and chi0 are "
            "the locked native constants"),
        "rest_DEL_norm": float(torch.linalg.vector_norm(rest_residual)),
        "first_shell_and_clock_case": case,
        "verdict": (
            "DIRECT_PRONG_SPACETIME_EXACT_SHIFT_PASS" if pass_update else
            "DIRECT_PRONG_SPACETIME_EXACT_SHIFT_OPEN"),
        "claim_boundary": (
            "One signed-axis exact-map research gate. A pass still requires "
            "all axes/signs, two unreset handoffs, conservation, rest "
            "spectrum, boundedness, and Dirac-current validation."),
    }
    output = HERE / (
        f"exact_translation_direct_prong_{args.mode}_axis{args.axis}_"
        f"{'plus' if args.sign > 0 else 'minus'}.json")
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n",
                      encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "mode": args.mode,
        "rest_DEL_norm": result["rest_DEL_norm"],
        "profile_residual": case["full_nonlinear_solve"][
            "final_relative_residual"],
        "precursor_norm": case["full_nonlinear_solve"][
            "correction_norm_over_state_norm"],
        "state_update_defect": update,
        "verdict": result["verdict"],
    }, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
