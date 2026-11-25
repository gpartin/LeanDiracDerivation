"""Density-oriented direct-prong symplectic C19 connection.

For each unoriented bond, the Psi-density contrast and the antisymmetric
same-component phase-transfer one-form both reverse sign when endpoints are
swapped.  Their product is therefore reciprocal and needs no selected axis.
The action connects Psi_a only to Psi_a and chi_a only to chi_a.

The natural scales are chi0 for Psi and B*chi0 for chi.  A dimensionless
factor is exposed only for the declared geometric screen; no fitted factor is
admitted by this file.
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


def squared_norm(value: torch.Tensor) -> torch.Tensor:
    return torch.sum(value.real**2 + value.imag**2, dim=-1)


def direct_one_form(left: torch.Tensor, right: torch.Tensor,
                    field: str) -> torch.Tensor:
    left_psi, left_chi = local.unpack_tensor(left, SIDE)
    right_psi, right_chi = local.unpack_tensor(right, SIDE)
    psi = (left_psi + right_psi) / 2
    rho = squared_norm(psi) / rank.R**2
    if field == "psi":
        value = psi
        delta = right_psi - left_psi
    elif field == "chi":
        value = (left_chi + right_chi) / 2
        delta = right_chi - left_chi
    else:
        raise ValueError(field)
    result = left.new_zeros((), dtype=torch.float64)
    for offset, weight in rank.native.DIRECTIONS:
        neighbor_value = torch.roll(
            value, shifts=offset, dims=(0, 1, 2))
        neighbor_delta = torch.roll(
            delta, shifts=offset, dims=(0, 1, 2))
        neighbor_rho = torch.roll(
            rho, shifts=offset, dims=(0, 1, 2))
        # Sum over matching component labels only.
        pairing = torch.sum(
            torch.conj(value) * neighbor_delta
            - torch.conj(neighbor_value) * delta, dim=-1)
        result = result + weight * torch.sum(
            (rho - neighbor_rho) * torch.imag(pairing))
    return result / rank.R**2


def interval(left: torch.Tensor, right: torch.Tensor, dt: float,
             mode: str, factor: float) -> torch.Tensor:
    left_psi, left_chi = local.unpack_tensor(left, SIDE)
    right_psi, right_chi = local.unpack_tensor(right, SIDE)
    psi_dot = (right_psi - left_psi) / dt
    chi_dot = (right_chi - left_chi) / dt
    kinetic = torch.sum(squared_norm(psi_dot)) / 2
    kinetic = kinetic + rank.B * torch.sum(squared_norm(chi_dot)) / 2
    action = dt * (kinetic - (four.orbit.potential(left)
                              + four.orbit.potential(right)) / 2)
    if mode in ("psi", "both"):
        action = action + factor * rank.R * direct_one_form(
            left, right, "psi")
    if mode in ("chi", "both"):
        action = action + factor * rank.B * rank.R * direct_one_form(
            left, right, "chi")
    return action


def del_three(previous: torch.Tensor, current: torch.Tensor,
              following: torch.Tensor, dt: float, mode: str,
              factor: float) -> torch.Tensor:
    def summed(left, middle, right):
        return (interval(left, middle, dt, mode, factor)
                + interval(middle, right, dt, mode, factor))
    return torch.func.grad(summed, argnums=1)(previous, current, following)


def profile_residual(raw: torch.Tensor, axis: int, sign: int, dt: float,
                     psi_phase: float, chi_phase: float, mode: str,
                     factor: float) -> torch.Tensor:
    following = precursor.translate(
        raw, axis, sign, psi_phase, chi_phase)
    previous = precursor.translate(
        raw, axis, -sign, -psi_phase, -chi_phase)
    return del_three(previous, raw, following, dt, mode, factor)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("psi", "chi", "both"),
                        default="both")
    parser.add_argument("--factor", type=float, default=1.0)
    parser.add_argument("--axis", type=int, default=0, choices=(0, 1, 2))
    parser.add_argument("--sign", type=int, default=1, choices=(-1, 1))
    parser.add_argument("--maxiter", type=int, default=20)
    parser.add_argument("--next-maxiter", type=int, default=80)
    parser.add_argument("--screen-only", action="store_true")
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
            raw, axis, sign, dt, psi_phase, chi_phase,
            args.mode, args.factor)

    def selected_del(previous, current, following, dt):
        return del_three(
            previous, current, following, dt, args.mode, args.factor)

    rest_phase = float(torch.arccos(torch.tensor(
        1 - row["best_dt"]**2 * rank.R**2 / 2)))
    rest_previous = precursor.translate(
        q0, args.axis, 0, -rest_phase, 0.0)
    rest_following = precursor.translate(
        q0, args.axis, 0, rest_phase, 0.0)
    rest_residual = selected_del(
        rest_previous, q0, rest_following, row["best_dt"]).detach()
    rigid = selected_profile(
        q0, args.axis, args.sign, row["best_dt"],
        row["best_phase_psi"], row["best_phase_chi"]).detach()
    canonical = profile_residual(
        q0, args.axis, args.sign, row["best_dt"],
        row["best_phase_psi"], row["best_phase_chi"], "both", 0.0).detach()
    rigid_relative = float(
        torch.linalg.vector_norm(rigid)
        / torch.linalg.vector_norm(canonical))

    summary = {
        "status": "V4_EXACT_TRANSLATION_DIRECT_PRONG_CONNECTION_GATE",
        "mode": args.mode,
        "factor": args.factor,
        "rest_DEL_norm": float(torch.linalg.vector_norm(rest_residual)),
        "rigid_relative_residual": rigid_relative,
    }
    if args.screen_only:
        print(json.dumps(summary, indent=2, allow_nan=False))
        return

    case = inverse.solve_shell_and_clock(
        shell, core, q0, args.axis, args.sign, row["best_dt"],
        row["best_phase_psi"], row["best_phase_chi"], args.maxiter,
        selected_profile, selected_del, args.next_maxiter)
    update = case["full_nonlinear_solve"]["state_update_defect"]
    passed = (update["postsolve_DEL_relative_residual"] < 1e-8
              and update["correction_over_exact_translation_step"] < 1e-6)
    result = {
        **summary,
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "target_pattern": (
            "same-assignment exact one-cell translation of the complete "
            "joint Psi-chi profile with unchanged eight-site core values"),
        "action_candidate": (
            "density-oriented reciprocal direct-component symplectic C19 "
            "connection; each component communicates only with its matching "
            "neighbor component"),
        "coefficient_provenance": (
            "screen factor times chi0 for Psi and B*chi0 for chi; factor "
            "is not fitted or admitted"),
        "first_shell_and_clock_case": case,
        "verdict": (
            "DIRECT_PRONG_CONNECTION_EXACT_SHIFT_PASS" if passed else
            "DIRECT_PRONG_CONNECTION_EXACT_SHIFT_OPEN"),
        "claim_boundary": (
            "Research screen only; no action admission without exact update, "
            "all directions, free repeated handoff, boundedness, conservation, "
            "rest spectrum, and Dirac-current validation."),
    }
    output = HERE / (
        f"exact_translation_direct_prong_connection_{args.mode}_"
        f"factor{args.factor:+g}_axis{args.axis}_"
        f"{'plus' if args.sign > 0 else 'minus'}.json")
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n",
                      encoding="utf-8")
    print(json.dumps({
        **summary,
        "profile_residual": case["full_nonlinear_solve"][
            "final_relative_residual"],
        "precursor_norm": case["full_nonlinear_solve"][
            "correction_norm_over_state_norm"],
        "state_update_defect": update,
        "verdict": result["verdict"],
    }, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
