"""Full complex-prong best-matching material action.

All real and imaginary parts of Psi_x/y/z and chi_x/y/z contribute directly
to a local C19 normal equation for one translation velocity.  The local U(1)
phase rate of each field is removed before solving, so internal clock rotation
does not masquerade as spatial motion.  The resulting velocity advects every
component without mixing labels and enters one positive reciprocal action.

This is a research gate, not a canonical V40 modification or admission.
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
SECOND_MOMENT = 12.0
# Research defaults.  Keeping these named permits geometry-derived ablations
# without changing the componentwise action or fitting a force to a target.
TRACE_REGULARIZER = rank.KAPPA
VACUUM_REGULARIZER = rank.KAPPA**3 * rank.R**2


def neighbor(value: torch.Tensor, offset: tuple[int, int, int]) -> torch.Tensor:
    return torch.roll(value, shifts=tuple(-x for x in offset),
                      dims=(0, 1, 2))


def squared_norm(value: torch.Tensor) -> torch.Tensor:
    return torch.sum(value.real**2 + value.imag**2, dim=-1)


def complex_gradient(value: torch.Tensor) -> torch.Tensor:
    """C19 gradient; final axes are internal component, spatial direction."""
    result = torch.zeros(value.shape + (3,), dtype=torch.complex128)
    for offset, weight in rank.native.DIRECTIONS:
        direction = torch.tensor(offset, dtype=torch.float64)
        result = result + weight * (
            neighbor(value, offset) - value)[..., :, None] * direction
    return result / SECOND_MOMENT


def remove_phase_velocity(value: torch.Tensor,
                          derivative: torch.Tensor) -> torch.Tensor:
    density = squared_norm(value)
    phase_numerator = torch.imag(torch.sum(
        torch.conj(value) * derivative, dim=-1))
    # Exact quotient on occupied sites; the vacuum directional derivative is
    # defined as zero.  A scale-free machine floor only prevents 0/0 and does
    # not alter the carrier phase rate.
    floor = torch.finfo(torch.float64).eps * rank.R**2
    phase_rate = torch.where(
        density > floor,
        phase_numerator / torch.clamp_min(density, floor),
        torch.zeros_like(density))
    return derivative - 1j * phase_rate[..., None] * value


def normal_blocks(value: torch.Tensor, derivative: torch.Tensor,
                  weight: float) -> tuple[torch.Tensor, torch.Tensor]:
    gradient = complex_gradient(value)
    reduced = remove_phase_velocity(value, derivative)
    gram = weight * torch.real(torch.einsum(
        "...ca,...cb->...ab", torch.conj(gradient), gradient))
    rhs = weight * torch.real(torch.einsum(
        "...ca,...c->...a", torch.conj(gradient), reduced))
    return gram, rhs


def best_matching_velocity(psi: torch.Tensor, psi_dot: torch.Tensor,
                           chi: torch.Tensor, chi_dot: torch.Tensor,
                           source: str) -> torch.Tensor:
    shape = psi.shape[:-1]
    gram = torch.zeros(shape + (3, 3), dtype=torch.float64)
    rhs = torch.zeros(shape + (3,), dtype=torch.float64)
    if source in ("psi", "both"):
        addition, vector = normal_blocks(psi, psi_dot, 1.0)
        gram, rhs = gram + addition, rhs + vector
    if source in ("chi", "both"):
        addition, vector = normal_blocks(chi, chi_dot, rank.B)
        gram, rhs = gram + addition, rhs + vector
    trace = torch.diagonal(gram, dim1=-2, dim2=-1).sum(-1)
    identity = torch.eye(3, dtype=torch.float64)
    regularizer = TRACE_REGULARIZER * trace / 3 + VACUUM_REGULARIZER
    system = gram + regularizer[..., None, None] * identity
    return -torch.linalg.solve(system, rhs[..., None])[..., 0]


def advect_same_components(value: torch.Tensor,
                           velocity: torch.Tensor) -> torch.Tensor:
    result = torch.zeros_like(value)
    for offset, weight in rank.native.DIRECTIONS:
        direction = torch.tensor(offset, dtype=torch.float64)
        velocity_mid = (velocity + neighbor(velocity, offset)) / 2
        along = torch.einsum("...a,a->...", velocity_mid, direction)
        result = result + weight * along[..., None] * (
            neighbor(value, offset) - value)
    return result / SECOND_MOMENT


def interval(left: torch.Tensor, right: torch.Tensor, dt: float,
             source: str) -> torch.Tensor:
    left_psi, left_chi = local.unpack_tensor(left, SIDE)
    right_psi, right_chi = local.unpack_tensor(right, SIDE)
    psi = (left_psi + right_psi) / 2
    chi = (left_chi + right_chi) / 2
    psi_dot = (right_psi - left_psi) / dt
    chi_dot = (right_chi - left_chi) / dt
    velocity = best_matching_velocity(
        psi, psi_dot, chi, chi_dot, source)
    material_psi = psi_dot + advect_same_components(psi, velocity)
    material_chi = chi_dot + advect_same_components(chi, velocity)
    kinetic = torch.sum(squared_norm(material_psi)) / 2
    kinetic = kinetic + rank.B * torch.sum(squared_norm(material_chi)) / 2
    return dt * (kinetic - (four.orbit.potential(left)
                            + four.orbit.potential(right)) / 2)


def del_three(previous: torch.Tensor, current: torch.Tensor,
              following: torch.Tensor, dt: float,
              source: str) -> torch.Tensor:
    def summed(left, middle, right):
        return (interval(left, middle, dt, source)
                + interval(middle, right, dt, source))
    return torch.func.grad(summed, argnums=1)(previous, current, following)


def profile_residual(raw: torch.Tensor, axis: int, sign: int, dt: float,
                     psi_phase: float, chi_phase: float,
                     source: str) -> torch.Tensor:
    following = precursor.translate(
        raw, axis, sign, psi_phase, chi_phase)
    previous = precursor.translate(
        raw, axis, -sign, -psi_phase, -chi_phase)
    return del_three(previous, raw, following, dt, source)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=("psi", "chi", "both"),
                        default="both")
    parser.add_argument("--axis", type=int, default=0, choices=(0, 1, 2))
    parser.add_argument("--sign", type=int, default=1, choices=(-1, 1))
    parser.add_argument("--maxiter", type=int, default=20)
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
            raw, axis, sign, dt, psi_phase, chi_phase, args.source)

    def selected_del(previous, current, following, dt):
        return del_three(previous, current, following, dt, args.source)

    rest_phase = float(torch.arccos(torch.tensor(
        1 - row["best_dt"]**2 * rank.R**2 / 2)))
    rest_previous = precursor.translate(
        q0, args.axis, 0, -rest_phase, 0.0)
    rest_following = precursor.translate(
        q0, args.axis, 0, rest_phase, 0.0)
    rest_residual = selected_del(
        rest_previous, q0, rest_following, row["best_dt"]).detach()

    case = inverse.solve_shell_and_clock(
        shell, core, q0, args.axis, args.sign, row["best_dt"],
        row["best_phase_psi"], row["best_phase_chi"], args.maxiter,
        selected_profile, selected_del, args.next_maxiter)
    update = case["full_nonlinear_solve"]["state_update_defect"]
    passed = (update["postsolve_DEL_relative_residual"] < 1e-8
              and update["correction_over_exact_translation_step"] < 1e-6)
    result = {
        "status": "V4_EXACT_TRANSLATION_MULTICHANNEL_PRONG_GATE",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "velocity_source": args.source,
        "target_pattern": (
            "same-assignment exact one-cell translation of the complete "
            "joint Psi-chi carrier with unchanged eight-site core values"),
        "action_candidate": (
            "positive best-matching material kinetic action; all real and "
            "imaginary parts of the six native complex prongs determine a "
            "common local velocity and every prong advects without mixing"),
        "coefficient_provenance": (
            "C19 second moment 12, locked B and kappa, and the unique "
            "regularized least-squares velocity; no target coefficient"),
        "rest_DEL_norm": float(torch.linalg.vector_norm(rest_residual)),
        "first_shell_and_clock_case": case,
        "verdict": (
            "MULTICHANNEL_PRONG_EXACT_SHIFT_PASS" if passed else
            "MULTICHANNEL_PRONG_EXACT_SHIFT_OPEN"),
        "claim_boundary": (
            "Research gate only. No action admission without exact update, "
            "all axes/signs, two free handoffs, boundedness, conservation, "
            "rest spectrum, and Dirac-current validation."),
    }
    output = HERE / (
        f"exact_translation_multichannel_prong_{args.source}_"
        f"axis{args.axis}_{'plus' if args.sign > 0 else 'minus'}.json")
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n",
                      encoding="utf-8")
    print(json.dumps({
        "status": result["status"], "velocity_source": args.source,
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
