"""Local density-flow material action for direct native-prong transport.

The interval midpoint derives the least-norm local velocity satisfying the
Psi-density continuity relation, using the C19 gradient.  That same velocity
advects every complex component without mixing labels:

    D_t Psi_a = PsiDot_a + v.rgrad_19 Psi_a
    D_t chi_a = chiDot_a + v.rgrad_19 chi_a.

The kinetic action is the positive square of these material velocities.  The
construction is local, reciprocal through one action, cubic covariant, time
reversal invariant, and exactly reduces to canonical inertia when rhoDot=0.
It is tested here against the locked exact one-cell translation map and is not
installed in canonical V40.
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
VELOCITY_SCALE = 1.0
CONSENSUS_PASSES = 0


def neighbor(value: torch.Tensor, offset: tuple[int, int, int]) -> torch.Tensor:
    return torch.roll(value, shifts=tuple(-x for x in offset),
                      dims=(0, 1, 2))


def squared_norm(value: torch.Tensor) -> torch.Tensor:
    return torch.sum(value.real**2 + value.imag**2, dim=-1)


def c19_gradient(value: torch.Tensor) -> torch.Tensor:
    result = torch.zeros(value.shape + (3,), dtype=torch.float64)
    for offset, weight in rank.native.DIRECTIONS:
        direction = torch.tensor(offset, dtype=torch.float64)
        result = result + weight * (
            neighbor(value, offset) - value)[..., None] * direction
    return result / SECOND_MOMENT


def c19_average(value: torch.Tensor) -> torch.Tensor:
    return sum(weight * neighbor(value, offset)
               for offset, weight in rank.native.DIRECTIONS
               ) / rank.C19_WEIGHT_SUM


def continuity_velocity(psi: torch.Tensor,
                        psi_dot: torch.Tensor) -> torch.Tensor:
    rho = squared_norm(psi)
    rho_dot = 2 * torch.real(torch.sum(
        torch.conj(psi) * psi_dot, dim=-1))
    gradient = c19_gradient(rho)
    gradient_sq = torch.sum(gradient**2, dim=-1)
    # Tikhonov scale uses only the native kappa and local C19 density scale.
    # The kappa^3 vacuum floor removes a 0/0 without affecting a carrier edge.
    density_scale_sq = rho**2 + c19_average(rho**2)
    denominator = (gradient_sq + rank.KAPPA * density_scale_sq
                   + rank.KAPPA**3 * rank.R**4)
    velocity = (-VELOCITY_SCALE * rho_dot[..., None] * gradient
                / denominator[..., None])
    for _ in range(CONSENSUS_PASSES):
        velocity = c19_average(velocity)
    return velocity


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
             mode: str) -> torch.Tensor:
    left_psi, left_chi = local.unpack_tensor(left, SIDE)
    right_psi, right_chi = local.unpack_tensor(right, SIDE)
    psi = (left_psi + right_psi) / 2
    chi = (left_chi + right_chi) / 2
    psi_dot = (right_psi - left_psi) / dt
    chi_dot = (right_chi - left_chi) / dt
    velocity = continuity_velocity(psi, psi_dot)
    if mode in ("psi", "both"):
        psi_dot = psi_dot + advect_same_components(psi, velocity)
    if mode in ("chi", "both"):
        chi_dot = chi_dot + advect_same_components(chi, velocity)
    kinetic = torch.sum(squared_norm(psi_dot)) / 2
    kinetic = kinetic + rank.B * torch.sum(squared_norm(chi_dot)) / 2
    return dt * (kinetic - (four.orbit.potential(left)
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
    parser.add_argument("--maxiter", type=int, default=20)
    parser.add_argument("--next-maxiter", type=int, default=100)
    parser.add_argument("--velocity-scale", type=float, default=1.0)
    parser.add_argument("--consensus-passes", type=int, default=0,
                        choices=(0, 1, 2))
    args = parser.parse_args()
    global VELOCITY_SCALE, CONSENSUS_PASSES
    VELOCITY_SCALE = args.velocity_scale
    CONSENSUS_PASSES = args.consensus_passes
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
        "status": "V4_EXACT_TRANSLATION_DENSITY_FLOW_PRONG_GATE",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "mode": args.mode,
        "velocity_scale": args.velocity_scale,
        "consensus_passes": args.consensus_passes,
        "target_pattern": (
            "same-assignment exact one-cell translation of the complete "
            "joint Psi-chi carrier with unchanged eight-site core values"),
        "action_candidate": (
            "positive material kinetic action with local velocity derived "
            "from Psi-density continuity and C19 gradient; every Psi/chi "
            "component advects only into its matching component"),
        "coefficient_provenance": (
            "C19 second moment 12, locked kappa regularization, and unit "
            "material derivative from the continuity equation"),
        "rest_DEL_norm": float(torch.linalg.vector_norm(rest_residual)),
        "first_shell_and_clock_case": case,
        "verdict": (
            "DENSITY_FLOW_PRONG_EXACT_SHIFT_PASS" if passed else
            "DENSITY_FLOW_PRONG_EXACT_SHIFT_OPEN"),
        "claim_boundary": (
            "Research gate only. No action admission without exact update, "
            "all axes/signs, two free handoffs, boundedness, conservation, "
            "rest spectrum, and Dirac-current validation."),
    }
    output = HERE / (
        f"exact_translation_density_flow_prong_{args.mode}_"
        f"scale{args.velocity_scale:g}_consensus{args.consensus_passes}_"
        f"axis{args.axis}_"
        f"{'plus' if args.sign > 0 else 'minus'}.json")
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n",
                      encoding="utf-8")
    print(json.dumps({
        "status": result["status"], "mode": args.mode,
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
