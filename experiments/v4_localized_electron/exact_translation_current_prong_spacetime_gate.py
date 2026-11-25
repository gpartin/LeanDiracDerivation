"""Current-directed, component-preserving C19 spacetime communication gate.

Every native complex component is transported without internal-frame mixing:
Psi_a talks only to neighboring Psi_a and chi_a only to neighboring chi_a.
Unlike an undirected neighbor average, the bond weights are derived from the
native Psi phase current.  A density-transfer activation makes this extra
connection vanish on the exact fixed-center rest orbit.  The interval action
is the average of forward and backward positive kinetic squares.

This is a research-action screen against the locked literal translation map.
It does not modify or promote canonical V40.
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
FACE_FACTOR = 6.0


def neighbor(value: torch.Tensor, offset: tuple[int, int, int]) -> torch.Tensor:
    return torch.roll(value, shifts=tuple(-x for x in offset),
                      dims=(0, 1, 2))


def squared_norm(value: torch.Tensor) -> torch.Tensor:
    return torch.sum(value.real**2 + value.imag**2, dim=-1)


def psi_current_velocity(psi: torch.Tensor) -> torch.Tensor:
    """Native C19 phase current divided by the regular native density."""
    current = torch.zeros(psi.shape[:-1] + (3,), dtype=torch.float64)
    for offset, weight in rank.native.DIRECTIONS:
        direction = torch.tensor(offset, dtype=torch.float64)
        flux = torch.imag(torch.sum(
            torch.conj(psi) * neighbor(psi, offset), dim=-1))
        current = current + weight * flux[..., None] * direction
    current = current / SECOND_MOMENT
    density = squared_norm(psi) + rank.KAPPA * rank.R**2
    return current / density[..., None]


def advect_same_components(value: torch.Tensor,
                           velocity: torch.Tensor) -> torch.Tensor:
    """C19 approximation to v.grad(value), never mixing component labels."""
    result = torch.zeros_like(value)
    for offset, weight in rank.native.DIRECTIONS:
        direction = torch.tensor(offset, dtype=torch.float64)
        velocity_neighbor = neighbor(velocity, offset)
        along = torch.einsum(
            "...a,a->...", (velocity + velocity_neighbor) / 2, direction)
        result = result + weight * along[..., None] * (
            neighbor(value, offset) - value)
    return result / SECOND_MOMENT


def activation(left_psi: torch.Tensor, right_psi: torch.Tensor,
               dt: float) -> torch.Tensor:
    psi = (left_psi + right_psi) / 2
    psi_dot = (right_psi - left_psi) / dt
    rho_dot = 2 * torch.real(torch.sum(
        torch.conj(psi) * psi_dot, dim=-1))
    rho_psi = squared_norm(psi)
    transfer = rho_dot / (
        2 * rank.R * (rho_psi + rank.KAPPA * rank.R**2))
    return FACE_FACTOR * transfer**2 / (1 + FACE_FACTOR * transfer**2)


def linked_velocities(left: torch.Tensor, right: torch.Tensor,
                      left_psi: torch.Tensor, right_psi: torch.Tensor,
                      amount: torch.Tensor, dt: float) -> torch.Tensor:
    delta = (right - left) / dt
    left_transport = advect_same_components(
        left, psi_current_velocity(left_psi))
    right_transport = advect_same_components(
        right, psi_current_velocity(right_psi))
    # Two interval orientations make the kinetic square time symmetric.
    forward = delta + amount[..., None] * left_transport
    backward = -delta + amount[..., None] * right_transport
    return torch.stack((forward, backward), dim=0)


def interval(left: torch.Tensor, right: torch.Tensor, dt: float,
             mode: str) -> torch.Tensor:
    left_psi, left_chi = local.unpack_tensor(left, SIDE)
    right_psi, right_chi = local.unpack_tensor(right, SIDE)
    amount = activation(left_psi, right_psi, dt)

    psi_dot = (right_psi - left_psi) / dt
    if mode in ("psi", "both"):
        velocities = linked_velocities(
            left_psi, right_psi, left_psi, right_psi, amount, dt)
        psi_kinetic = torch.sum(squared_norm(velocities)) / 4
    else:
        psi_kinetic = torch.sum(squared_norm(psi_dot)) / 2

    chi_dot = (right_chi - left_chi) / dt
    if mode in ("chi", "both"):
        velocities = linked_velocities(
            left_chi, right_chi, left_psi, right_psi, amount, dt)
        chi_kinetic = rank.B * torch.sum(squared_norm(velocities)) / 4
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
    parser.add_argument("--maxiter", type=int, default=20)
    parser.add_argument("--next-maxiter", type=int, default=80)
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
    pass_update = (
        update["postsolve_DEL_relative_residual"] < 1e-8
        and update["correction_over_exact_translation_step"] < 1e-6)
    result = {
        "status": "V4_EXACT_TRANSLATION_CURRENT_PRONG_SPACETIME_GATE",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "mode": args.mode,
        "target_pattern": (
            "same-assignment exact one-cell translation of the complete "
            "joint Psi-chi profile with unchanged eight-site core values"),
        "action_candidate": (
            "positive reciprocal time-symmetric interval kinetic action; "
            "native Psi bond current selects C19 communication direction, "
            "while each complex component retains its own label"),
        "component_routing": [
            "Psi_x to Psi_x", "Psi_y to Psi_y", "Psi_z to Psi_z",
            "chi_x to chi_x", "chi_y to chi_y", "chi_z to chi_z"],
        "direction_source": (
            "native U(1)-invariant Psi C19 bond current; no chosen axis or "
            "destination appears in the action"),
        "coefficient_provenance": (
            "sum_C19 w n_a n_b=12 delta_ab; factor 6 from the six face "
            "directions; kappa and chi0 are locked native constants"),
        "rest_DEL_norm": float(torch.linalg.vector_norm(rest_residual)),
        "first_shell_and_clock_case": case,
        "verdict": (
            "CURRENT_PRONG_EXACT_SHIFT_PASS" if pass_update else
            "CURRENT_PRONG_EXACT_SHIFT_OPEN"),
        "claim_boundary": (
            "One signed-axis exact-map research gate. No action admission "
            "without all axes/signs, two free handoffs, conservation, rest "
            "spectrum, boundedness, and Dirac-current validation."),
    }
    output = HERE / (
        f"exact_translation_current_prong_{args.mode}_axis{args.axis}_"
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
