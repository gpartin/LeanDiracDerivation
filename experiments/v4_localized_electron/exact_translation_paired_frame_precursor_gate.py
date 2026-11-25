"""Test paired Psi/chi live-frame broadcast on the exact translation map.

The previous bond-frame candidate transported chi velocity but left Psi with
strictly local inertia.  This reciprocal action trial replaces the Psi kinetic
square by the same native live-frame blend during density transfer.  At the
rest orbit rhoPsiDot is zero, so the addition vanishes exactly and the prior
rest action is recovered.

This is an explicit research action candidate, not a canonical change.  The
eight occupied carrier sites remain fixed while a first-shell precursor is
solved against the literal one-cell translation recurrence.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.sparse.linalg import LinearOperator, lsmr
import torch

import chi_bond_frame_transport_probe as bond_frame
import density_mold_connection_probe as local
import edge_snap_action_rank as rank
import edge_snap_transfer_four_update_orbit as transfer
import exact_translation_bond_frame_precursor_gate as chi_gate
import exact_translation_precursor_linear_gate as precursor


HERE = Path(__file__).resolve().parent
SIDE = precursor.SIDE
FRAME_FACTOR = 6.0
DUAL_REGULARIZER = rank.KAPPA


def dual_frame(psi: torch.Tensor, chi: torch.Tensor) -> torch.Tensor:
    """Dimensionless live three-prong frame [chi, Psi, chi wedge Psi]."""
    regulator = rank.KAPPA * rank.R**2
    area = torch.linalg.cross(chi, psi, dim=-1)
    rho_chi = torch.sum(local.squared_norm(chi), dim=-1)
    rho_psi = torch.sum(local.squared_norm(psi), dim=-1)
    rho_area = torch.sum(local.squared_norm(area), dim=-1)
    chi_unit = chi / torch.sqrt(rho_chi + regulator)[..., None]
    psi_unit = psi / torch.sqrt(rho_psi + regulator)[..., None]
    area_unit = area / torch.sqrt(
        rho_area + regulator**2)[..., None]
    return torch.stack((chi_unit, psi_unit, area_unit), dim=-1)


def dual_transported_average(psi: torch.Tensor, chi: torch.Tensor,
                             velocity: torch.Tensor) -> torch.Tensor:
    """Parallel transport using the exact regularized dual live frame."""
    frame_i = dual_frame(psi, chi)
    identity = torch.eye(3, dtype=torch.complex128)
    result = torch.zeros_like(velocity)
    for offset, weight in rank.native.DIRECTIONS:
        frame_j = torch.roll(frame_i, shifts=offset, dims=(0, 1, 2))
        velocity_j = torch.roll(velocity, shifts=offset, dims=(0, 1, 2))
        adjoint_j = torch.conj(frame_j).transpose(-1, -2)
        gram_j = adjoint_j @ frame_j + DUAL_REGULARIZER * identity
        coordinates = torch.linalg.solve(
            gram_j, torch.einsum("...ab,...b->...a", adjoint_j, velocity_j))
        mapped = torch.einsum("...ab,...b->...a", frame_i, coordinates)
        result = result + weight * mapped
    return result / rank.C19_WEIGHT_SUM


def paired_interval(left: torch.Tensor, right: torch.Tensor,
                    dt: float) -> torch.Tensor:
    base = transfer.interval_action(left, right, dt, 0.0)
    left_psi, left_chi = local.unpack_tensor(left, SIDE)
    right_psi, right_chi = local.unpack_tensor(right, SIDE)
    psi = (left_psi + right_psi) / 2
    chi = (left_chi + right_chi) / 2
    psi_dot = (right_psi - left_psi) / dt
    rho_dot = 2 * torch.real(torch.sum(
        torch.conj(psi) * psi_dot, dim=-1))
    rho_psi = torch.sum(local.squared_norm(psi), dim=-1)
    transfer_rate = rho_dot / (
        2 * rank.R * (rho_psi + rank.KAPPA * rank.R**2))
    activation = (FRAME_FACTOR * transfer_rate**2
                  / (1 + FRAME_FACTOR * transfer_rate**2))
    transported = bond_frame.transported_average(psi, chi, psi_dot)
    paired_velocity = ((1 - activation)[..., None] * psi_dot
                       + activation[..., None] * transported)
    replacement = torch.sum(local.squared_norm(paired_velocity))
    canonical = torch.sum(local.squared_norm(psi_dot))
    return base + dt * (replacement - canonical) / 2


def del_three(previous: torch.Tensor, current: torch.Tensor,
              following: torch.Tensor, dt: float) -> torch.Tensor:
    def summed(left, middle, right):
        return (paired_interval(left, middle, dt)
                + paired_interval(middle, right, dt))
    return torch.func.grad(summed, argnums=1)(previous, current, following)


def profile_residual(raw: torch.Tensor, axis: int, sign: int, dt: float,
                     psi_phase: float, chi_phase: float) -> torch.Tensor:
    following = precursor.translate(
        raw, axis, sign, psi_phase, chi_phase)
    previous = precursor.translate(
        raw, axis, -sign, -psi_phase, -chi_phase)
    return del_three(previous, raw, following, dt)


def solve_shell_and_clock(shell: np.ndarray, core: np.ndarray,
                          q0: torch.Tensor, axis: int, sign: int,
                          dt0: float, psi_phase0: float, chi_phase0: float,
                          maxiter: int, profile_function=profile_residual,
                          del_function=del_three, next_maxiter: int | None = None
                          ) -> dict:
    if next_maxiter is None:
        next_maxiter = maxiter
    indices = np.flatnonzero(precursor.raw_coordinate_mask(shell))
    field_count = indices.size
    zero = torch.tensor(0.0, dtype=torch.float64)
    initial_raw_dt = torch.tensor(
        precursor.orbit.raw_from_dt(dt0), dtype=torch.float64)

    def residual(raw, raw_dt, delta_psi_phase, delta_chi_phase):
        return profile_function(
            raw, axis, sign,
            precursor.orbit.dt_from_raw(raw_dt),
            psi_phase0 + delta_psi_phase,
            chi_phase0 + delta_chi_phase)

    inputs = (q0, initial_raw_dt, zero, zero)
    residual0 = residual(*inputs).detach()
    baseline_norm = float(torch.linalg.vector_norm(residual0))

    def split_direction(values):
        values = np.asarray(values, dtype=float)
        raw = torch.zeros_like(q0)
        raw[torch.tensor(indices, dtype=torch.long)] = torch.tensor(
            values[:field_count], dtype=torch.float64)
        scalars = tuple(torch.tensor(value, dtype=torch.float64)
                        for value in values[field_count:])
        return (raw, *scalars)

    def matvec(values):
        _value, image = torch.autograd.functional.jvp(
            residual, inputs, split_direction(values),
            create_graph=False, strict=False)
        return image.detach().numpy()

    def rmatvec(values):
        variables = (
            q0.detach().clone().requires_grad_(True),
            initial_raw_dt.detach().clone().requires_grad_(True),
            zero.detach().clone().requires_grad_(True),
            zero.detach().clone().requires_grad_(True),
        )
        probe = torch.tensor(np.asarray(values, dtype=float),
                             dtype=torch.float64)
        gradients = torch.autograd.grad(
            torch.dot(residual(*variables), probe), variables)
        return np.concatenate((
            gradients[0].detach().numpy()[indices],
            np.asarray([float(value.detach()) for value in gradients[1:]])))

    operator = LinearOperator(
        (q0.numel(), field_count + 3), matvec=matvec, rmatvec=rmatvec,
        dtype=np.float64)
    solution = lsmr(operator, -residual0.numpy(), atol=1e-10, btol=1e-10,
                    maxiter=maxiter, conlim=1e10)
    direction = split_direction(solution[0])
    linear = residual0.numpy() + matvec(solution[0])
    rows = []
    for scale in np.concatenate((np.linspace(0.0, 0.1, 11),
                                 np.linspace(0.2, 1.0, 9))):
        values = tuple(base + float(scale) * delta
                       for base, delta in zip(inputs, direction, strict=True))
        image = residual(*values).detach()
        rows.append({
            "scale": float(scale),
            "relative_residual": float(
                torch.linalg.vector_norm(image) / baseline_norm),
        })
    best = min(rows, key=lambda item: item["relative_residual"])
    best_values = tuple(base + best["scale"] * delta
                        for base, delta in zip(inputs, direction, strict=True))
    best_residual = residual(*best_values).detach()
    raw_correction_norm = float(torch.linalg.vector_norm(direction[0]))

    # Continue the same exact-map problem nonlinearly.  Only shell values and
    # the three legitimate clock variables are optimized; the carrier core is
    # never a parameter and therefore remains bit-for-bit unchanged.
    shell_values = torch.tensor(
        (best["scale"] * solution[0][:field_count]), dtype=torch.float64,
        requires_grad=True)
    raw_dt = torch.tensor(float(best_values[1]), dtype=torch.float64,
                          requires_grad=True)
    delta_psi = torch.tensor(float(best_values[2]), dtype=torch.float64,
                             requires_grad=True)
    delta_chi = torch.tensor(float(best_values[3]), dtype=torch.float64,
                             requires_grad=True)
    optimizer = torch.optim.LBFGS(
        [shell_values, raw_dt, delta_psi, delta_chi], lr=0.3,
        max_iter=maxiter, max_eval=2 * maxiter,
        tolerance_grad=1e-11, tolerance_change=1e-15,
        history_size=40, line_search_fn="strong_wolfe")
    trace = []

    def nonlinear_state():
        current = q0.clone()
        current[torch.tensor(indices, dtype=torch.long)] += shell_values
        return current

    def closure():
        optimizer.zero_grad()
        image = residual(nonlinear_state(), raw_dt, delta_psi, delta_chi)
        loss = torch.sum(image**2) / baseline_norm**2
        loss.backward()
        trace.append(float(torch.sqrt(loss.detach())))
        return loss

    nonlinear_initial = float(torch.sqrt(closure().detach()))
    optimizer.step(closure)
    nonlinear_current = nonlinear_state().detach()
    nonlinear_image = residual(
        nonlinear_current, raw_dt, delta_psi, delta_chi).detach()
    nonlinear_final = float(
        torch.linalg.vector_norm(nonlinear_image) / baseline_norm)
    nonlinear_correction = nonlinear_current - q0

    solved_dt = precursor.orbit.dt_from_raw(raw_dt.detach())
    solved_psi_phase = psi_phase0 + delta_psi.detach()
    solved_chi_phase = chi_phase0 + delta_chi.detach()
    previous = precursor.translate(
        nonlinear_current, axis, -sign, -solved_psi_phase,
        -solved_chi_phase)
    target_following = precursor.translate(
        nonlinear_current, axis, sign, solved_psi_phase,
        solved_chi_phase)

    def following_residual(following):
        return del_function(previous, nonlinear_current, following, solved_dt)

    target_residual = following_residual(target_following).detach()

    def next_matvec(values):
        direction = torch.tensor(np.asarray(values, dtype=float),
                                 dtype=torch.float64)
        _value, image = torch.autograd.functional.jvp(
            following_residual, target_following, direction,
            create_graph=False, strict=False)
        return image.detach().numpy()

    def next_rmatvec(values):
        probe = torch.tensor(np.asarray(values, dtype=float),
                             dtype=torch.float64)
        variable = target_following.detach().clone().requires_grad_(True)
        gradient = torch.autograd.grad(
            torch.dot(following_residual(variable), probe), variable)[0]
        return gradient.detach().numpy()

    next_operator = LinearOperator(
        (q0.numel(), q0.numel()), matvec=next_matvec,
        rmatvec=next_rmatvec, dtype=np.float64)
    next_solution = lsmr(
        next_operator, -target_residual.numpy(), atol=1e-10, btol=1e-10,
        maxiter=next_maxiter, conlim=1e10)
    next_correction = torch.tensor(next_solution[0], dtype=torch.float64)
    desired_step = target_following - nonlinear_current
    state_update_defect = {
        "lsmr_stop_code": int(next_solution[1]),
        "lsmr_iterations": int(next_solution[2]),
        "linearized_next_layer_correction_norm": float(
            torch.linalg.vector_norm(next_correction)),
        "correction_over_exact_translation_step": float(
            torch.linalg.vector_norm(next_correction)
            / max(float(torch.linalg.vector_norm(desired_step)), 1e-300)),
        "postsolve_DEL_relative_residual": float(
            next_solution[3]
            / max(float(torch.linalg.vector_norm(target_residual)), 1e-300)),
    }
    return {
        "free_site_count": int(np.count_nonzero(shell)),
        "free_real_coordinate_count": int(field_count),
        "lsmr_stop_code": int(solution[1]),
        "lsmr_iterations": int(solution[2]),
        "linear_relative_residual": float(
            np.linalg.norm(linear) / baseline_norm),
        "correction_norm_over_state_norm": float(
            raw_correction_norm / torch.linalg.vector_norm(q0)),
        "linear_bounded_dt_coordinate_change": float(direction[1]),
        "linear_psi_phase_change": float(direction[2]),
        "linear_chi_phase_change": float(direction[3]),
        "best_nonlinear": best,
        "best_dt": float(precursor.orbit.dt_from_raw(best_values[1])),
        "best_psi_phase": float(psi_phase0 + best_values[2]),
        "best_chi_phase": float(chi_phase0 + best_values[3]),
        "best_nonlinear_residual_diagnostics": (
            precursor.residual_diagnostics(best_residual, core, shell)),
        "full_nonlinear_solve": {
            "initial_relative_residual": nonlinear_initial,
            "final_relative_residual": nonlinear_final,
            "minimum_trace_residual": min(trace),
            "optimizer_evaluations": len(trace),
            "correction_norm_over_state_norm": float(
                torch.linalg.vector_norm(nonlinear_correction)
                / torch.linalg.vector_norm(q0)),
            "dt": float(solved_dt),
            "dt_bounds": [precursor.orbit.DT_MIN, precursor.orbit.DT_MAX],
            "psi_phase": float(psi_phase0 + delta_psi.detach()),
            "chi_phase": float(chi_phase0 + delta_chi.detach()),
            "residual_diagnostics": precursor.residual_diagnostics(
                nonlinear_image, core, shell),
            "state_update_defect": state_update_defect,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--axis", type=int, default=0, choices=(0, 1, 2))
    parser.add_argument("--sign", type=int, default=1, choices=(-1, 1))
    parser.add_argument("--maxiter", type=int, default=120)
    parser.add_argument("--next-maxiter", type=int)
    parser.add_argument("--action", choices=("chi_only", "chi_dual", "paired"),
                        default="paired")
    parser.add_argument("--dual-regularizer", type=float, default=rank.KAPPA)
    args = parser.parse_args()
    rank.lock_inertia()
    chi_gate.configure_action()
    if args.action == "chi_dual":
        global DUAL_REGULARIZER
        DUAL_REGULARIZER = args.dual_regularizer
        bond_frame.transported_average = dual_transported_average

    evidence = json.loads((HERE / "edge_snap_verlet_gate.json").read_text(
        encoding="utf-8"))
    row = next(item for item in evidence["rows"]
               if item["axis"] == args.axis and item["sign"] == args.sign)
    amplitude_sq = rank.R**2 / (2 + 54 / rank.R**2)
    psi, chi = rank.native.carrier_fields(SIDE, amplitude_sq)
    q0 = torch.tensor(rank.pack(psi, chi), dtype=torch.float64)
    core, shell = precursor.site_masks(psi)

    selected_profile = (profile_residual if args.action == "paired"
                        else chi_gate.profile_residual)
    selected_del = del_three if args.action == "paired" else chi_gate.del_three

    def function(value: torch.Tensor) -> torch.Tensor:
        return selected_profile(
            value, args.axis, args.sign, row["best_dt"],
            row["best_phase_psi"], row["best_phase_chi"])

    residual0 = function(q0).detach()
    case = precursor.solve_case(
        "C19_first_shell", shell, core, shell, q0, residual0,
        function, args.maxiter)
    clock_case = solve_shell_and_clock(
        shell, core, q0, args.axis, args.sign, row["best_dt"],
        row["best_phase_psi"], row["best_phase_chi"], args.maxiter,
        selected_profile, selected_del, args.next_maxiter)
    best_residual = min(
        case["best_nonlinear"]["relative_residual"],
        clock_case["best_nonlinear"]["relative_residual"],
        clock_case["full_nonlinear_solve"]["final_relative_residual"])
    result = {
        "status": ("V4_EXACT_TRANSLATION_PAIRED_FRAME_PRECURSOR_GATE"
                   if args.action == "paired" else
                   "V4_EXACT_TRANSLATION_CHI_DUAL_FRAME_PRECURSOR_GATE"
                   if args.action == "chi_dual" else
                   "V4_EXACT_TRANSLATION_CHI_FRAME_PRECURSOR_GATE"),
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "axis": args.axis,
        "sign": args.sign,
        "target_pattern": (
            "same eight-site Psi-chi carrier values and entire native "
            "profile translated exactly one cell"),
        "action_candidate": (
            ("prior chi density-mold/live-frame kinetic action plus a "
             "positive Psi kinetic square transported through the same live "
             "chi, Psi, chi-wedge-Psi bond frame during handoff")
            if args.action == "paired" else
            ("positive reciprocal chi density-mold kinetic action using "
             "the regularized inverse Gram matrix of the live chi, Psi, "
             "chi-wedge-Psi frame on every C19 bond")
            if args.action == "chi_dual" else
            ("positive reciprocal chi density-mold/live-frame kinetic "
             "action transporting chi through the live chi, Psi, "
             "chi-wedge-Psi bond frame")),
        "rest_reduction": (
            "rhoPsiDot=0 on the exact rest orbit, hence activation=0 and "
            "the paired Psi kinetic square equals the canonical one"),
        "coefficient_provenance": (
            "factor 6 from the six C19 face transport directions"),
        "dual_frame_regularizer": (
            args.dual_regularizer if args.action == "chi_dual" else None),
        "core_values_held_exact_during_inverse_gate": True,
        "dt": row["best_dt"],
        "psi_phase": row["best_phase_psi"],
        "chi_phase": row["best_phase_chi"],
        "compact_profile_del_norm": float(
            torch.linalg.vector_norm(residual0)),
        "first_shell_case": case,
        "first_shell_and_clock_case": clock_case,
        "verdict": (
            "FULL_FRAME_PRECURSOR_ADVANCES_TO_NONLINEAR_PROFILE_SOLVE"
            if (best_residual < 0.1
                and clock_case["correction_norm_over_state_norm"] < 0.5)
            else "FULL_FRAME_PRECURSOR_DOES_NOT_CLOSE_EXACT_SHIFT"),
        "claim_boundary": (
            "One signed-axis inverse-profile action screen. A favorable "
            "result requires action boundedness/well-posedness, a nonlinear "
            "exact profile, all axes/signs, free evolution, conservation, "
            "and Dirac-current validation."),
    }
    sign_label = "plus" if args.sign > 0 else "minus"
    regularizer_label = (f"_reg{args.dual_regularizer:.3e}"
                         if args.action == "chi_dual" else "")
    output = HERE / (
        f"exact_translation_{args.action}_frame_precursor_axis{args.axis}_"
        f"{sign_label}{regularizer_label}.json")
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n",
                      encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "compact_profile_del_norm": result["compact_profile_del_norm"],
        "linear_relative_residual": case["linear_relative_residual"],
        "correction_norm_over_state_norm": case[
            "correction_norm_over_state_norm"],
        "best_nonlinear": case["best_nonlinear"],
        "clock_solved_best_nonlinear": clock_case["best_nonlinear"],
        "clock_solved_values": {
            "dt": clock_case["best_dt"],
            "psi_phase": clock_case["best_psi_phase"],
            "chi_phase": clock_case["best_chi_phase"],
        },
        "full_nonlinear_solve": clock_case["full_nonlinear_solve"],
        "residual_diagnostics": case[
            "best_nonlinear_residual_diagnostics"],
        "verdict": result["verdict"],
    }, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
