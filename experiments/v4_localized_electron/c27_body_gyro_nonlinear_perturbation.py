"""Unrestricted nonlinear implicit evolution of the C27/gyro carrier.

This is the actual variational update.  It solves the discrete Euler--Lagrange
equation for every following Psi/chi coordinate.  The calculation is performed
in the exact co-rotating frame only to make the rest orbit stationary; no
state projection, held site, reset, or fitted readout is used.  A fixed exact-
rest Jacobian is used as a quasi-Newton preconditioner, with a full-Jacobian
fallback if required.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.linalg import lu_factor, lu_solve

import c27_body_gyro_basis_probe as candidate
import c27_body_gyro_rest_spectrum as spectrum
import relative_kinetic_probe as relative


HERE = Path(__file__).resolve().parent
SOLVER_TOLERANCE = 1e-8


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cycles", type=int, default=1)
    parser.add_argument("--clock-fraction", type=float, default=0.01)
    parser.add_argument("--signs", nargs="+", type=float, default=(-1.0, 1.0))
    parser.add_argument(
        "--noncoplanar-fraction", type=float, default=None,
        help=("load this solved fraction from "
              "c19_triangle_noncoplanar_branch_probe.npz"))
    parser.add_argument(
        "--output", default="c27_body_gyro_nonlinear_perturbation.json")
    args = parser.parse_args()

    spectrum.configure()
    candidate.base.inverse.rank.lock_inertia()
    _, current, _, theta = candidate.base.inverse.rest_layers()
    q0 = np.asarray(current, dtype=np.float64)
    if args.noncoplanar_fraction is not None:
        archive = np.load(
            HERE / "c19_triangle_noncoplanar_branch_probe.npz")
        fractions = archive["fractions"]
        index = int(np.argmin(abs(fractions - args.noncoplanar_fraction)))
        if abs(float(fractions[index]) - args.noncoplanar_fraction) > 1e-12:
            raise ValueError(
                f"missing noncoplanar fraction {args.noncoplanar_fraction}")
        q0 = np.asarray(archive["states"][index], dtype=np.float64)
    a = 2 * spectrum.DT / np.sin(theta)
    coefficients = np.asarray((
        a + candidate.base.inverse.rank.B / 4,
        a - candidate.base.inverse.rank.B / 4,
        -a / 2,
        0.0,
    ))
    del_residual, _ = candidate.make_derivatives(1.0, coefficients)
    rotation = torch.tensor(relative.raw_rotation_matrix(
        spectrum.SIDE, theta, 0.0, 1), dtype=torch.float64)
    inverse_rotation = rotation.T

    def co_residual(previous: torch.Tensor, current_state: torch.Tensor,
                    following: torch.Tensor) -> torch.Tensor:
        return del_residual(
            inverse_rotation @ previous,
            current_state,
            rotation @ following)

    q0_tensor = torch.tensor(q0, dtype=torch.float64)
    rest_residual = co_residual(q0_tensor, q0_tensor, q0_tensor)
    rest_rms = float(np.linalg.norm(rest_residual.detach().numpy())
                     / np.sqrt(q0.size))
    following_jacobian = torch.autograd.functional.jacobian(
        lambda following: co_residual(
            q0_tensor, q0_tensor, following),
        q0_tensor.detach().requires_grad_(True), vectorize=True,
    ).detach().numpy()
    fixed_factor = lu_factor(following_jacobian)

    psi0, chi0 = relative.unpack_raw(q0, spectrum.SIDE)
    carrier = np.sum(abs(psi0) ** 2, axis=-1) > 1e-12
    steps_per_cycle = int(round(2 * np.pi / theta))
    total_steps = args.cycles * steps_per_cycle
    rows = []

    def phase_aligned_shape(state: np.ndarray) -> dict:
        psi, chi = relative.unpack_raw(state, spectrum.SIDE)
        psi_phase = float(np.angle(np.sum(
            np.conj(psi0[..., :2]) * psi[..., :2])))
        chi_phase = float(np.angle(np.sum(
            np.conj(chi0[..., :2]) * chi[..., :2])))
        inverse = relative.raw_rotation_matrix(
            spectrum.SIDE, psi_phase, chi_phase, -1)
        aligned = inverse @ state
        return {
            "relative_shape_residual": float(
                np.linalg.norm(aligned - q0) / np.linalg.norm(q0)),
            "removed_psi_phase": psi_phase,
            "removed_chi_phase": chi_phase,
        }

    def deviation_decomposition(current_state: np.ndarray,
                                previous: np.ndarray) -> dict:
        powers = {"psi": 0.0, "chi": 0.0}
        carrier_powers = {"psi": 0.0, "chi": 0.0}
        component_powers = {
            "psi": np.zeros(3), "chi": np.zeros(3)}
        for state in (current_state, previous):
            dpsi, dchi = relative.unpack_raw(state - q0, spectrum.SIDE)
            for name, value in (("psi", dpsi), ("chi", dchi)):
                site = np.sum(abs(value) ** 2, axis=-1)
                powers[name] += float(np.sum(site))
                carrier_powers[name] += float(np.sum(site[carrier]))
                component_powers[name] += np.sum(
                    abs(value) ** 2, axis=(0, 1, 2))
        total = max(sum(powers.values()), 1e-300)
        return {
            "field_fractions": {
                name: value / total for name, value in powers.items()},
            "carrier_fractions_of_field": {
                name: carrier_powers[name] / max(powers[name], 1e-300)
                for name in powers},
            "component_fractions_of_total": {
                name: [float(value / total)
                       for value in component_powers[name]]
                for name in component_powers},
        }

    def solve_following(previous: np.ndarray, current_state: np.ndarray,
                        initial: np.ndarray):
        following = initial.copy()
        trace = []
        full_jacobian_uses = 0
        for iteration in range(14):
            tensors = tuple(torch.tensor(value, dtype=torch.float64)
                            for value in (previous, current_state, following))
            value = co_residual(*tensors).detach().numpy()
            rms = float(np.linalg.norm(value) / np.sqrt(value.size))
            trace.append(rms)
            if rms < SOLVER_TOLERANCE:
                return following, trace, full_jacobian_uses, True
            correction = lu_solve(fixed_factor, -value)
            accepted = False
            for power in range(10):
                step = 0.5**power
                trial = following + step * correction
                trial_tensors = tuple(torch.tensor(v, dtype=torch.float64)
                                      for v in (previous, current_state, trial))
                trial_value = co_residual(*trial_tensors).detach().numpy()
                trial_rms = float(
                    np.linalg.norm(trial_value) / np.sqrt(trial_value.size))
                if trial_rms < rms:
                    following = trial
                    accepted = True
                    break
            if accepted:
                continue
            following_tensor = torch.tensor(
                following, dtype=torch.float64, requires_grad=True)
            exact = torch.autograd.functional.jacobian(
                lambda next_state: co_residual(
                    torch.tensor(previous, dtype=torch.float64),
                    torch.tensor(current_state, dtype=torch.float64),
                    next_state),
                following_tensor, vectorize=True).detach().numpy()
            full_jacobian_uses += 1
            exact_correction = np.linalg.lstsq(
                exact, -value, rcond=1e-10)[0]
            for power in range(12):
                step = 0.5**power
                trial = following + step * exact_correction
                trial_value = co_residual(
                    torch.tensor(previous, dtype=torch.float64),
                    torch.tensor(current_state, dtype=torch.float64),
                    torch.tensor(trial, dtype=torch.float64)).detach().numpy()
                trial_rms = float(
                    np.linalg.norm(trial_value) / np.sqrt(trial_value.size))
                if trial_rms < rms:
                    following = trial
                    accepted = True
                    break
            if not accepted:
                break
        return following, trace, full_jacobian_uses, False

    for sign in args.signs:
        delta = sign * args.clock_fraction * theta
        internal_previous = relative.raw_rotation_matrix(
            spectrum.SIDE, 0.0, delta, -1) @ q0
        previous = internal_previous
        current_state = q0.copy()
        initial_phase_state = np.r_[current_state - q0, previous - q0]
        initial_norm = max(np.linalg.norm(initial_phase_state), 1e-300)
        samples = []
        maximum_amplification = 1.0
        maximum_state = (current_state.copy(), previous.copy())
        maximum_step = 0
        maximum_del_rms = rest_rms
        maximum_aligned_shape = 0.0
        total_full_jacobians = 0
        converged = True
        for step_index in range(1, total_steps + 1):
            predictor = current_state + (current_state - previous)
            following, trace, full_uses, step_converged = solve_following(
                previous, current_state, predictor)
            total_full_jacobians += full_uses
            maximum_del_rms = max(maximum_del_rms, trace[-1])
            if not step_converged:
                converged = False
                samples.append({
                    "step": step_index,
                    "solver_trace": trace,
                    "solver_converged": False,
                })
                break
            previous, current_state = current_state, following
            phase_state = np.r_[current_state - q0, previous - q0]
            amplification = float(np.linalg.norm(phase_state) / initial_norm)
            aligned_shape = phase_aligned_shape(current_state)
            maximum_aligned_shape = max(
                maximum_aligned_shape,
                aligned_shape["relative_shape_residual"])
            maximum_amplification = max(maximum_amplification, amplification)
            if amplification >= maximum_amplification:
                maximum_state = (current_state.copy(), previous.copy())
                maximum_step = step_index
            if (step_index == 1 or step_index % 5 == 0
                    or step_index % steps_per_cycle == 0
                    or step_index == total_steps):
                psi, chi = relative.unpack_raw(current_state, spectrum.SIDE)
                rho_psi = np.sum(abs(psi) ** 2, axis=-1)
                rho_chi = np.sum(abs(chi) ** 2, axis=-1)
                samples.append({
                    "step": step_index,
                    "cycle": float(step_index / steps_per_cycle),
                    "phase_space_amplification": amplification,
                    "phase_aligned_shape": aligned_shape,
                    "solver_final_DEL_rms": trace[-1],
                    "solver_iterations": len(trace),
                    "solver_converged": True,
                    "carrier_psi_density_minmax": [
                        float(np.min(rho_psi[carrier])),
                        float(np.max(rho_psi[carrier]))],
                    "exterior_psi_density_rms": float(np.sqrt(
                        np.mean(rho_psi[~carrier] ** 2))),
                    "carrier_chi_density_minmax": [
                        float(np.min(rho_chi[carrier])),
                        float(np.max(rho_chi[carrier]))],
                    "exterior_chi_density_rms_change": float(np.sqrt(np.mean(
                        (rho_chi[~carrier]
                         - np.sum(abs(chi0[~carrier])**2, axis=-1))**2))),
                })
        rows.append({
            "sign": sign,
            "chi_clock_fraction": args.clock_fraction,
            "converged_all_steps": converged,
            "completed_steps": int(samples[-1]["step"] if samples else 0),
            "maximum_phase_space_amplification": maximum_amplification,
            "maximum_amplification_step": maximum_step,
            "maximum_amplification_decomposition": deviation_decomposition(
                *maximum_state),
            "maximum_DEL_rms": maximum_del_rms,
            "maximum_phase_aligned_shape_residual": maximum_aligned_shape,
            "full_jacobian_fallbacks": total_full_jacobians,
            "samples": samples,
        })

    result = {
        "status": "V4_C27_BODY_GYRO_NONLINEAR_PERTURBATION",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "side": spectrum.SIDE,
        "dt": spectrum.DT,
        "theta": float(theta),
        "steps_per_cycle": steps_per_cycle,
        "cycles_requested": args.cycles,
        "noncoplanar_fraction": args.noncoplanar_fraction,
        "rest_DEL_rms": rest_rms,
        "solver_tolerance": SOLVER_TOLERANCE,
        "rows": rows,
        "verdict": (
            "C27_BODY_GYRO_NONLINEAR_PERTURBATION_NUMERIC_PASS"
            if all(row["converged_all_steps"] for row in rows) else
            "C27_BODY_GYRO_NONLINEAR_UPDATE_FAILED"),
        "claim_boundary": (
            "Unrestricted nonlinear implicit evolution and solver closure "
            "only. A numerical pass is not yet the 100-cycle stability, "
            "translation, containment, conservation, or Dirac-current gate."),
    }
    output = HERE / args.output
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
