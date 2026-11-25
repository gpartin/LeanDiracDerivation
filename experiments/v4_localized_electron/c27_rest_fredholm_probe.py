"""Fredholm gate for a smooth joint-W C27 rest continuation.

The joint body-diagonal action closes the missing fixed-charge transport rank
but unbalances the old rest orbit.  This probe asks the necessary next
question: is its rest forcing in the range of the unchanged relative-orbit
Jacobian when the complete carrier/first-shell state and the Psi clock phase
may adjust?

The solve is matrix-free.  It uses every raw Psi/chi coordinate in the 4^3
carrier-plus-first-shell block of the side-five lattice and one clock variable.
No target transport or Dirac data enter the solve.  If the linearized residual
closes, scaled nonlinear controls must fall quadratically with the C27
continuation strength before a full branch solve is warranted.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.sparse.linalg import LinearOperator, lsmr
from torch.func import grad, jvp, vjp

import c27_body_diagonal_transport_probe as c27
import localized_carrier_contrast_inverse_launch as inverse


HERE = Path(__file__).resolve().parent
c27.MODE = "joint_W"


def interval_action(left: torch.Tensor, right: torch.Tensor,
                    strength: torch.Tensor) -> torch.Tensor:
    return (
        inverse.interval_action(left, right)
        - inverse.DT * strength
        * (c27.body_potential(left) + c27.body_potential(right)) / 2
    )


def three_layer_action(previous: torch.Tensor, current: torch.Tensor,
                       following: torch.Tensor,
                       strength: torch.Tensor) -> torch.Tensor:
    return interval_action(previous, current, strength) + interval_action(
        current, following, strength)


middle_residual = grad(three_layer_action, argnums=1)


def phase_rotate(raw: torch.Tensor, theta_psi: torch.Tensor,
                 theta_chi: torch.Tensor,
                 sign: float) -> torch.Tensor:
    psi, chi = inverse.current_audit.mold.unpack_tensor(raw, inverse.SIDE)
    psi_phase = torch.exp(1j * sign * theta_psi)
    chi_phase = torch.stack((
        torch.exp(1j * sign * theta_chi),
        torch.exp(1j * sign * theta_chi),
        torch.ones((), dtype=torch.complex128),
    ))
    return inverse.current_audit.pack_tensor(
        psi_phase * psi, chi_phase * chi)


def orbit_residual(raw: torch.Tensor, theta_psi: torch.Tensor,
                   theta_chi: torch.Tensor,
                   strength: torch.Tensor) -> torch.Tensor:
    previous = phase_rotate(raw, theta_psi, theta_chi, -1.0)
    following = phase_rotate(raw, theta_psi, theta_chi, 1.0)
    return middle_residual(previous, raw, following, strength)


def active_indices(scope: str) -> np.ndarray:
    shape = (inverse.SIDE, inverse.SIDE, inverse.SIDE, 4, 3)
    indices = np.arange(np.prod(shape)).reshape(shape)
    if scope == "full":
        return indices.reshape(-1)
    # Carrier centers and their complete first shell occupy coordinates 0:4.
    return indices[0:4, 0:4, 0:4].reshape(-1)


def carrier_overlap(reference: np.ndarray, trial: np.ndarray) -> float:
    reference_psi, reference_chi = inverse.rank.native.unpack(
        reference, inverse.SIDE)
    trial_psi, trial_chi = inverse.rank.native.unpack(trial, inverse.SIDE)
    mask = np.sum(np.abs(reference_psi)**2, axis=-1) > 1e-20
    left = np.concatenate((reference_psi[mask].reshape(-1),
                           reference_chi[mask].reshape(-1)))
    right = np.concatenate((trial_psi[mask].reshape(-1),
                            trial_chi[mask].reshape(-1)))
    return float(abs(np.vdot(left, right))
                 / np.sqrt(np.vdot(left, left).real
                           * np.vdot(right, right).real))


def residual_breakdown(vector: np.ndarray) -> dict[str, object]:
    data = vector.reshape(
        inverse.SIDE, inverse.SIDE, inverse.SIDE, 4, 3)
    total = float(np.vdot(vector, vector).real)
    field_power = {
        "Psi": float(np.sum(data[..., 0:2, :]**2)),
        "chi": float(np.sum(data[..., 2:4, :]**2)),
    }
    component_power = {
        f"component_{component}": float(np.sum(data[..., :, component]**2))
        for component in range(3)
    }
    site_power = np.sum(data**2, axis=(-1, -2))
    order = np.argsort(site_power.reshape(-1))[::-1][:8]
    top_sites = []
    for flat in order:
        site = tuple(int(value) for value in np.unravel_index(
            flat, site_power.shape))
        top_sites.append({
            "site": list(site),
            "power_fraction": float(site_power[site] / max(total, 1e-300)),
            "raw_components": data[site].tolist(),
        })
    return {
        "norm": float(np.linalg.norm(vector)),
        "field_power_fractions": {
            name: value / max(total, 1e-300)
            for name, value in field_power.items()
        },
        "component_power_fractions": {
            name: value / max(total, 1e-300)
            for name, value in component_power.items()
        },
        "top_sites": top_sites,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", choices=("first_shell", "full"),
                        default="first_shell")
    args = parser.parse_args()
    inverse.rank.lock_inertia()
    inverse.rest.SIDE = inverse.SIDE
    inverse.rest.DT = inverse.DT
    _, current, _, theta_value = inverse.rest_layers()
    base = torch.tensor(current, dtype=torch.float64)
    theta_psi = torch.tensor(theta_value, dtype=torch.float64)
    theta_chi = torch.tensor(0.0, dtype=torch.float64)
    zero = torch.tensor(0.0, dtype=torch.float64)
    one = torch.tensor(1.0, dtype=torch.float64)
    indices = active_indices(args.scope)
    variable_count = indices.size + 2

    baseline = orbit_residual(
        base, theta_psi, theta_chi, zero).detach().numpy()
    forcing = (
        orbit_residual(
            base, theta_psi, theta_chi, one).detach().numpy() - baseline)

    def matvec(vector: np.ndarray) -> np.ndarray:
        raw_direction = np.zeros(base.numel(), dtype=np.float64)
        raw_direction[indices] = vector[:-2]
        direction = torch.tensor(raw_direction, dtype=torch.float64)
        psi_clock_direction = torch.tensor(vector[-2], dtype=torch.float64)
        chi_clock_direction = torch.tensor(vector[-1], dtype=torch.float64)
        function = lambda raw, psi_clock, chi_clock: orbit_residual(
            raw, psi_clock, chi_clock, zero)
        _, image = jvp(
            function, (base, theta_psi, theta_chi),
            (direction, psi_clock_direction, chi_clock_direction))
        return image.detach().numpy()

    def rmatvec(vector: np.ndarray) -> np.ndarray:
        function = lambda raw, psi_clock, chi_clock: orbit_residual(
            raw, psi_clock, chi_clock, zero)
        _, pullback = vjp(function, base, theta_psi, theta_chi)
        raw_gradient, psi_clock_gradient, chi_clock_gradient = pullback(
            torch.tensor(vector, dtype=torch.float64))
        return np.concatenate((raw_gradient.detach().numpy()[indices],
                               [float(psi_clock_gradient.detach()),
                                float(chi_clock_gradient.detach())]))

    operator = LinearOperator(
        (base.numel(), variable_count), matvec=matvec,
        rmatvec=rmatvec, dtype=np.float64)
    solution = lsmr(
        operator, -forcing, atol=1e-11, btol=1e-11,
        conlim=1e10, maxiter=800, show=False)
    correction = solution[0]
    linear_residual = operator.matvec(correction) + forcing

    raw_correction = np.zeros(base.numel(), dtype=np.float64)
    raw_correction[indices] = correction[:-2]
    nonlinear_rows = []
    for strength in (1e-4, 1e-3, 1e-2):
        trial_raw = base + strength * torch.tensor(
            raw_correction, dtype=torch.float64)
        trial_theta_psi = theta_psi + strength * correction[-2]
        trial_theta_chi = theta_chi + strength * correction[-1]
        actual = orbit_residual(
            trial_raw, trial_theta_psi, trial_theta_chi,
            torch.tensor(strength, dtype=torch.float64)).detach().numpy()
        uncorrected = orbit_residual(
            base, theta_psi, theta_chi,
            torch.tensor(strength, dtype=torch.float64)).detach().numpy()
        nonlinear_rows.append({
            "strength": strength,
            "uncorrected_residual_norm": float(np.linalg.norm(uncorrected)),
            "corrected_residual_norm": float(np.linalg.norm(actual)),
            "corrected_over_uncorrected": float(
                np.linalg.norm(actual) / max(np.linalg.norm(uncorrected), 1e-30)),
            "corrected_over_strength_squared": float(
                np.linalg.norm(actual) / strength**2),
            "carrier_overlap": carrier_overlap(
                current, trial_raw.detach().numpy()),
        })

    relative_linear = float(
        np.linalg.norm(linear_residual) / max(np.linalg.norm(forcing), 1e-30))
    quadratic_consistent = (
        relative_linear < 1e-6
        and nonlinear_rows[0]["corrected_over_uncorrected"] < 1e-2
        and max(row["carrier_overlap"] for row in nonlinear_rows) > 0.99
    )
    result = {
        "status": "V4_C27_JOINT_W_REST_FREDHOLM_GATE",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "action_delta": "joint W body-diagonal positive Dirichlet energy",
        "solve_scope": args.scope,
        "active_state_coordinates": int(indices.size),
        "clock_coordinates": 2,
        "baseline_residual_norm": float(np.linalg.norm(baseline)),
        "unit_c27_forcing_norm": float(np.linalg.norm(forcing)),
        "lsmr_stop_code": int(solution[1]),
        "lsmr_iterations": int(solution[2]),
        "lsmr_reported_residual_norm": float(solution[3]),
        "linear_range_relative_residual": relative_linear,
        "linear_obstruction_breakdown": residual_breakdown(linear_residual),
        "correction_norm": float(np.linalg.norm(correction[:-2])),
        "psi_clock_correction": float(correction[-2]),
        "chi_clock_correction": float(correction[-1]),
        "nonlinear_controls": nonlinear_rows,
        "verdict": (
            "C27_JOINT_W_SMOOTH_REST_CONTINUATION_FEASIBLE"
            if quadratic_consistent else
            f"C27_JOINT_W_REST_CONTINUATION_OBSTRUCTED_IN_{args.scope.upper()}"),
        "claim_boundary": (
            "Matrix-free first-order range and small-strength nonlinear "
            "control only. A feasibility pass is not a solved finite C27 rest "
            "branch, spectrum pass, transport pass, or action admission."),
    }
    output = HERE / f"c27_rest_fredholm_probe_{args.scope}.json"
    output.write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    print(json.dumps({
        "verdict": result["verdict"],
        "lsmr_iterations": result["lsmr_iterations"],
        "linear_range_relative_residual": relative_linear,
        "psi_clock_correction": result["psi_clock_correction"],
        "chi_clock_correction": result["chi_clock_correction"],
        "nonlinear_controls": nonlinear_rows,
        "output": str(output),
    }, indent=2))


if __name__ == "__main__":
    main()
