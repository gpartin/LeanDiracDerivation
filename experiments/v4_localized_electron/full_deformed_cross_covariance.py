"""Unrestricted rest solve for the positive cross-covariance bond action."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.linalg import eig
from scipy.optimize import least_squares

import continuous_tangent as tangent
import field_berry_probe as field


HERE = Path(__file__).resolve().parent


def solve(side: int, rate: int, strength: float, max_nfev: int):
    dt = 1.0 / rate
    coefficients = tangent.candidate_coefficients(
        cross_covariance_bond_potential=strength)
    baseline = tangent.candidate_coefficients()
    guess = np.asarray((np.log(178.1135296575095),
                        21.128280812869438 * dt,
                        0.09336897508566924 * dt))
    base_solution, base_layers, base_residual = field.solve_rest(
        side, dt, baseline, guess)
    trial_solution, trial_layers, trial_rigid_residual = field.solve_rest(
        side, dt, coefficients, base_solution.x)
    initial = np.concatenate((trial_layers[1], trial_solution.x[1:]))
    scale = np.sqrt(trial_layers[1].size)
    psi_reference, chi_reference = (
        field.prior.bond.local.common.base.carrier_fields(side, 1.0))
    carrier_mask = field.prior.bond.local.common.base.norm2(psi_reference) > 0
    anchor = tuple(int(x) for x in np.argwhere(carrier_mask)[0])
    exterior = tuple(int(x) for x in np.argwhere(~carrier_mask)[0])

    def unpack(parameters):
        current = parameters[:-2]
        theta_psi, theta_chi = parameters[-2:]
        psi, chi = field.prior.bond.local.common.unpack_raw(current, side)
        previous = field.prior.bond.local.common.pack_raw(
            *field.prior.bond.local.common.rotate(
                psi, chi, theta_psi, theta_chi, -1))
        following = field.prior.bond.local.common.pack_raw(
            *field.prior.bond.local.common.rotate(
                psi, chi, theta_psi, theta_chi, 1))
        return current, previous, following, psi, chi

    def residual(parameters):
        current, previous, following, psi, chi = unpack(parameters)
        dynamic = field.del_residual(
            previous, current, following, side, dt, *coefficients) / scale
        psi_phase = np.vdot(psi_reference[anchor], psi[anchor]).imag
        chi_phase = chi[exterior][2].imag / field.prior.R
        return np.concatenate((dynamic, (psi_phase, chi_phase)))

    solution = least_squares(
        residual, initial, xtol=3e-11, ftol=3e-11, gtol=3e-11,
        x_scale="jac", diff_step=1e-5, max_nfev=max_nfev, verbose=2)
    current, previous, following, psi, chi = unpack(solution.x)
    raw_residual = field.del_residual(
        previous, current, following, side, dt, *coefficients)
    rms = float(np.linalg.norm(raw_residual) / np.sqrt(raw_residual.size))
    psi_scale = float(np.max(
        field.prior.bond.local.common.base.norm2(psi)))
    reference, _ = field.prior.bond.local.common.base.carrier_fields(
        side, psi_scale)
    total = float(np.sum(abs(psi)**2))
    result = {
        "status": "V4_FULL_DEFORMED_CROSS_COVARIANCE_RESEARCH",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "strength": strength,
        "rate": rate,
        "baseline_rigid_residual": base_residual,
        "trial_rigid_residual": trial_rigid_residual,
        "solver_success": bool(solution.success),
        "solver_nfev": int(solution.nfev),
        "rest_del_rms_residual": rms,
        "theta_psi": float(solution.x[-2]),
        "theta_chi": float(solution.x[-1]),
        "psi_carrier_overlap_sq": float(
            abs(np.vdot(reference[carrier_mask], psi[carrier_mask]))**2
            / (np.sum(abs(reference[carrier_mask])**2)
               * np.sum(abs(psi[carrier_mask])**2))),
        "psi_exterior_fraction": float(
            np.sum(abs(psi[~carrier_mask])**2) / total),
    }
    if rms <= 1e-7:
        layers = (previous, current, following)
        matrices = [field.jacobian(i, layers, side, dt, coefficients,
                                   step=5e-7)
                    for i in range(3)]
        rotation = field.prior.bond.local.common.raw_rotation_matrix(
            side, solution.x[-2], solution.x[-1], 1)
        a_minus = matrices[0] @ rotation.T
        a_zero = matrices[1]
        a_plus = matrices[2] @ rotation
        mass = -0.5 * dt * (a_plus + a_minus)
        gyro = -(a_plus - a_minus)
        stiffness = -(a_plus + a_zero + a_minus) / dt
        size = mass.shape[0]
        generator = np.block([
            [np.zeros_like(mass), np.eye(size)],
            [-np.linalg.solve(mass, stiffness),
             -np.linalg.solve(mass, gyro)],
        ])
        values = eig(generator, right=False, check_finite=False)
        result.update({
            "continuous_growth_rate": float(np.max(values.real)),
            "growth_count_above_1e_2": int(np.sum(values.real > 1e-2)),
            "growth_count_above_1e_4": int(np.sum(values.real > 1e-4)),
        })
    return result, solution.x


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--side", type=int, default=3)
    parser.add_argument("--rate", type=int, default=512)
    parser.add_argument("--strength", type=float, default=0.001)
    parser.add_argument("--max-nfev", type=int, default=20)
    parser.add_argument("--output", default="full_deformed_cross_covariance.json")
    args = parser.parse_args()
    field.set_chi_inertia(229.0)
    result, parameters = solve(
        args.side, args.rate, args.strength, args.max_nfev)
    (HERE / args.output).write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    np.savez_compressed(HERE / (Path(args.output).stem + ".npz"),
                        parameters=parameters)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
