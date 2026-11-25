"""Extract the continuous co-rotating tangent from the variational update.

The three discrete Euler--Lagrange Jacobians are expanded on one continuously
continued rigid branch.  This removes the logarithm of a nearly-unit Floquet
multiplier and separates a rest-branch switch from an action instability.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.linalg import eig

import field_berry_probe as field


HERE = Path(__file__).resolve().parent


def candidate_coefficients(parallel_potential: float = 0.0,
                           directional_area_potential: float = 0.0,
                           cross_covariance_bond_potential: float = 0.0
                           ) -> tuple[float, ...]:
    values = [0.0] * 47
    values[0] = -0.014
    values[1] = 0.0215
    values[2] = -0.0165
    values[3] = -2.0
    values[4] = 0.15
    values[5] = 0.015
    values[7] = 0.01
    values[9] = 3.1
    values[10] = -3.1
    values[20] = parallel_potential
    values[30] = 20.0
    values[32] = 20.0
    values[33] = 10.0
    values[40] = -0.0155
    values[41] = -0.00125
    values[44] = 1.0 / (12.0 * field.prior.R**2)
    values[45] = directional_area_potential
    values[46] = cross_covariance_bond_potential
    return tuple(values)


def classify(rate: int, jacobian_step: float, amplitude_sq: float,
             omega_psi: float, omega_chi: float,
             parallel_potential: float,
             directional_area_potential: float) -> dict:
    side = 3
    dt = 1.0 / rate
    coefficients = candidate_coefficients(
        parallel_potential, directional_area_potential)
    guess = np.asarray((np.log(amplitude_sq),
                        omega_psi * dt, omega_chi * dt))
    solution, layers, residual = field.solve_rest(
        side, dt, coefficients, guess)
    matrices = [field.jacobian(
        index, layers, side, dt, coefficients, step=jacobian_step)
        for index in range(3)]
    rotation = field.prior.bond.local.common.raw_rotation_matrix(
        side, solution.x[1], solution.x[2], 1)
    a_minus = matrices[0] @ rotation.T
    a_zero = matrices[1]
    a_plus = matrices[2] @ rotation

    # A+ x(t+dt) + A0 x(t) + A- x(t-dt) = 0.  Multiplication by
    # -1/dt gives the standard positive mass convention.
    mass = -0.5 * dt * (a_plus + a_minus)
    gyro = -(a_plus - a_minus)
    stiffness = -(a_plus + a_zero + a_minus) / dt
    mass_asymmetry = np.linalg.norm(mass - mass.T) / np.linalg.norm(mass)
    stiffness_asymmetry = (np.linalg.norm(stiffness - stiffness.T)
                           / np.linalg.norm(stiffness))
    gyro_symmetry = np.linalg.norm(gyro + gyro.T) / max(
        np.linalg.norm(gyro), 1e-30)
    mass_symmetric = (mass + mass.T) / 2
    stiffness_symmetric = (stiffness + stiffness.T) / 2
    mass_eigenvalues = np.linalg.eigvalsh(mass_symmetric)
    stiffness_eigenvalues = np.linalg.eigvalsh(stiffness_symmetric)

    dimension = mass.shape[0]
    acceleration_q = -np.linalg.solve(mass, stiffness)
    acceleration_v = -np.linalg.solve(mass, gyro)
    generator = np.block([
        [np.zeros_like(mass), np.eye(dimension)],
        [acceleration_q, acceleration_v],
    ])
    values, vectors = eig(generator, check_finite=False)
    order = np.argsort(values.real)[::-1]
    leading = values[order[0]]
    mode_raw = vectors[:dimension, order[0]]
    dpsi, dchi = field.prior.bond.local.common.unpack_raw(mode_raw, side)
    psi_power = np.sum(abs(dpsi)**2)
    chi_power = np.sum(abs(dchi)**2)
    total = float(psi_power + chi_power)
    chi_components = np.sum(abs(dchi)**2, axis=(0, 1, 2))
    fourier = (np.sum(abs(np.fft.fftn(dpsi, axes=(0, 1, 2)))**2,
                      axis=-1)
               + np.sum(abs(np.fft.fftn(dchi, axes=(0, 1, 2)))**2,
                        axis=-1))
    flat = int(np.argmax(fourier))
    index = tuple(int(x) for x in np.unravel_index(flat, fourier.shape))
    wave_numbers = 2 * np.pi * np.fft.fftfreq(side)
    wave = tuple(float(wave_numbers[x]) for x in index)
    symbol = 0.0
    for offset, weight in field.prior.bond.local.common.base.DIRECTIONS:
        symbol += weight * (1 - np.cos(np.dot(wave, offset)))
    return {
        "rate": rate,
        "dt": dt,
        "linearization_step": jacobian_step,
        "parallel_potential": parallel_potential,
        "directional_area_potential": directional_area_potential,
        "amplitude_sq": float(np.exp(solution.x[0])),
        "omega_psi": float(solution.x[1] / dt),
        "omega_chi": float(solution.x[2] / dt),
        "rest_del_rms_residual": residual,
        "continuous_growth_rate": float(leading.real),
        "continuous_leading_frequency": float(leading.imag),
        "largest_real_parts": [float(values[i].real) for i in order[:16]],
        "mass_condition_number": float(np.linalg.cond(mass)),
        "mass_minimum_eigenvalue": float(mass_eigenvalues[0]),
        "mass_maximum_eigenvalue": float(mass_eigenvalues[-1]),
        "mass_negative_count": int(np.sum(mass_eigenvalues < -1e-8)),
        "stiffness_minimum_eigenvalue": float(stiffness_eigenvalues[0]),
        "stiffness_maximum_eigenvalue": float(stiffness_eigenvalues[-1]),
        "stiffness_negative_count": int(np.sum(
            stiffness_eigenvalues < -1e-8)),
        "mass_relative_asymmetry": float(mass_asymmetry),
        "stiffness_relative_asymmetry": float(stiffness_asymmetry),
        "gyro_relative_symmetric_part": float(gyro_symmetry),
        "leading_mode_psi_fraction": float(psi_power / total),
        "leading_mode_chi_component_fractions": [
            float(x / total) for x in chi_components],
        "leading_fourier_index": index,
        "leading_wave_numbers": wave,
        "leading_c19_symbol": float(symbol),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rate", type=int, default=1024)
    parser.add_argument("--jacobian-step", type=float, default=5e-7)
    parser.add_argument("--amplitude-sq", type=float,
                        default=178.1135296575095)
    parser.add_argument("--omega-psi", type=float,
                        default=21.128280812869438)
    parser.add_argument("--omega-chi", type=float,
                        default=0.09336897508566924)
    parser.add_argument("--parallel-potential", type=float, default=0.0)
    parser.add_argument("--directional-area-potential", type=float,
                        default=0.0)
    parser.add_argument("--output", default="continuous_tangent.json")
    args = parser.parse_args()
    field.set_chi_inertia(229.0)
    result = {
        "status": "V4_CONTINUOUS_TANGENT_RESEARCH",
        "canonical_action_changed": False,
        "row": classify(args.rate, args.jacobian_step,
                        args.amplitude_sq, args.omega_psi, args.omega_chi,
                        args.parallel_potential,
                        args.directional_area_potential),
    }
    (HERE / args.output).write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
