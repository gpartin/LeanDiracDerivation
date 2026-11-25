"""Combine local and C19 bond-resolved gyroscopic phase couplings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.linalg import eig
from scipy.optimize import least_squares

import gyroscopic_probe as local


HERE = Path(__file__).resolve().parent
R = local.R
B = local.B
C19_WEIGHT_SUM = sum(weight for _, weight in local.common.base.DIRECTIONS)


def bond_gyro_derivatives(left_psi: np.ndarray, left_chi: np.ndarray,
                          right_psi: np.ndarray, right_chi: np.ndarray,
                          dt: float, gamma: float):
    """Derivatives of gamma Im[W^dag D19 R]/(chi0^2 sum w)."""
    psi = (left_psi + right_psi) / 2
    chi = (left_chi + right_chi) / 2
    velocity_psi = (right_psi - left_psi) / dt
    velocity_chi = (right_chi - left_chi) / dt
    area = np.cross(chi, psi)
    relative = (np.cross(chi, velocity_psi)
                - np.cross(velocity_chi, psi))
    darea = local.common.base.c19_dirichlet(area)
    drelative = local.common.base.c19_dirichlet(relative)
    scale = gamma / (R**2 * C19_WEIGHT_SUM)
    momentum_psi = scale * 1j * np.cross(darea, chi.conj())
    momentum_chi = scale * 1j * np.cross(darea, psi.conj())
    position_psi = scale * 1j * (
        np.cross(velocity_chi.conj(), darea)
        - np.cross(drelative, chi.conj()))
    position_chi = scale * 1j * (
        np.cross(velocity_psi.conj(), darea)
        - np.cross(psi.conj(), drelative))
    return momentum_psi, momentum_chi, position_psi, position_chi


def combined_derivatives(left_psi, left_chi, right_psi, right_chi, dt,
                         local_gamma, bond_gamma):
    lp, lc, lqp, lqc = local.gyro_derivatives(
        left_psi, left_chi, right_psi, right_chi, dt, local_gamma)
    bp, bc, bqp, bqc = bond_gyro_derivatives(
        left_psi, left_chi, right_psi, right_chi, dt, bond_gamma)
    # local.gyro_derivatives already contains the ordinary diagonal momenta.
    return lp + bp, lc + bc, lqp + bqp, lqc + bqc


def del_residual(previous, current, following, side, dt,
                 local_gamma, bond_gamma):
    pm_psi, pm_chi = local.common.unpack_raw(previous, side)
    q_psi, q_chi = local.common.unpack_raw(current, side)
    pp_psi, pp_chi = local.common.unpack_raw(following, side)
    minus = combined_derivatives(pm_psi, pm_chi, q_psi, q_chi, dt,
                                 local_gamma, bond_gamma)
    plus = combined_derivatives(q_psi, q_chi, pp_psi, pp_chi, dt,
                                local_gamma, bond_gamma)
    force_psi, force_chi = local.common.potential_force(q_psi, q_chi)
    return local.common.pack_raw(
        minus[0] - plus[0] + dt * (minus[2] + plus[2]) / 2 + dt * force_psi,
        minus[1] - plus[1] + dt * (minus[3] + plus[3]) / 2 + dt * force_chi)


def solve_rest(side, dt, local_gamma, bond_gamma, guess):
    scale = np.sqrt(12 * side**3)

    def residual(parameters):
        layers = local.ansatz_layers(side, np.exp(parameters[0]),
                                     parameters[1], parameters[2])
        return del_residual(*layers, side, dt, local_gamma, bond_gamma) / scale

    solution = least_squares(residual, guess, xtol=1e-13, ftol=1e-13,
                             gtol=1e-13, max_nfev=250)
    layers = local.ansatz_layers(side, np.exp(solution.x[0]),
                                 solution.x[1], solution.x[2])
    raw = del_residual(*layers, side, dt, local_gamma, bond_gamma)
    return solution, layers, float(np.linalg.norm(raw) / np.sqrt(raw.size))


def jacobian(argument, layers, side, dt, local_gamma, bond_gamma, step=2e-6):
    dimension = layers[0].size
    result = np.empty((dimension, dimension))
    for column in range(dimension):
        plus = [value.copy() for value in layers]
        minus = [value.copy() for value in layers]
        plus[argument][column] += step
        minus[argument][column] -= step
        result[:, column] = (
            del_residual(*plus, side, dt, local_gamma, bond_gamma)
            - del_residual(*minus, side, dt, local_gamma, bond_gamma)) / (2 * step)
    return result


def classify(side, dt, local_gamma, bond_gamma, guess):
    solution, layers, residual = solve_rest(
        side, dt, local_gamma, bond_gamma, guess)
    row = {
        "local_gamma": local_gamma,
        "bond_gamma": bond_gamma,
        "amplitude_sq": float(np.exp(solution.x[0])),
        "theta_psi": float(solution.x[1]),
        "theta_chi": float(solution.x[2]),
        "rest_del_rms_residual": residual,
        "rest_solver_success": bool(solution.success),
    }
    if residual > 1e-7:
        return row
    matrices = [jacobian(index, layers, side, dt, local_gamma, bond_gamma)
                for index in range(3)]
    rotation = local.common.raw_rotation_matrix(
        side, solution.x[1], solution.x[2], 1)
    lhs = matrices[2] @ rotation
    forward = -np.linalg.solve(lhs, matrices[1])
    backward = -np.linalg.solve(lhs, matrices[0] @ rotation.T)
    identity = np.eye(forward.shape[0])
    monodromy = np.block([[forward, backward],
                          [identity, np.zeros_like(identity)]])
    values, vectors = eig(monodromy, check_finite=False)
    absolute = abs(values)
    order = np.argsort(absolute)[::-1]
    radius = float(absolute[order[0]])
    leading_raw = vectors[:forward.shape[0], order[0]]
    dpsi, dchi = local.common.unpack_raw(leading_raw, side)
    leading = local.common.base.pack(dpsi, dchi)
    leading /= np.linalg.norm(leading)
    current_psi, current_chi = local.common.unpack_raw(layers[1], side)
    _, corner, radial, phase = local.common.base.internal_mode_bases(
        current_psi, current_chi)
    mode = leading.reshape(side, side, side, 4, 3)
    psi_weight = np.sum(abs(mode[..., :2, :])**2, axis=(-2, -1))
    chi_weight = np.sum(abs(mode[..., 2:, :])**2, axis=(-2, -1))
    chi_components = np.sum(abs(mode[..., 2:, :])**2, axis=(0, 1, 2, 3))
    total = float(np.sum(psi_weight + chi_weight))
    row.update({
        "spectral_radius": radius,
        "leading_growth_rate_per_unit_time": float(np.log(radius) / dt),
        "largest_growth_rates": [
            float(np.log(absolute[index]) / dt) for index in order[:12]],
        "leading_mode_corner_overlap_sq": float(np.sum(
            abs(corner.conj().T @ leading)**2).real),
        "leading_mode_radial_overlap_sq": float(np.sum(
            abs(radial.conj().T @ leading)**2).real),
        "leading_mode_phase_overlap_sq": float(np.sum(
            abs(phase.conj().T @ leading)**2).real),
        "leading_mode_psi_fraction": float(np.sum(psi_weight) / total),
        "leading_mode_chi_component_fractions": [
            float(value / total) for value in chi_components],
    })
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-gamma", type=float, default=-0.00425)
    parser.add_argument("--bond-gammas", nargs="+", type=float,
                        default=(-0.01, -0.003, 0.0, 0.003, 0.01))
    parser.add_argument("--side", type=int, default=3)
    parser.add_argument("--rate", type=int, default=256)
    parser.add_argument("--output", default="bond_gyroscopic_response.json")
    args = parser.parse_args()
    dt = 1 / args.rate
    psi, _, qp, qc, _ = local.common.base.rest_state(
        args.side, dt, 1.0, -0.8, 0.03, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0)
    guess = np.asarray((np.log(local.common.base.norm2(psi).max()),
                        np.angle(qp), np.angle(qc)))
    rows = [classify(args.side, dt, args.local_gamma, gamma, guess.copy())
            for gamma in args.bond_gammas]
    result = {
        "status": "C19_BOND_GYROSCOPIC_ACTION_RESEARCH",
        "canonical_action_changed": False,
        "c19_weight_sum": C19_WEIGHT_SUM,
        "local_gamma": args.local_gamma,
        "rows": rows,
    }
    (HERE / args.output).write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
