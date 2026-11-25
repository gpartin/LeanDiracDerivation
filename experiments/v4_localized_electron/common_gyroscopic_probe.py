"""Add the independent common-orientation Berry term Im[W^dag W_dot]."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.linalg import eig
from scipy.optimize import least_squares

import bond_gyroscopic_probe as bond


HERE = Path(__file__).resolve().parent
R = bond.R


def common_derivatives(left_psi, left_chi, right_psi, right_chi, dt, gamma):
    psi = (left_psi + right_psi) / 2
    chi = (left_chi + right_chi) / 2
    velocity_psi = (right_psi - left_psi) / dt
    velocity_chi = (right_chi - left_chi) / dt
    area = np.cross(chi, psi)
    area_velocity = (np.cross(chi, velocity_psi)
                     + np.cross(velocity_chi, psi))
    scale = gamma / R**2
    return (
        velocity_psi + scale * 1j * np.cross(area, chi.conj()),
        bond.B * velocity_chi + scale * 1j * np.cross(psi.conj(), area),
        -scale * 1j * (np.cross(area_velocity, chi.conj())
                       + np.cross(velocity_chi.conj(), area)),
        scale * 1j * (np.cross(velocity_psi.conj(), area)
                      - np.cross(psi.conj(), area_velocity)),
    )


def all_derivatives(left_psi, left_chi, right_psi, right_chi, dt,
                    relative_local, relative_bond, common_gamma):
    # The common derivative contains the ordinary momenta.  Add only the
    # gyroscopic increments from the relative local/bond channels.
    common = common_derivatives(
        left_psi, left_chi, right_psi, right_chi, dt, common_gamma)
    relative = bond.combined_derivatives(
        left_psi, left_chi, right_psi, right_chi, dt,
        relative_local, relative_bond)
    velocity_psi = (right_psi - left_psi) / dt
    velocity_chi = (right_chi - left_chi) / dt
    ordinary = (velocity_psi, bond.B * velocity_chi,
                np.zeros_like(velocity_psi), np.zeros_like(velocity_chi))
    return tuple(a + b - c for a, b, c in zip(common, relative, ordinary))


def del_residual(previous, current, following, side, dt,
                 relative_local, relative_bond, common_gamma):
    pm_psi, pm_chi = bond.local.common.unpack_raw(previous, side)
    q_psi, q_chi = bond.local.common.unpack_raw(current, side)
    pp_psi, pp_chi = bond.local.common.unpack_raw(following, side)
    minus = all_derivatives(pm_psi, pm_chi, q_psi, q_chi, dt,
                            relative_local, relative_bond, common_gamma)
    plus = all_derivatives(q_psi, q_chi, pp_psi, pp_chi, dt,
                           relative_local, relative_bond, common_gamma)
    force_psi, force_chi = bond.local.common.potential_force(q_psi, q_chi)
    return bond.local.common.pack_raw(
        minus[0] - plus[0] + dt * (minus[2] + plus[2]) / 2 + dt * force_psi,
        minus[1] - plus[1] + dt * (minus[3] + plus[3]) / 2 + dt * force_chi)


def solve_rest(side, dt, relative_local, relative_bond, common_gamma, guess):
    scale = np.sqrt(12 * side**3)

    def residual(parameters):
        layers = bond.local.ansatz_layers(
            side, np.exp(parameters[0]), parameters[1], parameters[2])
        return del_residual(*layers, side, dt, relative_local,
                            relative_bond, common_gamma) / scale

    solution = least_squares(residual, guess, xtol=1e-13, ftol=1e-13,
                             gtol=1e-13, max_nfev=250)
    layers = bond.local.ansatz_layers(
        side, np.exp(solution.x[0]), solution.x[1], solution.x[2])
    raw = del_residual(*layers, side, dt, relative_local,
                       relative_bond, common_gamma)
    return solution, layers, float(np.linalg.norm(raw) / np.sqrt(raw.size))


def jacobian(argument, layers, side, dt, relative_local, relative_bond,
             common_gamma, step=2e-6):
    dimension = layers[0].size
    result = np.empty((dimension, dimension))
    for column in range(dimension):
        plus = [value.copy() for value in layers]
        minus = [value.copy() for value in layers]
        plus[argument][column] += step
        minus[argument][column] -= step
        result[:, column] = (
            del_residual(*plus, side, dt, relative_local, relative_bond,
                         common_gamma)
            - del_residual(*minus, side, dt, relative_local, relative_bond,
                           common_gamma)) / (2 * step)
    return result


def classify(side, dt, relative_local, relative_bond, common_gamma, guess):
    solution, layers, residual = solve_rest(
        side, dt, relative_local, relative_bond, common_gamma, guess)
    row = {
        "relative_local_gamma": relative_local,
        "relative_bond_gamma": relative_bond,
        "common_gamma": common_gamma,
        "amplitude_sq": float(np.exp(solution.x[0])),
        "theta_psi": float(solution.x[1]),
        "theta_chi": float(solution.x[2]),
        "rest_del_rms_residual": residual,
        "rest_solver_success": bool(solution.success),
    }
    if residual > 1e-7:
        return row
    matrices = [jacobian(index, layers, side, dt, relative_local,
                         relative_bond, common_gamma) for index in range(3)]
    rotation = bond.local.common.raw_rotation_matrix(
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
    dpsi, dchi = bond.local.common.unpack_raw(leading_raw, side)
    leading = bond.local.common.base.pack(dpsi, dchi)
    leading /= np.linalg.norm(leading)
    current_psi, current_chi = bond.local.common.unpack_raw(layers[1], side)
    _, corner, radial, phase = bond.local.common.base.internal_mode_bases(
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
    parser.add_argument("--relative-local", type=float, default=-0.00425)
    parser.add_argument("--relative-bond", type=float, default=0.007)
    parser.add_argument("--common-gammas", nargs="+", type=float,
                        default=(-0.01, -0.003, 0.0, 0.003, 0.01))
    parser.add_argument("--side", type=int, default=3)
    parser.add_argument("--rate", type=int, default=256)
    parser.add_argument("--output", default="common_gyroscopic_response.json")
    args = parser.parse_args()
    dt = 1 / args.rate
    psi, _, qp, qc, _ = bond.local.common.base.rest_state(
        args.side, dt, 1.0, -0.8, 0.03, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0)
    guess = np.asarray((np.log(bond.local.common.base.norm2(psi).max()),
                        np.angle(qp), np.angle(qc)))
    rows = [classify(args.side, dt, args.relative_local, args.relative_bond,
                     gamma, guess.copy()) for gamma in args.common_gammas]
    result = {
        "status": "COMMON_GYROSCOPIC_ACTION_RESEARCH",
        "canonical_action_changed": False,
        "relative_local_gamma": args.relative_local,
        "relative_bond_gamma": args.relative_bond,
        "rows": rows,
    }
    (HERE / args.output).write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
