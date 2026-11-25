"""Add the elementary-plaquette gyroscopic channel to the local/bond action."""

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
PLAQUETTE_COUNT = len(bond.local.common.base.PLAQUETTES)


def plaquette_derivatives(left_psi, left_chi, right_psi, right_chi,
                          dt, gamma):
    psi = (left_psi + right_psi) / 2
    chi = (left_chi + right_chi) / 2
    velocity_psi = (right_psi - left_psi) / dt
    velocity_chi = (right_chi - left_chi) / dt
    area = np.cross(chi, psi)
    relative = (np.cross(chi, velocity_psi)
                - np.cross(velocity_chi, psi))
    darea = bond.local.common.base.plaquette_dirichlet(area)
    drelative = bond.local.common.base.plaquette_dirichlet(relative)
    scale = gamma / (R**2 * PLAQUETTE_COUNT)
    return (
        scale * 1j * np.cross(darea, chi.conj()),
        scale * 1j * np.cross(darea, psi.conj()),
        scale * 1j * (np.cross(velocity_chi.conj(), darea)
                      - np.cross(drelative, chi.conj())),
        scale * 1j * (np.cross(velocity_psi.conj(), darea)
                      - np.cross(psi.conj(), drelative)),
    )


def all_derivatives(left_psi, left_chi, right_psi, right_chi, dt,
                    local_gamma, bond_gamma, plaquette_gamma):
    prior = bond.combined_derivatives(
        left_psi, left_chi, right_psi, right_chi, dt,
        local_gamma, bond_gamma)
    extra = plaquette_derivatives(
        left_psi, left_chi, right_psi, right_chi, dt, plaquette_gamma)
    return tuple(a + b for a, b in zip(prior, extra))


def del_residual(previous, current, following, side, dt,
                 local_gamma, bond_gamma, plaquette_gamma):
    pm_psi, pm_chi = bond.local.common.unpack_raw(previous, side)
    q_psi, q_chi = bond.local.common.unpack_raw(current, side)
    pp_psi, pp_chi = bond.local.common.unpack_raw(following, side)
    minus = all_derivatives(pm_psi, pm_chi, q_psi, q_chi, dt,
                            local_gamma, bond_gamma, plaquette_gamma)
    plus = all_derivatives(q_psi, q_chi, pp_psi, pp_chi, dt,
                           local_gamma, bond_gamma, plaquette_gamma)
    force_psi, force_chi = bond.local.common.potential_force(q_psi, q_chi)
    return bond.local.common.pack_raw(
        minus[0] - plus[0] + dt * (minus[2] + plus[2]) / 2 + dt * force_psi,
        minus[1] - plus[1] + dt * (minus[3] + plus[3]) / 2 + dt * force_chi)


def solve_rest(side, dt, local_gamma, bond_gamma, plaquette_gamma, guess):
    scale = np.sqrt(12 * side**3)

    def residual(parameters):
        layers = bond.local.ansatz_layers(
            side, np.exp(parameters[0]), parameters[1], parameters[2])
        return del_residual(*layers, side, dt, local_gamma, bond_gamma,
                            plaquette_gamma) / scale

    solution = least_squares(residual, guess, xtol=1e-13, ftol=1e-13,
                             gtol=1e-13, max_nfev=250)
    layers = bond.local.ansatz_layers(
        side, np.exp(solution.x[0]), solution.x[1], solution.x[2])
    raw = del_residual(*layers, side, dt, local_gamma, bond_gamma,
                       plaquette_gamma)
    return solution, layers, float(np.linalg.norm(raw) / np.sqrt(raw.size))


def core_basis(side):
    psi, chi = bond.local.common.base.carrier_fields(side, 1.0)
    sites = [tuple(int(x) for x in site)
             for site in np.argwhere(bond.local.common.base.norm2(psi) > 0)]
    return psi, chi, sites


def symmetric_deformed_guess(amplitude_sq, theta_psi, theta_chi):
    amplitude = np.sqrt(amplitude_sq)
    values = [np.log(amplitude), np.log(amplitude)]
    for _ in range(1, 8):
        values.extend((amplitude, 0.0, amplitude, 0.0))
    values.extend((theta_psi, theta_chi))
    return np.asarray(values)


def deformed_fields(side, parameters):
    psi_basis, chi_basis, sites = core_basis(side)
    psi = np.zeros_like(psi_basis)
    chi = np.zeros_like(chi_basis)
    chi[..., 2] = R
    amplitudes_psi = [np.exp(parameters[0]) + 0j]
    amplitudes_chi = [np.exp(parameters[1]) + 0j]
    cursor = 2
    for _ in range(1, 8):
        amplitudes_psi.append(parameters[cursor] + 1j * parameters[cursor + 1])
        amplitudes_chi.append(parameters[cursor + 2] + 1j * parameters[cursor + 3])
        cursor += 4
    for site, ap, ac in zip(sites, amplitudes_psi, amplitudes_chi):
        psi[site] = ap * psi_basis[site]
        chi[site] = ac * chi_basis[site]
    return psi, chi, sites, np.asarray(amplitudes_psi), np.asarray(amplitudes_chi)


def deformed_layers(side, parameters):
    psi, chi, sites, ap, ac = deformed_fields(side, parameters)
    theta_psi, theta_chi = parameters[-2:]
    previous = bond.local.common.rotate(psi, chi, theta_psi, theta_chi, -1)
    following = bond.local.common.rotate(psi, chi, theta_psi, theta_chi, 1)
    return ((bond.local.common.pack_raw(*previous),
             bond.local.common.pack_raw(psi, chi),
             bond.local.common.pack_raw(*following)), sites, ap, ac)


def solve_deformed_rest(side, dt, local_gamma, bond_gamma, plaquette_gamma,
                        guess):
    psi_basis, chi_basis, sites = core_basis(side)

    def projected(parameters):
        layers, _, _, _ = deformed_layers(side, parameters)
        raw = del_residual(*layers, side, dt, local_gamma, bond_gamma,
                           plaquette_gamma)
        residual_psi, residual_chi = bond.local.common.unpack_raw(raw, side)
        coefficients = []
        for site in sites:
            coefficients.append(np.vdot(psi_basis[site], residual_psi[site]))
        for site in sites:
            coefficients.append(np.vdot(chi_basis[site], residual_chi[site]))
        return np.concatenate((np.real(coefficients), np.imag(coefficients)))

    solution = least_squares(projected, guess, xtol=1e-13, ftol=1e-13,
                             gtol=1e-13, max_nfev=1000)
    layers, sites, ap, ac = deformed_layers(side, solution.x)
    raw = del_residual(*layers, side, dt, local_gamma, bond_gamma,
                       plaquette_gamma)
    return (solution, layers, float(np.linalg.norm(raw) / np.sqrt(raw.size)),
            ap, ac)


def jacobian(argument, layers, side, dt, local_gamma, bond_gamma,
             plaquette_gamma, step=2e-6):
    dimension = layers[0].size
    result = np.empty((dimension, dimension))
    for column in range(dimension):
        plus = [value.copy() for value in layers]
        minus = [value.copy() for value in layers]
        plus[argument][column] += step
        minus[argument][column] -= step
        result[:, column] = (
            del_residual(*plus, side, dt, local_gamma, bond_gamma,
                         plaquette_gamma)
            - del_residual(*minus, side, dt, local_gamma, bond_gamma,
                           plaquette_gamma)) / (2 * step)
    return result


def classify(side, dt, local_gamma, bond_gamma, plaquette_gamma, guess):
    solution, layers, residual = solve_rest(
        side, dt, local_gamma, bond_gamma, plaquette_gamma, guess)
    row = {
        "local_gamma": local_gamma,
        "bond_gamma": bond_gamma,
        "plaquette_gamma": plaquette_gamma,
        "amplitude_sq": float(np.exp(solution.x[0])),
        "theta_psi": float(solution.x[1]),
        "theta_chi": float(solution.x[2]),
        "rest_del_rms_residual": residual,
        "rest_solver_success": bool(solution.success),
    }
    if residual > 1e-7:
        return row
    matrices = [jacobian(index, layers, side, dt, local_gamma, bond_gamma,
                         plaquette_gamma) for index in range(3)]
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


def classify_deformed(side, dt, local_gamma, bond_gamma, plaquette_gamma,
                      guess):
    solution, layers, residual, amplitudes_psi, amplitudes_chi = (
        solve_deformed_rest(side, dt, local_gamma, bond_gamma,
                            plaquette_gamma, guess))
    magnitudes_psi = abs(amplitudes_psi)
    magnitudes_chi = abs(amplitudes_chi)
    row = {
        "local_gamma": local_gamma,
        "bond_gamma": bond_gamma,
        "plaquette_gamma": plaquette_gamma,
        "theta_psi": float(solution.x[-2]),
        "theta_chi": float(solution.x[-1]),
        "rest_del_rms_residual": residual,
        "rest_solver_success": bool(solution.success),
        "psi_amplitude_relative_spread": float(
            (magnitudes_psi.max() - magnitudes_psi.min()) / magnitudes_psi.mean()),
        "chi_amplitude_relative_spread": float(
            (magnitudes_chi.max() - magnitudes_chi.min()) / magnitudes_chi.mean()),
        "psi_phase_spread": float(np.ptp(np.unwrap(np.angle(amplitudes_psi)))),
        "chi_phase_spread": float(np.ptp(np.unwrap(np.angle(amplitudes_chi)))),
    }
    if residual > 1e-7:
        return row
    matrices = [jacobian(index, layers, side, dt, local_gamma, bond_gamma,
                         plaquette_gamma) for index in range(3)]
    rotation = bond.local.common.raw_rotation_matrix(
        side, solution.x[-2], solution.x[-1], 1)
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
    parser.add_argument("--local-gamma", type=float, default=-0.00425)
    parser.add_argument("--bond-gamma", type=float, default=0.007)
    parser.add_argument("--plaquette-gammas", nargs="+", type=float,
                        default=(-0.01, -0.003, 0.0, 0.003, 0.01))
    parser.add_argument("--side", type=int, default=3)
    parser.add_argument("--rate", type=int, default=256)
    parser.add_argument("--deformed", action="store_true")
    parser.add_argument("--output", default="plaquette_gyroscopic_response.json")
    args = parser.parse_args()
    dt = 1 / args.rate
    psi, _, qp, qc, _ = bond.local.common.base.rest_state(
        args.side, dt, 1.0, -0.8, 0.03, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0)
    guess = np.asarray((np.log(bond.local.common.base.norm2(psi).max()),
                        np.angle(qp), np.angle(qc)))
    if args.deformed:
        deformed_guess = symmetric_deformed_guess(
            bond.local.common.base.norm2(psi).max(), np.angle(qp), np.angle(qc))
        rows = [classify_deformed(
            args.side, dt, args.local_gamma, args.bond_gamma, gamma,
            deformed_guess.copy()) for gamma in args.plaquette_gammas]
    else:
        rows = [classify(args.side, dt, args.local_gamma, args.bond_gamma,
                         gamma, guess.copy()) for gamma in args.plaquette_gammas]
    result = {
        "status": "PLAQUETTE_GYROSCOPIC_ACTION_RESEARCH",
        "canonical_action_changed": False,
        "local_gamma": args.local_gamma,
        "bond_gamma": args.bond_gamma,
        "plaquette_count": PLAQUETTE_COUNT,
        "deformed_eight_ray_rest_solver": args.deformed,
        "rows": rows,
    }
    (HERE / args.output).write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
