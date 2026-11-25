"""Screen the lowest local antisymmetric Psi--chi phase coupling.

The research Lagrangian adds

  gamma/chi0^2 Im[(chi x Psi)^dagger
    (chi x Psi_dot - chi_dot x Psi)].

It is real, local, cubic, phase invariant, linear in velocity, and changes the
symplectic form without changing the Hamiltonian energy.  The implicit update
is obtained from the discrete Euler--Lagrange equation, not inserted as a
force.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.linalg import eig
from scipy.optimize import least_squares

import relative_kinetic_probe as common


HERE = Path(__file__).resolve().parent
R = common.R
B = common.B


def gyro_derivatives(left_psi: np.ndarray, left_chi: np.ndarray,
                     right_psi: np.ndarray, right_chi: np.ndarray,
                     dt: float, gamma: float):
    """Return p=dT/dqdot and dLgyro/dq at the interval midpoint."""
    psi = (left_psi + right_psi) / 2
    chi = (left_chi + right_chi) / 2
    velocity_psi = (right_psi - left_psi) / dt
    velocity_chi = (right_chi - left_chi) / dt
    area = np.cross(chi, psi)
    relative = (np.cross(chi, velocity_psi)
                - np.cross(velocity_chi, psi))
    scale = gamma / R**2
    momentum_psi = velocity_psi + scale * 1j * np.cross(area, chi.conj())
    momentum_chi = B * velocity_chi + scale * 1j * np.cross(area, psi.conj())
    position_psi = scale * 1j * (
        np.cross(velocity_chi.conj(), area)
        - np.cross(relative, chi.conj()))
    position_chi = scale * 1j * (
        np.cross(velocity_psi.conj(), area)
        - np.cross(psi.conj(), relative))
    return momentum_psi, momentum_chi, position_psi, position_chi


def del_residual(previous: np.ndarray, current: np.ndarray, following: np.ndarray,
                 side: int, dt: float, gamma: float) -> np.ndarray:
    pm_psi, pm_chi = common.unpack_raw(previous, side)
    q_psi, q_chi = common.unpack_raw(current, side)
    pp_psi, pp_chi = common.unpack_raw(following, side)
    pminus_psi, pminus_chi, tqminus_psi, tqminus_chi = gyro_derivatives(
        pm_psi, pm_chi, q_psi, q_chi, dt, gamma)
    pplus_psi, pplus_chi, tqplus_psi, tqplus_chi = gyro_derivatives(
        q_psi, q_chi, pp_psi, pp_chi, dt, gamma)
    force_psi, force_chi = common.potential_force(q_psi, q_chi)
    return common.pack_raw(
        pminus_psi - pplus_psi
        + dt * (tqminus_psi + tqplus_psi) / 2 + dt * force_psi,
        pminus_chi - pplus_chi
        + dt * (tqminus_chi + tqplus_chi) / 2 + dt * force_chi,
    )


def ansatz_layers(side: int, amplitude_sq: float, theta_psi: float,
                  theta_chi: float):
    psi, chi = common.base.carrier_fields(side, amplitude_sq)
    previous = common.rotate(psi, chi, theta_psi, theta_chi, -1)
    following = common.rotate(psi, chi, theta_psi, theta_chi, 1)
    return (common.pack_raw(*previous), common.pack_raw(psi, chi),
            common.pack_raw(*following))


def solve_rest(side: int, dt: float, gamma: float, guess: np.ndarray):
    scale = np.sqrt(12 * side**3)

    def residual(parameters: np.ndarray) -> np.ndarray:
        layers = ansatz_layers(side, np.exp(parameters[0]),
                               parameters[1], parameters[2])
        return del_residual(*layers, side, dt, gamma) / scale

    solution = least_squares(residual, guess, xtol=1e-13, ftol=1e-13,
                             gtol=1e-13, max_nfev=250)
    layers = ansatz_layers(side, np.exp(solution.x[0]),
                           solution.x[1], solution.x[2])
    raw = del_residual(*layers, side, dt, gamma)
    return solution, layers, float(np.linalg.norm(raw) / np.sqrt(raw.size))


def jacobian(argument: int, layers: tuple[np.ndarray, np.ndarray, np.ndarray],
             side: int, dt: float, gamma: float,
             step: float = 2e-6) -> np.ndarray:
    dimension = layers[0].size
    result = np.empty((dimension, dimension))
    for column in range(dimension):
        plus = [value.copy() for value in layers]
        minus = [value.copy() for value in layers]
        plus[argument][column] += step
        minus[argument][column] -= step
        result[:, column] = (
            del_residual(*plus, side, dt, gamma)
            - del_residual(*minus, side, dt, gamma)) / (2 * step)
    return result


def classify(side: int, dt: float, gamma: float, guess: np.ndarray,
             compute_spectrum: bool):
    solution, layers, residual = solve_rest(side, dt, gamma, guess)
    row = {
        "gamma": gamma,
        "amplitude_sq": float(np.exp(solution.x[0])),
        "theta_psi": float(solution.x[1]),
        "theta_chi": float(solution.x[2]),
        "rest_del_rms_residual": residual,
        "rest_solver_success": bool(solution.success),
        "rest_solver_nfev": int(solution.nfev),
    }
    if not compute_spectrum or residual > 1e-7:
        return row, solution.x
    previous_jacobian = jacobian(0, layers, side, dt, gamma)
    current_jacobian = jacobian(1, layers, side, dt, gamma)
    following_jacobian = jacobian(2, layers, side, dt, gamma)
    rotation = common.raw_rotation_matrix(
        side, solution.x[1], solution.x[2], 1)
    inverse_rotation = rotation.T
    lhs = following_jacobian @ rotation
    forward = -np.linalg.solve(lhs, current_jacobian)
    backward = -np.linalg.solve(lhs, previous_jacobian @ inverse_rotation)
    identity = np.eye(forward.shape[0])
    zero = np.zeros_like(identity)
    monodromy = np.block([[forward, backward], [identity, zero]])
    values, vectors = eig(monodromy, check_finite=False)
    absolute_values = abs(values)
    order = np.argsort(absolute_values)[::-1]
    radius = float(absolute_values[order[0]])
    moduli = np.sort(absolute_values)
    leading_raw = vectors[:forward.shape[0], order[0]]
    dpsi, dchi = common.unpack_raw(leading_raw, side)
    leading = common.base.pack(dpsi, dchi)
    leading /= np.linalg.norm(leading)
    current_psi, current_chi = common.unpack_raw(layers[1], side)
    translations = common.base.translation_vectors(current_psi, current_chi)
    global_internal, corner_internal, corner_radial, corner_phase = (
        common.base.internal_mode_bases(current_psi, current_chi))
    shape = (side, side, side, 4, 3)
    mode = leading.reshape(shape)
    psi_weight = np.sum(abs(mode[..., :2, :])**2, axis=(-2, -1))
    chi_weight = np.sum(abs(mode[..., 2:, :])**2, axis=(-2, -1))
    psi_components = np.sum(abs(mode[..., :2, :])**2, axis=(0, 1, 2, 3))
    chi_components = np.sum(abs(mode[..., 2:, :])**2, axis=(0, 1, 2, 3))
    core = common.base.norm2(current_psi) > 1e-20
    total_weight = float(np.sum(psi_weight + chi_weight))
    row.update({
        "spectral_radius": radius,
        "leading_growth_rate_per_unit_time": float(np.log(radius) / dt),
        "reciprocal_pairing_defect": float(np.max(
            np.abs(moduli * moduli[::-1] - 1))),
        "linearization_step": 2e-6,
        "largest_growth_rates": [
            float(np.log(absolute_values[index]) / dt) for index in order[:12]],
        "leading_mode_translation_overlap_sq": float(np.sum(
            abs(translations.conj().T @ leading)**2).real),
        "leading_mode_global_radial_phase_overlap_sq": float(np.sum(
            abs(global_internal.conj().T @ leading)**2).real),
        "leading_mode_corner_radial_phase_overlap_sq": float(np.sum(
            abs(corner_internal.conj().T @ leading)**2).real),
        "leading_mode_corner_radial_overlap_sq": float(np.sum(
            abs(corner_radial.conj().T @ leading)**2).real),
        "leading_mode_corner_phase_overlap_sq": float(np.sum(
            abs(corner_phase.conj().T @ leading)**2).real),
        "leading_mode_psi_fraction": float(np.sum(psi_weight) / total_weight),
        "leading_mode_psi_component_fractions": [
            float(value / total_weight) for value in psi_components],
        "leading_mode_chi_component_fractions": [
            float(value / total_weight) for value in chi_components],
        "leading_mode_core_fraction": float(
            np.sum((psi_weight + chi_weight)[core]) / total_weight),
    })
    return row, solution.x


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gammas", nargs="+", type=float,
                        default=(0.0, -0.03, 0.03, -0.1, 0.1))
    parser.add_argument("--side", type=int, default=3)
    parser.add_argument("--rate", type=int, default=256)
    parser.add_argument("--spectrum", action="store_true")
    parser.add_argument("--output", default="gyroscopic_response.json")
    args = parser.parse_args()
    dt = 1 / args.rate
    psi, _, qp, qc, _ = common.base.rest_state(
        args.side, dt, 1.0, -0.8, 0.03, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0)
    initial = np.asarray((np.log(common.base.norm2(psi).max()),
                          np.angle(qp), np.angle(qc)))
    rows = []
    for gamma in args.gammas:
        row, _ = classify(args.side, dt, gamma, initial.copy(), args.spectrum)
        rows.append(row)
    result = {
        "status": "LOCAL_GYROSCOPIC_ACTION_RESEARCH",
        "canonical_action_changed": False,
        "research_action": "gamma Im[(chi x Psi)^dagger (chi x Psi_dot - chi_dot x Psi)] / chi0^2",
        "hamiltonian": "ordinary positive kinetic energy plus U; the velocity-linear term cancels in the Legendre transform",
        "rows": rows,
    }
    (HERE / args.output).write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
