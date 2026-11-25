"""Screen a native relative Psi--chi kinetic coupling.

The tested research action adds

  eta/(2 chi0^2) |chi x Psi_dot - chi_dot x Psi|^2

to the kinetic energy.  Because its kinetic metric depends on the fields, the
update is derived from a self-adjoint discrete action rather than inserted into
ordinary Verlet.  This script solves the exact rotating rest ansatz and then
linearizes the resulting implicit discrete Euler--Lagrange map.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.linalg import eigvals
from scipy.optimize import least_squares

import localized_stability as base


HERE = Path(__file__).resolve().parent
R = 19.0
B = 229.0
base.B = B


def pack_raw(psi: np.ndarray, chi: np.ndarray) -> np.ndarray:
    return np.stack((psi.real, psi.imag, chi.real, chi.imag), axis=-2).reshape(-1)


def unpack_raw(vector: np.ndarray, side: int) -> tuple[np.ndarray, np.ndarray]:
    data = vector.reshape(side, side, side, 4, 3)
    return (data[..., 0, :] + 1j * data[..., 1, :],
            data[..., 2, :] + 1j * data[..., 3, :])


def rotate(psi: np.ndarray, chi: np.ndarray, theta_psi: float,
           theta_chi: float, exponent: int) -> tuple[np.ndarray, np.ndarray]:
    pf = np.exp(1j * exponent * np.asarray(
        (theta_psi, theta_psi, theta_psi - theta_chi)))
    cf = np.exp(1j * exponent * np.asarray((theta_chi, theta_chi, 0.0)))
    return psi * pf, chi * cf


def potential_force(psi: np.ndarray, chi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Raw negative potential gradient for the best bounded diagnostic action."""
    accel_psi, accel_chi = base.force(
        psi, chi, bond_quartic=-0.8, bond_sextic=0.03)
    return accel_psi, B * accel_chi


def kinetic_derivatives(left_psi: np.ndarray, left_chi: np.ndarray,
                        right_psi: np.ndarray, right_chi: np.ndarray,
                        dt: float, eta: float):
    """Return p=dT/dqdot and dT/dq at the interval midpoint."""
    psi = (left_psi + right_psi) / 2
    chi = (left_chi + right_chi) / 2
    velocity_psi = (right_psi - left_psi) / dt
    velocity_chi = (right_chi - left_chi) / dt
    relative = (np.cross(chi, velocity_psi)
                - np.cross(velocity_chi, psi))
    scale = eta / R**2
    momentum_psi = velocity_psi + scale * np.cross(relative, chi.conj())
    momentum_chi = B * velocity_chi + scale * np.cross(relative, psi.conj())
    position_psi = scale * np.cross(velocity_chi.conj(), relative)
    position_chi = scale * np.cross(velocity_psi.conj(), relative)
    return momentum_psi, momentum_chi, position_psi, position_chi


def del_residual(previous: np.ndarray, current: np.ndarray, following: np.ndarray,
                 side: int, dt: float, eta: float) -> np.ndarray:
    """Discrete Euler--Lagrange residual of the reversible action step.

    The discrete Lagrangian uses midpoint kinetic energy and trapezoidal
    potential energy.  At eta=0 this reduces exactly to centered Verlet.
    """
    pm_psi, pm_chi = unpack_raw(previous, side)
    q_psi, q_chi = unpack_raw(current, side)
    pp_psi, pp_chi = unpack_raw(following, side)
    pminus_psi, pminus_chi, tqminus_psi, tqminus_chi = kinetic_derivatives(
        pm_psi, pm_chi, q_psi, q_chi, dt, eta)
    pplus_psi, pplus_chi, tqplus_psi, tqplus_chi = kinetic_derivatives(
        q_psi, q_chi, pp_psi, pp_chi, dt, eta)
    force_psi, force_chi = potential_force(q_psi, q_chi)
    return pack_raw(
        pminus_psi - pplus_psi
        + dt * (tqminus_psi + tqplus_psi) / 2 + dt * force_psi,
        pminus_chi - pplus_chi
        + dt * (tqminus_chi + tqplus_chi) / 2 + dt * force_chi,
    )


def ansatz_layers(side: int, amplitude_sq: float, theta_psi: float,
                  theta_chi: float):
    psi, chi = base.carrier_fields(side, amplitude_sq)
    previous = rotate(psi, chi, theta_psi, theta_chi, -1)
    following = rotate(psi, chi, theta_psi, theta_chi, 1)
    return pack_raw(*previous), pack_raw(psi, chi), pack_raw(*following)


def solve_rest(side: int, dt: float, eta: float, guess: np.ndarray):
    scale = np.sqrt(12 * side**3)

    def residual(parameters: np.ndarray) -> np.ndarray:
        amplitude_sq = np.exp(parameters[0])
        layers = ansatz_layers(
            side, amplitude_sq, parameters[1], parameters[2])
        return del_residual(*layers, side, dt, eta) / scale

    solution = least_squares(residual, guess, xtol=1e-13, ftol=1e-13,
                             gtol=1e-13, max_nfev=200)
    layers = ansatz_layers(side, np.exp(solution.x[0]),
                           solution.x[1], solution.x[2])
    raw = del_residual(*layers, side, dt, eta)
    return solution, layers, float(np.linalg.norm(raw) / np.sqrt(raw.size))


def raw_rotation_matrix(side: int, theta_psi: float, theta_chi: float,
                        exponent: int) -> np.ndarray:
    dimension = 12 * side**3
    result = np.empty((dimension, dimension))
    basis = np.zeros(dimension)
    for column in range(dimension):
        basis[column] = 1
        psi, chi = unpack_raw(basis, side)
        result[:, column] = pack_raw(*rotate(
            psi, chi, theta_psi, theta_chi, exponent))
        basis[column] = 0
    return result


def jacobian(argument: int, layers: tuple[np.ndarray, np.ndarray, np.ndarray],
             side: int, dt: float, eta: float, step: float = 2e-6) -> np.ndarray:
    dimension = layers[0].size
    result = np.empty((dimension, dimension))
    for column in range(dimension):
        plus = [value.copy() for value in layers]
        minus = [value.copy() for value in layers]
        plus[argument][column] += step
        minus[argument][column] -= step
        result[:, column] = (
            del_residual(*plus, side, dt, eta)
            - del_residual(*minus, side, dt, eta)) / (2 * step)
    return result


def classify(side: int, dt: float, eta: float, guess: np.ndarray,
             compute_spectrum: bool):
    solution, layers, residual = solve_rest(side, dt, eta, guess)
    row = {
        "eta": eta,
        "amplitude_sq": float(np.exp(solution.x[0])),
        "theta_psi": float(solution.x[1]),
        "theta_chi": float(solution.x[2]),
        "rest_del_rms_residual": residual,
        "rest_solver_success": bool(solution.success),
        "rest_solver_nfev": int(solution.nfev),
    }
    if not compute_spectrum or residual > 1e-7:
        return row, solution.x
    previous_jacobian = jacobian(0, layers, side, dt, eta)
    current_jacobian = jacobian(1, layers, side, dt, eta)
    following_jacobian = jacobian(2, layers, side, dt, eta)
    rotation = raw_rotation_matrix(
        side, solution.x[1], solution.x[2], 1)
    inverse_rotation = rotation.T
    lhs = following_jacobian @ rotation
    forward = -np.linalg.solve(lhs, current_jacobian)
    backward = -np.linalg.solve(lhs, previous_jacobian @ inverse_rotation)
    identity = np.eye(forward.shape[0])
    zero = np.zeros_like(identity)
    monodromy = np.block([[forward, backward], [identity, zero]])
    values = eigvals(monodromy, check_finite=False)
    radius = float(np.max(abs(values)))
    row.update({
        "spectral_radius": radius,
        "leading_growth_rate_per_unit_time": float(np.log(radius) / dt),
        "reciprocal_pairing_defect": float(np.max(
            np.abs(np.sort(abs(values)) * np.sort(abs(values))[::-1] - 1))),
        "linearization_step": 2e-6,
    })
    return row, solution.x


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--etas", nargs="+", type=float,
                        default=(0.0, 0.03, 0.1, 0.3, 1.0, 3.0))
    parser.add_argument("--side", type=int, default=3)
    parser.add_argument("--rate", type=int, default=256)
    parser.add_argument("--spectrum", action="store_true")
    parser.add_argument("--output", default="relative_kinetic_response.json")
    args = parser.parse_args()
    base.B = B
    dt = 1 / args.rate
    psi, chi, qp, qc, _ = base.rest_state(
        args.side, dt, 1.0, -0.8, 0.03, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0)
    guess = np.asarray((np.log(base.norm2(psi).max()),
                        np.angle(qp), np.angle(qc)))
    rows = []
    for eta in args.etas:
        row, guess = classify(args.side, dt, eta, guess, args.spectrum)
        rows.append(row)
    result = {
        "status": "RELATIVE_KINETIC_ACTION_RESEARCH",
        "canonical_action_changed": False,
        "research_action": "eta |chi x Psi_dot - chi_dot x Psi|^2 / (2 chi0^2)",
        "integrator": "implicit self-adjoint midpoint-kinetic trapezoidal-potential discrete Euler-Lagrange update",
        "rows": rows,
    }
    (HERE / args.output).write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
