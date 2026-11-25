"""Solve the unrestricted relative-periodic rest orbit for an action trial.

This is a research diagnostic.  Every Psi/chi component on the periodic box
is free; only one Psi phase and the exterior chi phase are gauge-fixed.  It is
used when an admissible action deforms the eight-site carrier or generates a
self-consistent halo, so a failed rigid three-parameter ansatz is not mistaken
for a failed action.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.linalg import eig
from scipy.optimize import least_squares

import field_berry_probe as field


HERE = Path(__file__).resolve().parent


def coefficients(psi_gradient_kinetic: float,
                 psi_orientation_kinetic: float,
                 chi_orientation_kinetic: float):
    """Current best exact-rest action plus the requested deformation probe."""
    values = [0.0] * 38
    values[0:11] = (-0.011, 0.017, -0.0165, -1.8, 0.1, 0.015, 0.0,
                    0.007, 0.0, 3.1, -3.1)
    values[17] = psi_gradient_kinetic
    values[36] = psi_orientation_kinetic
    values[37] = chi_orientation_kinetic
    return tuple(values)


def solve(side: int, dt: float, eta: float, psi_orientation: float,
          chi_orientation: float, max_nfev: int,
          initial_parameters=None):
    coeffs = coefficients(eta, psi_orientation, chi_orientation)
    psi0, chi0 = field.prior.bond.local.common.base.carrier_fields(side, 1.0)
    mask = field.prior.bond.local.common.base.norm2(psi0) > 0
    carrier_sites = [tuple(int(value) for value in site)
                     for site in np.argwhere(mask)]
    anchor = carrier_sites[0]
    exterior = tuple(int(value) for value in np.argwhere(~mask)[0])

    base_psi, _, qp, qc, _ = field.prior.bond.local.common.base.rest_state(
        side, dt, 1.0, -0.8, 0.03, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0)
    guess = np.asarray((
        np.log(field.prior.bond.local.common.base.norm2(base_psi).max()),
        np.angle(qp), np.angle(qc)))
    rigid, layers, _ = field.solve_rest(side, dt, coeffs, guess)
    initial = np.concatenate((layers[1], rigid.x[1:]))
    if initial_parameters is not None:
        if initial_parameters.shape != initial.shape:
            raise ValueError("continuation state has the wrong dimension")
        initial = initial_parameters
    scale = np.sqrt(layers[1].size)
    psi_scale = np.sqrt(np.exp(rigid.x[0]))

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
            previous, current, following, side, dt, *coeffs) / scale
        psi_phase = np.vdot(psi0[anchor], psi[anchor]).imag / psi_scale
        chi_phase = chi[exterior][2].imag / field.prior.R
        return np.concatenate((dynamic, (psi_phase, chi_phase)))

    solution = least_squares(
        residual, initial, xtol=3e-11, ftol=3e-11, gtol=3e-11,
        x_scale="jac", max_nfev=max_nfev, verbose=2)
    current, previous, following, psi, chi = unpack(solution.x)
    raw = field.del_residual(
        previous, current, following, side, dt, *coeffs)
    rms = float(np.linalg.norm(raw) / np.sqrt(raw.size))

    psi_reference, chi_reference = field.prior.bond.local.common.base.carrier_fields(
        side, float(np.max(field.prior.bond.local.common.base.norm2(psi))))
    psi_total = float(np.sum(abs(psi)**2))
    psi_exterior = float(np.sum(abs(psi[~mask])**2))
    chi_vacuum = np.zeros_like(chi)
    chi_vacuum[..., 2] = field.prior.R
    result = {
        "status": "FULL_DEFORMED_REST_RESEARCH",
        "eta_psi_gradient": eta,
        "eta_psi_orientation": psi_orientation,
        "eta_chi_orientation": chi_orientation,
        "solver_success": bool(solution.success),
        "solver_status": int(solution.status),
        "solver_nfev": int(solution.nfev),
        "cost": float(solution.cost),
        "optimality": float(solution.optimality),
        "rest_del_rms_residual": rms,
        "theta_psi": float(solution.x[-2]),
        "theta_chi": float(solution.x[-1]),
        "psi_exterior_fraction": psi_exterior / psi_total,
        "psi_carrier_overlap_sq": float(
            abs(np.vdot(psi_reference[mask], psi[mask]))**2
            / (np.sum(abs(psi_reference[mask])**2)
               * np.sum(abs(psi[mask])**2))),
        "chi_exterior_deviation_rms": float(np.linalg.norm(
            (chi - chi_vacuum)[~mask]) / np.sqrt(3 * np.count_nonzero(~mask))),
    }
    if rms <= 1e-7:
        orbit_layers = (previous, current, following)
        matrices = [field.jacobian(index, orbit_layers, side, dt, coeffs)
                    for index in range(3)]
        rotation = field.prior.bond.local.common.raw_rotation_matrix(
            side, solution.x[-2], solution.x[-1], 1)
        lhs = matrices[2] @ rotation
        forward = -np.linalg.solve(lhs, matrices[1])
        backward = -np.linalg.solve(lhs, matrices[0] @ rotation.T)
        identity = np.eye(forward.shape[0])
        monodromy = np.block([
            [forward, backward],
            [identity, np.zeros_like(identity)]])
        eigenvalues = eig(monodromy, right=False, check_finite=False)
        radius = float(np.max(abs(eigenvalues)))
        result.update({
            "spectral_radius": radius,
            "leading_growth_rate_per_unit_time": float(np.log(radius) / dt),
        })
    return result, solution.x


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eta", type=float, default=0.01)
    parser.add_argument("--psi-orientation", type=float, default=0.0)
    parser.add_argument("--chi-orientation", type=float, default=0.0)
    parser.add_argument("--side", type=int, default=3)
    parser.add_argument("--rate", type=int, default=256)
    parser.add_argument("--max-nfev", type=int, default=30)
    parser.add_argument("--initial-npz")
    parser.add_argument("--output", default="full_deformed_rest_result.json")
    args = parser.parse_args()
    initial = None
    if args.initial_npz:
        initial = np.load(args.initial_npz)["parameters"]
    result, parameters = solve(
        args.side, 1 / args.rate, args.eta, args.psi_orientation,
        args.chi_orientation, args.max_nfev, initial)
    (HERE / args.output).write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    np.savez_compressed(
        HERE / (Path(args.output).stem + ".npz"), parameters=parameters)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
