"""Unrestricted fixed-Psi-charge branch with exactly static complex chi.

All ten native global field symmetries are quotiented by a fixed orthogonal
slice.  This removes the singular gauges that contaminate a direct nonlinear
solve near omega_chi=0.  No field site, halo, or exterior value is held.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.linalg import null_space
from scipy.optimize import least_squares

import canonical_constrained_index as canonical_index
import localized_stability as native
import unrestricted_binding_profile as binding


HERE = Path(__file__).resolve().parent


def harmonic_chi_seed(chi: np.ndarray, core: np.ndarray):
    """C19 harmonic extension of core chi, used only as an initial guess."""
    side = chi.shape[0]
    sites = side**3
    matrix = np.empty((sites, sites))
    basis = np.zeros((side, side, side, 1))
    for column in range(sites):
        basis.reshape(-1)[column] = 1.0
        matrix[:, column] = native.c19_dirichlet(basis)[..., 0].reshape(-1)
        basis.reshape(-1)[column] = 0.0
    core_flat = core.reshape(-1)
    exterior_flat = ~core_flat
    result = chi.reshape(sites, 3).copy()
    block = matrix[np.ix_(exterior_flat, exterior_flat)]
    coupling = matrix[np.ix_(exterior_flat, core_flat)]
    for component in range(3):
        result[exterior_flat, component] = np.linalg.solve(
            block, -coupling @ result[core_flat, component])
    # The local vacuum energy selects magnitude R but not orientation.
    exterior_norm = np.sqrt(np.sum(abs(result[exterior_flat])**2, axis=-1))
    valid = exterior_norm > 1e-12
    result[np.flatnonzero(exterior_flat)[valid]] *= (
        native.R / exterior_norm[valid])[:, None]
    return result.reshape(chi.shape)


def solve_stage(initial, side, lam, coupling, chi_stiffness, psi_stiffness,
                target_charge,
                reference_psi, reference_chi, symmetry, complement, max_nfev):
    size = 12 * side**3
    scale = np.sqrt(size)
    reference = native.pack(reference_psi, reference_chi)

    def unpack(parameters):
        psi, chi = native.unpack(parameters[:size], side)
        return psi, chi, float(parameters[-1])

    def full_stationary(parameters):
        psi, chi, omega = unpack(parameters)
        fp, fc = binding.action_force(
            psi, chi, lam, coupling, chi_stiffness, psi_stiffness)
        return native.pack(fp + omega**2 * psi, fc) / scale

    def residual(parameters):
        psi, _, omega = unpack(parameters)
        stationarity = complement.T @ full_stationary(parameters)
        gauges = symmetry.T @ (parameters[:size] - reference)
        charge = np.asarray(((omega * float(np.sum(abs(psi)**2))
                              - target_charge)
                             / max(abs(target_charge), 1.0),))
        return np.concatenate((stationarity, gauges, charge))

    solution = least_squares(
        residual, initial, jac="2-point", method="trf", x_scale="jac",
        ftol=2e-11, xtol=2e-11, gtol=2e-11, max_nfev=max_nfev,
        verbose=0)
    psi, chi, omega = unpack(solution.x)
    full = full_stationary(solution.x)
    return solution, psi, chi, omega, full, residual(solution.x)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--side", type=int, default=3)
    parser.add_argument("--lambda-self", type=float, default=0.25)
    parser.add_argument("--coupling", type=float, default=1.0)
    parser.add_argument("--stiffness-path", nargs="+", type=float,
                        default=(0.0, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4,
                                 3e-4, 1 / 1197))
    parser.add_argument("--psi-stiffness-path", nargs="+", type=float,
                        default=(0.0,))
    parser.add_argument("--max-nfev", type=int, default=120)
    parser.add_argument("--output", default="unrestricted_static_chi_profile.json")
    args = parser.parse_args()
    native.B = 1197.0
    amplitude_sq = native.R**2 / (2 + 54 / native.R**2)
    psi0, chi0, omega0, omega_chi = binding.rigid_seed(
        args.side, args.lambda_self, args.coupling, amplitude_sq)
    if abs(omega_chi) > 1e-6:
        raise RuntimeError("derived static-chi seed is not static")
    size = 12 * args.side**3
    _, tangent_matrix = canonical_index.symmetry_directions(psi0, chi0)
    tangent_u, singular_values, _ = np.linalg.svd(
        tangent_matrix, full_matrices=False)
    # Keep the complete independent native group orbit.
    tangent_rank = int(np.sum(singular_values > singular_values[0] * 1e-11))
    symmetry = tangent_u[:, :tangent_rank]
    complement = null_space(symmetry.T, rcond=1e-11)
    target_charge = omega0 * float(np.sum(abs(psi0)**2))
    initial = np.concatenate((native.pack(psi0, chi0), (omega0,)))
    rows = []
    last = None
    mask = native.norm2(psi0) > 0
    vacuum = np.zeros_like(chi0)
    vacuum[..., 2] = native.R
    chi_path = list(args.stiffness_path)
    psi_path = list(args.psi_stiffness_path)
    if len(chi_path) == 1:
        chi_path *= len(psi_path)
    if len(psi_path) == 1:
        psi_path *= len(chi_path)
    if len(chi_path) != len(psi_path):
        raise ValueError("chi and psi stiffness paths must match or broadcast")
    for stiffness, psi_stiffness in zip(chi_path, psi_path):
        if stiffness > 0 and rows and rows[-1]["chi_stiffness"] == 0:
            seed_psi, seed_chi = native.unpack(initial[:size], args.side)
            seed_chi = harmonic_chi_seed(seed_chi, mask)
            initial[:size] = native.pack(seed_psi, seed_chi)
        solution, psi, chi, omega, full, raw = solve_stage(
            initial, args.side, args.lambda_self, args.coupling, stiffness,
            psi_stiffness,
            target_charge, psi0, chi0, symmetry, complement, args.max_nfev)
        psi_total = max(float(np.sum(abs(psi)**2)), 1e-30)
        overlap = (abs(np.vdot(psi0[mask], psi[mask]))**2
                   / max(float(np.sum(abs(psi0[mask])**2)
                               * np.sum(abs(psi[mask])**2)), 1e-30))
        row = {
            "chi_stiffness": stiffness,
            "psi_stiffness": psi_stiffness,
            "solver_success": bool(solution.success),
            "solver_status": int(solution.status),
            "solver_nfev": int(solution.nfev),
            "cost": float(solution.cost),
            "optimality": float(solution.optimality),
            "stationary_rms": float(np.linalg.norm(full) / np.sqrt(size)),
            "largest_stationary_component": float(np.max(abs(full))),
            "reduced_residual_norm": float(np.linalg.norm(raw)),
            "omega_psi": omega,
            "psi_charge_relative_error": float(raw[-1]),
            "psi_carrier_overlap_sq": float(overlap),
            "psi_exterior_fraction": float(np.sum(abs(psi[~mask])**2)
                                           / psi_total),
            "chi_exterior_deviation_rms": float(np.linalg.norm(
                (chi - vacuum)[~mask])
                / np.sqrt(3 * np.count_nonzero(~mask))),
            "harmonic_halo_initialization": bool(stiffness > 0),
        }
        rows.append(row)
        if np.linalg.norm(raw) > 1e-7:
            break
        initial = solution.x
        last = (psi, chi, solution.x)
    result = {
        "status": "V4_UNRESTRICTED_STATIC_CHI_FIXED_CHARGE_BRANCH",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "lambda_self": args.lambda_self,
        "density_coupling": args.coupling,
        "derived_amplitude_sq": float(amplitude_sq),
        "target_psi_charge": float(target_charge),
        "symmetry_rank_removed": tangent_rank,
        "configuration_dimension": size,
        "rows": rows,
        "claim_boundary": (
            "Existence and carrier identity only; no stability, movement, or "
            "Dirac promotion follows without the full constrained spectrum."),
    }
    output = HERE / args.output
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n",
                      encoding="utf-8")
    if last is not None:
        np.savez_compressed(output.with_suffix(".npz"), psi=last[0], chi=last[1],
                            parameters=last[2])
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
