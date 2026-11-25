"""Continue the exact canonical eight-site branch through charge ratio.

The previous prepared orbit imposed equal Psi and chi Noether charges, which
selects one amplitude.  The field equations themselves admit a continuous
rigid eight-site relative-equilibrium branch.  This script derives both
frequencies from the native forces at each amplitude and applies the complete
continuous spectrum and constrained energy--momentum index.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.linalg import eig, null_space

import canonical_constrained_index as canonical_index
import constrained_energy_index as reduced
import localized_stability as canonical


HERE = Path(__file__).resolve().parent


def classify(side: int, amplitude_sq: float):
    psi, chi = canonical.carrier_fields(side, amplitude_sq)
    force_psi, force_chi = canonical.force(psi, chi)
    mask = canonical.norm2(psi) > 0
    dynamic_chi = np.zeros_like(chi)
    dynamic_chi[mask] = chi[mask]
    psi_frequency_sq = float(
        -np.vdot(psi, force_psi).real / np.vdot(psi, psi).real)
    chi_frequency_sq = float(
        -np.vdot(dynamic_chi, force_chi).real
        / np.vdot(dynamic_chi, dynamic_chi).real)
    if psi_frequency_sq <= 0 or chi_frequency_sq <= 0:
        return {
            "amplitude_sq": amplitude_sq,
            "status": "NO_REAL_RELATIVE_EQUILIBRIUM",
            "psi_frequency_sq": psi_frequency_sq,
            "chi_frequency_sq": chi_frequency_sq,
        }
    omega_psi, omega_chi = np.sqrt(psi_frequency_sq), np.sqrt(chi_frequency_sq)
    force_residual = max(
        np.linalg.norm(force_psi + psi_frequency_sq * psi)
        / max(np.linalg.norm(force_psi), 1e-30),
        np.linalg.norm(force_chi + chi_frequency_sq * dynamic_chi)
        / max(np.linalg.norm(force_chi), 1e-30))
    matrices = canonical.tangent_matrices(
        psi, chi, 0.0, 0.0, 1.0, 1.0)
    force_jacobian = matrices[1] + matrices[2]
    gamma = canonical_index.rotation_generator(side, omega_psi, omega_chi)
    size = force_jacobian.shape[0]
    mass = np.eye(size)
    gyro = 2 * gamma
    stiffness = gamma @ gamma - force_jacobian
    generator = np.block([
        [np.zeros_like(mass), np.eye(size)],
        [-stiffness, -gyro],
    ])
    omega_form = np.block([
        [gyro, mass],
        [-mass, np.zeros_like(mass)],
    ])
    hessian = np.block([
        [stiffness, np.zeros_like(mass)],
        [np.zeros_like(mass), mass],
    ])
    _, q_tangents = canonical_index.symmetry_directions(psi, chi)
    tangents = np.vstack((q_tangents, np.zeros_like(q_tangents)))
    tangents, _, tangent_rank = reduced.independent_columns(tangents)
    constraints, _, constraint_rank = reduced.independent_columns(
        np.column_stack((tangents, omega_form @ tangents)))
    basis = null_space(constraints.T, rcond=1e-10)
    reduced_hessian = basis.T @ hessian @ basis
    _, index = reduced.index_counts(reduced_hessian)
    values = eig(generator, right=False, check_finite=False)
    charge_psi = omega_psi * float(np.sum(abs(psi)**2))
    charge_chi = canonical.B * omega_chi * float(np.sum(abs(dynamic_chi)**2))
    return {
        "amplitude_sq": amplitude_sq,
        "status": "EXACT_CANONICAL_BRANCH",
        "psi_frequency_sq": psi_frequency_sq,
        "chi_frequency_sq": chi_frequency_sq,
        "omega_psi": float(omega_psi),
        "omega_chi": float(omega_chi),
        "native_force_relative_residual": float(force_residual),
        "charge_psi": charge_psi,
        "charge_chi": charge_chi,
        "charge_ratio_psi_over_chi": charge_psi / charge_chi,
        "maximum_growth": float(np.max(values.real)),
        "growth_count_above_1e_2": int(np.sum(values.real > 1e-2)),
        "growth_count_above_1e_4": int(np.sum(values.real > 1e-4)),
        "constrained_hessian_index": index,
        "group_orbit_rank": tangent_rank,
        "constraint_and_orbit_rank": constraint_rank,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--side", type=int, default=3)
    parser.add_argument("--amplitudes", nargs="+", type=float,
                        default=[168.08, 175, 200, 250, 300, 361, 450, 600, 900])
    parser.add_argument("--output", default="canonical_charge_branch_scan.json")
    args = parser.parse_args()
    canonical.B = 1197.0
    rows = [classify(args.side, value) for value in args.amplitudes]
    result = {
        "status": "V40_CANONICAL_CHARGE_BRANCH_SCAN",
        "canonical_action_changed": False,
        "rows": rows,
        "claim_boundary": (
            "Exact rigid eight-site charge branch only; a failed row is not "
            "a no-go for fully deformed or time-periodic carriers."),
    }
    (HERE / args.output).write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "rows": [{
            "amplitude_sq": row["amplitude_sq"],
            "status": row["status"],
            "charge_ratio": row.get("charge_ratio_psi_over_chi"),
            "maximum_growth": row.get("maximum_growth"),
            "negative_index": row.get("constrained_hessian_index", {}).get(
                "negative"),
        } for row in rows],
        "output": str(HERE / args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
