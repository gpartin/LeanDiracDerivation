"""Feasibility scan of the minimal bounded local self-binding action family.

Replace only the coefficient of rho_Psi^2 in the canonical local energy:

  Ulocal = 1/4 (rhoChi-R^2)^2 + g/2 rhoChi rhoPsi
           + lambda/4 rhoPsi^2.

For g=1 the canonical action is lambda=1.  Eliminating chi locally gives a
strict self-binding window precisely when 0 < lambda < g^2, while the full
two-field energy remains bounded for lambda>0.  No prepared geometry occurs in
this action family.
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


def modified_force(psi, chi, lam, coupling):
    fp, fc = canonical.force(psi, chi)
    p = canonical.norm2(psi)[..., None]
    c = canonical.norm2(chi)[..., None]
    fp += -((lam - 1) * p + (coupling - 1) * c) * psi
    fc += -(coupling - 1) * p * chi / canonical.B
    return fp, fc


def modified_jacobian(psi, chi, lam, coupling):
    matrices = canonical.tangent_matrices(
        psi, chi, 0.0, 0.0, 1.0, 1.0)
    result = matrices[1] + matrices[2]
    side = psi.shape[0]
    size = result.shape[0]
    delta = np.empty_like(result)
    p = canonical.norm2(psi)[..., None]
    c = canonical.norm2(chi)[..., None]
    basis = np.zeros(size)
    for column in range(size):
        basis[column] = 1
        dp_field, dc_field = canonical.unpack(basis, side)
        dp = 2 * np.real(np.sum(psi.conj() * dp_field,
                                axis=-1))[..., None]
        dc = 2 * np.real(np.sum(chi.conj() * dc_field,
                                axis=-1))[..., None]
        dfp = -((lam - 1) * (dp * psi + p * dp_field)
                + (coupling - 1) * (dc * psi + c * dp_field))
        dfc = -(coupling - 1) * (dp * chi + p * dc_field) / canonical.B
        delta[:, column] = canonical.pack(dfp, dfc)
        basis[column] = 0
    return result + delta


def classify(side, lam, coupling, amplitude_sq):
    psi, chi = canonical.carrier_fields(side, amplitude_sq)
    fp, fc = modified_force(psi, chi, lam, coupling)
    mask = canonical.norm2(psi) > 0
    dynamic_chi = np.zeros_like(chi)
    dynamic_chi[mask] = chi[mask]
    wp2 = float(-np.vdot(psi, fp).real / np.vdot(psi, psi).real)
    wc2 = float(-np.vdot(dynamic_chi, fc).real
                / np.vdot(dynamic_chi, dynamic_chi).real)
    row = {"lambda": lam, "coupling": coupling,
           "amplitude_sq": amplitude_sq,
           "self_binding_margin": coupling**2 - lam,
           "psi_frequency_sq": wp2, "chi_frequency_sq": wc2}
    if wp2 <= 0 or wc2 <= 0:
        row["status"] = "NO_REAL_RELATIVE_EQUILIBRIUM"
        return row
    wp, wc = np.sqrt(wp2), np.sqrt(wc2)
    force_residual = max(
        np.linalg.norm(fp + wp2 * psi) / max(np.linalg.norm(fp), 1e-30),
        np.linalg.norm(fc + wc2 * dynamic_chi) / max(np.linalg.norm(fc), 1e-30))
    force_jacobian = modified_jacobian(psi, chi, lam, coupling)
    gamma = canonical_index.rotation_generator(side, wp, wc)
    size = force_jacobian.shape[0]
    mass = np.eye(size)
    gyro = 2 * gamma
    stiffness = gamma @ gamma - force_jacobian
    generator = np.block([
        [np.zeros_like(mass), np.eye(size)], [-stiffness, -gyro]])
    omega_form = np.block([
        [gyro, mass], [-mass, np.zeros_like(mass)]])
    hessian = np.block([
        [stiffness, np.zeros_like(mass)],
        [np.zeros_like(mass), mass]])
    _, qt = canonical_index.symmetry_directions(psi, chi)
    tangents = np.vstack((qt, np.zeros_like(qt)))
    tangents, _, _ = reduced.independent_columns(tangents)
    constraints, _, _ = reduced.independent_columns(
        np.column_stack((tangents, omega_form @ tangents)))
    rbasis = null_space(constraints.T, rcond=1e-10)
    _, index = reduced.index_counts(rbasis.T @ hessian @ rbasis)
    values = eig(generator, right=False, check_finite=False)
    row.update({
        "status": "EXACT_RESEARCH_BRANCH",
        "native_force_relative_residual": float(force_residual),
        "maximum_growth": float(np.max(values.real)),
        "growth_count_above_1e_2": int(np.sum(values.real > 1e-2)),
        "growth_count_above_1e_4": int(np.sum(values.real > 1e-4)),
        "constrained_negative_index": index["negative"],
        "constrained_zero_index": index["zero"],
    })
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--side", type=int, default=3)
    parser.add_argument("--lambdas", nargs="+", type=float,
                        default=[0.81, 0.5625, 0.36, 0.25])
    parser.add_argument("--output", default="local_binding_branch_scan.json")
    args = parser.parse_args()
    canonical.B = 1197.0
    rows = []
    for lam in args.lambdas:
        center = canonical.R**2 / np.sqrt(lam)
        for factor in (0.75, 1.0, 1.25):
            rows.append(classify(args.side, lam, 1.0, center * factor))
    result = {
        "status": "V4_LOCAL_BINDING_ACTION_FEASIBILITY",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "analytic_condition": "0 < lambda < coupling^2",
        "rows": rows,
        "claim_boundary": (
            "Rigid eight-site relative-equilibrium feasibility only; a "
            "survivor still requires coefficient provenance, unrestricted "
            "deformation, nonlinear evolution, and movement/Dirac gates."),
    }
    (HERE / args.output).write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "rows": [{
            "lambda": row["lambda"],
            "amplitude_sq": row["amplitude_sq"],
            "status": row["status"],
            "maximum_growth": row.get("maximum_growth"),
            "negative_index": row.get("constrained_negative_index"),
        } for row in rows],
        "output": str(HERE / args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
