"""Rank-test the declared reciprocal action basis against translation pinning.

This is an inverse Hessian screen at the exact canonical localized rest orbit.
It does not fit a Dirac trajectory.  It asks whether the Hessians of generic
real cubic/C19 invariants already present in the V4 research vocabulary span
the native force required to turn the three lattice translations into neutral
directions of the co-rotating update.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.optimize import lsq_linear

import canonical_first_shell_phase_embedding as phase
import edge_snap_action_rank as rank
import spectral_psi_translation_repair as psi_spectral


HERE = Path(__file__).resolve().parent
DT = 1.0 / 1024.0


def response(part: np.ndarray, tangent: np.ndarray,
             inverse: np.ndarray) -> np.ndarray:
    return inverse @ (part @ tangent)


def main() -> None:
    rank.lock_inertia()
    phase.DT = DT
    amplitude_sq = rank.R**2 / (2 + 54 / rank.R**2)
    psi, chi = rank.native.carrier_fields(phase.SIDE, amplitude_sq)
    matrices = rank.native.tangent_matrices(
        psi, chi, 0.0, 0.0, 1.0, 1.0)
    names = [
        "local_potential", "C19_area_bond", "density_balance_square",
        "parallel_overlap_square", "onsite_area_quartic",
        "onsite_area_sextic", "C19_area_curvature_square",
        "plaquette_area_curvature_square", "spectral_area_square",
        "spectral_Psi_square",
    ]
    parts = list(matrices[1:]) + [
        psi_spectral.spectral_psi_force_matrix(phase.SIDE)]
    theta = float(np.arccos(1 - DT**2 * rank.R**2 / 2))
    rotation = rank.native.rotation_matrix(
        phase.SIDE, np.exp(1j * theta), 1.0 + 0.0j, 1)
    inverse = rotation.T
    translations = rank.native.translation_vectors(psi, chi)

    targets = []
    columns = [[] for _ in names]
    axis_rows = []
    for axis in range(3):
        tangent = translations[:, axis]
        kinematic = ((2 * inverse - inverse @ inverse - np.eye(
            inverse.shape[0])) @ tangent) / DT**2
        incumbent = (kinematic + response(matrices[0], tangent, inverse))
        targets.append(-incumbent)
        correlations = {}
        for index, (name, part) in enumerate(zip(names, parts)):
            value = response(part, tangent, inverse)
            columns[index].append(value)
            correlations[name] = float(np.dot(value, -incumbent) / max(
                np.linalg.norm(value) * np.linalg.norm(incumbent), 1e-300))
        axis_rows.append({
            "axis": axis,
            "incumbent_pinning_norm": float(np.linalg.norm(incumbent)),
            "basis_correlations_with_required_repair": correlations,
        })

    target = np.concatenate(targets)
    matrix = np.column_stack([np.concatenate(column) for column in columns])
    singular = np.linalg.svd(matrix, compute_uv=False)
    tolerance = max(matrix.shape) * singular[0] * np.finfo(float).eps
    unconstrained, *_ = np.linalg.lstsq(matrix, target, rcond=1e-12)
    unconstrained_relative = float(
        np.linalg.norm(matrix @ unconstrained - target) / np.linalg.norm(target))

    # Positive-square subspace.  Local and canonical area coefficients are
    # fixed; the onsite polynomial terms are excluded because neither is a
    # separately bounded positive square.  All retained columns are Hessians
    # of nonnegative energies at the declared sign.
    positive_indices = [2, 3, 6, 7, 8, 9]
    positive_matrix = matrix[:, positive_indices]
    positive_fit = lsq_linear(
        positive_matrix, target, bounds=(0.0, np.inf),
        tol=1e-13, lsmr_tol=1e-13, max_iter=5000)
    positive_relative = float(np.linalg.norm(
        positive_matrix @ positive_fit.x - target) / np.linalg.norm(target))

    result = {
        "status": "V4_CANONICAL_TRANSLATION_ACTION_BASIS_RANK",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "dt": DT,
        "basis": names,
        "axis_rows": axis_rows,
        "response_rank": int(np.sum(singular > tolerance)),
        "singular_values": [float(x) for x in singular],
        "unconstrained_coefficients": {
            name: float(value) for name, value in zip(names, unconstrained)},
        "unconstrained_relative_residual": unconstrained_relative,
        "positive_square_basis": [names[i] for i in positive_indices],
        "positive_square_coefficients": {
            names[i]: float(value)
            for i, value in zip(positive_indices, positive_fit.x)},
        "positive_square_relative_residual": positive_relative,
        "positive_fit_success": bool(positive_fit.success),
        "verdict": (
            "DECLARED_POSITIVE_ACTION_BASIS_CLOSES_TRANSLATION_ZERO_MODE"
            if positive_relative < 1e-8 else
            "DECLARED_POSITIVE_ACTION_BASIS_CANNOT_CLOSE_TRANSLATION_ZERO_MODE"),
        "claim_boundary": (
            "Linear inverse-Hessian screen at the exact rest orbit only. "
            "A small residual would still require exact action variation, "
            "rest re-solution, full-spectrum stability, nonlinear transport, "
            "Dirac compression, and Lean proof."),
    }
    output = HERE / "canonical_translation_action_basis_rank.json"
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n",
                      encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
