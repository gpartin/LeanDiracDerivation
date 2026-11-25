"""Test native direct-field C19 bonds against the saved moving-orbit defect.

This is a response diagnostic, not an admitted action.  It asks whether the
positive reciprocal potentials

  U_f = (1 / (2 W_f)) sum_(i,n in f) w_n |field_(i+n)-field_i|^2

for the face and edge C19 bond orbits point toward closure of the freely
optimized four-update handoff.  No target force is installed in an evolution.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.optimize import nnls

import edge_snap_action_rank as rank
import edge_snap_residual_action_projection as projection


HERE = Path(__file__).resolve().parent
DT = 0.1
INPUT = "edge_snap_transfer_charge_four_update_orbit"


def class_dirichlet(field: np.ndarray, taxicab: int) -> np.ndarray:
    return sum(
        weight * (field - rank.native.shift(field, offset))
        for offset, weight in rank.native.DIRECTIONS
        if sum(abs(value) for value in offset) == taxicab
    )


def potential_column(triples, field_index: int, taxicab: int) -> np.ndarray:
    """Discrete Euler--Lagrange column for a unit positive bond energy."""
    weight_sum = sum(
        weight for offset, weight in rank.native.DIRECTIONS
        if sum(abs(value) for value in offset) == taxicab
    )
    pieces = []
    for _, current, _ in triples:
        response = -(2 * DT / weight_sum) * class_dirichlet(
            current[field_index], taxicab)
        zeros = np.zeros_like(response)
        pieces.append(rank.pack(response if field_index == 0 else zeros,
                                response if field_index == 1 else zeros))
    return np.concatenate(pieces)


def metrics(column: np.ndarray, target: np.ndarray) -> dict:
    norm = np.linalg.norm(column)
    correlation = float(np.dot(column, target) / (norm * np.linalg.norm(target)))
    coefficient = float(np.dot(column, target) / (norm * norm))
    positive_coefficient = max(coefficient, 0.0)
    positive_residual = np.linalg.norm(
        positive_coefficient * column - target) / np.linalg.norm(target)
    return {
        "correlation_with_required_DEL_response": correlation,
        "unconstrained_optimal_coefficient": coefficient,
        "positive_optimal_coefficient": positive_coefficient,
        "positive_relative_residual": float(positive_residual),
        "positive_explained_squared_fraction": float(1 - positive_residual**2),
    }


def main() -> None:
    rank.lock_inertia()
    triples, data = projection.layers_from_artifact(INPUT)
    target = -data["residuals"].reshape(-1)
    names = ["psi_faces", "psi_edges", "chi_faces", "chi_edges"]
    columns = [
        potential_column(triples, 0, 1),
        potential_column(triples, 0, 2),
        potential_column(triples, 1, 1),
        potential_column(triples, 1, 2),
    ]
    matrix = np.column_stack(columns)
    norms = np.linalg.norm(matrix, axis=0)
    normalized = matrix / norms
    normalized_coefficients, residual_norm = nnls(
        normalized, target / np.linalg.norm(target))
    physical_coefficients = normalized_coefficients * np.linalg.norm(target) / norms

    equal_field = columns[0] + columns[1] + columns[2] + columns[3]
    equal_wave_speed = columns[0] + columns[1] + rank.B * (
        columns[2] + columns[3])
    result = {
        "status": "V4_DIRECT_FIELD_BOND_FREE_RESPONSE",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "input_orbit": INPUT,
        "chi_inertia": rank.B,
        "candidate": {
            "description": (
                "Separate positive reciprocal direct-field C19 Dirichlet "
                "energies for native complex Psi and complex chi, split into "
                "the cubic face and edge bond orbits."),
            "normalization": (
                "Each class is divided by its directed C19 weight sum; the "
                "Euler-Lagrange force is therefore -2 D_class/W_class."),
            "contains_corner_or_dirac_labels": False,
        },
        "single_columns": {
            name: metrics(column, target)
            for name, column in zip(names, columns)
        },
        "positive_four_column_fit": {
            "coefficients": {
                name: float(value)
                for name, value in zip(names, physical_coefficients)
                if value > 1e-12
            },
            "relative_residual": float(residual_norm),
            "explained_squared_fraction": float(1 - residual_norm**2),
            "rank": int(np.linalg.matrix_rank(normalized, tol=1e-11)),
        },
        "equal_coefficient_controls": {
            "all_fields": metrics(equal_field, target),
            "equal_native_wave_speed_chi_weighted_by_B": metrics(
                equal_wave_speed, target),
        },
        "verdict": "RESPONSE_DIAGNOSTIC_ONLY",
        "claim_boundary": (
            "A favorable response would only justify a freely re-solved action "
            "test. It would not establish an exact rest orbit, stability, "
            "translation, Dirac motion, or coefficient provenance."),
    }
    path = HERE / "direct_field_bond_free_response.json"
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
