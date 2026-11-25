"""Can a fixed rotation of the two cubical copies remove their mixing?

For each spatial axis, the exact rest-preserving Bargmann connection maps the
same eight abstract component labels into the full native field space.  A
geometry-derived relabeling of the two copies must be one fixed rank-four
grading on those eight labels, valid for every axis.  Consequently it must
commute with every full-response Gram operator Y_a^T Y_b.

This script computes that common commutant from the unprojected native images.
It does not fit a Dirac endpoint and it does not change the action.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

import c19_bargmann_connection_copy_gate as gate
import c19_triangle_same_embedding_closure as embedding
import localized_carrier_contrast_inverse_launch as inverse


HERE = Path(__file__).resolve().parent


def nullspace(matrix: np.ndarray, relative: float = 1.0e-10) -> tuple[np.ndarray, np.ndarray]:
    _u, singular, vh = np.linalg.svd(matrix, full_matrices=True)
    tolerance = relative * max(float(singular[0]), 1.0)
    rank = int(np.sum(singular > tolerance))
    return vh[rank:].T, singular


def commutator(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return left @ right - right @ left


def vectorize(matrix: np.ndarray) -> np.ndarray:
    return matrix.reshape(-1, order="F")


def unvectorize(vector: np.ndarray) -> np.ndarray:
    return vector.reshape((8, 8), order="F")


def relative_commutator(operator: np.ndarray, grading: np.ndarray) -> float:
    return float(np.linalg.norm(commutator(operator, grading))
                 / max(np.linalg.norm(operator), 1.0e-300))


def main() -> None:
    inverse.rank.lock_inertia()
    inverse.rest.SIDE = inverse.SIDE
    inverse.rest.DT = inverse.DT
    previous, current, following, theta = inverse.rest_layers()
    layers = tuple(torch.tensor(value, dtype=torch.float64)
                   for value in (previous, current, following))

    targets = []
    images = []
    for axis in range(3):
        _labels, axis_targets = embedding.component_targets(
            axis, np.exp(1j * theta))
        targets.append(axis_targets)
        images.append(gate.images(*layers, axis_targets))

    operators = []
    operator_labels = []
    for first in range(3):
        for second in range(3):
            gram = images[first].T @ images[second]
            gram = gram / max(np.linalg.norm(gram), 1.0e-300)
            operators.append(gram)
            operator_labels.append(f"Y{first}^T_Y{second}")

    # vec(XG-GX) = (G^T kron I - I kron G) vec(X).
    identity8 = np.eye(8)
    equations = np.vstack([
        np.kron(operator.T, identity8) - np.kron(identity8, operator)
        for operator in operators
    ])
    common, singular = nullspace(equations)

    # A flavor grading is self-adjoint in the response metric.  Enforce real
    # symmetry inside the common commutant before looking for a 4+4 split.
    symmetry_equations = np.column_stack([
        vectorize(unvectorize(common[:, column])
                  - unvectorize(common[:, column]).T)
        for column in range(common.shape[1])
    ])
    symmetric_coefficients, symmetric_singular = nullspace(
        symmetry_equations, relative=1.0e-9)
    symmetric_basis = common @ symmetric_coefficients

    identity_vector = vectorize(identity8)
    if symmetric_basis.shape[1]:
        identity_projection = symmetric_basis @ (
            np.linalg.pinv(symmetric_basis) @ identity_vector)
        identity_in_symmetric_commutant = float(
            np.linalg.norm(identity_projection - identity_vector)
            / np.linalg.norm(identity_vector))
    else:
        identity_in_symmetric_commutant = 1.0

    # Remove the scalar identity direction. Any remaining symmetric direction
    # is a necessary precursor of a nontrivial rotated copy grading.
    traceless_columns = []
    for column in range(symmetric_basis.shape[1]):
        matrix = unvectorize(symmetric_basis[:, column])
        matrix = (matrix + matrix.T) / 2
        matrix -= np.trace(matrix) * identity8 / 8
        if np.linalg.norm(matrix) > 1.0e-9:
            traceless_columns.append(vectorize(matrix))
    traceless_rank = (0 if not traceless_columns else int(np.linalg.matrix_rank(
        np.column_stack(traceless_columns), tol=1.0e-8)))

    original_grading = np.diag([1.0] * 4 + [-1.0] * 4)
    original_defects = {
        label: relative_commutator(operator, original_grading)
        for label, operator in zip(operator_labels, operators)
    }
    joint_images = np.column_stack(images)
    joint_singular = np.linalg.svd(joint_images, compute_uv=False)
    joint_tolerance = 1.0e-10 * max(float(joint_singular[0]), 1.0)
    joint_rank = int(np.sum(joint_singular > joint_tolerance))

    result = {
        "status": "V4_C19_BARGMANN_ROTATED_COPY_AUDIT",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "axis_target_gram_defects_from_identity": [
            float(np.linalg.norm(target.T @ target
                                 / (np.trace(target.T @ target) / 8)
                                 - identity8))
            for target in targets
        ],
        "full_response_norms": [float(np.linalg.norm(value))
                                for value in images],
        "per_axis_response_ranks": [
            int(np.linalg.matrix_rank(value, tol=joint_tolerance))
            for value in images
        ],
        "joint_three_axis_response_rank": joint_rank,
        "joint_three_axis_singular_values": [
            float(value) for value in joint_singular
        ],
        "common_commutant_dimension": int(common.shape[1]),
        "commutant_equation_smallest_singular_values": [
            float(value) for value in singular[-8:]
        ],
        "symmetric_common_commutant_dimension": int(
            symmetric_basis.shape[1]),
        "symmetric_equation_singular_values": [
            float(value) for value in symmetric_singular
        ],
        "identity_relative_residual": identity_in_symmetric_commutant,
        "traceless_symmetric_commutant_rank": traceless_rank,
        "original_copy_grading_commutator_defects": original_defects,
        "verdict": (
            "NONTRIVIAL_ROTATED_COPY_GRADING_EXISTS"
            if traceless_rank > 0 else
            "NO_FIXED_ROTATION_SEPARATES_BARGMANN_COPIES"),
        "claim_boundary": (
            "Necessary simultaneous-response condition for a fixed real "
            "rotation of the two existing copies. Failure rules out relabeling "
            "inside the same eight abstract components for this Bargmann "
            "connection; it does not rule out a different native embedding "
            "with additional internal modes or a different action."),
    }
    output = HERE / "c19_bargmann_rotated_copy_audit.json"
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n",
                      encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
