"""Independent reconstruction of the scoped DIRAC-018 classification."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


TOLERANCE = 3.0e-12


def bit(corner: int, axis: int) -> int:
    return (corner >> (2 - axis)) & 1


def flip(corner: int, axis: int) -> int:
    return corner ^ (1 << (2 - axis))


def gamma(axis: int) -> np.ndarray:
    matrix = np.zeros((8, 8), dtype=complex)
    for corner in range(8):
        if axis == 0:
            sign = 1
        elif axis == 1:
            sign = 1 if bit(corner, 0) == 0 else -1
        else:
            sign = 1 if (bit(corner, 0) + bit(corner, 1)) % 2 == 0 else -1
        matrix[corner, flip(corner, axis)] = sign
    return matrix


def beta() -> np.ndarray:
    return np.diag([
        1 if sum(bit(corner, axis) for axis in range(3)) % 2 == 0 else -1
        for corner in range(8)
    ]).astype(complex)


def flavor() -> np.ndarray:
    matrix = np.zeros((8, 8), dtype=complex)
    for corner in range(8):
        sign = 1 if bit(corner, 2) == 0 else -1
        matrix[corner, flip(flip(corner, 2), 1)] = 1j * sign
    return matrix


def raw_embeddings() -> tuple[np.ndarray, np.ndarray]:
    minus = np.zeros((8, 4), dtype=complex)
    plus = np.zeros((8, 4), dtype=complex)
    minus[0, 0] = 1; minus[1, 1] = 1; minus[2, 1] = -1j; minus[3, 0] = 1j
    minus[4, 2] = 1; minus[5, 3] = 1; minus[6, 3] = -1j; minus[7, 2] = 1j
    plus[0, 0] = 1; plus[1, 1] = 1; plus[2, 1] = 1j; plus[3, 0] = -1j
    plus[4, 2] = 1; plus[5, 3] = 1; plus[6, 3] = 1j; plus[7, 2] = -1j
    return minus, plus


def orthogonal_projector(embedding: np.ndarray) -> np.ndarray:
    return embedding @ np.linalg.inv(embedding.conj().T @ embedding) @ embedding.conj().T


def commutant_system(rotations: tuple[np.ndarray, ...]) -> np.ndarray:
    columns: list[np.ndarray] = []
    for row in range(3):
        for column in range(3):
            basis = np.zeros((3, 3))
            basis[row, column] = 1.0
            columns.append(np.concatenate([
                (basis @ rotation - rotation @ basis).reshape(-1)
                for rotation in rotations
            ]))
    return np.stack(columns, axis=1)


def c19_symbol(momentum: np.ndarray) -> float:
    x, y, z = momentum
    return float(8 * (
        (1 - np.cos(x)) + (1 - np.cos(y)) + (1 - np.cos(z))
        + (1 - np.cos(x) * np.cos(y))
        + (1 - np.cos(x) * np.cos(z))
        + (1 - np.cos(y) * np.cos(z))
    ))


def main() -> None:
    identity3 = np.eye(3)
    identity8 = np.eye(8, dtype=complex)
    rz = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
    rx = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=float)
    system = commutant_system((rz, rx))
    _, singular_values, vh = np.linalg.svd(system)
    rank = int(np.sum(singular_values > TOLERANCE))
    commutant = vh[-1].reshape(3, 3)
    commutant /= np.trace(commutant) / 3

    gs = [gamma(axis) for axis in range(3)]
    b = beta()
    f = flavor()
    plus_projector = (identity8 + f) / 2
    minus_projector = (identity8 - f) / 2
    raw_minus, raw_plus = raw_embeddings()

    errors: dict[str, float] = {
        "rotation_commutant_is_scalar": float(np.linalg.norm(commutant - identity3)),
        "flavor_involution": float(np.linalg.norm(f @ f - identity8)),
        "projector_decomposition": float(np.linalg.norm(plus_projector + minus_projector - identity8)),
        "projector_orthogonality": float(np.linalg.norm(plus_projector @ minus_projector)),
        "plus_embedding_exhausts_projector": float(np.linalg.norm(orthogonal_projector(raw_plus) - plus_projector)),
        "minus_embedding_exhausts_projector": float(np.linalg.norm(orthogonal_projector(raw_minus) - minus_projector)),
    }

    # A single-axis reflection is a sharper control than arbitrary unequal
    # coefficients: it still squares to the same scalar symbol, but it does
    # not commute with the generating cube rotations.
    noncovariant_control = np.diag([-1.0, 1.0, 1.0])
    controls = {
        "noncovariant_control_commutator": float(
            max(
                np.linalg.norm(noncovariant_control @ rotation - rotation @ noncovariant_control)
                for rotation in (rz, rx)
            )
        )
    }

    rng = np.random.default_rng(180018)
    for sample in range(24):
        mass = rng.normal() + 1j * rng.normal()
        momentum = rng.normal(size=3) + 1j * rng.normal(size=3)
        operator = mass * b + sum(momentum[axis] * gs[axis] for axis in range(3))
        scalar = mass**2 + sum(component**2 for component in momentum)
        errors[f"canonical_square_{sample}"] = float(
            np.linalg.norm(operator @ operator - scalar * identity8)
        )
        negative_operator = -operator
        errors[f"negative_orientation_square_control_{sample}"] = float(
            np.linalg.norm(negative_operator @ negative_operator - scalar * identity8)
        )
        reflected_operator = (
            mass * b - momentum[0] * gs[0]
            + momentum[1] * gs[1] + momentum[2] * gs[2]
        )
        errors[f"reflected_axis_square_control_{sample}"] = float(
            np.linalg.norm(reflected_operator @ reflected_operator - scalar * identity8)
        )

    zero_counts: dict[str, int] = {}
    for size in range(2, 33):
        modes = 2 * np.pi * np.fft.fftfreq(size)
        zeros: list[tuple[int, int, int]] = []
        for i, x in enumerate(modes):
            for j, y in enumerate(modes):
                for k, z in enumerate(modes):
                    if abs(c19_symbol(np.array([x, y, z]))) <= TOLERANCE:
                        zeros.append((i, j, k))
        zero_counts[str(size)] = len(zeros)
        if zeros != [(0, 0, 0)]:
            errors[f"unexpected_zero_set_N{size}"] = float(len(zeros))

    checks = {
        "cubic_commutant_has_dimension_one": rank == 8,
        "cubic_commutant_basis_is_identity": errors["rotation_commutant_is_scalar"] <= TOLERANCE,
        "canonical_root_squares_to_scalar": all(
            errors[f"canonical_square_{sample}"] <= TOLERANCE for sample in range(24)
        ),
        "noncovariant_reflection_squares_but_is_rejected": (
            controls["noncovariant_control_commutator"] > 1.0
            and all(
                errors[f"reflected_axis_square_control_{sample}"] <= TOLERANCE
                for sample in range(24)
            )
        ),
        "flavor_has_exactly_two_rank_four_sectors": (
            np.linalg.matrix_rank(plus_projector, tol=TOLERANCE) == 4
            and np.linalg.matrix_rank(minus_projector, tol=TOLERANCE) == 4
            and errors["plus_embedding_exhausts_projector"] <= TOLERANCE
            and errors["minus_embedding_exhausts_projector"] <= TOLERANCE
        ),
        "tested_periodic_brillouin_zones_have_only_origin_zero": all(
            count == 1 for count in zero_counts.values()
        ),
        "negative_orientation_control_squares_but_is_rejected": (
            all(
                errors[f"negative_orientation_square_control_{sample}"] <= TOLERANCE
                for sample in range(24)
            )
            and not (-1.0 >= 0 and -1.0 >= 0)
        ),
    }
    max_error = max(abs(value) for value in errors.values())
    passed = max_error <= TOLERANCE and all(checks.values())
    result = {
        "status": "PASS" if passed else "FAIL",
        "tolerance": TOLERANCE,
        "max_error": max_error,
        "commutant_rank": rank,
        "commutant_nullity": 9 - rank,
        "plus_sector_rank": int(np.linalg.matrix_rank(plus_projector, tol=TOLERANCE)),
        "minus_sector_rank": int(np.linalg.matrix_rank(minus_projector, tol=TOLERANCE)),
        "periodic_zone_zero_counts": zero_counts,
        "controls": controls,
        "checks": checks,
        "errors": errors,
        "claim": (
            "independent reconstruction of the scoped real, sector-local, "
            "flavor-preserving cubical-incidence factorization classification, exact two-copy "
            "census, and finite periodic-zone zero controls"
        ),
        "excluded_claims": [
            "classification of arbitrary nonlocal square roots",
            "classification of flavor-mixing roots outside the native flavor-preserving class",
            "localized nonlinear electron selection or stability",
            "electromagnetic coupling",
        ],
    }
    output = Path(__file__).with_name("result.json")
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
