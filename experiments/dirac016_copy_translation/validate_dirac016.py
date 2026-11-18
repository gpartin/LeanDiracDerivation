"""Independent matrix reconstruction of the DIRAC-016 copy exchange."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


TOLERANCE = 1.0e-12


def bit(corner: int, axis: int) -> int:
    return (corner >> (2 - axis)) & 1


def flip(corner: int, axis: int) -> int:
    return corner ^ (1 << (2 - axis))


def permutation(axis: int) -> np.ndarray:
    matrix = np.zeros((8, 8), dtype=complex)
    for corner in range(8):
        matrix[corner, flip(corner, axis)] = 1
    return matrix


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


def dirac_embeddings() -> tuple[np.ndarray, np.ndarray]:
    raw_minus, raw_plus = raw_embeddings()
    to_minus = np.array([
        [1, 0, 0, 0],
        [0, 0, 0, -1],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
    ], dtype=complex)
    to_plus = np.array([
        [0, -1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0],
    ], dtype=complex)
    return raw_minus @ to_minus, raw_plus @ to_plus


def half_shift() -> np.ndarray:
    return np.array([
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, 0],
    ], dtype=complex)


def main() -> None:
    identity8 = np.eye(8, dtype=complex)
    identity4 = np.eye(4, dtype=complex)
    translations = [permutation(axis) for axis in range(3)]
    flavor_matrix = flavor()
    minus, plus = dirac_embeddings()
    shift = half_shift()

    errors: dict[str, float] = {}
    for axis, translation in enumerate(translations):
        errors[f"axis_{axis}_involution"] = float(
            np.linalg.norm(translation @ translation - identity8)
        )
        errors[f"axis_{axis}_unitarity"] = float(
            np.linalg.norm(translation.conj().T @ translation - identity8)
        )
    errors["axis_0_flavor_preservation"] = float(
        np.linalg.norm(flavor_matrix @ translations[0] - translations[0] @ flavor_matrix)
    )
    errors["axis_1_flavor_preservation"] = float(
        np.linalg.norm(flavor_matrix @ translations[1] - translations[1] @ flavor_matrix)
    )
    errors["axis_2_flavor_exchange"] = float(
        np.linalg.norm(flavor_matrix @ translations[2] + translations[2] @ flavor_matrix)
    )
    errors["minus_to_plus"] = float(
        np.linalg.norm(translations[2] @ minus - plus @ shift)
    )
    errors["plus_to_minus"] = float(
        np.linalg.norm(translations[2] @ plus - minus @ shift)
    )
    errors["half_shift_involution"] = float(np.linalg.norm(shift @ shift - identity4))
    errors["minus_rank_defect"] = float(4 - np.linalg.matrix_rank(minus))
    errors["plus_rank_defect"] = float(4 - np.linalg.matrix_rank(plus))
    errors["signed_incidence_preserves_flavor"] = float(
        np.linalg.norm(flavor_matrix @ gamma(2) - gamma(2) @ flavor_matrix)
    )

    c19_directions: list[tuple[tuple[int, int, int], int]] = []
    for axis in range(3):
        for sign in (-1, 1):
            vector = [0, 0, 0]
            vector[axis] = sign
            c19_directions.append((tuple(vector), 2))
    for first, second in ((0, 1), (0, 2), (1, 2)):
        for first_sign in (-1, 1):
            for second_sign in (-1, 1):
                vector = [0, 0, 0]
                vector[first] = first_sign
                vector[second] = second_sign
                c19_directions.append((tuple(vector), 1))
    exchange_weight = sum(weight for vector, weight in c19_directions if vector[2] != 0)
    preserving_weight = sum(weight for vector, weight in c19_directions if vector[2] == 0)
    errors["c19_exchange_weight"] = float(abs(exchange_weight - 12))
    errors["c19_preserving_weight"] = float(abs(preserving_weight - 12))

    checks = {
        "all_translations_are_unitary_involutions": all(
            errors[f"axis_{axis}_involution"] <= TOLERANCE
            and errors[f"axis_{axis}_unitarity"] <= TOLERANCE
            for axis in range(3)
        ),
        "axis_2_exchanges_both_rank_four_copies": (
            errors["minus_to_plus"] <= TOLERANCE
            and errors["plus_to_minus"] <= TOLERANCE
            and errors["minus_rank_defect"] == 0
            and errors["plus_rank_defect"] == 0
        ),
        "coordinate_half_shift_is_involutive": errors["half_shift_involution"] <= TOLERANCE,
        "signed_incidence_preserves_copy": errors["signed_incidence_preserves_flavor"] <= TOLERANCE,
        "c19_copy_class_weights_are_balanced": exchange_weight == preserving_weight == 12,
    }
    max_error = max(abs(value) for value in errors.values())
    result = {
        "status": "PASS" if max_error <= TOLERANCE and all(checks.values()) else "FAIL",
        "tolerance": TOLERANCE,
        "max_error": max_error,
        "checks": checks,
        "errors": errors,
        "c19_exchange_weight": exchange_weight,
        "c19_preserving_weight": preserving_weight,
        "claim": "independent matrix reconstruction of the scoped DIRAC-016 cube-copy exchange",
        "excluded_claims": [
            "native time evolution performs one copy exchange per update",
            "localized carrier transport",
            "physical purpose for doubling",
        ],
    }
    output = Path(__file__).with_name("result.json")
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
