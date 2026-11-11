"""Independent matrix check of the Lean DIRAC-010 representation map."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def bit(corner: int, axis: int) -> int:
    shifts = (2, 1, 0)
    return (corner >> shifts[axis]) & 1


def flip(corner: int, axis: int) -> int:
    return corner ^ (1 << (2 - axis))


def gamma(axis: int) -> np.ndarray:
    out = np.zeros((8, 8), dtype=complex)
    for corner in range(8):
        if axis == 0:
            sign = 1
        elif axis == 1:
            sign = 1 if bit(corner, 0) == 0 else -1
        else:
            sign = 1 if (bit(corner, 0) + bit(corner, 1)) % 2 == 0 else -1
        out[corner, flip(corner, axis)] = sign
    return out


def beta() -> np.ndarray:
    return np.diag([
        1 if sum(bit(corner, axis) for axis in range(3)) % 2 == 0 else -1
        for corner in range(8)
    ]).astype(complex)


def flavor() -> np.ndarray:
    out = np.zeros((8, 8), dtype=complex)
    for corner in range(8):
        sign = 1 if bit(corner, 2) == 0 else -1
        out[corner, flip(flip(corner, 2), 1)] = 1j * sign
    return out


def embeddings() -> tuple[np.ndarray, np.ndarray]:
    minus = np.zeros((8, 4), dtype=complex)
    plus = np.zeros((8, 4), dtype=complex)
    minus[0, 0] = 1; minus[1, 1] = 1; minus[2, 1] = -1j; minus[3, 0] = 1j
    minus[4, 2] = 1; minus[5, 3] = 1; minus[6, 3] = -1j; minus[7, 2] = 1j
    plus[0, 0] = 1; plus[1, 1] = 1; plus[2, 1] = 1j; plus[3, 0] = -1j
    plus[4, 2] = 1; plus[5, 3] = 1; plus[6, 3] = 1j; plus[7, 2] = -1j
    return minus, plus


def main() -> None:
    gs = [gamma(axis) for axis in range(3)]
    b = beta()
    f = flavor()
    ident = np.eye(8, dtype=complex)
    minus, plus = embeddings()
    errors: dict[str, float] = {}
    for axis, g in enumerate(gs):
        errors[f"gamma_{axis}_square"] = float(np.linalg.norm(g @ g - ident))
        errors[f"beta_gamma_{axis}"] = float(np.linalg.norm(b @ g + g @ b))
        errors[f"flavor_gamma_{axis}"] = float(np.linalg.norm(f @ g - g @ f))
    for first in range(3):
        for second in range(first + 1, 3):
            errors[f"gamma_{first}_{second}"] = float(
                np.linalg.norm(gs[first] @ gs[second] + gs[second] @ gs[first])
            )
    errors["beta_square"] = float(np.linalg.norm(b @ b - ident))
    errors["flavor_square"] = float(np.linalg.norm(f @ f - ident))
    errors["minus_flavor"] = float(np.linalg.norm(f @ minus + minus))
    errors["plus_flavor"] = float(np.linalg.norm(f @ plus - plus))
    errors["minus_rank_defect"] = float(4 - np.linalg.matrix_rank(minus))
    errors["plus_rank_defect"] = float(4 - np.linalg.matrix_rank(plus))
    max_error = max(abs(value) for value in errors.values())
    result = {
        "status": "PASS" if max_error < 1e-12 else "FAIL",
        "max_error": max_error,
        "errors": errors,
        "claim": "independent matrix reproduction of the native cubical Clifford algebra and both rank-four copies",
        "excluded_claims": ["physical mass selection", "localized electron", "operational spin half"],
    }
    output = Path(__file__).with_name("result.json")
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

