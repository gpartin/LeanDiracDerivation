"""Independent numerical reconstruction of the DIRAC-011 identities."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


CHI0 = 19.0
C = np.sqrt(12.0)


def bit(corner: int, axis: int) -> int:
    return (corner >> (2 - axis)) & 1


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


def beta8() -> np.ndarray:
    return np.diag([
        1 if sum(bit(corner, axis) for axis in range(3)) % 2 == 0 else -1
        for corner in range(8)
    ]).astype(complex)


def embeddings() -> tuple[np.ndarray, np.ndarray]:
    minus = np.zeros((8, 4), dtype=complex)
    plus = np.zeros((8, 4), dtype=complex)
    minus[0, 0] = 1; minus[1, 1] = 1; minus[2, 1] = -1j; minus[3, 0] = 1j
    minus[4, 2] = 1; minus[5, 3] = 1; minus[6, 3] = -1j; minus[7, 2] = 1j
    plus[0, 0] = 1; plus[1, 1] = 1; plus[2, 1] = 1j; plus[3, 0] = -1j
    plus[4, 2] = 1; plus[5, 3] = 1; plus[6, 3] = 1j; plus[7, 2] = -1j
    standard_to_minus = np.array([
        [1, 0, 0, 0], [0, 0, 0, -1], [0, 0, 1, 0], [0, 1, 0, 0]
    ], dtype=complex)
    standard_to_plus = np.array([
        [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0]
    ], dtype=complex)
    return minus @ standard_to_minus, plus @ standard_to_plus


def standard_matrices() -> tuple[np.ndarray, list[np.ndarray]]:
    beta = np.diag([1, 1, -1, -1]).astype(complex)
    ax = np.array([
        [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0]
    ], dtype=complex)
    ay = np.array([
        [0, 0, 0, -1j], [0, 0, 1j, 0], [0, -1j, 0, 0], [1j, 0, 0, 0]
    ], dtype=complex)
    az = np.array([
        [0, 0, 1, 0], [0, 0, 0, -1], [1, 0, 0, 0], [0, -1, 0, 0]
    ], dtype=complex)
    return beta, [ax, ay, az]


def h8(mass: float, momentum: np.ndarray) -> np.ndarray:
    gs = [gamma(axis) for axis in range(3)]
    return mass * beta8() + C * sum(momentum[a] * gs[a] for a in range(3))


def h4(mass: float, momentum: np.ndarray) -> np.ndarray:
    beta, alpha = standard_matrices()
    px, py, pz = -C * momentum[2], -C * momentum[1], C * momentum[0]
    return mass * beta + px * alpha[0] + py * alpha[1] + pz * alpha[2]


def c19_transport(momentum: np.ndarray) -> float:
    k0, k1, k2 = momentum
    return 8.0 * (
        (1 - np.cos(k0)) + (1 - np.cos(k1)) + (1 - np.cos(k2))
        + (1 - np.cos(k0) * np.cos(k1))
        + (1 - np.cos(k0) * np.cos(k2))
        + (1 - np.cos(k1) * np.cos(k2))
    )


def main() -> None:
    rng = np.random.default_rng(11015)
    errors: dict[str, float] = {}
    minus, plus = embeddings()
    identity8 = np.eye(8, dtype=complex)

    for case in range(12):
        mass = float(rng.uniform(0.2, 21.0))
        momentum = rng.uniform(-0.35, 0.35, size=3)
        amplitude = rng.normal(size=8) + 1j * rng.normal(size=8)
        native_h = h8(mass, momentum)
        standard_h = h4(mass, momentum)
        exact_scalar = mass * mass + c19_transport(momentum)
        continuum_scalar = mass * mass + 12.0 * float(momentum @ momentum)
        remainder = exact_scalar - continuum_scalar

        errors[f"case_{case}_square"] = float(np.linalg.norm(
            native_h @ native_h - continuum_scalar * identity8
        ))
        errors[f"case_{case}_minus_copy"] = float(np.linalg.norm(
            native_h @ minus - minus @ standard_h
        ))
        errors[f"case_{case}_plus_copy"] = float(np.linalg.norm(
            native_h @ plus - plus @ standard_h
        ))

        for label, sign in (("positive", -1.0), ("negative", 1.0)):
            velocity = sign * 1j * native_h @ amplitude
            graph_derivative = np.concatenate([
                velocity,
                sign * 1j * native_h @ velocity - remainder * amplitude,
            ])
            exact_derivative = np.concatenate([
                velocity,
                -exact_scalar * amplitude,
            ])
            errors[f"case_{case}_{label}_graph"] = float(np.linalg.norm(
                graph_derivative - exact_derivative
            ))

    amplitude_size = 11.0
    chi_size = np.sqrt(CHI0 * CHI0 - amplitude_size * amplitude_size)
    native_frequency = np.sqrt(amplitude_size**2 + chi_size**2)
    selected_mass = np.sqrt(native_frequency**2)
    errors["native_frequency_minus_chi0"] = abs(native_frequency - CHI0)
    errors["selected_mass_minus_chi0"] = abs(selected_mass - CHI0)

    max_error = max(errors.values())
    result = {
        "status": "PASS" if max_error <= 1.0e-11 else "FAIL",
        "max_error": max_error,
        "tolerance": 1.0e-11,
        "cases": 12,
        "errors": errors,
        "claim": (
            "independent reconstruction of both frequency graphs, both retained "
            "rank-four copies, the exact C19 remainder, and nonnegative native "
            "rest-mass selection"
        ),
        "excluded_claims": [
            "full continuum Dirac evolution",
            "Dirac density or current",
            "physical spin half",
            "electron identity",
        ],
    }
    output = Path(__file__).with_name("result.json")
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
