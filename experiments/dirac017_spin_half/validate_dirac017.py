"""Independent matrix reconstruction of the DIRAC-017 spin representation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


TOLERANCE = 2.0e-12


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


def restricted(operator: np.ndarray, embedding: np.ndarray) -> np.ndarray:
    gram_inverse = np.linalg.inv(embedding.conj().T @ embedding)
    return gram_inverse @ embedding.conj().T @ operator @ embedding


def rotation(axis: int, angle: float, planes: list[np.ndarray]) -> np.ndarray:
    return np.cos(angle / 2) * np.eye(8) - np.sin(angle / 2) * planes[axis]


def rotated_momentum(axis: int, angle: float, momentum: np.ndarray) -> np.ndarray:
    px, py, pz = momentum
    cosine, sine = np.cos(angle), np.sin(angle)
    if axis == 0:
        return np.array([cosine * px + sine * py, -sine * px + cosine * py, pz])
    if axis == 1:
        return np.array([cosine * px + sine * pz, py, -sine * px + cosine * pz])
    return np.array([px, cosine * py - sine * pz, sine * py + cosine * pz])


def hamiltonian(mass: complex, momentum: np.ndarray, gs: list[np.ndarray], b: np.ndarray) -> np.ndarray:
    px, py, pz = momentum
    return mass * b + pz * gs[0] - py * gs[1] - px * gs[2]


def main() -> None:
    identity = np.eye(8, dtype=complex)
    gs = [gamma(axis) for axis in range(3)]
    planes = [gs[1] @ gs[2], gs[2] @ gs[0], gs[0] @ gs[1]]
    sigma = [-1j * plane for plane in planes]
    spins = [0.5 * component for component in sigma]
    b = beta()
    minus, plus = dirac_embeddings()

    errors: dict[str, float] = {}
    cyclic = ((0, 1, 2), (1, 2, 0), (2, 0, 1))
    for axis in range(3):
        errors[f"plane_{axis}_square"] = float(np.linalg.norm(planes[axis] @ planes[axis] + identity))
        errors[f"spin_{axis}_square"] = float(np.linalg.norm(spins[axis] @ spins[axis] - 0.25 * identity))
        errors[f"spin_{axis}_self_adjoint"] = float(np.linalg.norm(spins[axis].conj().T - spins[axis]))
        errors[f"spin_{axis}_commutes_beta"] = float(np.linalg.norm(spins[axis] @ b - b @ spins[axis]))
    for first, second, third in cyclic:
        errors[f"spin_commutator_{first}{second}"] = float(
            np.linalg.norm(spins[first] @ spins[second] - spins[second] @ spins[first] - 1j * spins[third])
        )
    errors["casimir"] = float(np.linalg.norm(sum(component @ component for component in spins) - 0.75 * identity))

    for copy_name, embedding in (("minus", minus), ("plus", plus)):
        for axis in range(3):
            spin4 = restricted(spins[axis], embedding)
            errors[f"{copy_name}_spin_{axis}_intertwiner"] = float(
                np.linalg.norm(spins[axis] @ embedding - embedding @ spin4)
            )
        spectrum = np.sort(np.linalg.eigvalsh(restricted(spins[0], embedding)))
        errors[f"{copy_name}_spin0_spectrum"] = float(
            np.linalg.norm(spectrum - np.array([-0.5, -0.5, 0.5, 0.5]))
        )

    quarter_turns = [(identity - plane) / np.sqrt(2) for plane in planes]
    for axis, quarter in enumerate(quarter_turns):
        errors[f"quarter_{axis}_unitarity"] = float(np.linalg.norm(quarter.conj().T @ quarter - identity))
        errors[f"quarter_{axis}_four"] = float(np.linalg.norm(np.linalg.matrix_power(quarter, 4) + identity))
        errors[f"quarter_{axis}_eight"] = float(np.linalg.norm(np.linalg.matrix_power(quarter, 8) - identity))

    sample_angles = (0.0, 0.31, 1.07, np.pi / 2, 2 * np.pi, 4 * np.pi)
    for axis in range(3):
        for index, angle in enumerate(sample_angles):
            turn = rotation(axis, angle, planes)
            errors[f"rotation_{axis}_{index}_unitarity"] = float(np.linalg.norm(turn.conj().T @ turn - identity))
        errors[f"rotation_{axis}_two_pi"] = float(np.linalg.norm(rotation(axis, 2 * np.pi, planes) + identity))
        errors[f"rotation_{axis}_four_pi"] = float(np.linalg.norm(rotation(axis, 4 * np.pi, planes) - identity))
        errors[f"rotation_{axis}_composition"] = float(
            np.linalg.norm(rotation(axis, 0.37, planes) @ rotation(axis, -0.91, planes) - rotation(axis, -0.54, planes))
        )

    mass = 0.73 - 0.11j
    momentum = np.array([0.21 + 0.03j, -0.34 + 0.07j, 0.16 - 0.05j])
    angle = 0.83
    for axis in range(3):
        turn = rotation(axis, angle, planes)
        rotated = rotated_momentum(axis, angle, momentum)
        errors[f"continuous_hamiltonian_covariance_{axis}"] = float(
            np.linalg.norm(turn @ hamiltonian(mass, momentum, gs, b) - hamiltonian(mass, rotated, gs, b) @ turn)
        )

    checks = {
        "native_spin_algebra_reconstructed": all(errors[f"spin_commutator_{a}{b}"] <= TOLERANCE for a, b, _ in cyclic),
        "spin_half_spectrum_on_both_copies": errors["minus_spin0_spectrum"] <= TOLERANCE and errors["plus_spin0_spectrum"] <= TOLERANCE,
        "quarter_turn_is_unitary_and_double_valued": all(
            errors[f"quarter_{axis}_unitarity"] <= TOLERANCE
            and errors[f"quarter_{axis}_four"] <= TOLERANCE
            and errors[f"quarter_{axis}_eight"] <= TOLERANCE
            for axis in range(3)
        ),
        "continuous_rotations_are_unitary_with_two_pi_sign_and_four_pi_return": all(
            errors[f"rotation_{axis}_two_pi"] <= TOLERANCE
            and errors[f"rotation_{axis}_four_pi"] <= TOLERANCE
            for axis in range(3)
        ),
        "continuous_hamiltonian_covariance_reconstructed": all(
            errors[f"continuous_hamiltonian_covariance_{axis}"] <= TOLERANCE
            for axis in range(3)
        ),
    }
    max_error = max(abs(value) for value in errors.values())
    result = {
        "status": "PASS" if max_error <= TOLERANCE and all(checks.values()) else "FAIL",
        "tolerance": TOLERANCE,
        "max_error": max_error,
        "checks": checks,
        "errors": errors,
        "claim": "independent matrix reconstruction of the conditional DIRAC-017 cubical spin-one-half representation",
        "excluded_claims": [
            "localized dynamical spin",
            "electromagnetic coupling or g-factor",
            "physical electron identity",
        ],
    }
    output = Path(__file__).with_name("result.json")
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
