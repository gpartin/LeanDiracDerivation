"""Independent matrix checks for DIRAC-026 through DIRAC-030."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np


OUT = Path(__file__).with_name("result.json")
TOL = 5.0e-11
I4 = np.eye(4, dtype=complex)
I = 1j
BETA = np.diag([1, 1, -1, -1]).astype(complex)
AX = np.array([[0, 0, 0, 1], [0, 0, 1, 0],
               [0, 1, 0, 0], [1, 0, 0, 0]], dtype=complex)
AY = np.array([[0, 0, 0, -I], [0, 0, I, 0],
               [0, -I, 0, 0], [I, 0, 0, 0]], dtype=complex)
AZ = np.array([[0, 0, 1, 0], [0, 0, 0, -1],
               [1, 0, 0, 0], [0, -1, 0, 0]], dtype=complex)
GAMMA_CUBE = [AZ, -AY, -AX]
SPIN = [(-I / 2) * GAMMA_CUBE[1] @ GAMMA_CUBE[2],
        (-I / 2) * GAMMA_CUBE[2] @ GAMMA_CUBE[0],
        (-I / 2) * GAMMA_CUBE[0] @ GAMMA_CUBE[1]]
BOOST = [(I / 2) * matrix for matrix in GAMMA_CUBE]
GAMMA = [BETA] + [BETA @ matrix for matrix in GAMMA_CUBE]
CHARGE_MATRIX = I * BETA @ AY


def error(first: np.ndarray, second: np.ndarray) -> float:
    return float(np.max(np.abs(first - second)))


def comm(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    return first @ second - second @ first


def sigma(mu: int, nu: int) -> np.ndarray:
    return (I / 2) * comm(GAMMA[mu], GAMMA[nu])


def slash(four: np.ndarray) -> np.ndarray:
    cov = np.array([four[0], -four[1], -four[2], -four[3]])
    return sum(cov[index] * GAMMA[index] for index in range(4))


def charge(state: np.ndarray) -> np.ndarray:
    return CHARGE_MATRIX @ state.conjugate()


def main() -> int:
    rng = np.random.default_rng(26030)
    errors: dict[str, float] = {}

    # DIRAC-026: Lorentz algebra, finite boost and mass shell.
    errors["026_boost_commutator"] = max(
        error(comm(BOOST[0], BOOST[1]), -I * SPIN[2]),
        error(comm(BOOST[1], BOOST[2]), -I * SPIN[0]),
        error(comm(BOOST[2], BOOST[0]), -I * SPIN[1]),
    )
    errors["026_rotation_boost"] = max(
        error(comm(SPIN[0], BOOST[1]), I * BOOST[2]),
        error(comm(SPIN[1], BOOST[2]), I * BOOST[0]),
        error(comm(SPIN[2], BOOST[0]), I * BOOST[1]),
    )
    boost_errors = []
    for axis in range(3):
        for rapidity in (-1.1, -0.35, 0.4, 1.25):
            matrix = (math.cosh(rapidity / 2) * I4 +
                      math.sinh(rapidity / 2) * GAMMA_CUBE[axis])
            inverse = (math.cosh(rapidity / 2) * I4 -
                       math.sinh(rapidity / 2) * GAMMA_CUBE[axis])
            rest = np.array([1, 0, 0, 0], dtype=complex)
            state = matrix @ rest
            energy = 19.0 * math.cosh(rapidity)
            momentum_coefficient = 19.0 * math.sinh(rapidity)
            hamiltonian = 19.0 * BETA + momentum_coefficient * GAMMA_CUBE[axis]
            boost_errors.extend([
                error(inverse @ matrix, I4),
                error(hamiltonian @ state, energy * state),
                abs(energy**2 - momentum_coefficient**2 - 19.0**2),
            ])
    errors["026_finite_boost"] = max(boost_errors)

    # DIRAC-027: Pauli--Lubanski Casimir for generic four-momenta.
    casimir_errors = []
    for _ in range(12):
        four = rng.normal(size=4)
        energy, px, py, pz = four
        w0 = px * SPIN[0] + py * SPIN[1] + pz * SPIN[2]
        wx = energy * SPIN[0] - py * BOOST[2] + pz * BOOST[1]
        wy = energy * SPIN[1] - pz * BOOST[0] + px * BOOST[2]
        wz = energy * SPIN[2] - px * BOOST[1] + py * BOOST[0]
        actual = w0 @ w0 - wx @ wx - wy @ wy - wz @ wz
        invariant = energy**2 - px**2 - py**2 - pz**2
        casimir_errors.append(error(actual, -(3.0 / 4.0) * invariant * I4))
    errors["027_pauli_lubanski"] = max(casimir_errors)

    # DIRAC-028: the antisymmetric mixed boost term is half the commutator.
    errors["028_thomas_half"] = error(
        0.5 * comm(BOOST[0], BOOST[1]), (-I / 2) * SPIN[2])

    # DIRAC-029: anti-linear involution and Hamiltonian branch exchange.
    charge_errors = []
    for _ in range(12):
        state = rng.normal(size=4) + I * rng.normal(size=4)
        momentum = rng.normal(size=3)
        hplus = 19.0 * BETA + sum(
            momentum[index] * GAMMA_CUBE[index] for index in range(3))
        hminus = 19.0 * BETA - sum(
            momentum[index] * GAMMA_CUBE[index] for index in range(3))
        charge_errors.extend([
            error(charge(charge(state)), state),
            abs(np.vdot(charge(state), charge(state)) - np.vdot(state, state)),
            error(hminus @ charge(state), -charge(hplus @ state)),
        ])
    errors["029_charge_conjugation"] = max(charge_errors)

    # DIRAC-030: operator Gordon identity for generic incoming/outgoing momenta.
    gordon_errors = []
    for _ in range(12):
        q = rng.normal(size=4)
        p = rng.normal(size=4)
        qcov = np.array([q[0], -q[1], -q[2], -q[3]])
        pcov = np.array([p[0], -p[1], -p[2], -p[3]])
        for mu in range(4):
            left = slash(q) @ GAMMA[mu] + GAMMA[mu] @ slash(p)
            right = (q[mu] + p[mu]) * I4 + I * sum(
                (qcov[nu] - pcov[nu]) * sigma(mu, nu)
                for nu in range(4))
            gordon_errors.append(error(left, right))
    errors["030_gordon_kernel"] = max(gordon_errors)

    # Controls are deliberately wrong reconstructions and must remain finite.
    controls = {
        "wrong_thomas_factor": error(
            comm(BOOST[0], BOOST[1]), (-I / 2) * SPIN[2]),
        "pauli_lubanski_without_boosts": error(
            SPIN[0] @ SPIN[0], -(3.0 / 4.0) * I4),
        "charge_without_momentum_reversal": error(
            (19.0 * BETA + 0.7 * GAMMA_CUBE[0]) @
                charge(np.array([1, 2, 3, 4], dtype=complex)),
            -charge((19.0 * BETA + 0.7 * GAMMA_CUBE[0]) @
                np.array([1, 2, 3, 4], dtype=complex))),
        "gordon_without_spin_term": error(
            slash(np.array([2.0, 0.3, -0.2, 0.4])) @ GAMMA[1] +
                GAMMA[1] @ slash(np.array([1.7, -0.4, 0.1, 0.2])),
            (-0.1) * I4),
    }
    checks = {name: value <= TOL for name, value in errors.items()}
    checks["negative_controls_rejected"] = all(
        math.isfinite(value) and value > 1.0e-3 for value in controls.values())
    max_error = max(errors.values())
    payload = {
        "schema_version": "1.0.0",
        "stages": ["DIRAC-026", "DIRAC-027", "DIRAC-028", "DIRAC-029", "DIRAC-030"],
        "status": "PASS" if all(checks.values()) else "FAIL",
        "tolerance": TOL,
        "max_error": max_error,
        "errors": errors,
        "checks": checks,
        "negative_controls": controls,
        "native_framework_changed": False,
        "excluded_claims": [
            "exact finite-spacing Lorentz invariance",
            "localized nonlinear electron",
            "electric charge or interacting positron",
            "electromagnetic magnetic moment or g factor",
            "SI calibration",
        ],
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
