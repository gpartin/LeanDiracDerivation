"""Independent reconstruction of the DIRAC-024 projective-cycle bridge."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


TOLERANCE = 2.0e-11
CHI0 = 19.0
C_EFF = np.sqrt(12.0)
OMEGA_ZITTER = 2.0 * CHI0
ZITTER_PERIOD = 2.0 * np.pi / OMEGA_ZITTER
ZITTER_AMPLITUDE = C_EFF / OMEGA_ZITTER


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


def dirac_embeddings() -> tuple[np.ndarray, np.ndarray]:
    raw_minus = np.zeros((8, 4), dtype=complex)
    raw_plus = np.zeros((8, 4), dtype=complex)
    raw_minus[0, 0] = 1; raw_minus[1, 1] = 1
    raw_minus[2, 1] = -1j; raw_minus[3, 0] = 1j
    raw_minus[4, 2] = 1; raw_minus[5, 3] = 1
    raw_minus[6, 3] = -1j; raw_minus[7, 2] = 1j
    raw_plus[0, 0] = 1; raw_plus[1, 1] = 1
    raw_plus[2, 1] = 1j; raw_plus[3, 0] = -1j
    raw_plus[4, 2] = 1; raw_plus[5, 3] = 1
    raw_plus[6, 3] = 1j; raw_plus[7, 2] = -1j
    to_minus = np.array([
        [1, 0, 0, 0], [0, 0, 0, -1],
        [0, 0, 1, 0], [0, 1, 0, 0],
    ], dtype=complex)
    to_plus = np.array([
        [0, -1, 0, 0], [0, 0, 1, 0],
        [0, 0, 0, 1], [1, 0, 0, 0],
    ], dtype=complex)
    return raw_minus @ to_minus, raw_plus @ to_plus


def coherent_state(time: float) -> np.ndarray:
    normalization = 1.0 / np.sqrt(2.0)
    return normalization * np.array([
        np.exp(-1j * CHI0 * time), 0.0,
        np.exp(1j * CHI0 * time), 0.0,
    ], dtype=complex)


def zitter_velocity(time: float) -> float:
    return C_EFF * np.cos(OMEGA_ZITTER * time)


def zitter_position(time: float) -> float:
    return ZITTER_AMPLITUDE * np.sin(OMEGA_ZITTER * time)


def rotation(axis: int, angle: float, planes: list[np.ndarray]) -> np.ndarray:
    return np.cos(angle / 2.0) * np.eye(8) - np.sin(angle / 2.0) * planes[axis]


def main() -> None:
    gs = [gamma(axis) for axis in range(3)]
    planes = [gs[1] @ gs[2], gs[2] @ gs[0], gs[0] @ gs[1]]
    minus, plus = dirac_embeddings()
    sample_times = (-0.41, 0.0, 0.073, 0.52, 1.17)
    errors: dict[str, float] = {}

    errors["period_equals_pi_over_chi0"] = abs(ZITTER_PERIOD - np.pi / CHI0)
    for index, time in enumerate(sample_times):
        state = coherent_state(time)
        state_one = coherent_state(time + ZITTER_PERIOD)
        state_two = coherent_state(time + 2.0 * ZITTER_PERIOD)
        errors[f"state_antiperiod_{index}"] = float(np.linalg.norm(state_one + state))
        errors[f"state_two_cycle_return_{index}"] = float(np.linalg.norm(state_two - state))
        errors[f"density_return_{index}"] = abs(np.vdot(state_one, state_one).real - np.vdot(state, state).real)
        errors[f"velocity_return_{index}"] = abs(zitter_velocity(time + ZITTER_PERIOD) - zitter_velocity(time))
        errors[f"position_return_{index}"] = abs(zitter_position(time + ZITTER_PERIOD) - zitter_position(time))
        for axis in range(3):
            turn_2pi = rotation(axis, 2.0 * np.pi, planes)
            turn_4pi = rotation(axis, 4.0 * np.pi, planes)
            for copy_name, embedding in (("minus", minus), ("plus", plus)):
                errors[f"{copy_name}_{axis}_2pi_bridge_{index}"] = float(
                    np.linalg.norm(turn_2pi @ embedding @ state - embedding @ state_one)
                )
                errors[f"{copy_name}_{axis}_4pi_bridge_{index}"] = float(
                    np.linalg.norm(turn_4pi @ embedding @ state - embedding @ state_two)
                )

    errors["half_period_velocity_reversal"] = abs(
        zitter_velocity(ZITTER_PERIOD / 2.0) + zitter_velocity(0.0)
    )
    half_period_return_defect = abs(
        zitter_velocity(ZITTER_PERIOD / 2.0) - zitter_velocity(0.0)
    )

    beta = np.diag([1.0, 1.0, -1.0, -1.0]).astype(complex)
    rest_hamiltonian = CHI0 * beta
    gram_inverse = np.linalg.inv(minus.conj().T @ minus)
    spin0 = gram_inverse @ minus.conj().T @ (-0.5j * planes[0]) @ minus
    e0 = np.array([1.0, 0.0, 0.0, 0.0], dtype=complex)
    e2 = np.array([0.0, 0.0, 1.0, 0.0], dtype=complex)
    errors["time_generator_positive_branch"] = float(np.linalg.norm(rest_hamiltonian @ e0 - CHI0 * e0))
    errors["time_generator_negative_branch"] = float(np.linalg.norm(rest_hamiltonian @ e2 + CHI0 * e2))
    errors["spin_generator_same_on_positive_branch"] = float(np.linalg.norm(spin0 @ e0 + 0.5 * e0))
    errors["spin_generator_same_on_negative_branch"] = float(np.linalg.norm(spin0 @ e2 + 0.5 * e2))

    wrong_period = 0.9 * ZITTER_PERIOD
    wrong_period_defect = min(
        float(np.linalg.norm(coherent_state(time + wrong_period) + coherent_state(time)))
        for time in sample_times
    )

    max_error = max(abs(value) for value in errors.values())
    checks = {
        "one_zitter_cycle_negates_the_spinor": all(
            errors[f"state_antiperiod_{index}"] <= TOLERANCE
            for index in range(len(sample_times))
        ),
        "two_zitter_cycles_return_the_spinor": all(
            errors[f"state_two_cycle_return_{index}"] <= TOLERANCE
            for index in range(len(sample_times))
        ),
        "density_velocity_and_position_return_after_one_cycle": all(
            errors[f"density_return_{index}"] <= TOLERANCE
            and errors[f"velocity_return_{index}"] <= TOLERANCE
            and errors[f"position_return_{index}"] <= TOLERANCE
            for index in range(len(sample_times))
        ),
        "two_pi_and_four_pi_bridges_hold_on_both_copies": all(
            errors[f"{copy_name}_{axis}_2pi_bridge_{index}"] <= TOLERANCE
            and errors[f"{copy_name}_{axis}_4pi_bridge_{index}"] <= TOLERANCE
            for copy_name in ("minus", "plus")
            for axis in range(3)
            for index in range(len(sample_times))
        ),
        "half_period_is_reversal_not_return": (
            errors["half_period_velocity_reversal"] <= TOLERANCE
            and half_period_return_defect > 1.0
        ),
        "time_and_spin_generators_remain_distinct": all(
            errors[name] <= TOLERANCE
            for name in (
                "time_generator_positive_branch",
                "time_generator_negative_branch",
                "spin_generator_same_on_positive_branch",
                "spin_generator_same_on_negative_branch",
            )
        ),
        "wrong_period_is_rejected": wrong_period_defect > 0.1,
    }
    checks = {name: bool(value) for name, value in checks.items()}
    result = {
        "schema_version": "1.0.0",
        "status": "PASS" if max_error <= TOLERANCE and all(checks.values()) else "FAIL",
        "tolerance": TOLERANCE,
        "max_error": max_error,
        "sample_count": len(sample_times),
        "native_values": {
            "rest_angular_frequency": CHI0,
            "zitter_angular_frequency": OMEGA_ZITTER,
            "zitter_period": ZITTER_PERIOD,
            "two_cycle_spinor_period": 2.0 * ZITTER_PERIOD,
            "c19_continuum_speed": C_EFF,
        },
        "checks": checks,
        "errors": errors,
        "negative_controls": {
            "half_period_return_defect": half_period_return_defect,
            "wrong_period_scale": 0.9,
            "wrong_period_min_antiperiod_defect": wrong_period_defect,
        },
        "claim": "independent reconstruction that one derived zitter observable cycle is a spinor antiperiod and matches the endpoint action of a cubical 2pi rotation on both copies",
        "excluded_claims": [
            "time evolution is identical to spatial rotation",
            "zitterbewegung is the physical cause of spin one-half",
            "exact finite-spacing localized-electron motion",
            "electromagnetic coupling or a magnetic g-factor",
        ],
    }
    output = Path(__file__).with_name("result.json")
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
