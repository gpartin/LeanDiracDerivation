"""Independent matrix reconstruction of the DIRAC-025 velocity-clock identity."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


TOLERANCE = 5.0e-11
CHI0 = 19.0
C_EFF = np.sqrt(12.0)


def standard_matrices() -> tuple[np.ndarray, list[np.ndarray]]:
    """Return beta and the three standard alpha matrices used downstream."""
    beta = np.diag([1.0, 1.0, -1.0, -1.0]).astype(complex)
    alpha_x = np.array(
        [[0, 0, 0, 1], [0, 0, 1, 0],
         [0, 1, 0, 0], [1, 0, 0, 0]], dtype=complex
    )
    alpha_y = np.array(
        [[0, 0, 0, -1j], [0, 0, 1j, 0],
         [0, -1j, 0, 0], [1j, 0, 0, 0]], dtype=complex
    )
    alpha_z = np.array(
        [[0, 0, 1, 0], [0, 0, 0, -1],
         [1, 0, 0, 0], [0, -1, 0, 0]], dtype=complex
    )
    return beta, [alpha_z, -alpha_y, -alpha_x]


def analyze(momentum: np.ndarray) -> tuple[dict[str, float], int]:
    beta, velocity_axes = standard_matrices()
    magnitude = float(np.linalg.norm(momentum))
    direction = momentum / magnitude
    hamiltonian = CHI0 * beta
    longitudinal_velocity = np.zeros((4, 4), dtype=complex)
    for axis in range(3):
        hamiltonian += C_EFF * momentum[axis] * velocity_axes[axis]
        longitudinal_velocity += C_EFF * direction[axis] * velocity_axes[axis]

    expected_energy = np.sqrt(CHI0**2 + C_EFF**2 * magnitude**2)
    expected_mean = C_EFF**2 * magnitude / expected_energy
    expected_clock = CHI0 / expected_energy
    expected_variance = C_EFF**2 * expected_clock**2
    expected_speed_sq = expected_mean**2

    eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian)
    positive_indices = np.where(eigenvalues > 0)[0]
    errors: dict[str, float] = {
        "operator_square": float(
            np.linalg.norm(
                longitudinal_velocity @ longitudinal_velocity
                - C_EFF**2 * np.eye(4)
            )
        ),
        "positive_energy_spectrum": float(
            np.max(np.abs(eigenvalues[positive_indices] - expected_energy))
        ),
    }
    for spin_index, eigen_index in enumerate(positive_indices):
        state = eigenvectors[:, eigen_index]
        density = float(np.vdot(state, state).real)
        mean = float(np.vdot(state, longitudinal_velocity @ state).real / density)
        second = float(
            np.vdot(
                state,
                longitudinal_velocity @ longitudinal_velocity @ state,
            ).real / density
        )
        variance = second - mean**2
        rms_ratio = np.sqrt(max(variance, 0.0)) / C_EFF
        speed_fraction_sq = mean**2 / C_EFF**2
        errors[f"spin_{spin_index}_mean"] = abs(mean - expected_mean)
        errors[f"spin_{spin_index}_second_moment"] = abs(second - C_EFF**2)
        errors[f"spin_{spin_index}_variance"] = abs(variance - expected_variance)
        errors[f"spin_{spin_index}_rms_clock"] = abs(rms_ratio - expected_clock)
        errors[f"spin_{spin_index}_pythagorean"] = abs(
            rms_ratio**2 + speed_fraction_sq - 1.0
        )
        errors[f"spin_{spin_index}_speed"] = abs(
            speed_fraction_sq - expected_speed_sq / C_EFF**2
        )
    return errors, len(positive_indices)


def momentum_for_speed_fraction(beta: float) -> np.ndarray:
    gamma_inverse = np.sqrt(1.0 - beta**2)
    magnitude = CHI0 * beta / (C_EFF * gamma_inverse)
    direction = np.array([2.0, -3.0, 6.0]) / 7.0
    return magnitude * direction


def main() -> None:
    momenta = [
        np.array([0.11, -0.27, 0.43]),
        np.array([1.0, 2.0, 3.0]),
        np.array([-4.0, 0.5, 2.25]),
        np.array([7.0, -3.0, -5.0]),
        np.array([0.03, 0.07, -0.02]),
    ]
    errors: dict[str, float] = {}
    spin_state_count = 0
    for sample_index, momentum in enumerate(momenta):
        sample_errors, states = analyze(momentum)
        spin_state_count += states
        for name, value in sample_errors.items():
            errors[f"sample_{sample_index}_{name}"] = value

    checkpoints: dict[str, dict[str, float]] = {}
    for label, beta_value, expected_clock in (
        ("three_four_five", 3.0 / 5.0, 4.0 / 5.0),
        ("four_three_five", 4.0 / 5.0, 3.0 / 5.0),
        ("sqrt_three_half", np.sqrt(3.0) / 2.0, 1.0 / 2.0),
    ):
        momentum = momentum_for_speed_fraction(beta_value)
        energy = np.sqrt(CHI0**2 + C_EFF**2 * np.dot(momentum, momentum))
        clock = CHI0 / energy
        checkpoints[label] = {
            "speed_fraction": float(beta_value),
            "expected_clock_ratio": float(expected_clock),
            "observed_clock_ratio": float(clock),
            "error": float(abs(clock - expected_clock)),
        }
        errors[f"checkpoint_{label}"] = abs(clock - expected_clock)

    beta, velocity_axes = standard_matrices()
    control_momentum = np.array([1.0, 2.0, -2.0])
    magnitude = float(np.linalg.norm(control_momentum))
    raw_velocity = sum(
        C_EFF * control_momentum[axis] * velocity_axes[axis]
        for axis in range(3)
    )
    unnormalized_direction_defect = float(
        np.linalg.norm(raw_velocity @ raw_velocity - C_EFF**2 * np.eye(4))
    )
    direction = control_momentum / magnitude
    transverse = np.array([-direction[1], direction[0], 0.0])
    transverse /= np.linalg.norm(transverse)
    hamiltonian = CHI0 * beta + sum(
        C_EFF * control_momentum[axis] * velocity_axes[axis]
        for axis in range(3)
    )
    eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian)
    state = eigenvectors[:, np.where(eigenvalues > 0)[0][0]]
    transverse_velocity = sum(
        C_EFF * transverse[axis] * velocity_axes[axis]
        for axis in range(3)
    )
    transverse_mean = abs(float(np.vdot(state, transverse_velocity @ state).real))
    longitudinal_expected = C_EFF**2 * magnitude / np.sqrt(
        CHI0**2 + C_EFF**2 * magnitude**2
    )
    wrong_mass_clock_error = abs(
        18.0 / np.sqrt(18.0**2 + C_EFF**2 * magnitude**2)
        - CHI0 / np.sqrt(CHI0**2 + C_EFF**2 * magnitude**2)
    )

    max_error = max(errors.values())
    negative_controls = {
        "unnormalized_direction_operator_square_defect": unnormalized_direction_defect,
        "transverse_direction_longitudinal_relation_defect": abs(
            transverse_mean - longitudinal_expected
        ),
        "wrong_rest_mass_clock_defect": wrong_mass_clock_error,
    }
    checks = {
        "arbitrary_three_direction_momenta_pass": max_error <= TOLERANCE,
        "both_positive_energy_spin_states_pass": spin_state_count == 2 * len(momenta),
        "known_relativistic_ratios_pass": all(
            checkpoint["error"] <= TOLERANCE
            for checkpoint in checkpoints.values()
        ),
        "unnormalized_direction_is_rejected": unnormalized_direction_defect > 1.0,
        "transverse_substitution_is_rejected": (
            abs(transverse_mean - longitudinal_expected) > 0.1
        ),
        "wrong_rest_mass_is_rejected": wrong_mass_clock_error > 1.0e-3,
    }
    checks = {name: bool(value) for name, value in checks.items()}
    result = {
        "schema_version": "1.0.0",
        "status": "PASS" if max_error <= TOLERANCE and all(checks.values()) else "FAIL",
        "tolerance": TOLERANCE,
        "max_error": max_error,
        "sample_count": len(momenta),
        "spin_state_count": spin_state_count,
        "native_values": {
            "rest_angular_frequency": CHI0,
            "c19_continuum_speed": C_EFF,
            "c19_continuum_speed_squared": C_EFF**2,
        },
        "checks": checks,
        "errors": errors,
        "known_checkpoints": checkpoints,
        "negative_controls": negative_controls,
        "claim": (
            "independent reconstruction that the longitudinal velocity "
            "variance of each positive-frequency continuum mode encodes the "
            "same dimensionless factor as its moving phase clock"
        ),
        "excluded_claims": [
            "a classical hidden path moving at c_eff",
            "an exact finite-spacing native velocity identity",
            "a localized nonlinear electron",
            "an SI calibration",
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
