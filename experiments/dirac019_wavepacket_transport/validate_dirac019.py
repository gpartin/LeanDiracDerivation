"""Independent numerical reconstruction of DIRAC-019 packet transport.

The formal theorem is momentum-space and linear.  This validator reconstructs
its identities from explicit matrices and also Fourier-synthesizes a narrow
real-space packet as a downstream diagnostic.  It does not simulate a
localized nonlinear electron or add a translation rule.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


CHI0 = 19.0
C19_SPEED_SQ = 12.0
C19_SPEED = np.sqrt(C19_SPEED_SQ)
TOLERANCE = 2.0e-11


def dirac_matrices() -> tuple[np.ndarray, list[np.ndarray]]:
    beta = np.diag([1.0, 1.0, -1.0, -1.0]).astype(complex)
    alpha_x = np.array([
        [0, 0, 0, 1], [0, 0, 1, 0],
        [0, 1, 0, 0], [1, 0, 0, 0],
    ], dtype=complex)
    alpha_y = np.array([
        [0, 0, 0, -1j], [0, 0, 1j, 0],
        [0, -1j, 0, 0], [1j, 0, 0, 0],
    ], dtype=complex)
    alpha_z = np.array([
        [0, 0, 1, 0], [0, 0, 0, -1],
        [1, 0, 0, 0], [0, -1, 0, 0],
    ], dtype=complex)
    velocities = [C19_SPEED * alpha_z, -C19_SPEED * alpha_y,
                  -C19_SPEED * alpha_x]
    return beta, velocities


def hamiltonian(momentum: np.ndarray) -> np.ndarray:
    beta, velocities = dirac_matrices()
    return CHI0 * beta + sum(
        momentum[axis] * velocities[axis] for axis in range(3)
    )


def frequency(momentum: np.ndarray) -> float:
    return float(np.sqrt(CHI0**2 + C19_SPEED_SQ * momentum @ momentum))


def recurrence_phase(dt: float, energy: float) -> complex:
    argument = dt * energy
    if argument**2 > 4.0:
        raise ValueError("outside the stable centered-recurrence branch")
    cosine = 1.0 - argument**2 / 2.0
    sine = argument * np.sqrt(1.0 - argument**2 / 4.0)
    return complex(cosine, -sine)


def positive_eigenspinor_x(momentum_x: float) -> np.ndarray:
    """Smooth positive-frequency eigenvector for momentum (kx,0,0)."""
    energy = float(np.sqrt(CHI0**2 + C19_SPEED_SQ * momentum_x**2))
    upper = np.sqrt((energy + CHI0) / (2.0 * energy))
    lower = C19_SPEED * momentum_x / np.sqrt(
        2.0 * energy * (energy + CHI0)
    )
    return np.array([upper, 0.0, lower, 0.0], dtype=complex)


def packet_center(field: np.ndarray) -> tuple[float, float]:
    density = np.sum(np.abs(field) ** 2, axis=1)
    positions = np.arange(field.shape[0], dtype=float)
    total = float(np.sum(density))
    return float(np.sum(positions * density) / total), total


def real_space_packet_control(dt: float, duration: float = 6.0) -> dict[str, float]:
    size = 512
    momenta = 2.0 * np.pi * np.fft.fftfreq(size)
    center_momentum = 0.55
    width = 0.055
    initial_center = size / 3.0
    envelope = (
        np.exp(-0.5 * ((momenta - center_momentum) / width) ** 2)
        * np.exp(-1j * momenta * initial_center)
    )
    spinors = np.stack([positive_eigenspinor_x(value) for value in momenta])
    coefficients = envelope[:, None] * spinors
    weights = np.sum(np.abs(coefficients) ** 2, axis=1)
    energies = np.sqrt(CHI0**2 + C19_SPEED_SQ * momenta**2)
    velocities = C19_SPEED_SQ * momenta / energies
    predicted_velocity = float(np.sum(weights * velocities) / np.sum(weights))

    steps = int(round(duration / dt))
    actual_duration = steps * dt
    phases = np.array([recurrence_phase(dt, value) for value in energies])
    initial = np.fft.ifft(coefficients, axis=0)
    advanced = np.fft.ifft(coefficients * phases[:, None] ** steps, axis=0)
    initial_position, initial_density = packet_center(initial)
    final_position, final_density = packet_center(advanced)
    measured_velocity = (final_position - initial_position) / actual_duration
    return {
        "dt": dt,
        "steps": steps,
        "duration": actual_duration,
        "predicted_velocity": predicted_velocity,
        "measured_velocity": measured_velocity,
        "velocity_error": abs(measured_velocity - predicted_velocity),
        "density_drift": abs(final_density - initial_density) / initial_density,
    }


def main() -> None:
    rng = np.random.default_rng(19019)
    _, velocity_operators = dirac_matrices()
    identity = np.eye(4, dtype=complex)
    errors: dict[str, float] = {}
    speed_margins: list[float] = []
    generator_ratios: list[float] = []

    for case in range(48):
        momentum = rng.uniform(-0.8, 0.8, size=3)
        energy = frequency(momentum)
        eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian(momentum))
        positive_indices = np.flatnonzero(eigenvalues > 0.0)
        dt = float(rng.uniform(0.002, 0.018))
        phase = recurrence_phase(dt, energy)
        errors[f"case_{case}_phase_norm"] = abs(abs(phase) ** 2 - 1.0)

        branch = (
            (1.0 - (dt * energy) ** 2 / 2.0) * identity
            - 1j * dt * np.sqrt(1.0 - (dt * energy) ** 2 / 4.0)
            * hamiltonian(momentum)
        )
        errors[f"case_{case}_branch_unitarity"] = float(
            np.linalg.norm(branch.conj().T @ branch - identity)
        )

        packet_density = 0.0
        advanced_density = 0.0
        packet_current = np.zeros(3)
        advanced_current = np.zeros(3)
        velocity_moment = np.zeros(3)
        for branch_index, eigen_index in enumerate(positive_indices):
            amplitude = eigenvectors[:, eigen_index]
            amplitude *= rng.normal() + 1j * rng.normal()
            density = float(np.vdot(amplitude, amplitude).real)
            evolved = np.linalg.matrix_power(branch, 37) @ amplitude
            packet_density += density
            advanced_density += float(np.vdot(evolved, evolved).real)
            for axis in range(3):
                packet_current[axis] += float(
                    np.vdot(amplitude, velocity_operators[axis] @ amplitude).real
                )
                advanced_current[axis] += float(
                    np.vdot(evolved, velocity_operators[axis] @ evolved).real
                )
                velocity_moment[axis] += (
                    C19_SPEED_SQ * momentum[axis] / energy * density
                )
            errors[f"case_{case}_branch_{branch_index}_eigen"] = float(
                np.linalg.norm(
                    hamiltonian(momentum) @ amplitude - energy * amplitude
                )
            )
        errors[f"case_{case}_all_step_density"] = abs(
            advanced_density - packet_density
        )
        errors[f"case_{case}_packet_current"] = float(
            np.linalg.norm(packet_current - velocity_moment)
        )
        errors[f"case_{case}_all_step_current"] = float(
            np.linalg.norm(advanced_current - velocity_moment)
        )

        velocity = C19_SPEED_SQ * momentum / energy
        speed_sq = float(velocity @ velocity)
        speed_margins.append(C19_SPEED_SQ - speed_sq)

        defects: list[float] = []
        for refined_dt in (dt, dt / 2.0, dt / 4.0):
            refined_phase = recurrence_phase(refined_dt, energy)
            defects.append(abs((refined_phase - 1.0) / refined_dt + 1j * energy))
        generator_ratios.extend([
            defects[0] / defects[1], defects[1] / defects[2]
        ])

    # These controls ensure the theorem's two essential hypotheses are doing
    # work rather than being decorative.
    unstable_rejected = False
    try:
        recurrence_phase(0.13, CHI0)
    except ValueError:
        unstable_rejected = True
    negative_values, negative_vectors = np.linalg.eigh(
        hamiltonian(np.array([0.3, -0.2, 0.1]))
    )
    negative_index = int(np.argmin(negative_values))
    negative_energy = float(negative_values[negative_index])
    positive_energy = frequency(np.array([0.3, -0.2, 0.1]))
    negative_as_positive_residual = float(np.linalg.norm(
        hamiltonian(np.array([0.3, -0.2, 0.1]))
        @ negative_vectors[:, negative_index]
        - positive_energy * negative_vectors[:, negative_index]
    ))

    packet_controls = [real_space_packet_control(dt) for dt in (0.02, 0.01, 0.005)]
    packet_errors = [entry["velocity_error"] for entry in packet_controls]
    max_identity_error = max(errors.values())
    checks = {
        "explicit_matrix_identities_within_tolerance": bool(
            max_identity_error <= TOLERANCE
        ),
        "all_step_current_is_preserved": bool(all(
            value <= TOLERANCE for name, value in errors.items()
            if name.endswith("all_step_current")
        )),
        "all_massive_modes_strictly_subluminal": bool(
            min(speed_margins) > 0.0
        ),
        "generator_defect_converges_first_order_or_better": bool(
            min(generator_ratios) > 1.95
        ),
        "unstable_timestep_is_rejected": unstable_rejected,
        "negative_frequency_control_is_not_admitted_as_positive": bool(
            negative_energy < 0.0 and negative_as_positive_residual > CHI0
        ),
        "fourier_packet_density_is_preserved": bool(
            max(entry["density_drift"] for entry in packet_controls) < 2.0e-13
        ),
        "fourier_packet_velocity_converges_to_current_prediction": bool(
            packet_errors[2] < packet_errors[1] < packet_errors[0]
            and packet_errors[2] < 1.0e-3
        ),
    }
    passed = all(checks.values())
    result = {
        "status": "PASS" if passed else "FAIL",
        "tolerance": TOLERANCE,
        "matrix_cases": 48,
        "max_identity_error": max_identity_error,
        "minimum_speed_margin": min(speed_margins),
        "minimum_generator_refinement_ratio": min(generator_ratios),
        "negative_frequency_control_residual": negative_as_positive_residual,
        "real_space_packet_controls": packet_controls,
        "checks": checks,
        "claim": (
            "independent reconstruction of stable all-step positive-frequency "
            "spectral-family norm preservation, density-weighted current/group velocity, "
            "strict massive speed bound, and recurrence-generator convergence"
        ),
        "excluded_claims": [
            "nonlinear localized electron transport",
            "native selection of a positive-frequency packet",
            "uniform finite-time continuum error bound",
            "exact native finite-spacing packet evolution",
            "unrestricted live-chi evolution",
            "electromagnetic motion or response",
        ],
    }
    output = Path(__file__).with_name("result.json")
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
