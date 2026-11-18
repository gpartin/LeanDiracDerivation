"""Independent validation of DIRAC-020 coherent spatial transport.

The Lean theorem is linear and uses an explicit normalized finite Fourier
frame.  This script reconstructs that frame numerically, checks Parseval
density and integrated-current identities, verifies the formal finite-update
pointwise error bound, and follows a narrow packet in real space.  It does not
simulate a nonlinear localized electron or unrestricted live-chi dynamics.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


CHI0 = 19.0
C19_SPEED_SQ = 12.0
C19_SPEED = np.sqrt(C19_SPEED_SQ)
TOLERANCE = 3.0e-11


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
    return beta, [C19_SPEED * alpha_z, -C19_SPEED * alpha_y,
                  -C19_SPEED * alpha_x]


def frequency(momentum_x: np.ndarray | float) -> np.ndarray | float:
    return np.sqrt(CHI0**2 + C19_SPEED_SQ * np.asarray(momentum_x) ** 2)


def recurrence_phase(dt: float, energy: np.ndarray | float) -> np.ndarray:
    argument = dt * np.asarray(energy)
    if np.any(argument**2 > 4.0):
        raise ValueError("outside the stable centered-recurrence branch")
    return (1.0 - argument**2 / 2.0) - 1j * argument * np.sqrt(
        1.0 - argument**2 / 4.0
    )


def positive_eigenspinor_x(momentum_x: np.ndarray) -> np.ndarray:
    energy = frequency(momentum_x)
    upper = np.sqrt((energy + CHI0) / (2.0 * energy))
    lower = C19_SPEED * momentum_x / np.sqrt(
        2.0 * energy * (energy + CHI0)
    )
    result = np.zeros((momentum_x.size, 4), dtype=complex)
    result[:, 0] = upper
    result[:, 2] = lower
    return result


def normalized_fourier_frame(size: int) -> np.ndarray:
    sites = np.arange(size, dtype=float)[:, None]
    modes = np.arange(size, dtype=float)[None, :]
    return np.exp(2j * np.pi * sites * modes / size) / np.sqrt(size)


def packet_center(field: np.ndarray) -> tuple[float, float]:
    density = np.sum(np.abs(field) ** 2, axis=1)
    positions = np.arange(field.shape[0], dtype=float)
    total = float(np.sum(density))
    return float(np.sum(positions * density) / total), total


def coherent_packet_case(dt: float, duration: float = 6.0) -> dict[str, float]:
    size = 256
    frame = normalized_fourier_frame(size)
    momenta = 2.0 * np.pi * np.fft.fftfreq(size)
    character_momentum_phase_error = float(np.max(np.abs(
        frame[1, :] / frame[0, :] - np.exp(1j * momenta)
    )))
    center_momentum = 0.55
    width = 0.055
    initial_center = size / 3.0
    envelope = (
        np.exp(-0.5 * ((momenta - center_momentum) / width) ** 2)
        * np.exp(-1j * momenta * initial_center)
    )
    spinors = positive_eigenspinor_x(momenta)
    coefficients = envelope[:, None] * spinors
    energies = frequency(momenta)
    phases = recurrence_phase(dt, energies)
    exact_phases = np.exp(-1j * dt * energies)
    steps = int(round(duration / dt))
    actual_duration = steps * dt

    initial = frame @ coefficients
    evolved_coefficients = coefficients * phases[:, None] ** steps
    evolved = frame @ evolved_coefficients
    exact = frame @ (coefficients * exact_phases[:, None] ** steps)

    beta, velocity_operators = dirac_matrices()
    del beta
    spectral_density = float(np.sum(np.abs(evolved_coefficients) ** 2))
    spatial_density = float(np.sum(np.abs(evolved) ** 2))
    spectral_current = float(sum(
        np.vdot(value, velocity_operators[0] @ value).real
        for value in evolved_coefficients
    ))
    spatial_current = float(sum(
        np.vdot(value, velocity_operators[0] @ value).real
        for value in evolved
    ))
    weights = np.sum(np.abs(coefficients) ** 2, axis=1)
    velocities = C19_SPEED_SQ * momenta / energies
    velocity_moment = float(np.sum(weights * velocities))
    predicted_velocity = velocity_moment / float(np.sum(weights))

    point_error = np.abs(evolved - exact)
    phase_error_bound = steps * np.abs(phases - exact_phases)
    component_bound = np.abs(frame) @ (
        phase_error_bound[:, None] * np.abs(coefficients)
    )
    maximum_bound_violation = float(np.max(point_error - component_bound))

    initial_position, initial_density = packet_center(initial)
    final_position, final_density = packet_center(evolved)
    measured_velocity = (final_position - initial_position) / actual_duration
    return {
        "dt": dt,
        "steps": steps,
        "duration": actual_duration,
        "fourier_orthonormality_error": float(np.linalg.norm(
            frame.conj().T @ frame - np.eye(size)
        )),
        "character_momentum_phase_error": character_momentum_phase_error,
        "parseval_error": abs(spatial_density - spectral_density),
        "integrated_current_error": abs(spatial_current - spectral_current),
        "current_velocity_moment_error": abs(spectral_current - velocity_moment),
        "maximum_finite_step_bound_violation": maximum_bound_violation,
        "predicted_velocity": predicted_velocity,
        "measured_velocity": measured_velocity,
        "velocity_error": abs(measured_velocity - predicted_velocity),
        "density_drift": abs(final_density - initial_density) / initial_density,
    }


def main() -> None:
    cases = [coherent_packet_case(dt) for dt in (0.02, 0.01, 0.005)]
    velocity_errors = [entry["velocity_error"] for entry in cases]

    # Negative controls: omitting Fourier normalization must break Parseval,
    # and a momentum-symmetric packet must have zero net transport even though
    # it still evolves.  Motion is therefore derived, not guaranteed by label.
    size = 32
    raw_frame = normalized_fourier_frame(size) * np.sqrt(size)
    rng = np.random.default_rng(20020)
    random_coefficients = rng.normal(size=(size, 4)) + 1j * rng.normal(
        size=(size, 4)
    )
    raw_parseval_ratio = float(
        np.sum(np.abs(raw_frame @ random_coefficients) ** 2)
        / np.sum(np.abs(random_coefficients) ** 2)
    )
    symmetric_momenta = np.array([-0.5, 0.5])
    symmetric_energies = frequency(symmetric_momenta)
    symmetric_velocity = C19_SPEED_SQ * symmetric_momenta / symmetric_energies
    symmetric_net_velocity = float(np.sum(symmetric_velocity))
    unstable_rejected = False
    try:
        recurrence_phase(0.13, CHI0)
    except ValueError:
        unstable_rejected = True

    checks = {
        "explicit_fourier_frame_is_orthonormal": bool(max(
            entry["fourier_orthonormality_error"] for entry in cases
        ) < TOLERANCE),
        "fourier_character_matches_mode_momentum": bool(max(
            entry["character_momentum_phase_error"] for entry in cases
        ) < TOLERANCE),
        "coherent_spatial_parseval_identity": bool(max(
            entry["parseval_error"] for entry in cases
        ) < TOLERANCE),
        "integrated_spatial_current_equals_spectral_current": bool(max(
            entry["integrated_current_error"] for entry in cases
        ) < TOLERANCE),
        "spectral_current_equals_density_weighted_velocity": bool(max(
            entry["current_velocity_moment_error"] for entry in cases
        ) < TOLERANCE),
        "finite_step_pointwise_error_bound_holds": bool(max(
            entry["maximum_finite_step_bound_violation"] for entry in cases
        ) < TOLERANCE),
        "coherent_packet_density_is_preserved": bool(max(
            entry["density_drift"] for entry in cases
        ) < 3.0e-13),
        "coherent_packet_velocity_converges": bool(
            velocity_errors[2] < velocity_errors[1] < velocity_errors[0]
            and velocity_errors[2] < 1.0e-3
        ),
        "unnormalized_frame_fails_parseval_control": bool(
            abs(raw_parseval_ratio - size) < 1.0e-10
            and abs(raw_parseval_ratio - 1.0) > 1.0
        ),
        "zero_velocity_moment_gives_no_transport_control": bool(
            abs(symmetric_net_velocity) < 1.0e-14
        ),
        "unstable_timestep_is_rejected": unstable_rejected,
    }
    passed = all(checks.values())
    result = {
        "status": "PASS" if passed else "FAIL",
        "tolerance": TOLERANCE,
        "coherent_packet_controls": cases,
        "negative_controls": {
            "raw_fourier_parseval_ratio": raw_parseval_ratio,
            "symmetric_net_velocity": symmetric_net_velocity,
            "unstable_timestep_rejected": unstable_rejected,
        },
        "checks": checks,
        "claim": (
            "independent reconstruction of normalized finite Fourier "
            "synthesis with character-momentum compatibility, coherent "
            "spatial Parseval density, integrated "
            "current/group-velocity transport, and the finite-update "
            "pointwise recurrence error bound"
        ),
        "excluded_claims": [
            "exact native finite-spacing Dirac evolution",
            "uniform fixed-physical-time analytic convergence rate",
            "nonlinear localized electron transport",
            "unrestricted live-chi evolution",
            "electron formation or identity",
            "electromagnetic response",
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
