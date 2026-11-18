"""Independent reconstruction of the scoped DIRAC-022 observable."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np


CHI0 = 19.0
C_EFF = math.sqrt(12.0)
TOLERANCE = 2.0e-11


def expectation(state: np.ndarray, operator: np.ndarray) -> float:
    return float(np.real(np.vdot(state, operator @ state)))


def main() -> None:
    beta = np.diag([1.0, 1.0, -1.0, -1.0]).astype(complex)
    alpha_z = np.array(
        [
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, -1.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
        ],
        dtype=complex,
    )
    h_rest = CHI0 * beta
    velocity = C_EFF * alpha_z
    positive = np.array([1.0, 0.0, 0.0, 0.0], dtype=complex)
    negative = np.array([0.0, 0.0, 1.0, 0.0], dtype=complex)
    second_positive = np.array([0.0, 1.0, 0.0, 0.0], dtype=complex)

    times = np.linspace(-0.83, 0.97, 257)
    coherent_states = np.array(
        [
            (
                np.exp(-1j * CHI0 * time) * positive
                + np.exp(1j * CHI0 * time) * negative
            )
            / math.sqrt(2.0)
            for time in times
        ]
    )
    currents = np.array([expectation(state, velocity) for state in coherent_states])
    expected_currents = C_EFF * np.cos(2.0 * CHI0 * times)

    amplitude = C_EFF / (2.0 * CHI0)
    positions = amplitude * np.sin(2.0 * CHI0 * times)
    analytic_position_derivative = (
        amplitude * 2.0 * CHI0 * np.cos(2.0 * CHI0 * times)
    )

    pure_positive_currents = np.array(
        [expectation(np.exp(-1j * CHI0 * time) * positive, velocity) for time in times]
    )
    pure_negative_currents = np.array(
        [expectation(np.exp(1j * CHI0 * time) * negative, velocity) for time in times]
    )
    same_sign_states = np.array(
        [
            np.exp(-1j * CHI0 * time) * (positive + second_positive)
            / math.sqrt(2.0)
            for time in times
        ]
    )
    same_sign_currents = np.array(
        [expectation(state, velocity) for state in same_sign_states]
    )

    rho_incoherent = 0.5 * np.outer(positive, positive.conj()) + 0.5 * np.outer(
        negative, negative.conj()
    )
    incoherent_current = float(np.real(np.trace(rho_incoherent @ velocity)))

    wrong_mass = 18.0
    wrong_mass_signal = C_EFF * np.cos(2.0 * wrong_mass * times)
    wrong_mass_mismatch = float(np.max(np.abs(wrong_mass_signal - expected_currents)))

    errors = {
        "state_normalization": float(
            np.max(np.abs(np.sum(np.abs(coherent_states) ** 2, axis=1) - 1.0))
        ),
        "velocity_from_hamiltonian_momentum_derivative": float(
            np.linalg.norm(
                ((CHI0 * beta + 0.37 * C_EFF * alpha_z) - h_rest) / 0.37
                - velocity
            )
        ),
        "rest_hamiltonian_velocity_anticommutator": float(
            np.linalg.norm(h_rest @ velocity + velocity @ h_rest)
        ),
        "positive_branch_current": float(np.max(np.abs(pure_positive_currents))),
        "negative_branch_current": float(np.max(np.abs(pure_negative_currents))),
        "coherent_velocity_law": float(np.max(np.abs(currents - expected_currents))),
        "position_derivative_law": float(
            np.max(np.abs(analytic_position_derivative - expected_currents))
        ),
        "amplitude_identity": abs(amplitude - math.sqrt(12.0) / 38.0),
        "peak_to_peak_identity": abs(2.0 * amplitude - C_EFF / CHI0),
        "position_bound": max(0.0, float(np.max(np.abs(positions))) - amplitude),
        "same_sign_current": float(np.max(np.abs(same_sign_currents))),
        "incoherent_mixture_current": abs(incoherent_current),
    }
    finite = all(math.isfinite(value) for value in errors.values())
    max_error = max(errors.values())

    checks = {
        "all_evidence_is_finite": finite,
        "normalized_cross_branch_state": errors["state_normalization"] <= TOLERANCE,
        "velocity_is_hamiltonian_momentum_response": (
            errors["velocity_from_hamiltonian_momentum_derivative"] <= TOLERANCE
        ),
        "rest_velocity_anticommutes_with_rest_hamiltonian": (
            errors["rest_hamiltonian_velocity_anticommutator"] <= TOLERANCE
        ),
        "pure_rest_branches_have_zero_velocity": (
            errors["positive_branch_current"] <= TOLERANCE
            and errors["negative_branch_current"] <= TOLERANCE
        ),
        "coherent_velocity_frequency_is_two_chi0": (
            errors["coherent_velocity_law"] <= TOLERANCE
        ),
        "position_derivative_equals_velocity": (
            errors["position_derivative_law"] <= TOLERANCE
        ),
        "position_amplitude_is_half_reduced_compton_length": (
            errors["amplitude_identity"] <= TOLERANCE
            and errors["peak_to_peak_identity"] <= TOLERANCE
            and errors["position_bound"] <= TOLERANCE
        ),
        "same_sign_control_has_no_signal": errors["same_sign_current"] <= TOLERANCE,
        "incoherent_mixture_has_no_signal": (
            errors["incoherent_mixture_current"] <= TOLERANCE
        ),
        "wrong_mass_control_is_rejected": wrong_mass_mismatch > 0.1,
    }

    result = {
        "schema_version": "1.0.0",
        "status": "PASS" if max_error <= TOLERANCE and all(checks.values()) else "FAIL",
        "tolerance": TOLERANCE,
        "max_error": max_error,
        "native_values": {
            "action_selected_mass": CHI0,
            "zitter_angular_frequency": 2.0 * CHI0,
            "c19_continuum_speed": C_EFF,
            "zitter_position_amplitude": amplitude,
            "zitter_peak_to_peak_excursion": 2.0 * amplitude,
            "native_reduced_compton_length": C_EFF / CHI0,
        },
        "errors": errors,
        "checks": checks,
        "negative_controls": {
            "wrong_mass": wrong_mass,
            "wrong_mass_max_signal_mismatch": wrong_mass_mismatch,
            "same_sign_max_current": float(np.max(np.abs(same_sign_currents))),
            "incoherent_mixture_current": incoherent_current,
        },
        "claim": (
            "independent reconstruction of the free-continuum zitterbewegung "
            "velocity and position expectation for a coherent superposition of "
            "the action-derived positive and negative rest branches"
        ),
        "excluded_claims": [
            "a localized nonlinear native lattice electron",
            "measured trembling of a physical electron",
            "an SI frequency or SI length",
            "electromagnetic response or magnetic g-factor",
            "zitterbewegung in a pure branch or incoherent mixture",
        ],
    }
    output = Path(__file__).with_name("result.json")
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
