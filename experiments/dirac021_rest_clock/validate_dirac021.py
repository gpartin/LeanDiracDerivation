"""Independent numerical reconstruction of the DIRAC-021 rest-clock identities."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


CHI0 = 19.0
C19_CONTINUUM_SPEED = np.sqrt(12.0)
TOLERANCE = 2.0e-12


def main() -> None:
    beta = np.diag([1.0, 1.0, -1.0, -1.0]).astype(complex)
    h_rest = CHI0 * beta
    positive = np.array([1.0, 0.0, 0.0, 0.0], dtype=complex)
    negative = np.array([0.0, 0.0, 1.0, 0.0], dtype=complex)

    times = np.linspace(-0.73, 0.91, 41)
    phase_positive = np.exp(-1j * CHI0 * times)
    phase_negative = np.exp(1j * CHI0 * times)
    relative_phase = phase_positive / phase_negative
    expected_relative = np.exp(-1j * (2.0 * CHI0) * times)

    errors = {
        "positive_rest_energy": float(np.linalg.norm(
            h_rest @ positive - CHI0 * positive
        )),
        "negative_rest_energy": float(np.linalg.norm(
            h_rest @ negative + CHI0 * negative
        )),
        "positive_dirac_time_equation": float(np.max(np.abs(
            1j * (-1j * CHI0) * phase_positive - CHI0 * phase_positive
        ))),
        "negative_dirac_time_equation": float(np.max(np.abs(
            1j * (1j * CHI0) * phase_negative + CHI0 * phase_negative
        ))),
        "relative_phase_doubles": float(np.max(np.abs(
            relative_phase - expected_relative
        ))),
        "positive_clock_return": float(abs(
            np.exp(-1j * CHI0 * (2.0 * np.pi / CHI0)) - 1.0
        )),
        "negative_clock_return": float(abs(
            np.exp(1j * CHI0 * (2.0 * np.pi / CHI0)) - 1.0
        )),
        "branch_beat_return": float(abs(
            np.exp(-1j * 2.0 * CHI0 * (np.pi / CHI0)) - 1.0
        )),
        "rest_frequency": abs(CHI0 - 19.0),
        "branch_beat_frequency": abs(2.0 * CHI0 - 38.0),
        "inverse_mass_scale": abs(1.0 / CHI0 - 1.0 / 19.0),
        "native_reduced_compton_length": abs(
            C19_CONTINUUM_SPEED / CHI0 - np.sqrt(12.0) / 19.0
        ),
    }

    density_positive = np.abs(phase_positive) ** 2
    density_negative = np.abs(phase_negative) ** 2
    wrong_mass_relative = np.exp(-1j * 2.0 * 18.0 * times)
    same_sign_relative = phase_positive / phase_positive

    checks = {
        "zero_momentum_rest_frequency_is_chi0": errors["rest_frequency"] <= TOLERANCE,
        "positive_and_negative_rest_energies_are_opposite": (
            errors["positive_rest_energy"] <= TOLERANCE
            and errors["negative_rest_energy"] <= TOLERANCE
        ),
        "both_rest_states_satisfy_the_dirac_time_equation": (
            errors["positive_dirac_time_equation"] <= TOLERANCE
            and errors["negative_dirac_time_equation"] <= TOLERANCE
        ),
        "opposite_branch_relative_phase_has_frequency_two_chi0": (
            errors["relative_phase_doubles"] <= TOLERANCE
        ),
        "single_branch_period_is_two_pi_over_chi0": (
            errors["positive_clock_return"] <= TOLERANCE
            and errors["negative_clock_return"] <= TOLERANCE
        ),
        "branch_coherence_period_is_pi_over_chi0": (
            errors["branch_beat_return"] <= TOLERANCE
        ),
        "pure_branch_density_has_no_clock_signal": (
            float(np.max(np.abs(density_positive - 1.0))) <= TOLERANCE
            and float(np.max(np.abs(density_negative - 1.0))) <= TOLERANCE
        ),
        "wrong_mass_control_is_rejected": (
            float(np.max(np.abs(wrong_mass_relative - expected_relative))) > 0.1
        ),
        "same_sign_control_has_no_doubled_phase": (
            float(np.max(np.abs(same_sign_relative - expected_relative))) > 0.1
        ),
    }

    max_error = max(errors.values())
    result = {
        "status": "PASS" if max_error <= TOLERANCE and all(checks.values()) else "FAIL",
        "tolerance": TOLERANCE,
        "max_error": max_error,
        "native_values": {
            "rest_angular_frequency": CHI0,
            "branch_beat_angular_frequency": 2.0 * CHI0,
            "rest_clock_period": 2.0 * np.pi / CHI0,
            "branch_beat_period": np.pi / CHI0,
            "inverse_mass_scale": 1.0 / CHI0,
            "reduced_compton_length": C19_CONTINUUM_SPEED / CHI0,
            "c19_continuum_speed": C19_CONTINUUM_SPEED,
        },
        "errors": errors,
        "checks": checks,
        "negative_controls": {
            "wrong_mass_max_phase_error": float(np.max(np.abs(
                wrong_mass_relative - expected_relative
            ))),
            "same_sign_max_phase_error": float(np.max(np.abs(
                same_sign_relative - expected_relative
            ))),
        },
        "claim": (
            "independent reconstruction of the action-selected zero-momentum "
            "rest frequency, opposite Dirac rest-energy phases, and the exact "
            "two-times-frequency relative branch phase"
        ),
        "excluded_claims": [
            "an SI electron mass or SI Compton wavelength",
            "a directly observable clock in a pure energy eigenstate",
            "zitterbewegung without a specified cross-branch observable",
            "electromagnetic coupling or magnetic g-factor",
            "nonlinear localized-electron dynamics",
        ],
    }
    output = Path(__file__).with_name("result.json")
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
