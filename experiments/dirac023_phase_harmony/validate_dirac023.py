"""Independent reconstruction of the scoped DIRAC-023 identities."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np


CHI0 = 19.0
C_EFF_SQ = 12.0
C_EFF = math.sqrt(C_EFF_SQ)
TOLERANCE = 2.0e-11


def mode_frequency(momentum: np.ndarray) -> float:
    return math.sqrt(CHI0**2 + C_EFF_SQ * float(momentum @ momentum))


def transport_velocity(momentum: np.ndarray) -> np.ndarray:
    return C_EFF_SQ * momentum / mode_frequency(momentum)


def main() -> None:
    momenta = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.25, 0.0, 0.0],
            [-0.8, 0.3, 0.1],
            [2.0, -1.5, 0.7],
            [0.0, 0.6, -0.9],
            [4.0, 3.0, -2.0],
        ],
        dtype=float,
    )

    dispersion_errors: list[float] = []
    phase_rate_errors: list[float] = []
    energy_clock_errors: list[float] = []
    clock_speed_errors: list[float] = []
    directional_reciprocity_errors: list[float] = []
    subluminal_margins: list[float] = []

    for momentum in momenta:
        energy = mode_frequency(momentum)
        velocity = transport_velocity(momentum)
        speed_sq = float(velocity @ velocity)
        phase_rate = energy - float(momentum @ velocity)
        clock_ratio = phase_rate / CHI0
        energy_ratio = energy / CHI0

        dispersion_errors.append(
            abs(energy**2 - (CHI0**2 + C_EFF_SQ * float(momentum @ momentum)))
        )
        phase_rate_errors.append(abs(phase_rate - CHI0**2 / energy))
        energy_clock_errors.append(abs(energy_ratio * clock_ratio - 1.0))
        clock_speed_errors.append(
            abs(clock_ratio**2 - (1.0 - speed_sq / C_EFF_SQ))
        )
        subluminal_margins.append(C_EFF_SQ - speed_sq)

        for axis, component in enumerate(momentum):
            if component != 0.0:
                phase_velocity = energy / component
                directional_reciprocity_errors.append(
                    abs(velocity[axis] * phase_velocity - C_EFF_SQ)
                )

    zitter_amplitude = C_EFF / (2.0 * CHI0)
    zitter_angular_frequency = 2.0 * CHI0
    zitter_scale_lock_error = abs(
        zitter_amplitude * zitter_angular_frequency - C_EFF
    )
    rest_phase_rate_error = abs(
        mode_frequency(momenta[0])
        - float(momenta[0] @ transport_velocity(momenta[0]))
        - CHI0
    )

    wrong_transport_scale = 0.9
    wrong_transport_defects: list[float] = []
    for momentum in momenta[1:]:
        energy = mode_frequency(momentum)
        wrong_velocity = (
            wrong_transport_scale * C_EFF_SQ * momentum / energy
        )
        for axis, component in enumerate(momentum):
            if component != 0.0:
                wrong_transport_defects.append(
                    abs(wrong_velocity[axis] * (energy / component) - C_EFF_SQ)
                )

    wrong_rest_frequency = 18.0
    wrong_rest_reciprocity_defects: list[float] = []
    for momentum in momenta:
        energy = mode_frequency(momentum)
        velocity = transport_velocity(momentum)
        phase_rate = energy - float(momentum @ velocity)
        wrong_rest_reciprocity_defects.append(
            abs((energy / wrong_rest_frequency) *
                (phase_rate / wrong_rest_frequency) - 1.0)
        )

    errors = {
        "dispersion_identity": max(dispersion_errors),
        "packet_center_phase_rate": max(phase_rate_errors),
        "energy_clock_reciprocity": max(energy_clock_errors),
        "relativistic_clock_factor": max(clock_speed_errors),
        "directional_group_phase_reciprocity": max(
            directional_reciprocity_errors
        ),
        "rest_phase_rate": rest_phase_rate_error,
        "zitter_amplitude_frequency_lock": zitter_scale_lock_error,
    }
    finite = all(math.isfinite(value) for value in errors.values())
    max_error = max(errors.values())
    min_subluminal_margin = min(subluminal_margins)
    wrong_transport_max_defect = max(wrong_transport_defects)
    wrong_rest_min_defect = min(wrong_rest_reciprocity_defects)

    checks = {
        "all_evidence_is_finite": finite,
        "dispersion_is_inherited": errors["dispersion_identity"] <= TOLERANCE,
        "packet_center_phase_rate_is_mass_square_over_energy": (
            errors["packet_center_phase_rate"] <= TOLERANCE
        ),
        "energy_and_clock_ratios_are_reciprocal": (
            errors["energy_clock_reciprocity"] <= TOLERANCE
        ),
        "clock_slowdown_matches_speed_factor": (
            errors["relativistic_clock_factor"] <= TOLERANCE
        ),
        "directional_group_phase_product_is_c_eff_square": (
            errors["directional_group_phase_reciprocity"] <= TOLERANCE
        ),
        "rest_control_returns_action_clock": (
            errors["rest_phase_rate"] <= TOLERANCE
        ),
        "all_sampled_modes_are_subluminal": min_subluminal_margin > 0.0,
        "zitter_amplitude_times_frequency_is_c_eff": (
            errors["zitter_amplitude_frequency_lock"] <= TOLERANCE
        ),
        "wrong_transport_scale_is_rejected": wrong_transport_max_defect > 0.1,
        "wrong_rest_frequency_is_rejected": wrong_rest_min_defect > 0.1,
    }
    checks = {name: bool(value) for name, value in checks.items()}

    result = {
        "schema_version": "1.0.0",
        "status": (
            "PASS" if max_error <= TOLERANCE and all(checks.values()) else "FAIL"
        ),
        "tolerance": TOLERANCE,
        "max_error": max_error,
        "sample_count": len(momenta),
        "native_values": {
            "action_selected_rest_frequency": CHI0,
            "c19_continuum_speed_squared": C_EFF_SQ,
            "c19_continuum_speed": C_EFF,
            "zitter_angular_frequency": zitter_angular_frequency,
            "zitter_amplitude": zitter_amplitude,
            "zitter_amplitude_frequency_product": (
                zitter_amplitude * zitter_angular_frequency
            ),
        },
        "errors": errors,
        "checks": checks,
        "diagnostics": {
            "minimum_sampled_c_eff_squared_minus_speed_squared": (
                min_subluminal_margin
            ),
        },
        "negative_controls": {
            "wrong_transport_scale": wrong_transport_scale,
            "wrong_transport_max_reciprocity_defect": wrong_transport_max_defect,
            "wrong_rest_frequency": wrong_rest_frequency,
            "wrong_rest_min_energy_clock_defect": wrong_rest_min_defect,
        },
        "claim": (
            "independent reconstruction of continuum directional group/phase "
            "velocity reciprocity and packet-center clock dilation from the "
            "already-derived C19 dispersion, current velocity, and rest clock"
        ),
        "excluded_claims": [
            "an exact finite-spacing phase-harmony identity",
            "an SI-calibrated electron clock",
            "a localized nonlinear native lattice electron",
            "unrestricted live-chi localized-electron motion",
            "electromagnetic response or magnetic g-factor",
            "zitterbewegung physically causes spin one-half",
        ],
    }
    output = Path(__file__).with_name("result.json")
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
