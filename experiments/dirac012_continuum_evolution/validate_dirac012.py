"""Independent numerical reconstruction of the DIRAC-012 bounds."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


CHI0 = 19.0


def c19_directions() -> list[tuple[np.ndarray, float]]:
    directions: list[tuple[np.ndarray, float]] = []
    for axis in range(3):
        for sign in (-1, 1):
            vector = np.zeros(3)
            vector[axis] = sign
            directions.append((vector, 2.0))
    for a, b in ((0, 1), (0, 2), (1, 2)):
        for sa in (-1, 1):
            for sb in (-1, 1):
                vector = np.zeros(3)
                vector[a], vector[b] = sa, sb
                directions.append((vector, 1.0))
    return directions


def c19_transport(momentum: np.ndarray) -> float:
    return float(sum(
        weight * abs(1.0 - np.exp(1j * float(momentum @ direction))) ** 2
        for direction, weight in c19_directions()
    ))


def quadratic_transport(momentum: np.ndarray) -> float:
    return 12.0 * float(momentum @ momentum)


def local_site_potential(rho_psi: float) -> float:
    rho_chi = CHI0**2
    return (
        (rho_chi - CHI0**2) ** 2 / 4.0
        + rho_chi * rho_psi / 2.0
        + rho_psi**2 / 4.0
    )


def main() -> None:
    rng = np.random.default_rng(12015)
    errors: dict[str, float] = {}
    bound_margins: list[float] = []
    normalized_bound_margins: list[float] = []
    convergence_ratios: list[float] = []
    normalized_convergence_ratios: list[float] = []

    for case in range(32):
        momentum = rng.uniform(-0.18, 0.18, size=3)
        radius = float(np.max(np.abs(momentum)))
        dt = float(rng.uniform(0.004, 0.025))
        frequency = float(np.sqrt(CHI0**2 + quadratic_transport(momentum)))
        assert 2.0 * radius <= 1.0
        assert abs(dt * frequency) <= 1.0

        exact_transport = c19_transport(momentum)
        spatial_remainder = exact_transport - quadratic_transport(momentum)
        time_remainder = (
            2.0 * (1.0 - np.cos(dt * frequency))
            - (dt * frequency) ** 2
        )
        residual = (
            2.0 * (1.0 - np.cos(dt * frequency))
            - dt**2 * (CHI0**2 + exact_transport)
        )
        split = time_remainder - dt**2 * spatial_remainder
        bound = (
            (5.0 / 48.0) * abs(dt * frequency) ** 4
            + dt**2 * 40.0 * radius**4
        )

        amplitude = rng.normal() + 1j * rng.normal()
        phase_defect = (
            np.exp(-1j * dt * frequency) * amplitude
            - 2.0 * amplitude
            + np.exp(1j * dt * frequency) * amplitude
            + dt**2 * (CHI0**2 + exact_transport) * amplitude
        )

        eps = float(rng.uniform(1.0e-4, 0.15))
        rho = eps**2 * abs(amplitude) ** 2
        actual_modal_potential = (
            local_site_potential(rho)
            + eps**2 * abs(amplitude) ** 2 * exact_transport / 2.0
        )
        derived_modal_potential = (
            eps**2 * abs(amplitude) ** 2
            * (CHI0**2 + exact_transport) / 2.0
            + eps**4 * abs(amplitude) ** 4 / 4.0
        )

        errors[f"case_{case}_residual_split"] = abs(residual - split)
        errors[f"case_{case}_centered_recurrence"] = abs(
            phase_defect + residual * amplitude
        )
        errors[f"case_{case}_action_expansion"] = abs(
            actual_modal_potential - derived_modal_potential
        )
        bound_margins.append(bound - abs(residual))
        normalized_bound = (
            (5.0 / 48.0) * abs(dt * frequency) ** 4 / dt**2
            + 40.0 * radius**4
        )
        normalized_bound_margins.append(
            normalized_bound - abs(residual / dt**2)
        )

        half_momentum = momentum / 2.0
        half_dt = dt / 2.0
        half_frequency = float(np.sqrt(
            CHI0**2 + quadratic_transport(half_momentum)
        ))
        half_residual = abs(
            2.0 * (1.0 - np.cos(half_dt * half_frequency))
            - half_dt**2 * (CHI0**2 + c19_transport(half_momentum))
        )
        if half_residual > 1.0e-18:
            convergence_ratios.append(abs(residual) / half_residual)
            normalized_convergence_ratios.append(
                abs(residual / dt**2) /
                abs(half_residual / half_dt**2)
            )

    max_error = max(errors.values())
    minimum_bound_margin = min(bound_margins)
    minimum_normalized_bound_margin = min(normalized_bound_margins)
    minimum_refinement_ratio = min(convergence_ratios)
    minimum_normalized_refinement_ratio = min(normalized_convergence_ratios)
    checks = {
        "identity_error_within_tolerance": bool(max_error <= 1.0e-11),
        "every_proved_bound_holds": bool(
            minimum_bound_margin >= -1.0e-14
            and minimum_normalized_bound_margin >= -1.0e-12
        ),
        "residual_decreases_under_joint_refinement": bool(
            minimum_refinement_ratio > 8.0
        ),
        "normalized_equation_error_decreases_under_joint_refinement": bool(
            minimum_normalized_refinement_ratio > 3.0
        ),
        "c19_direction_count_is_18": len(c19_directions()) == 18,
        "c19_weight_sum_is_24":
            abs(sum(weight for _, weight in c19_directions()) - 24.0) <= 1.0e-15,
    }
    passed = all(checks.values())
    result = {
        "status": "PASS" if passed else "FAIL",
        "max_error": max_error,
        "tolerance": 1.0e-11,
        "cases": 32,
        "minimum_bound_margin": minimum_bound_margin,
        "minimum_normalized_bound_margin": minimum_normalized_bound_margin,
        "minimum_refinement_ratio": minimum_refinement_ratio,
        "minimum_normalized_refinement_ratio":
            minimum_normalized_refinement_ratio,
        "checks": checks,
        "errors": errors,
        "claim": (
            "independent reconstruction of the action-derived tangent "
            "coefficient, centered native recurrence, and combined explicit "
            "C19 spatial plus finite-timestep continuum bound"
        ),
        "excluded_claims": [
            "exact finite-lattice Dirac evolution",
            "Dirac density or current",
            "physical spin half",
            "localized electron identity",
        ],
    }
    output = Path(__file__).with_name("result.json")
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
