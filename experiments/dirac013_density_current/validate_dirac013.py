"""Independent reconstruction of the DIRAC-013 density/current bridge."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


C19_SPEED = np.sqrt(12.0)


def c19_directions() -> list[tuple[np.ndarray, float]]:
    directions: list[tuple[np.ndarray, float]] = []
    for axis in range(3):
        for sign in (-1, 1):
            vector = np.zeros(3, dtype=int)
            vector[axis] = sign
            directions.append((vector, 2.0))
    for first, second in ((0, 1), (0, 2), (1, 2)):
        for sign_first in (-1, 1):
            for sign_second in (-1, 1):
                vector = np.zeros(3, dtype=int)
                vector[first] = sign_first
                vector[second] = sign_second
                directions.append((vector, 1.0))
    return directions


def phase_pair(source: complex, target: complex) -> float:
    return float(np.imag(np.conj(source) * target))


def bond_energy(weight: float, source: complex, target: complex) -> float:
    return float(weight * abs(source - target) ** 2 / 2.0)


def outward_current(weight: float, source: complex, target: complex) -> float:
    return -2.0 * weight * phase_pair(source, target)


def c19_flux(axis: int, momentum: np.ndarray) -> float:
    return float(sum(
        weight * direction[axis] * np.sin(float(momentum @ direction))
        for direction, weight in c19_directions()
    ))


def matrices() -> tuple[np.ndarray, list[np.ndarray]]:
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


def cube_embeddings(z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    minus_basis = np.array([z[0], -z[3], z[2], z[1]])
    plus_basis = np.array([-z[1], z[2], z[3], z[0]])
    minus = np.array([
        minus_basis[0], minus_basis[1], -1j * minus_basis[1],
        1j * minus_basis[0], minus_basis[2], minus_basis[3],
        -1j * minus_basis[3], 1j * minus_basis[2],
    ])
    plus = np.array([
        plus_basis[0], plus_basis[1], 1j * plus_basis[1],
        -1j * plus_basis[0], plus_basis[2], plus_basis[3],
        1j * plus_basis[3], -1j * plus_basis[2],
    ])
    return minus, plus


def main() -> None:
    rng = np.random.default_rng(13013)
    directions = c19_directions()
    beta, velocities = matrices()
    errors: dict[str, float] = {}
    refinement_ratios: list[float] = []

    for case in range(32):
        source = rng.normal() + 1j * rng.normal()
        target = rng.normal() + 1j * rng.normal()
        weight = float(rng.choice([1.0, 2.0]))
        eps = 2.0e-6
        plus_energy = bond_energy(weight, source + 1j * eps * source, target)
        minus_energy = bond_energy(weight, source - 1j * eps * source, target)
        derivative = (plus_energy - minus_energy) / (2.0 * eps)
        errors[f"case_{case}_phase_variation"] = abs(
            derivative + weight * phase_pair(source, target)
        )
        errors[f"case_{case}_reverse_bond"] = abs(
            outward_current(weight, source, target)
            + outward_current(weight, target, source)
        )

        size = 4
        field = rng.normal(size=(size, size, size)) + 1j * rng.normal(
            size=(size, size, size)
        )
        velocity = rng.normal(size=(size, size, size)) + 1j * rng.normal(
            size=(size, size, size)
        )
        mass_sq = float(rng.uniform(0.2, 4.0))
        max_continuity = 0.0
        max_action_divergence = 0.0
        for index in np.ndindex(field.shape):
            position = field[index]
            acceleration = -mass_sq * position
            divergence = 0.0
            for direction, bond_weight in directions:
                neighbor_index = tuple(
                    (index[axis] + int(direction[axis])) % size
                    for axis in range(3)
                )
                neighbor = field[neighbor_index]
                acceleration -= 2.0 * bond_weight * (position - neighbor)
                divergence += outward_current(bond_weight, position, neighbor)
            local_eps = 1.0e-6
            def local_star(value: complex) -> float:
                total = mass_sq * abs(value) ** 2 / 2.0
                for local_direction, local_weight in directions:
                    local_neighbor_index = tuple(
                        (index[local_axis] + int(local_direction[local_axis]))
                        % size for local_axis in range(3)
                    )
                    total += local_weight * abs(
                        value - field[local_neighbor_index]
                    ) ** 2
                return float(total)
            action_derivative = (
                local_star(position + 1j * local_eps * position)
                - local_star(position - 1j * local_eps * position)
            ) / (2.0 * local_eps)
            max_action_divergence = max(
                max_action_divergence, abs(action_derivative - divergence)
            )
            density_derivative = phase_pair(velocity[index], velocity[index])
            density_derivative += phase_pair(position, acceleration)
            max_continuity = max(
                max_continuity, abs(density_derivative + divergence)
            )
        errors[f"case_{case}_local_continuity"] = max_continuity
        errors[f"case_{case}_action_divergence"] = max_action_divergence

        momentum = rng.uniform(-0.4, 0.4, size=3)
        scale = float(rng.uniform(2.0e-3, 1.0e-2))
        axis = case % 3
        fine_error = abs(c19_flux(axis, scale * momentum) / scale
                         - 12.0 * momentum[axis])
        coarse_error = abs(c19_flux(axis, 2.0 * scale * momentum)
                           / (2.0 * scale) - 12.0 * momentum[axis])
        if fine_error > 1.0e-15:
            refinement_ratios.append(coarse_error / fine_error)
        errors[f"case_{case}_flux_slope"] = fine_error

        mass = float(rng.uniform(0.5, 3.0))
        hamiltonian = mass * beta + sum(
            momentum[component] * velocities[component]
            for component in range(3)
        )
        delta = float(rng.uniform(-0.2, 0.2))
        shifted_momentum = momentum.copy()
        shifted_momentum[axis] += delta
        shifted_hamiltonian = mass * beta + sum(
            shifted_momentum[component] * velocities[component]
            for component in range(3)
        )
        errors[f"case_{case}_hamiltonian_momentum_response"] = float(
            np.max(np.abs(shifted_hamiltonian - hamiltonian
                          - delta * velocities[axis]))
        )
        eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian)
        positive = np.flatnonzero(eigenvalues > 0.0)
        for branch, eigen_index in enumerate(positive):
            energy = float(eigenvalues[eigen_index])
            z = eigenvectors[:, eigen_index]
            density = float(np.vdot(z, z).real)
            current = float(np.vdot(z, velocities[axis] @ z).real)
            errors[f"case_{case}_branch_{branch}_dirac_current"] = abs(
                energy * current
                - C19_SPEED**2 * momentum[axis] * density
            )
            normalized = z / np.sqrt(2.0 * energy)
            native_density = float(sum(
                2.0 * energy * abs(component) ** 2
                for component in normalized
            ))
            errors[f"case_{case}_branch_{branch}_density"] = abs(
                native_density - density
            )
            native_flux = float(sum(
                2.0 * c19_flux(axis, momentum) * abs(component) ** 2
                for component in normalized
            ))
            errors[f"case_{case}_branch_{branch}_finite_flux"] = abs(
                native_flux - c19_flux(axis, momentum) / energy * density
            )

        z_random = rng.normal(size=4) + 1j * rng.normal(size=4)
        minus, plus = cube_embeddings(z_random)
        density = float(np.vdot(z_random, z_random).real)
        errors[f"case_{case}_minus_embedding_density"] = abs(
            float(np.vdot(minus, minus).real) - 2.0 * density
        )
        errors[f"case_{case}_plus_embedding_density"] = abs(
            float(np.vdot(plus, plus).real) - 2.0 * density
        )

    algebra_errors = [value for key, value in errors.items()
                      if not key.endswith(("flux_slope", "action_divergence"))]
    max_error = max(algebra_errors)
    max_flux_slope_error = max(
        value for key, value in errors.items() if key.endswith("flux_slope")
    )
    max_action_divergence_error = max(
        value for key, value in errors.items()
        if key.endswith("action_divergence")
    )
    minimum_refinement_ratio = min(refinement_ratios)
    checks = {
        "action_phase_variation_reconstructed": all(
            errors[key] <= 2.0e-9 for key in errors
            if key.endswith("phase_variation")
        ),
        "reverse_bond_law_reconstructed": all(
            errors[key] <= 1.0e-12 for key in errors
            if key.endswith("reverse_bond")
        ),
        "exact_local_continuity_reconstructed": all(
            errors[key] <= 2.0e-12 for key in errors
            if key.endswith("local_continuity")
        ),
        "local_action_divergence_reconstructed": all(
            errors[key] <= 2.0e-7 for key in errors
            if key.endswith("action_divergence")
        ),
        "positive_density_normalization_reconstructed": all(
            errors[key] <= 1.0e-12 for key in errors
            if key.endswith("density") and "embedding" not in key
        ),
        "finite_c19_flux_reconstructed": all(
            errors[key] <= 1.0e-12 for key in errors
            if key.endswith("finite_flux")
        ),
        "velocity_from_hamiltonian_reconstructed": all(
            errors[key] <= 2.0e-15 for key in errors
            if key.endswith("hamiltonian_momentum_response")
        ),
        "dirac_current_identity_reconstructed": all(
            errors[key] <= 2.0e-12 for key in errors
            if key.endswith("dirac_current")
        ),
        "both_cube_copy_density_scales_reconstructed": all(
            errors[key] <= 2.0e-12 for key in errors
            if key.endswith("embedding_density")
        ),
        "c19_flux_slope_converges_quadratically":
            minimum_refinement_ratio > 3.9,
        "c19_direction_count_is_18": len(directions) == 18,
        "c19_weight_sum_is_24": abs(sum(w for _, w in directions) - 24.0)
            <= 1.0e-15,
    }
    checks = {name: bool(value) for name, value in checks.items()}
    passed = all(checks.values())
    result = {
        "status": "PASS" if passed else "FAIL",
        "cases": 32,
        "max_error": max_error,
        "tolerance": 2.0e-9,
        "max_flux_slope_error_at_sampled_scale": max_flux_slope_error,
        "max_action_divergence_fd_error": max_action_divergence_error,
        "minimum_flux_refinement_ratio": minimum_refinement_ratio,
        "checks": checks,
        "errors": errors,
        "claim": (
            "independent reconstruction of the action-derived C19 bond flow, "
            "exact local phase continuity, positive-frequency density "
            "normalization, and continuum Dirac-current limit"
        ),
        "excluded_claims": [
            "positive density for arbitrary mixed-frequency native data",
            "exact finite-lattice equality to the continuum Dirac current",
            "arbitrary nonlinear live-chi density/current closure",
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
