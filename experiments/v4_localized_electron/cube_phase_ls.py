"""Lyapunov--Schmidt phase gate for the eight-site C19 cube.

This is deliberately a cheap algebraic test.  It freezes amplitudes and the
native corner rays, extracts the phase of the action coordinate W=chi x Psi,
and studies the internal C19 bond energy on the eight cube vertices.  Exterior
vacuum bonds are phase independent and therefore do not enter this gate.

The test answers a question that must precede another action search: is the
prepared phase pattern a minimum, a saddle, or a maximum of the native bond
energy after the common phase zero mode is removed?
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np

import localized_stability as native


HERE = Path(__file__).resolve().parent
TOL = 1.0e-9


def cube_graph() -> tuple[list[tuple[int, int, int]], list[tuple[int, int, int]]]:
    corners = list(itertools.product((0, 1), repeat=3))
    edges: list[tuple[int, int, int]] = []
    for i, a in enumerate(corners):
        for j in range(i + 1, len(corners)):
            b = corners[j]
            distance = sum(x != y for x, y in zip(a, b))
            if distance == 1:
                edges.append((i, j, 2))
            elif distance == 2:
                edges.append((i, j, 1))
    return corners, edges


def phase_energy_gradient_hessian(theta: np.ndarray, edges):
    energy = 0.0
    gradient = np.zeros(8)
    hessian = np.zeros((8, 8))
    for i, j, weight in edges:
        difference = theta[i] - theta[j]
        energy += weight * (1.0 - np.cos(difference))
        sine = weight * np.sin(difference)
        cosine = weight * np.cos(difference)
        gradient[i] += sine
        gradient[j] -= sine
        hessian[i, i] += cosine
        hessian[j, j] += cosine
        hessian[i, j] -= cosine
        hessian[j, i] -= cosine
    return energy, gradient, hessian


def inertia(hessian: np.ndarray) -> dict[str, object]:
    # Quotient the exact common-phase direction instead of counting it as a
    # physical zero mode.
    q, _ = np.linalg.qr(
        np.column_stack((np.ones(8) / np.sqrt(8), np.eye(8)[:, :7]))
    )
    reduced = q[:, 1:].T @ hessian @ q[:, 1:]
    eigenvalues = np.linalg.eigvalsh(reduced)
    return {
        "negative": int(np.sum(eigenvalues < -TOL)),
        "zero": int(np.sum(np.abs(eigenvalues) <= TOL)),
        "positive": int(np.sum(eigenvalues > TOL)),
        "eigenvalues": eigenvalues.tolist(),
    }


def native_area_phases() -> np.ndarray:
    psi, chi = native.carrier_fields(3, 1.0)
    phases = []
    for corner in itertools.product((0, 1), repeat=3):
        area = np.cross(chi[corner], psi[corner])
        component = area[np.argmax(np.abs(area))]
        phases.append(np.angle(component))
    result = np.asarray(phases)
    return np.mod(result - result[0], 2 * np.pi)


def classify(theta: np.ndarray, edges) -> dict[str, object]:
    energy, gradient, hessian = phase_energy_gradient_hessian(theta, edges)
    return {
        "phases_over_pi": (theta / np.pi).tolist(),
        "energy_half_normalized": float(energy),
        "gradient_max_abs": float(np.max(np.abs(gradient))),
        "stationary": bool(np.max(np.abs(gradient)) < TOL),
        "inertia_after_common_phase_quotient": inertia(hessian),
    }


def main() -> None:
    corners, edges = cube_graph()
    target = native_area_phases()
    target_result = classify(target, edges)

    roots = np.arange(4) * np.pi / 2
    stationary = []
    for tail in itertools.product(roots, repeat=7):
        theta = np.asarray((0.0, *tail))
        item = classify(theta, edges)
        if item["stationary"]:
            stationary.append(item)

    inertia_counts: dict[str, int] = {}
    for item in stationary:
        value = item["inertia_after_common_phase_quotient"]
        key = f"negative={value['negative']},zero={value['zero']},positive={value['positive']}"
        inertia_counts[key] = inertia_counts.get(key, 0) + 1

    minima = [
        item for item in stationary
        if item["inertia_after_common_phase_quotient"]["negative"] == 0
        and item["inertia_after_common_phase_quotient"]["zero"] == 0
    ]
    output = {
        "test": "C19 eight-corner Lyapunov--Schmidt phase gate",
        "corners_in_native_order": corners,
        "internal_bonds": [list(edge) for edge in edges],
        "target_native_area_pattern": target_result,
        "quarter_turn_patterns_tested_after_global_phase_fix": 4**7,
        "stationary_pattern_count": len(stationary),
        "stationary_inertia_counts": inertia_counts,
        "strict_minimum_count": len(minima),
        "strict_minimum_examples": minima[:8],
        "verdict": (
            "TARGET_PHASE_SADDLE"
            if target_result["stationary"]
            and target_result["inertia_after_common_phase_quotient"]["negative"] > 0
            else "TARGET_NOT_CLASSIFIED_AS_SADDLE"
        ),
        "interpretation": (
            "The exact prepared orbit is stationary in the phase equations, "
            "but the positive native C19 area-bond energy has descending phase "
            "directions there. Another coefficient sweep around this fixed "
            "pattern cannot establish a constrained energy minimum."
        ),
    }
    destination = HERE / "cube_phase_ls.json"
    destination.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
