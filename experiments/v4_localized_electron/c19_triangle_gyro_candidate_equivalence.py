"""Verify the complete C19-triangle form of the C27/body-gyro oracle."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

import c19_triangle_body_factorization as body_triangle
import c19_triangle_gyro_factorization as gyro_triangle
import c27_body_gyro_basis_probe as direct
import c27_body_gyro_rest_spectrum as spectrum


HERE = Path(__file__).resolve().parent


def coefficients(theta: float) -> torch.Tensor:
    a = 2 * spectrum.DT / np.sin(theta)
    return torch.tensor((
        a + direct.base.inverse.rank.B / 4,
        a - direct.base.inverse.rank.B / 4,
        -a / 2,
        0.0,
    ), dtype=torch.float64)


def triangle_interval(left: torch.Tensor, right: torch.Tensor,
                      values: torch.Tensor) -> torch.Tensor:
    area, common, relative = direct.midpoint_kinematics(left, right)
    terms = (
        direct.local_pair(area, common),
        direct.local_pair(area, relative),
        gyro_triangle.triangle_pair(area, common),
        gyro_triangle.triangle_pair(area, relative),
    )
    gyro = sum(coefficient * term for coefficient, term in zip(values, terms))
    left_potential = body_triangle.triangle_terms(left)[2]
    right_potential = body_triangle.triangle_terms(right)[2]
    return (
        direct.base.inverse.interval_action(left, right)
        - direct.base.inverse.DT * (left_potential + right_potential) / 2
        + gyro
    )


def classify(label: str, left_raw: np.ndarray, right_raw: np.ndarray,
             values: torch.Tensor) -> dict:
    left = torch.tensor(left_raw, dtype=torch.float64, requires_grad=True)
    right = torch.tensor(right_raw, dtype=torch.float64, requires_grad=True)
    direct_value = direct.interval_action(left, right, 1.0, values)
    triangle_value = triangle_interval(left, right, values)
    direct_grad = torch.autograd.grad(
        direct_value, (left, right), retain_graph=True)
    triangle_grad = torch.autograd.grad(
        triangle_value, (left, right), retain_graph=True)
    value_scale = max(abs(float(direct_value.detach())), 1.0)
    gradient_scale = max(np.sqrt(sum(float(torch.sum(value**2))
                                     for value in direct_grad)), 1.0)
    gradient_error = np.sqrt(sum(float(torch.sum((first-second)**2))
                                 for first, second in zip(
                                     direct_grad, triangle_grad)))
    return {
        "label": label,
        "direct_interval_action": float(direct_value.detach()),
        "triangle_interval_action": float(triangle_value.detach()),
        "relative_action_error": abs(float(
            direct_value.detach() - triangle_value.detach())) / value_scale,
        "relative_first_variation_error": gradient_error / gradient_scale,
    }


def main() -> None:
    spectrum.configure()
    direct.base.inverse.rank.lock_inertia()
    previous, current, following, theta = direct.base.inverse.rest_layers()
    values = coefficients(theta)
    rng = np.random.default_rng(4019020)
    random_left = rng.normal(size=current.size)
    random_right = rng.normal(size=current.size)
    rows = [
        classify("exact_rest_previous_interval", previous, current, values),
        classify("exact_rest_following_interval", current, following, values),
        classify("deterministic_random_interval", random_left, random_right,
                 values),
    ]
    passed = all(
        row["relative_action_error"] < 1e-12
        and row["relative_first_variation_error"] < 1e-12
        for row in rows)
    result = {
        "status": "V4_C19_TRIANGLE_GYRO_CANDIDATE_EQUIVALENCE",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "coefficients": [float(value) for value in values],
        "rows": rows,
        "verdict": (
            "FULL_C27_GYRO_ORACLE_HAS_EXACT_C19_TRIANGLE_ACTION"
            if passed else "FULL_C19_TRIANGLE_ACTION_EQUIVALENCE_FAILED"),
        "claim_boundary": (
            "Exact interval-action and first-variation identity. The C19 "
            "triangle action remains an unadmitted research candidate and "
            "has not passed 100-cycle stability or translation gates."),
    }
    output = HERE / "c19_triangle_gyro_candidate_equivalence.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
