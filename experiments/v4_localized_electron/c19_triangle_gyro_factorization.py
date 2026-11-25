"""Factor the body W gyros through C19 face--edge paths.

The body common/relative one-forms pair W_i with the corresponding velocity
at W_(i+b), and conversely.  Reconstructing the endpoint by either of the
three C19 face+edge paths gives an exact identity.  Averaging the paths keeps
cubic symmetry and shows that the candidate needs a corner correlation of
C19 bonds, not a distinct fundamental C27 stencil.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import torch

import c27_body_gyro_basis_probe as candidate
import c27_body_gyro_rest_spectrum as spectrum


HERE = Path(__file__).resolve().parent
BODY_OFFSETS = tuple(itertools.product((-1, 1), repeat=3))


def decompositions(body):
    for axis in range(3):
        face = tuple(body[index] if index == axis else 0
                     for index in range(3))
        edge = tuple(body[index] if index != axis else 0
                     for index in range(3))
        yield face, edge


def triangle_pair(area: torch.Tensor, velocity: torch.Tensor) -> torch.Tensor:
    result = torch.zeros((), dtype=torch.float64)
    for body in BODY_OFFSETS:
        endpoint_area_direct = torch.roll(
            area, shifts=body, dims=(0, 1, 2))
        endpoint_velocity_direct = torch.roll(
            velocity, shifts=body, dims=(0, 1, 2))
        for face, _edge in decompositions(body):
            middle_area = torch.roll(area, shifts=face, dims=(0, 1, 2))
            middle_velocity = torch.roll(
                velocity, shifts=face, dims=(0, 1, 2))
            endpoint_area = middle_area + (
                endpoint_area_direct - middle_area)
            endpoint_velocity = middle_velocity + (
                endpoint_velocity_direct - middle_velocity)
            first = torch.sum(
                torch.conj(area) * endpoint_velocity, dim=-1)
            second = torch.sum(
                torch.conj(endpoint_area) * velocity, dim=-1)
            result = result + torch.sum(torch.imag(first + second))
    # Average three paths and retain the original directed-bond /4 factor.
    return candidate.base.inverse.DT * result / 12


def values(left: torch.Tensor, right: torch.Tensor):
    area, common, relative = candidate.midpoint_kinematics(left, right)
    return (
        candidate.body_pair(area, common),
        triangle_pair(area, common),
        candidate.body_pair(area, relative),
        triangle_pair(area, relative),
    )


def classify(label: str, left_raw: np.ndarray,
             right_raw: np.ndarray) -> dict:
    left = torch.tensor(left_raw, dtype=torch.float64, requires_grad=True)
    right = torch.tensor(right_raw, dtype=torch.float64, requires_grad=True)
    direct_common, triangle_common, direct_relative, triangle_relative = (
        values(left, right))
    rows = []
    for name, direct, triangle in (
            ("common", direct_common, triangle_common),
            ("relative", direct_relative, triangle_relative)):
        direct_grad = torch.autograd.grad(
            direct, (left, right), retain_graph=True)
        triangle_grad = torch.autograd.grad(
            triangle, (left, right), retain_graph=True)
        gradient_error = np.sqrt(sum(float(torch.sum(
            (first - second)**2)) for first, second in zip(
                direct_grad, triangle_grad)))
        gradient_scale = max(np.sqrt(sum(float(torch.sum(first**2))
                                         for first in direct_grad)), 1.0)
        direct_value = float(direct.detach())
        triangle_value = float(triangle.detach())
        rows.append({
            "channel": name,
            "direct_body_one_form": direct_value,
            "triangle_path_one_form": triangle_value,
            "relative_value_error": abs(direct_value - triangle_value)
                                    / max(abs(direct_value), 1.0),
            "relative_first_variation_error": gradient_error / gradient_scale,
        })
    return {"label": label, "channels": rows}


def main() -> None:
    spectrum.configure()
    candidate.base.inverse.rank.lock_inertia()
    previous, current, following, _ = candidate.base.inverse.rest_layers()
    rng = np.random.default_rng(4020)
    random_left = rng.normal(size=current.size)
    random_right = rng.normal(size=current.size)
    rows = [
        classify("exact_rest_previous_interval", previous, current),
        classify("exact_rest_following_interval", current, following),
        classify("deterministic_random_interval", random_left, random_right),
    ]
    passed = all(
        channel["relative_value_error"] < 1e-12
        and channel["relative_first_variation_error"] < 1e-12
        for row in rows for channel in row["channels"])
    result = {
        "status": "V4_C19_TRIANGLE_GYRO_FACTORIZATION",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "identity": (
            "each body common/relative W one-form equals the average over "
            "three C19 face+edge endpoint reconstructions"),
        "rows": rows,
        "verdict": (
            "C27_BODY_GYROS_EXACTLY_FACTOR_THROUGH_C19_TRIANGLES"
            if passed else "C19_TRIANGLE_GYRO_FACTORIZATION_FAILED"),
        "claim_boundary": (
            "Exact discrete one-form and first-variation identity only. It "
            "does not admit coefficients or prove stability, translation, "
            "one-hop execution, or Dirac-current closure."),
    }
    output = HERE / "c19_triangle_gyro_factorization.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
