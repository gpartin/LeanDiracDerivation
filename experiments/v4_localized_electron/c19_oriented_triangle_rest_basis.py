"""Screen the complete lowest oriented face--edge gyro basis at rest.

For every body sign triple and each face+complementary-edge decomposition, the
spatial axial vector ``face x edge`` is contracted with the native internal
axial vector ``conj(W_face) x V_edge``.  Both transform with the same cubic
rotation, so their dot product is a scalar; no fixed internal/spatial frame is
introduced.  Real/imaginary, common/relative, and exchange-even/odd variants
form the tested basis.

This file only calculates the rest-force range and nullspace.  A surviving
combination is not admitted until its odd momentum response and stability are
tested.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import torch
from torch.func import grad

import c27_body_gyro_basis_probe as incumbent


HERE = Path(__file__).resolve().parent
BODY_OFFSETS = tuple(itertools.product((-1, 1), repeat=3))
BASIS_NAMES = tuple(
    f"{velocity}_{exchange}_{part}"
    for velocity in ("common", "relative")
    for exchange in ("plus", "minus")
    for part in ("real", "imag")
)


def decompositions(body: tuple[int, int, int]):
    for axis in range(3):
        face = tuple(body[index] if index == axis else 0 for index in range(3))
        edge = tuple(body[index] if index != axis else 0 for index in range(3))
        yield face, edge


def oriented_pair(area: torch.Tensor, velocity: torch.Tensor,
                  exchange_sign: float, use_imag: bool) -> torch.Tensor:
    total = torch.zeros((), dtype=torch.float64)
    for body in BODY_OFFSETS:
        for face, edge in decompositions(body):
            area_face = torch.roll(area, shifts=face, dims=(0, 1, 2))
            area_edge = torch.roll(area, shifts=edge, dims=(0, 1, 2))
            velocity_face = torch.roll(
                velocity, shifts=face, dims=(0, 1, 2))
            velocity_edge = torch.roll(
                velocity, shifts=edge, dims=(0, 1, 2))
            first = torch.linalg.cross(
                torch.conj(area_face), velocity_edge, dim=-1)
            second = torch.linalg.cross(
                torch.conj(area_edge), velocity_face, dim=-1)
            internal = first + exchange_sign * second
            spatial = torch.tensor(
                np.cross(np.asarray(face), np.asarray(edge)),
                dtype=torch.float64)
            contraction = torch.sum(internal * spatial, dim=-1)
            value = torch.imag(contraction) if use_imag else torch.real(contraction)
            total = total + torch.sum(value)
    return incumbent.base.inverse.DT * total / 24.0


def basis(left: torch.Tensor, right: torch.Tensor) -> tuple[torch.Tensor, ...]:
    area, common, relative = incumbent.midpoint_kinematics(left, right)
    result = []
    for velocity in (common, relative):
        for exchange_sign in (1.0, -1.0):
            for use_imag in (False, True):
                result.append(oriented_pair(
                    area, velocity, exchange_sign, use_imag))
    return tuple(result)


def incumbent_coefficients(theta: float) -> np.ndarray:
    dt = incumbent.base.inverse.DT
    a = 2.0 * dt / np.sin(theta)
    return np.asarray((
        a + incumbent.base.inverse.rank.B / 4.0,
        a - incumbent.base.inverse.rank.B / 4.0,
        -a / 2.0,
        0.0,
    ))


def interval_action(left: torch.Tensor, right: torch.Tensor,
                    oriented_coefficients: torch.Tensor,
                    symmetric_coefficients: torch.Tensor) -> torch.Tensor:
    return (incumbent.interval_action(
        left, right, 1.0, symmetric_coefficients)
        + sum(coefficient * term for coefficient, term in zip(
            oriented_coefficients, basis(left, right))))


def del_residual(previous: torch.Tensor, current: torch.Tensor,
                 following: torch.Tensor, oriented: torch.Tensor,
                 symmetric: torch.Tensor) -> torch.Tensor:
    action = (interval_action(previous, current, oriented, symmetric)
              + interval_action(current, following, oriented, symmetric))
    return torch.autograd.grad(action, current, create_graph=True)[0]


def main() -> None:
    incumbent.base.inverse.rank.lock_inertia()
    incumbent.base.inverse.rest.SIDE = incumbent.base.inverse.SIDE
    incumbent.base.inverse.rest.DT = incumbent.base.inverse.DT
    previous, current, following, theta = incumbent.base.inverse.rest_layers()
    layers = tuple(torch.tensor(value, dtype=torch.float64, requires_grad=True)
                   for value in (previous, current, following))
    symmetric = torch.tensor(
        incumbent_coefficients(theta), dtype=torch.float64)
    zero = torch.zeros(len(BASIS_NAMES), dtype=torch.float64)
    baseline = del_residual(*layers, zero, symmetric).detach().numpy()
    columns = []
    for index in range(len(BASIS_NAMES)):
        unit = zero.clone()
        unit[index] = 1.0
        columns.append(
            del_residual(*layers, unit, symmetric).detach().numpy() - baseline)
    matrix = np.column_stack(columns)
    _, singular, vh = np.linalg.svd(matrix, full_matrices=True)
    threshold = max(float(singular[0]), 1.0) * 1e-10
    force_rank = int(np.sum(singular > threshold))
    null = vh[force_rank:].T
    null_residuals = [float(np.linalg.norm(matrix @ null[:, index])
                            / max(np.linalg.norm(matrix), 1e-300))
                      for index in range(null.shape[1])]
    rng = np.random.default_rng(401906)
    random_left = torch.tensor(
        rng.normal(size=current.size), dtype=torch.float64, requires_grad=True)
    random_right = torch.tensor(
        rng.normal(size=current.size), dtype=torch.float64, requires_grad=True)
    random_terms = basis(random_left, random_right)
    random_sum = sum(random_terms)
    random_gradients = torch.autograd.grad(
        random_sum, (random_left, random_right))
    random_values = [float(value.detach()) for value in random_terms]
    random_gradient_norm = float(np.sqrt(sum(
        float(torch.sum(value**2)) for value in random_gradients)))
    identically_cancelled = bool(
        max(abs(value) for value in random_values) < 1e-12
        and random_gradient_norm < 1e-12)
    result = {
        "status": "V4_C19_ORIENTED_TRIANGLE_REST_BASIS",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "basis_names": list(BASIS_NAMES),
        "construction": (
            "(face cross edge) dot [conj(W_face) cross V_edge +/- "
            "conj(W_edge) cross V_face], with common/relative velocity and "
            "real/imaginary contractions, summed over all cubic images"),
        "incumbent_rest_DEL_rms": float(
            np.linalg.norm(baseline) / np.sqrt(baseline.size)),
        "rest_force_rank": force_rank,
        "rest_force_singular_values": [float(value) for value in singular],
        "rest_preserving_null_dimension": int(null.shape[1]),
        "rest_preserving_null_basis": null.tolist(),
        "null_relative_residuals": null_residuals,
        "random_interval_basis_values": random_values,
        "random_interval_summed_gradient_norm": random_gradient_norm,
        "verdict": (
            "PAIRWISE_ORIENTED_C19_TRIANGLE_BASIS_CANCELS_IDENTICALLY"
            if identically_cancelled else
            "ORIENTED_C19_TRIANGLE_HAS_REST_PRESERVING_SUBSPACE"
            if null.shape[1] > 0 else
            "ORIENTED_C19_TRIANGLE_REST_FORCE_FULL_RANK"),
        "claim_boundary": (
            "Rest-force and random-field identity screen only. Exact "
            "cancellation retires this pairwise cubic sum; it is not a no-go "
            "for genuine three-field loop invariants."),
    }
    output = HERE / "c19_oriented_triangle_rest_basis.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "verdict": result["verdict"],
        "incumbent_rest_DEL_rms": result["incumbent_rest_DEL_rms"],
        "rest_force_rank": force_rank,
        "rest_preserving_null_dimension": int(null.shape[1]),
        "singular_values": result["rest_force_singular_values"],
        "output": str(output),
    }, indent=2))


if __name__ == "__main__":
    main()
