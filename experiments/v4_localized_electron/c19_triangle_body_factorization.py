"""Factor the C27 joint-W body bonds into C19 face--edge paths.

Every body diagonal b=(sx,sy,sz) has three decompositions b=f+e, where f is
one signed face direction and e is the complementary signed edge direction.
For W=chi x Psi / chi0,

  W(i+b)-W(i) = [W(i+f)-W(i)] + [W(i+b)-W(i+f)].

Both brackets are C19 bonds.  Averaging the squared path sum over the three
decompositions is therefore exactly the positive C27 body Dirichlet energy,
but expressed as a local correlation between consecutive C19 face and edge
bonds.  This script verifies both energy and first-variation identities on
the rest carrier and deterministic random fields.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import torch

import c27_body_gyro_basis_probe as candidate
import c27_body_gyro_rest_spectrum as spectrum
import relative_kinetic_probe as relative


HERE = Path(__file__).resolve().parent
BODY_OFFSETS = tuple(itertools.product((-1, 1), repeat=3))
FACE_OFFSETS = tuple(
    tuple(sign if axis == selected else 0 for axis in range(3))
    for selected in range(3) for sign in (-1, 1))
EDGE_OFFSETS = tuple(
    offset for offset in itertools.product((-1, 0, 1), repeat=3)
    if sum(value != 0 for value in offset) == 2)


def squared_norm(value: torch.Tensor) -> torch.Tensor:
    return torch.real(torch.conj(value) * value)


def field(raw: torch.Tensor) -> torch.Tensor:
    psi, chi = candidate.fields(raw)
    return torch.linalg.cross(chi, psi, dim=-1) / candidate.base.inverse.rank.R


def body_energy(raw: torch.Tensor) -> torch.Tensor:
    value = field(raw)
    result = torch.zeros((), dtype=torch.float64)
    for body in BODY_OFFSETS:
        difference = torch.roll(value, shifts=body, dims=(0, 1, 2)) - value
        result = result + torch.sum(squared_norm(difference))
    return result / 4


def decompositions(body):
    for axis in range(3):
        face = tuple(body[index] if index == axis else 0
                     for index in range(3))
        edge = tuple(body[index] if index != axis else 0
                     for index in range(3))
        yield face, edge


def triangle_terms(raw: torch.Tensor):
    value = field(raw)
    diagonal = torch.zeros((), dtype=torch.float64)
    corner = torch.zeros((), dtype=torch.float64)
    total = torch.zeros((), dtype=torch.float64)
    for body in BODY_OFFSETS:
        endpoint = torch.roll(value, shifts=body, dims=(0, 1, 2))
        for face, _edge in decompositions(body):
            middle = torch.roll(value, shifts=face, dims=(0, 1, 2))
            first = middle - value
            second = endpoint - middle
            diagonal = diagonal + torch.sum(
                squared_norm(first) + squared_norm(second))
            cross = 2 * torch.real(torch.sum(
                torch.conj(first) * second, dim=-1))
            corner = corner + torch.sum(cross)
            total = total + torch.sum(squared_norm(first + second))
    # Three paths per body direction and the same directed-bond /4 convention.
    return diagonal / 12, corner / 12, total / 12


def direct_c19_diagonal(raw: torch.Tensor):
    value = field(raw)
    face = torch.zeros((), dtype=torch.float64)
    edge = torch.zeros((), dtype=torch.float64)
    for offset in FACE_OFFSETS:
        difference = torch.roll(value, shifts=offset, dims=(0, 1, 2)) - value
        face = face + torch.sum(squared_norm(difference))
    for offset in EDGE_OFFSETS:
        difference = torch.roll(value, shifts=offset, dims=(0, 1, 2)) - value
        edge = edge + torch.sum(squared_norm(difference))
    return face / 3 + edge / 6


def classify(label: str, raw: np.ndarray) -> dict:
    variable = torch.tensor(raw, dtype=torch.float64, requires_grad=True)
    body = body_energy(variable)
    diagonal, corner, triangle = triangle_terms(variable)
    c19_diagonal = direct_c19_diagonal(variable)
    gradients = [torch.autograd.grad(value, variable, retain_graph=True)[0]
                 for value in (body, triangle)]
    body_value = float(body.detach())
    triangle_value = float(triangle.detach())
    scale = max(abs(body_value), 1.0)
    gradient_scale = max(float(torch.linalg.vector_norm(gradients[0])), 1.0)
    return {
        "label": label,
        "body_energy": body_value,
        "triangle_path_energy": triangle_value,
        "c19_diagonal_energy": float(c19_diagonal.detach()),
        "triangle_diagonal_energy": float(diagonal.detach()),
        "triangle_corner_cross_energy": float(corner.detach()),
        "relative_energy_identity_error": abs(
            triangle_value - body_value) / scale,
        "relative_diagonal_counting_error": abs(float(
            diagonal.detach() - c19_diagonal.detach())) / scale,
        "relative_first_variation_error": float(torch.linalg.vector_norm(
            gradients[1] - gradients[0])) / gradient_scale,
    }


def main() -> None:
    spectrum.configure()
    candidate.base.inverse.rank.lock_inertia()
    _, current, _, _ = candidate.base.inverse.rest_layers()
    rng = np.random.default_rng(4019)
    random_raw = rng.normal(size=current.size)
    rows = [
        classify("exact_rest_carrier", current),
        classify("deterministic_random_full_field", random_raw),
    ]
    passed = all(
        row["relative_energy_identity_error"] < 1e-12
        and row["relative_diagonal_counting_error"] < 1e-12
        and row["relative_first_variation_error"] < 1e-12
        for row in rows)
    result = {
        "status": "V4_C19_TRIANGLE_BODY_FACTORIZATION",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "identity": (
            "Ubody(W)=(1/3) sum over the three face+edge C19 paths "
            "of |Dface W + Dedge W|^2, with the directed-bond /4 convention"),
        "diagonal_counting_identity": (
            "path diagonal = (1/3) sum_directed_faces |Df W|^2 "
            "+ (1/6) sum_directed_edges |De W|^2"),
        "new_information": (
            "the transport-closing part beyond reweighted C19 stiffness is "
            "the real cross correlation of consecutive face and edge bonds"),
        "rows": rows,
        "verdict": (
            "C27_BODY_RESPONSE_EXACTLY_FACTORS_THROUGH_C19_TRIANGLES"
            if passed else "C19_TRIANGLE_FACTORIZATION_FAILED"),
        "claim_boundary": (
            "Exact action and first-variation identity only. This does not "
            "admit the coefficient, prove one-hop execution, stability, "
            "nonlinear translation, or Dirac-current closure."),
    }
    output = HERE / "c19_triangle_body_factorization.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
