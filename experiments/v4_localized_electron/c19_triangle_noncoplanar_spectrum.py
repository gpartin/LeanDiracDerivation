"""Complete spectrum along the solved noncoplanar C19 rest family."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.linalg import eigvals
import torch

import c19_triangle_gyro_candidate_equivalence as triangle
import c27_body_gyro_basis_probe as direct
import c27_body_gyro_rest_spectrum as spectrum
import relative_kinetic_probe as relative


HERE = Path(__file__).resolve().parent
SELECTED_FRACTIONS = (0.0, 1e-3, 1e-2, 3e-2)


def classify(state: np.ndarray, fraction: float, theta: float,
             values: torch.Tensor, rotation_np: np.ndarray) -> dict:
    previous = rotation_np.T @ state
    following = rotation_np @ state
    layers = tuple(torch.tensor(value, dtype=torch.float64)
                   for value in (previous, state, following))

    def action(pm: torch.Tensor, q: torch.Tensor,
               pp: torch.Tensor) -> torch.Tensor:
        return (triangle.triangle_interval(pm, q, values)
                + triangle.triangle_interval(q, pp, values))

    variables = tuple(value.detach().requires_grad_(True) for value in layers)
    scalar = action(*variables)
    gradient = torch.autograd.grad(
        scalar, variables[1], create_graph=False)[0].detach().numpy()
    blocks = torch.autograd.functional.hessian(
        action, variables, vectorize=True)
    previous_block = blocks[1][0].detach().numpy()
    current_block = blocks[1][1].detach().numpy()
    following_block = blocks[1][2].detach().numpy()
    lhs = following_block @ rotation_np
    singular = np.linalg.svd(lhs, compute_uv=False)
    rank = int(np.sum(singular > singular[0] * 1e-12))
    row = {
        "fraction_of_rest_norm": fraction,
        "rest_DEL_rms": float(np.linalg.norm(gradient) / np.sqrt(gradient.size)),
        "following_block_rank": rank,
        "following_block_dimension": int(lhs.shape[0]),
        "following_block_condition_number": float(
            singular[0] / max(singular[-1], 1e-300)),
    }
    if rank != lhs.shape[0]:
        row.update({"spectral_radius": None, "growth_rate": None})
        return row
    forward = -np.linalg.solve(lhs, current_block)
    backward = -np.linalg.solve(lhs, previous_block @ rotation_np.T)
    identity = np.eye(forward.shape[0])
    monodromy = np.block([
        [forward, backward],
        [identity, np.zeros_like(identity)],
    ])
    eigenvalues = eigvals(monodromy, check_finite=False)
    moduli = np.sort(abs(eigenvalues))
    row.update({
        "spectral_radius": float(moduli[-1]),
        "growth_rate": float(np.log(moduli[-1]) / spectrum.DT),
        "reciprocal_pairing_defect": float(np.max(
            abs(moduli * moduli[::-1] - 1))),
        "count_modulus_above_1_plus_1e_minus_7": int(np.sum(
            moduli > 1 + 1e-7)),
        "count_modulus_above_1_plus_1e_minus_6": int(np.sum(
            moduli > 1 + 1e-6)),
        "largest_moduli": [float(value) for value in moduli[-12:][::-1]],
    })
    return row


def main() -> None:
    spectrum.configure()
    direct.base.inverse.rank.lock_inertia()
    _previous, rest, _following, theta = direct.base.inverse.rest_layers()
    values = triangle.coefficients(theta)
    rotation = relative.raw_rotation_matrix(
        spectrum.SIDE, theta, 0.0, 1)
    archive = np.load(
        HERE / "c19_triangle_noncoplanar_branch_probe.npz")
    fractions = archive["fractions"]
    states = archive["states"]
    selected = [(0.0, rest)]
    for target in SELECTED_FRACTIONS[1:]:
        index = int(np.argmin(abs(fractions - target)))
        if abs(float(fractions[index]) - target) > 1e-12:
            raise RuntimeError(f"missing branch fraction {target}")
        selected.append((target, states[index]))
    rows = [classify(state, fraction, theta, values, rotation)
            for fraction, state in selected]
    baseline = rows[0]
    improved = [row for row in rows[1:]
                if row["rest_DEL_rms"] < 1e-8
                and row["spectral_radius"] is not None
                and row["spectral_radius"] < baseline["spectral_radius"] - 1e-7]
    result = {
        "status": "V4_C19_TRIANGLE_NONCOPLANAR_SPECTRUM",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "rows": rows,
        "improved_fractions": [
            row["fraction_of_rest_norm"] for row in improved],
        "verdict": (
            "NONCOPLANAR_REST_BRANCH_IMPROVES_FULL_SPECTRUM"
            if improved else
            "NONCOPLANAR_REST_BRANCH_DOES_NOT_IMPROVE_FULL_SPECTRUM"
        ),
        "claim_boundary": (
            "Side-three complete two-layer spectrum along one solved branch. "
            "A spectral improvement still requires box/timestep refinement, "
            "nonlinear stability, transport, and Dirac-current tests."
        ),
    }
    output = HERE / "c19_triangle_noncoplanar_spectrum.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
