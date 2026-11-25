"""Complete two-layer spectrum along the canonical noncoplanar rest branch."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.linalg import eigvals
import torch

import canonical_noncoplanar_bifurcation_audit as audit
import localized_stability as native
import relative_kinetic_probe as relative


HERE = Path(__file__).resolve().parent


def classify(state: np.ndarray, fraction: float, theta: float,
             rotation: np.ndarray) -> dict:
    previous = rotation.T @ state
    following = rotation @ state
    layers = tuple(torch.tensor(value, dtype=torch.float64,
                                requires_grad=True)
                   for value in (previous, state, following))

    def action(pm: torch.Tensor, q: torch.Tensor,
               pp: torch.Tensor) -> torch.Tensor:
        return (audit.interval_action(pm, q)
                + audit.interval_action(q, pp))

    scalar = action(*layers)
    gradient = torch.autograd.grad(
        scalar, layers[1], create_graph=False)[0].detach().numpy()
    blocks = torch.autograd.functional.hessian(
        action, layers, vectorize=True)
    previous_block = blocks[1][0].detach().numpy()
    current_block = blocks[1][1].detach().numpy()
    following_block = blocks[1][2].detach().numpy()
    lhs = following_block @ rotation
    singular = np.linalg.svd(lhs, compute_uv=False)
    rank = int(np.sum(singular > singular[0] * 1e-12))
    row = {
        "fraction_of_rest_norm": fraction,
        "rest_DEL_rms": float(np.linalg.norm(gradient)
                              / np.sqrt(gradient.size)),
        "following_block_rank": rank,
        "following_block_dimension": int(lhs.shape[0]),
        "following_block_condition_number": float(
            singular[0] / max(singular[-1], 1e-300)),
    }
    if rank != lhs.shape[0]:
        row.update({"spectral_radius": None, "growth_rate": None})
        return row
    forward = -np.linalg.solve(lhs, current_block)
    backward = -np.linalg.solve(lhs, previous_block @ rotation.T)
    identity = np.eye(forward.shape[0])
    monodromy = np.block([
        [forward, backward],
        [identity, np.zeros_like(identity)],
    ])
    eigenvalues = eigvals(monodromy, check_finite=False)
    moduli = np.sort(abs(eigenvalues))
    row.update({
        "spectral_radius": float(moduli[-1]),
        "growth_rate": float(np.log(moduli[-1]) / audit.DT),
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
    archive = np.load(HERE / "canonical_noncoplanar_augmented_branch.npz")
    rest = archive["rest"]
    fractions = archive["fractions"]
    states = archive["states"]
    theta = float(np.arccos(1 - audit.DT**2 * audit.CHI0**2 / 2))
    rotation = relative.raw_rotation_matrix(audit.SIDE, theta, 0.0, 1)
    selected = [(0.0, rest)] + [
        (float(fraction), state)
        for fraction, state in zip(fractions, states, strict=True)]
    rows = [classify(state, fraction, theta, rotation)
            for fraction, state in selected]
    baseline = rows[0]
    improved = [row for row in rows[1:]
                if row["rest_DEL_rms"] < 1e-9
                and row["spectral_radius"] is not None
                and row["spectral_radius"]
                < baseline["spectral_radius"] - 1e-7]
    stable = [row for row in rows[1:]
              if row["spectral_radius"] is not None
              and row["spectral_radius"] < 1 + 1e-7]
    result = {
        "status": "V4_CANONICAL_NONCOPLANAR_SPECTRUM",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "rows": rows,
        "improved_fractions": [row["fraction_of_rest_norm"]
                               for row in improved],
        "stable_fractions": [row["fraction_of_rest_norm"]
                             for row in stable],
        "verdict": (
            "CANONICAL_NONCOPLANAR_BRANCH_IMPROVES_FULL_SPECTRUM"
            if improved else
            "CANONICAL_NONCOPLANAR_BRANCH_DOES_NOT_IMPROVE_FULL_SPECTRUM"),
        "claim_boundary": (
            "Complete side-three two-layer linear spectrum. No nonlinear "
            "stability, translation, current, or box-refinement claim."),
    }
    output = HERE / "canonical_noncoplanar_spectrum.json"
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n",
                      encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
