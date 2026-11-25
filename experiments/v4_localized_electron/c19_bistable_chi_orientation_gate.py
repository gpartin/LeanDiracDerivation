"""Targeted C19 repair of the exterior-chi orientation null channel.

The C19-triangle/gyro candidate sees chi outside the carrier only through
W = chi x Psi.  Hence W vanishes in the Psi vacuum and exterior chi may turn
without potential cost.  A plain chi spring is inadmissible for the retained
rest orbit because that orbit contains both parallel exterior bonds and
orthogonal carrier--exterior bonds.

This gate tests the smallest polynomial bond energy with *both* relations as
minima.  For one C19 bond let

    s = |chi_i^dagger chi_j|^2,
    p = |chi_i|^2 |chi_j|^2.

Cauchy--Schwarz gives 0 <= s <= p, so s (p-s) is nonnegative and vanishes at
both orthogonal and parallel endpoint orientations.  The term contains no
carrier label, ray, projector, preferred direction, or Dirac object.  The
normalization chi0^-4 makes its dimension match the quartic local potential.

Unit strength is a native-normalization response test, not an admitted fitted
coefficient.  The script first verifies positivity and exact-rest closure,
then evaluates the complete two-layer spectrum.
"""

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
STRENGTHS = (0.0, 1.0)


def chi_orientation_potential(raw: torch.Tensor) -> torch.Tensor:
    _psi, chi = direct.fields(raw)
    result = torch.zeros((), dtype=torch.float64)
    radius = direct.base.inverse.rank.R
    for offset, weight in direct.base.inverse.rank.native.DIRECTIONS:
        neighbor = torch.roll(chi, shifts=offset, dims=(0, 1, 2))
        overlap = torch.sum(torch.conj(chi) * neighbor, dim=-1)
        s = torch.real(torch.conj(overlap) * overlap)
        rho = torch.sum(torch.real(torch.conj(chi) * chi), dim=-1)
        neighbor_rho = torch.sum(
            torch.real(torch.conj(neighbor) * neighbor), dim=-1)
        result = result + weight * torch.sum(s * (rho * neighbor_rho - s))
    # DIRECTIONS is directed; divide by two for each undirected C19 bond.
    return result / (2 * radius**4)


def interval_action(left: torch.Tensor, right: torch.Tensor,
                    strength: float, values: torch.Tensor) -> torch.Tensor:
    return (
        triangle.triangle_interval(left, right, values)
        - spectrum.DT * strength * (
            chi_orientation_potential(left)
            + chi_orientation_potential(right)) / 2
    )


def classify(strength: float, layers: tuple[torch.Tensor, ...],
             theta: float, values: torch.Tensor) -> dict:
    def action(previous: torch.Tensor, current: torch.Tensor,
               following: torch.Tensor) -> torch.Tensor:
        return (interval_action(previous, current, strength, values)
                + interval_action(current, following, strength, values))

    variables = tuple(value.detach().requires_grad_(True) for value in layers)
    scalar = action(*variables)
    gradient = torch.autograd.grad(
        scalar, variables[1], create_graph=False)[0].detach().numpy()
    blocks = torch.autograd.functional.hessian(
        action, variables, vectorize=True)
    previous_block = blocks[1][0].detach().numpy()
    current_block = blocks[1][1].detach().numpy()
    following_block = blocks[1][2].detach().numpy()
    rotation = relative.raw_rotation_matrix(spectrum.SIDE, theta, 0.0, 1)
    lhs = following_block @ rotation
    singular = np.linalg.svd(lhs, compute_uv=False)
    rank = int(np.sum(singular > singular[0] * 1e-12))
    row = {
        "strength": strength,
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
        "growth_rate": float(np.log(moduli[-1]) / spectrum.DT),
        "reciprocal_pairing_defect": float(np.max(
            abs(moduli * moduli[::-1] - 1))),
        "count_modulus_above_1_plus_1e_minus_7": int(np.sum(
            moduli > 1 + 1e-7)),
        "largest_moduli": [float(value) for value in moduli[-12:][::-1]],
    })
    return row


def main() -> None:
    spectrum.configure()
    direct.base.inverse.rank.lock_inertia()
    previous, current, following, theta = direct.base.inverse.rest_layers()
    layers = tuple(torch.tensor(value, dtype=torch.float64)
                   for value in (previous, current, following))
    values = triangle.coefficients(theta)

    rng = np.random.default_rng(4019031)
    random_raw = torch.tensor(
        rng.normal(size=current.size), dtype=torch.float64)
    random_positive = float(chi_orientation_potential(random_raw))
    rest_potential = float(chi_orientation_potential(layers[1]))
    rows = [classify(strength, layers, theta, values)
            for strength in STRENGTHS]
    baseline, trial = rows
    rest_preserved = trial["rest_DEL_rms"] < 1e-8
    positive = random_positive >= -1e-10 and rest_potential >= -1e-10
    improved = (
        trial["spectral_radius"] is not None
        and baseline["spectral_radius"] is not None
        and trial["spectral_radius"] < baseline["spectral_radius"] - 1e-7
    )
    result = {
        "status": "V4_C19_BISTABLE_CHI_ORIENTATION_GATE",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "action_delta": (
            "sum_C19 w |chi_i^dagger chi_j|^2 "
            "(rho_chi_i rho_chi_j-|chi_i^dagger chi_j|^2) / chi0^4"
        ),
        "coefficient_rule": (
            "unit native normalization response screen; no coefficient is "
            "admitted or fit to a Dirac or stability target"
        ),
        "random_positive_control": random_positive,
        "exact_rest_potential": rest_potential,
        "rows": rows,
        "rest_preserved": rest_preserved,
        "positive_control_pass": positive,
        "spectrum_improved": improved,
        "verdict": (
            "C19_BISTABLE_CHI_ORIENTATION_TARGETED_IMPROVEMENT"
            if positive and rest_preserved and improved else
            "C19_BISTABLE_CHI_ORIENTATION_NO_IMPROVEMENT"
        ),
        "claim_boundary": (
            "Targeted full-spectrum response gate only. A positive result "
            "would still require coefficient provenance, nonlinear 100-cycle "
            "stability, translation, and same-readout Dirac-current closure."
        ),
    }
    output = HERE / "c19_bistable_chi_orientation_gate.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
