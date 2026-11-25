"""Final chi-only C19 stiffness test using the exact cube overlap set.

The exact rest carrier has normalized chi bond overlaps x in {0, 1/3, 1}.
The lowest nonnegative scalar polynomial on 0 <= x <= 1 with precisely those
three zero sets is x (3x-1)^2 (1-x).  In homogeneous polynomial form this is

  s (3s-p)^2 (p-s),

where s=|chi_i^dagger chi_j|^2 and p=rho_i rho_j.  Division by chi0^12
restores quartic-field dimensions; division by the total directed C19 weight
24 is the native neighbor average.  No coefficient is fitted.
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


def geometric_potential(raw: torch.Tensor) -> torch.Tensor:
    _psi, chi = direct.fields(raw)
    result = torch.zeros((), dtype=torch.float64)
    radius = direct.base.inverse.rank.R
    total_weight = sum(
        weight for _offset, weight
        in direct.base.inverse.rank.native.DIRECTIONS)
    for offset, weight in direct.base.inverse.rank.native.DIRECTIONS:
        neighbor = torch.roll(chi, shifts=offset, dims=(0, 1, 2))
        overlap = torch.sum(torch.conj(chi) * neighbor, dim=-1)
        s = torch.real(torch.conj(overlap) * overlap)
        rho = torch.sum(torch.real(torch.conj(chi) * chi), dim=-1)
        neighbor_rho = torch.sum(
            torch.real(torch.conj(neighbor) * neighbor), dim=-1)
        p = rho * neighbor_rho
        result = result + weight * torch.sum(s * (3*s-p)**2 * (p-s))
    # Directed bonds are averaged by the total directed C19 weight.
    return result / (total_weight * radius**12)


def interval_action(left: torch.Tensor, right: torch.Tensor,
                    strength: float, values: torch.Tensor) -> torch.Tensor:
    return (
        triangle.triangle_interval(left, right, values)
        - spectrum.DT * strength * (
            geometric_potential(left) + geometric_potential(right)) / 2
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
    rng = np.random.default_rng(4019032)
    random_raw = torch.tensor(
        rng.normal(size=current.size), dtype=torch.float64)
    random_positive = float(geometric_potential(random_raw))
    rest_potential = float(geometric_potential(layers[1]))
    rows = [classify(strength, layers, theta, values)
            for strength in (0.0, 1.0)]
    baseline, trial = rows
    rest_preserved = trial["rest_DEL_rms"] < 1e-8
    positive = random_positive >= -1e-10 and rest_potential >= -1e-10
    improved = (
        trial["spectral_radius"] is not None
        and baseline["spectral_radius"] is not None
        and trial["spectral_radius"] < baseline["spectral_radius"] - 1e-7
    )
    result = {
        "status": "V4_C19_GEOMETRIC_CHI_ORIENTATION_GATE",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "action_delta": (
            "C19 average of s(3s-p)^2(p-s)/chi0^12, "
            "s=|chi_i^dagger chi_j|^2, p=rho_i rho_j"
        ),
        "coefficient_rule": (
            "zeros {0,1/3,1} from the exact cube chi-overlap audit; "
            "unit strength after C19-weight averaging; no fitted parameter"
        ),
        "random_positive_control": random_positive,
        "exact_rest_potential": rest_potential,
        "rows": rows,
        "rest_preserved": rest_preserved,
        "positive_control_pass": positive,
        "spectrum_improved": improved,
        "verdict": (
            "C19_GEOMETRIC_CHI_ORIENTATION_TARGETED_IMPROVEMENT"
            if positive and rest_preserved and improved else
            "C19_GEOMETRIC_CHI_ORIENTATION_NO_IMPROVEMENT"
        ),
        "claim_boundary": (
            "Final chi-only stiffness response gate. Failure retires this "
            "specific geometry-derived chi-only polynomial, not all joint "
            "Psi--chi kinetic or symplectic action classes."
        ),
    }
    output = HERE / "c19_geometric_chi_orientation_gate.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
