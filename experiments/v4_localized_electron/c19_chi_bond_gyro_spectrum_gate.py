"""C19 chi-bond gyroscopic gate for the exterior orientation null channel.

The term is the cubic average of the native Berry one-form

  Im <chi_i, dot(chi_j)>

over directed C19 neighbors.  The directed sum is reciprocal because every
bond orientation and its reverse occur with the same weight.  It contains no
potential energy, vanishes on the static-chi rest trajectory, and has nonzero
symplectic curvature.  The native dimensional coefficient B*chi0 equates its
clock scale to the chi inertia B at the only declared substrate frequency
chi0.  Both signs are tested; neither is fitted to spectral output.
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


def bond_one_form(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    _left_psi, left_chi = direct.fields(left)
    _right_psi, right_chi = direct.fields(right)
    chi = (left_chi + right_chi) / 2
    chi_velocity = (right_chi - left_chi) / spectrum.DT
    total_weight = sum(
        weight for _offset, weight
        in direct.base.inverse.rank.native.DIRECTIONS)
    result = torch.zeros((), dtype=torch.float64)
    for offset, weight in direct.base.inverse.rank.native.DIRECTIONS:
        neighbor_velocity = torch.roll(
            chi_velocity, shifts=offset, dims=(0, 1, 2))
        pairing = torch.sum(torch.conj(chi) * neighbor_velocity, dim=-1)
        result = result + weight * torch.sum(torch.imag(pairing))
    return spectrum.DT * result / total_weight


def interval_action(left: torch.Tensor, right: torch.Tensor,
                    coefficient: float, values: torch.Tensor) -> torch.Tensor:
    return (triangle.triangle_interval(left, right, values)
            + coefficient * bond_one_form(left, right))


def classify(coefficient: float, layers: tuple[torch.Tensor, ...],
             theta: float, values: torch.Tensor) -> dict:
    def action(previous: torch.Tensor, current: torch.Tensor,
               following: torch.Tensor) -> torch.Tensor:
        return (interval_action(previous, current, coefficient, values)
                + interval_action(current, following, coefficient, values))

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
        "coefficient": coefficient,
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
    natural = (direct.base.inverse.rank.B
               * direct.base.inverse.rank.R)
    coefficients = (0.0, natural, -natural)
    rows = [classify(value, layers, theta, values)
            for value in coefficients]
    baseline_radius = rows[0]["spectral_radius"]
    survivors = [row for row in rows[1:]
                 if row["rest_DEL_rms"] < 1e-8
                 and row["spectral_radius"] is not None
                 and row["spectral_radius"] < baseline_radius - 1e-7]
    result = {
        "status": "V4_C19_CHI_BOND_GYRO_SPECTRUM_GATE",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "action_delta": (
            "(B chi0) times the C19-weight average of "
            "Im <chi_i, dot(chi_j)>"
        ),
        "coefficient_rule": (
            "magnitude B*chi0 from native chi inertia and substrate clock; "
            "both signs tested without fitting"
        ),
        "rows": rows,
        "verdict": (
            "C19_CHI_BOND_GYRO_TARGETED_IMPROVEMENT"
            if survivors else "C19_CHI_BOND_GYRO_NO_IMPROVEMENT"
        ),
        "surviving_coefficients": [
            row["coefficient"] for row in survivors],
        "claim_boundary": (
            "Full side-three spectrum response gate only. A survivor would "
            "still require nonlinear perturbation, box/timestep refinement, "
            "translation, and same-readout Dirac-current closure."
        ),
    }
    output = HERE / "c19_chi_bond_gyro_spectrum_gate.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
