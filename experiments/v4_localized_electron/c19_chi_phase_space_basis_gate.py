"""Exhaust the reduced C19 chi-only phase-space action basis.

Three independent, rest-inactive mechanisms remain after the chi potential
and neighbor Berry tests:

1. local Berry one-form Im <chi, dot chi>;
2. C19-gradient Berry one-form Im <D chi, D dot chi>;
3. positive C19-gradient kinetic metric |D dot chi|^2.

The two gyro signs use the native clock magnitude B*chi0.  The kinetic metric
uses unit strength after C19-weight averaging and includes B so its magnitude
is relative to the native chi inertia.  Unit Hessian blocks are built once;
all spectra are then exact linear combinations of action Hessians.
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


def chi_midpoint(left: torch.Tensor, right: torch.Tensor):
    _lp, left_chi = direct.fields(left)
    _rp, right_chi = direct.fields(right)
    return ((left_chi + right_chi) / 2,
            (right_chi - left_chi) / spectrum.DT)


def local_berry(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    chi, velocity = chi_midpoint(left, right)
    pairing = torch.sum(torch.conj(chi) * velocity, dim=-1)
    return spectrum.DT * torch.sum(torch.imag(pairing))


def gradient_berry(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    chi, velocity = chi_midpoint(left, right)
    total_weight = sum(w for _o, w
                       in direct.base.inverse.rank.native.DIRECTIONS)
    result = torch.zeros((), dtype=torch.float64)
    for offset, weight in direct.base.inverse.rank.native.DIRECTIONS:
        dchi = torch.roll(chi, shifts=offset, dims=(0, 1, 2)) - chi
        dvelocity = (torch.roll(
            velocity, shifts=offset, dims=(0, 1, 2)) - velocity)
        pairing = torch.sum(torch.conj(dchi) * dvelocity, dim=-1)
        result = result + weight * torch.sum(torch.imag(pairing))
    return spectrum.DT * result / total_weight


def gradient_kinetic(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    _chi, velocity = chi_midpoint(left, right)
    total_weight = sum(w for _o, w
                       in direct.base.inverse.rank.native.DIRECTIONS)
    result = torch.zeros((), dtype=torch.float64)
    for offset, weight in direct.base.inverse.rank.native.DIRECTIONS:
        difference = (torch.roll(
            velocity, shifts=offset, dims=(0, 1, 2)) - velocity)
        result = result + weight * torch.sum(torch.real(
            torch.conj(difference) * difference))
    return (spectrum.DT * direct.base.inverse.rank.B * result
            / (2 * total_weight))


def interval_term(name: str, left: torch.Tensor,
                  right: torch.Tensor) -> torch.Tensor:
    if name == "local_berry":
        return local_berry(left, right)
    if name == "gradient_berry":
        return gradient_berry(left, right)
    if name == "gradient_kinetic":
        return gradient_kinetic(left, right)
    raise ValueError(name)


def three_layer_term(name: str, previous: torch.Tensor,
                     current: torch.Tensor,
                     following: torch.Tensor) -> torch.Tensor:
    return (interval_term(name, previous, current)
            + interval_term(name, current, following))


def blocks(function, variables):
    scalar = function(*variables)
    gradient = torch.autograd.grad(
        scalar, variables[1], create_graph=False)[0].detach().numpy()
    hessian = torch.autograd.functional.hessian(
        function, variables, vectorize=True)
    return gradient, tuple(
        hessian[1][index].detach().numpy() for index in range(3))


def classify(label: str, coefficient: float, base_gradient: np.ndarray,
             base_blocks: tuple[np.ndarray, ...],
             delta_gradient: np.ndarray,
             delta_blocks: tuple[np.ndarray, ...],
             rotation: np.ndarray) -> dict:
    gradient = base_gradient + coefficient * delta_gradient
    previous_block, current_block, following_block = tuple(
        base + coefficient * delta
        for base, delta in zip(base_blocks, delta_blocks))
    lhs = following_block @ rotation
    singular = np.linalg.svd(lhs, compute_uv=False)
    rank = int(np.sum(singular > singular[0] * 1e-12))
    row = {
        "basis": label,
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
    variables = tuple(torch.tensor(value, dtype=torch.float64,
                                   requires_grad=True)
                      for value in (previous, current, following))
    values = triangle.coefficients(theta)

    def base_action(pm, q, pp):
        return (triangle.triangle_interval(pm, q, values)
                + triangle.triangle_interval(q, pp, values))

    base_gradient, base_blocks = blocks(base_action, variables)
    deltas = {}
    for name in ("local_berry", "gradient_berry", "gradient_kinetic"):
        deltas[name] = blocks(
            lambda pm, q, pp, name=name: three_layer_term(name, pm, q, pp),
            variables)
    rotation = relative.raw_rotation_matrix(spectrum.SIDE, theta, 0.0, 1)
    natural_gyro = direct.base.inverse.rank.B * direct.base.inverse.rank.R
    specifications = [
        ("baseline", 0.0, np.zeros_like(base_gradient),
         tuple(np.zeros_like(value) for value in base_blocks)),
        ("local_berry_positive", natural_gyro, *deltas["local_berry"]),
        ("local_berry_negative", -natural_gyro, *deltas["local_berry"]),
        ("gradient_berry_positive", natural_gyro,
         *deltas["gradient_berry"]),
        ("gradient_berry_negative", -natural_gyro,
         *deltas["gradient_berry"]),
        ("gradient_kinetic", 1.0, *deltas["gradient_kinetic"]),
    ]
    rows = [classify(label, coefficient, base_gradient, base_blocks,
                     delta_gradient, delta_blocks, rotation)
            for label, coefficient, delta_gradient, delta_blocks
            in specifications]
    baseline_radius = rows[0]["spectral_radius"]
    survivors = [row for row in rows[1:]
                 if row["rest_DEL_rms"] < 1e-8
                 and row["spectral_radius"] is not None
                 and row["spectral_radius"] < baseline_radius - 1e-7]
    result = {
        "status": "V4_C19_CHI_PHASE_SPACE_BASIS_GATE",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "bases": [
            "local chi Berry one-form",
            "C19-gradient chi Berry one-form",
            "positive C19-gradient chi kinetic metric",
        ],
        "coefficient_rule": (
            "gyro magnitude B*chi0 with both signs; positive kinetic unit "
            "strength relative to native B after C19 averaging; no fitting"
        ),
        "rows": rows,
        "surviving_bases": [row["basis"] for row in survivors],
        "verdict": (
            "C19_CHI_PHASE_SPACE_BASIS_HAS_IMPROVEMENT"
            if survivors else "C19_CHI_PHASE_SPACE_BASIS_NO_IMPROVEMENT"
        ),
        "claim_boundary": (
            "Reduced chi-only phase-space basis exhaustion at the complete "
            "side-three spectrum gate. It does not test general joint "
            "Psi--chi field-dependent kinetic connections."
        ),
    }
    output = HERE / "c19_chi_phase_space_basis_gate.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
