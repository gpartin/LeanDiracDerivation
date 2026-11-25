"""Full two-layer rest spectrum of the density-driven chi mold metric.

This is the admission gate following ``density_mold_connection_probe.py``.
It differentiates the complete self-adjoint discrete action, rather than
placing the field-dependent kinetic square inside ordinary Verlet.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.linalg import eigvals
import torch

import density_mold_connection_probe as mold
import edge_snap_action_rank as rank
import relative_kinetic_probe as relative


HERE = Path(__file__).resolve().parent
SIDE = 3
DT = 1.0 / 256.0


def potential(raw: torch.Tensor) -> torch.Tensor:
    psi, chi = mold.unpack_tensor(raw, SIDE)
    rho_psi = torch.real(torch.sum(torch.conj(psi) * psi, dim=-1))
    rho_chi = torch.real(torch.sum(torch.conj(chi) * chi, dim=-1))
    local = torch.sum(
        0.25 * (rho_chi - rank.R**2)**2
        + 0.5 * rho_chi * rho_psi
        + 0.25 * rho_psi**2)
    area = torch.linalg.cross(chi, psi, dim=-1)
    bond = torch.zeros((), dtype=torch.float64)
    for offset, weight in rank.native.DIRECTIONS:
        difference = area - torch.roll(
            area, shifts=offset, dims=(0, 1, 2))
        bond = bond + weight * torch.sum(mold.squared_norm(difference))
    return local + bond / (2 * rank.R**2)


def interval_action(left: torch.Tensor, right: torch.Tensor,
                    coefficient: float) -> torch.Tensor:
    left_psi, left_chi = mold.unpack_tensor(left, SIDE)
    right_psi, right_chi = mold.unpack_tensor(right, SIDE)
    psi = (left_psi + right_psi) / 2
    chi = (left_chi + right_chi) / 2
    psi_dot = (right_psi - left_psi) / DT
    chi_dot = (right_chi - left_chi) / DT
    rho_psi_dot = 2 * torch.real(torch.sum(
        torch.conj(psi) * psi_dot, dim=-1))
    rho_chi = torch.real(torch.sum(torch.conj(chi) * chi, dim=-1))
    denominator = rho_chi + rank.KAPPA * rank.R**2
    velocity = (
        chi_dot
        + coefficient * rho_psi_dot[..., None] * chi
        / (2 * denominator[..., None]))
    averaged = mold.adjacency_average(velocity)
    kinetic = 0.5 * torch.sum(mold.squared_norm(psi_dot))
    kinetic = kinetic + rank.B / 2 * (
        rank.KAPPA * torch.sum(mold.squared_norm(chi_dot))
        + torch.sum(mold.squared_norm(averaged)))
    return DT * (kinetic - (potential(left) + potential(right)) / 2)


def orbit_layers():
    amplitude_sq = rank.R**2 / (2 + 54 / rank.R**2)
    psi, chi = rank.native.carrier_fields(SIDE, amplitude_sq)
    theta = float(np.arccos(1 - DT**2 * rank.R**2 / 2))
    previous = rank.pack(np.exp(-1j * theta) * psi, chi)
    current = rank.pack(psi, chi)
    following = rank.pack(np.exp(1j * theta) * psi, chi)
    return tuple(torch.tensor(value, dtype=torch.float64)
                 for value in (previous, current, following)), theta


def classify(coefficient: float):
    layers, theta = orbit_layers()

    def two_interval_action(previous, current, following):
        return (interval_action(previous, current, coefficient)
                + interval_action(current, following, coefficient))

    variables = tuple(value.detach().requires_grad_(True) for value in layers)
    action = two_interval_action(*variables)
    rest_gradient = torch.autograd.grad(
        action, variables[1], create_graph=False)[0].detach().numpy()
    blocks = torch.autograd.functional.hessian(
        two_interval_action, variables, vectorize=True)
    previous_jacobian = blocks[1][0].detach().numpy()
    current_jacobian = blocks[1][1].detach().numpy()
    following_jacobian = blocks[1][2].detach().numpy()
    rotation = relative.raw_rotation_matrix(SIDE, theta, 0.0, 1)
    inverse_rotation = rotation.T
    lhs = following_jacobian @ rotation
    singular_values = np.linalg.svd(lhs, compute_uv=False)
    condition = float(singular_values[0] / max(singular_values[-1], 1e-300))
    rank_value = int(np.linalg.matrix_rank(
        lhs, tol=singular_values[0] * 1e-12))
    if rank_value < lhs.shape[0]:
        return {
            "coefficient": coefficient,
            "rest_del_rms_residual": float(
                np.linalg.norm(rest_gradient) / np.sqrt(rest_gradient.size)),
            "following_block_condition_number": condition,
            "following_block_rank": rank_value,
            "following_block_dimension": int(lhs.shape[0]),
            "smallest_singular_value": float(singular_values[-1]),
            "spectral_radius": None,
            "growth_rate_per_unit_time": None,
            "reciprocal_pairing_defect": None,
        }
    forward = -np.linalg.solve(lhs, current_jacobian)
    backward = -np.linalg.solve(
        lhs, previous_jacobian @ inverse_rotation)
    identity = np.eye(forward.shape[0])
    zero = np.zeros_like(identity)
    monodromy = np.block([[forward, backward], [identity, zero]])
    values = eigvals(monodromy, check_finite=False)
    moduli = np.sort(abs(values))
    radius = float(moduli[-1])
    return {
        "coefficient": coefficient,
        "rest_del_rms_residual": float(
            np.linalg.norm(rest_gradient) / np.sqrt(rest_gradient.size)),
        "following_block_condition_number": condition,
        "following_block_rank": rank_value,
        "following_block_dimension": int(lhs.shape[0]),
        "smallest_singular_value": float(singular_values[-1]),
        "spectral_radius": radius,
        "growth_rate_per_unit_time": float(np.log(radius) / DT),
        "reciprocal_pairing_defect": float(np.max(
            abs(moduli * moduli[::-1] - 1))),
    }


def main() -> None:
    rank.lock_inertia()
    rows = [classify(0.0), classify(mold.ALPHA)]
    incumbent = rows[0]["growth_rate_per_unit_time"]
    candidate = rows[1]["growth_rate_per_unit_time"]
    well_posed = incumbent is not None and candidate is not None
    relative_change = ((incumbent - candidate) / incumbent
                       if well_posed else None)
    preserved = well_posed and abs(relative_change) < 1e-8
    improved = well_posed and relative_change >= 1e-8
    result = {
        "status": ("V4_DENSITY_MOLD_REST_GATE_PASS" if improved else
                   "V4_DENSITY_MOLD_REST_GATE_FAIL"),
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "side": SIDE,
        "dt": DT,
        "action": (
            "T_chi=(B/2)[kappa|chiDot|^2+|A19 Vchi/24|^2], "
            "Vchi=chiDot+s rhoPsiDot chi/[2(rhoChi+kappa chi0^2)]"),
        "rows": rows,
        "candidate_improves_incumbent_rest_growth": improved,
        "candidate_preserves_incumbent_rest_growth": preserved,
        "relative_growth_improvement": (float(relative_change)
                                         if well_posed else None),
        "verdict": (
            "ADVANCE_TO_RELATIVE_PERIODIC_ORBIT_SEARCH"
            if improved or preserved else
            ("REST_UPDATE_NOT_WELL_POSED" if not well_posed else
             "RETIRE_DENSITY_MOLD_CONNECTION")),
        "claim_boundary": (
            "This is the complete side-3 two-layer linear rest spectrum of "
            "the implicit discrete action. It is not a nonlinear long-time "
            "stability or moving-electron pass."),
    }
    (HERE / "density_mold_rest_spectrum.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
