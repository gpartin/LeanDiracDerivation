"""Find generic potential partners for the movement-positive direct chi bond."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

import direct_chi_bond_rest_continuation as continuation
import edge_snap_action_rank as rank


HERE = Path(__file__).resolve().parent


def local_terms(raw: torch.Tensor) -> dict[str, torch.Tensor]:
    psi, chi = continuation.local.unpack_tensor(raw, continuation.SIDE)
    rho_psi = torch.sum(continuation.local.squared_norm(psi), dim=-1)
    rho_chi = torch.sum(continuation.local.squared_norm(chi), dim=-1)
    return {
        "local_chi_vacuum": torch.sum((rho_chi - rank.R**2)**2) / 4,
        "local_chi_psi": torch.sum(rho_chi * rho_psi) / 2,
        "local_psi_quartic": torch.sum(rho_psi**2) / 4,
    }


def bond_terms(raw: torch.Tensor) -> dict[str, torch.Tensor]:
    psi, chi = continuation.local.unpack_tensor(raw, continuation.SIDE)
    terms = {
        "direct_psi_bond": torch.zeros((), dtype=torch.float64),
        "direct_chi_bond": torch.zeros((), dtype=torch.float64),
        "area_bond": torch.zeros((), dtype=torch.float64),
        "parallel_overlap_bond": torch.zeros((), dtype=torch.float64),
    }
    wedge = torch.linalg.cross(chi, psi, dim=-1)
    for offset, weight in rank.native.DIRECTIONS:
        psi_neighbor = torch.roll(psi, shifts=offset, dims=(0, 1, 2))
        chi_neighbor = torch.roll(chi, shifts=offset, dims=(0, 1, 2))
        wedge_neighbor = torch.roll(wedge, shifts=offset, dims=(0, 1, 2))
        terms["direct_psi_bond"] = terms["direct_psi_bond"] + weight * torch.sum(
            continuation.local.squared_norm(psi_neighbor - psi))
        terms["direct_chi_bond"] = terms["direct_chi_bond"] + weight * torch.sum(
            continuation.local.squared_norm(chi_neighbor - chi))
        terms["area_bond"] = terms["area_bond"] + weight * torch.sum(
            continuation.local.squared_norm(wedge_neighbor - wedge))
        overlap = torch.sum(torch.conj(chi) * psi_neighbor, dim=-1)
        reverse = torch.sum(torch.conj(psi) * chi_neighbor, dim=-1)
        terms["parallel_overlap_bond"] = (
            terms["parallel_overlap_bond"] + weight * torch.sum(
                continuation.local.squared_norm(overlap - torch.conj(reverse))))
    normalization = 2 * rank.C19_WEIGHT_SUM
    for name in terms:
        terms[name] = terms[name] / normalization
    terms["area_bond"] = terms["area_bond"] / rank.R**2
    terms["direct_chi_bond"] = terms["direct_chi_bond"] * rank.B / 6
    return terms


def main() -> None:
    rank.lock_inertia()
    layers, theta = continuation.transfer_rest.rest.orbit_layers()
    reference = layers[1].detach()
    shape = (continuation.SIDE,) * 3 + (4, 3)
    fixed = np.asarray([
        np.ravel_multi_index((0, 0, 0, 1, 0), shape),
        np.ravel_multi_index((2, 2, 2, 3, 2), shape),
    ])
    free_mask = np.ones(reference.numel(), dtype=bool)
    free_mask[fixed] = False
    free = torch.tensor(np.flatnonzero(free_mask), dtype=torch.long)
    base = reference.clone()
    values = torch.tensor(np.concatenate((
        base[free].numpy(), np.asarray([float(theta), 0.0]))),
        dtype=torch.float64, requires_grad=True)

    def baseline_residual(parameters: torch.Tensor) -> torch.Tensor:
        current = base.clone()
        current[free] = parameters[:-2]
        return continuation.del_residual(
            current, parameters[-2], parameters[-1], 0.0) / rank.R

    jacobian = torch.autograd.functional.jacobian(
        baseline_residual, values, vectorize=True).detach().numpy()
    u, singular, _ = np.linalg.svd(jacobian, full_matrices=True)
    numerical_rank = int(np.sum(singular > singular[0] * 1e-12))
    cokernel = u[:, numerical_rank:]

    independent = reference.clone().requires_grad_(True)
    potentials = {**local_terms(independent), **bond_terms(independent)}
    names = list(potentials)
    columns = []
    for name in names:
        gradient = torch.autograd.grad(
            potentials[name], independent, retain_graph=True)[0]
        # Two half-step potential contributions give -DT grad(U) in the DEL.
        columns.append((-continuation.DT * gradient / rank.R).detach().numpy())
    matrix = np.column_stack(columns)
    projected = cokernel.T @ matrix
    direct_index = names.index("direct_chi_bond")
    partner_indices = [index for index in range(len(names))
                       if index != direct_index]
    target = -projected[:, direct_index]
    partner_matrix = projected[:, partner_indices]
    scales = np.linalg.norm(partner_matrix, axis=0)
    active = scales > 1e-14
    normalized = partner_matrix[:, active] / scales[active]
    normalized_coefficients, _, _, _ = np.linalg.lstsq(
        normalized, target, rcond=1e-12)
    coefficients = np.zeros(len(partner_indices))
    coefficients[active] = normalized_coefficients / scales[active]
    combined = projected[:, direct_index] + partner_matrix @ coefficients
    original_norm = np.linalg.norm(projected[:, direct_index])
    combined_norm = np.linalg.norm(combined)
    full_coefficients = {"direct_chi_bond": 1.0}
    full_coefficients.update({
        names[index]: float(coefficient)
        for index, coefficient in zip(partner_indices, coefficients)
        if abs(coefficient) > 1e-10
    })
    result = {
        "status": "V4_DIRECT_CHI_BOND_PAIRED_REST_BASIS",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "basis": names,
        "baseline_jacobian_rank": numerical_rank,
        "baseline_cokernel_dimension": int(cokernel.shape[1]),
        "projected_partner_rank": int(np.linalg.matrix_rank(
            partner_matrix, tol=1e-10)),
        "direct_chi_obstruction_norm": float(original_norm),
        "combined_obstruction_norm": float(combined_norm),
        "relative_obstruction_after_pairing": float(
            combined_norm / original_norm),
        "squared_obstruction_removed": float(
            1 - (combined_norm / original_norm)**2),
        "least_squares_coefficients": full_coefficients,
        "verdict": (
            "GENERIC_PARTNER_BASIS_CANCELS_REST_OBSTRUCTION"
            if combined_norm / original_norm < 1e-6 else
            "GENERIC_PARTNER_BASIS_INSUFFICIENT"),
        "claim_boundary": (
            "The coefficients solve only the first-order rest Fredholm "
            "condition. They are not yet derived, bounded, movement-tested, "
            "or admitted."),
    }
    (HERE / "direct_chi_bond_paired_rest_basis.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
