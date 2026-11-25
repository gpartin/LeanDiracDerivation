"""Inverse phase test for generic positive C19 two-field bond squares.

The canonical area bond leaves the retained eight-corner twist at a
high-index phase saddle.  Before invoking a fourth harmonic, this audit asks
whether the same curvature can emerge from the smallest positive bond terms
built from native Psi/chi vectors while preserving their two independent
global phase symmetries.

For face and edge bonds separately the tested squares are:

* ``|Psi_j-Psi_i|^2`` and ``|chi_j-chi_i|^2``;
* ``|Psi_i^dagger Psi_j +/- chi_i^dagger chi_j|^2``;
* ``|chi_i^dagger Psi_j +/- conjugate(Psi_i^dagger chi_j)|^2``.

All coefficients are constrained nonnegative.  The exact phase torque must
vanish.  A cutting-plane linear program then asks whether these Hessians can
make every non-global target phase direction positive.  The calculation is a
cheap action-basis gate, not an admission of any term.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
from scipy.linalg import null_space
from scipy.optimize import linprog
import torch

import localized_stability as native


HERE = Path(__file__).resolve().parent
torch.set_default_dtype(torch.float64)
CORNERS = list(itertools.product((0, 1), repeat=3))


def norm_sq(value: torch.Tensor) -> torch.Tensor:
    return torch.sum(value.real.square() + value.imag.square())


def bond_list(distance: int):
    return [(i, j) for i in range(8) for j in range(i + 1, 8)
            if sum(a != b for a, b in zip(CORNERS[i], CORNERS[j])) == distance]


def phased_fields(phases: torch.Tensor, psi0: torch.Tensor,
                  chi0: torch.Tensor):
    psi = psi0 * torch.exp(1j * phases[:8, None])
    chi = chi0 * torch.exp(1j * phases[8:, None])
    return psi, chi


def energy(phases: torch.Tensor, psi0: torch.Tensor, chi0: torch.Tensor,
           family: str, distance: int):
    psi, chi = phased_fields(phases, psi0, chi0)
    result = phases.new_zeros(())
    for i, j in bond_list(distance):
        if family == "psi_difference":
            value = psi[j] - psi[i]
        elif family == "chi_difference":
            value = chi[j] - chi[i]
        else:
            pp = torch.sum(psi[i].conj() * psi[j])
            cc = torch.sum(chi[i].conj() * chi[j])
            cp = torch.sum(chi[i].conj() * psi[j])
            pc = torch.sum(psi[i].conj() * chi[j])
            if family == "parallel_minus":
                value = pp - cc
            elif family == "parallel_plus":
                value = pp + cc
            elif family == "cross_minus":
                value = cp - pc.conj()
            elif family == "cross_plus":
                value = cp + pc.conj()
            else:
                raise ValueError(family)
        result = result + norm_sq(value) / 2
    return result


def canonical_area_energy(phases: torch.Tensor, psi0: torch.Tensor,
                          chi0: torch.Tensor):
    psi, chi = phased_fields(phases, psi0, chi0)
    area = torch.linalg.cross(chi, psi, dim=-1)
    result = phases.new_zeros(())
    for distance, weight in ((1, 2.0), (2, 1.0)):
        for i, j in bond_list(distance):
            result = result + weight * norm_sq(area[j] - area[i]) / 2
    return result


def derivatives(function, origin):
    gradient = torch.autograd.functional.jacobian(
        function, origin, vectorize=True).detach().numpy()
    hessian = torch.autograd.functional.hessian(
        function, origin, vectorize=True).detach().numpy()
    return gradient, (hessian + hessian.T) / 2


def solve_cutting_plane(base_hessian, gradients, hessians, quotient,
                        target_curvature=1.0, maximum_iterations=100):
    projected_base = quotient.T @ base_hessian @ quotient
    projected = [quotient.T @ hessian @ quotient for hessian in hessians]
    equality = np.vstack((np.column_stack(gradients),))
    # Remove numerically empty/redundant stationarity rows using an SVD basis.
    row_basis = null_space(equality).T
    # ``row_basis`` spans coefficient combinations satisfying G x = 0, but
    # linprog accepts the original equations more transparently.  Keep only
    # independent rows of G by pivoting its transpose.
    from scipy.linalg import qr
    _, _, pivots = qr(equality.T, pivoting=True, mode="economic")
    rank = int(np.linalg.matrix_rank(equality, tol=1e-10))
    independent = equality[np.asarray(pivots[:rank], dtype=int)] if rank else None
    cone_equalities = np.vstack((independent, np.ones(len(hessians))))
    cone_rhs = np.r_[np.zeros(rank), 1.0]
    cone_test = linprog(
        np.zeros(len(hessians)), A_eq=cone_equalities, b_eq=cone_rhs,
        bounds=[(0, None)] * len(hessians), method="highs")
    cone_summary = {
        "nonzero_stationary_positive_combination_exists": bool(cone_test.success),
        "message": cone_test.message,
    }
    if cone_test.success:
        cone_summary["example_coefficients"] = cone_test.x.tolist()
        cone_hessian = sum(
            coefficient * matrix
            for coefficient, matrix in zip(cone_test.x, projected))
        cone_summary["example_candidate_curvatures"] = (
            np.linalg.eigvalsh(cone_hessian).tolist())
        cone_summary["example_base_plus_candidate_curvatures"] = (
            np.linalg.eigvalsh(projected_base + cone_hessian).tolist())

    values, vectors = np.linalg.eigh(projected_base)
    constraints = [vectors[:, index] for index in range(len(values))
                   if values[index] < target_curvature]
    history = []
    solution = None
    for iteration in range(maximum_iterations):
        aub, bub = [], []
        for vector in constraints:
            response = np.asarray([vector @ matrix @ vector
                                   for matrix in projected])
            aub.append(-response)
            bub.append(float(vector @ projected_base @ vector
                             - target_curvature))
        solution = linprog(
            np.ones(len(hessians)),
            A_ub=np.asarray(aub) if aub else None,
            b_ub=np.asarray(bub) if bub else None,
            A_eq=independent,
            b_eq=np.zeros(rank) if rank else None,
            bounds=[(0, None)] * len(hessians), method="highs")
        if not solution.success:
            return {
                "feasible": False,
                "iterations": iteration + 1,
                "message": solution.message,
                "stationarity_rank": rank,
                "stationary_nonnegative_cone_dimension": int(row_basis.shape[0]),
                "stationary_positive_cone": cone_summary,
                "history": history,
            }
        combined = projected_base + sum(
            coefficient * matrix
            for coefficient, matrix in zip(solution.x, projected))
        values, vectors = np.linalg.eigh(combined)
        minimum = float(values[0])
        history.append({
            "iteration": iteration + 1,
            "minimum_curvature": minimum,
            "coefficient_l1": float(np.sum(solution.x)),
        })
        if minimum >= target_curvature - 1e-8:
            return {
                "feasible": True,
                "iterations": iteration + 1,
                "message": solution.message,
                "stationarity_rank": rank,
                "stationary_nonnegative_cone_dimension": int(row_basis.shape[0]),
                "stationary_positive_cone": cone_summary,
                "coefficients": solution.x.tolist(),
                "minimum_curvature": minimum,
                "all_curvatures": values.tolist(),
                "phase_torque_max_abs": float(np.max(np.abs(
                    equality @ solution.x))),
                "coefficient_l1": float(np.sum(solution.x)),
                "history": history,
            }
        constraints.append(vectors[:, 0])
    return {
        "feasible": False,
        "iterations": maximum_iterations,
        "message": "cutting-plane iteration limit",
        "stationarity_rank": rank,
        "stationary_positive_cone": cone_summary,
        "history": history,
    }


def main():
    psi_field, chi_field = native.carrier_fields(3, 1.0)
    sites = [tuple(int(x) for x in site)
             for site in np.argwhere(native.norm2(psi_field) > 0)]
    psi0 = torch.as_tensor(np.asarray([psi_field[site] for site in sites]))
    chi0 = torch.as_tensor(np.asarray([chi_field[site] for site in sites]))
    origin = torch.zeros(16, requires_grad=True)

    base_gradient, base_hessian = derivatives(
        lambda x: canonical_area_energy(x, psi0, chi0), origin)
    families = (
        "psi_difference", "chi_difference", "parallel_minus",
        "parallel_plus", "cross_minus", "cross_plus")
    labels = [(family, distance) for family in families for distance in (1, 2)]
    responses = [derivatives(
        lambda x, family=family, distance=distance: energy(
            x, psi0, chi0, family, distance), origin)
        for family, distance in labels]
    gradients = [item[0] for item in responses]
    hessians = [item[1] for item in responses]

    symmetries = np.zeros((16, 2))
    symmetries[:8, 0] = 1 / np.sqrt(8)
    symmetries[8:, 1] = 1 / np.sqrt(8)
    quotient = null_space(symmetries.T)
    solution = solve_cutting_plane(
        base_hessian, gradients, hessians, quotient)
    if solution.get("coefficients") is not None:
        solution["labeled_coefficients"] = [
            {"family": family, "distance": distance, "coefficient": float(value)}
            for (family, distance), value in zip(labels, solution["coefficients"])
            if value > 1e-10]
    base_values = np.linalg.eigvalsh(quotient.T @ base_hessian @ quotient)
    result = {
        "status": "V4_NATIVE_POSITIVE_BOND_PHASE_BASIS",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "tested_families": list(families),
        "labels": [[family, distance] for family, distance in labels],
        "canonical_phase_torque_max_abs": float(np.max(np.abs(base_gradient))),
        "canonical_quotient_curvatures": base_values.tolist(),
        "candidate_gradient_norms": [float(np.linalg.norm(x)) for x in gradients],
        "solution": solution,
        "verdict": (
            "GENERIC_POSITIVE_BOND_BASIS_CAN_STABILIZE_PHASES"
            if solution["feasible"] else
            "GENERIC_POSITIVE_BOND_BASIS_NO_GO"),
        "claim_boundary": (
            "Phase-only inverse gate.  Feasibility does not establish an "
            "exact full-field orbit, bounded complete spectrum, coefficient "
            "provenance, transport, or Dirac behavior."),
    }
    (HERE / "native_phase_basis.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
