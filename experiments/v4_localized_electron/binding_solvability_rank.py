"""Fredholm solvability test for native chi communication actions.

Before an expensive nonlinear continuation, project each complete action
gradient through the exact fixed-charge relative-equilibrium Jacobian.  A
nonzero cokernel component means that the old localized orbit has no smooth
branch under that action perturbation.  This distinguishes a missing response
channel from a coefficient/solver problem.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.optimize._numdiff import approx_derivative
import torch

import localized_stability as native
import unrestricted_binding_profile as branch


HERE = Path(__file__).resolve().parent
torch.set_default_dtype(torch.float64)


def torch_fields(x: torch.Tensor, side: int):
    data = x.reshape(side, side, side, 4, 3)
    psi = torch.complex(data[..., 0, :], data[..., 1, :])
    chi = torch.complex(data[..., 2, :], data[..., 3, :]) / np.sqrt(native.B)
    return psi, chi


def stiffness_energy(x: torch.Tensor, side: int, kind: str):
    psi, chi = torch_fields(x, side)
    p = torch.sum(torch.abs(psi)**2, dim=-1)
    energy = x.new_zeros(())
    for offset, weight in native.DIRECTIONS:
        axes = (0, 1, 2)
        shifts = tuple(-value for value in offset)
        neighbor_chi = torch.roll(chi, shifts, axes)
        neighbor_p = torch.roll(p, shifts, axes)
        if kind in ("uniform", "chi_equal_action"):
            difference = neighbor_chi - chi
            factor = 1.0
            scale = native.B if kind == "uniform" else 1.0
        elif kind == "psi_uniform":
            neighbor_psi = torch.roll(psi, shifts, axes)
            difference = neighbor_psi - psi
            factor = 1.0
            scale = 1.0
        elif kind in ("two_wave_equal_action", "two_wave_equal_speed"):
            neighbor_psi = torch.roll(psi, shifts, axes)
            psi_difference = neighbor_psi - psi
            chi_difference = neighbor_chi - chi
            chi_scale = (1.0 if kind == "two_wave_equal_action"
                         else native.B)
            energy = energy + (weight / (4 * branch.WEIGHT_SUM)
                               * (torch.sum(torch.abs(psi_difference)**2)
                                  + chi_scale * torch.sum(
                                      torch.abs(chi_difference)**2)))
            continue
        elif kind in ("joint_projector6", "joint_density6"):
            local_z = torch.cat((chi, psi), dim=-1)
            neighbor_psi = torch.roll(psi, shifts, axes)
            neighbor_z = torch.cat((neighbor_chi, neighbor_psi), dim=-1)
            local_matrix = (local_z[..., :, None]
                            * torch.conj(local_z[..., None, :]))
            neighbor_matrix = (neighbor_z[..., :, None]
                               * torch.conj(neighbor_z[..., None, :]))
            if kind == "joint_projector6":
                local_matrix = local_matrix / torch.sum(
                    torch.abs(local_z)**2, dim=-1)[..., None, None]
                neighbor_matrix = neighbor_matrix / torch.sum(
                    torch.abs(neighbor_z)**2,
                    dim=-1)[..., None, None]
                scale = native.R**2
            else:
                scale = 1 / native.R**2
            difference = neighbor_matrix - local_matrix
            factor = 1.0
        elif kind.startswith("inverse_mean_"):
            power = int(kind.rsplit("_", 1)[1])
            factor = (native.R**2 / (
                native.R**2 + (p + neighbor_p) / 2))**power
            difference = neighbor_chi - chi
            scale = native.B
        elif kind == "inverse_product":
            factor = (native.R**4 / (
                (native.R**2 + p) * (native.R**2 + neighbor_p)))
            difference = neighbor_chi - chi
            scale = native.B
        elif kind == "normalized_joint":
            local = chi / torch.sqrt(native.R**2 + p)[..., None]
            neighbor = neighbor_chi / torch.sqrt(
                native.R**2 + neighbor_p)[..., None]
            difference = neighbor - local
            factor = 1.0
            scale = native.B * native.R**2
        else:
            raise ValueError(kind)
        energy = energy + (weight * scale / (4 * branch.WEIGHT_SUM)
                           * torch.sum(factor * torch.sum(
                               torch.abs(difference)**2, dim=-1)))
    return energy


def action_force_vector(x: np.ndarray, side: int, kind: str):
    variable = torch.tensor(x, requires_grad=True)
    energy = stiffness_energy(variable, side, kind)
    gradient, = torch.autograd.grad(energy, variable)
    return -gradient.detach().numpy()


def build_problem(side: int, lam: float, coupling: float,
                  amplitude_sq: float):
    psi, chi, op, oc = branch.rigid_seed(side, lam, coupling, amplitude_sq)
    target = branch.charges(psi, chi, op, oc)
    size = 12 * side**3
    initial = np.concatenate((native.pack(psi, chi), (op, oc)))
    carrier_sites = np.argwhere(native.norm2(psi) > 0)
    anchor = tuple(int(value) for value in carrier_sites[0])
    block = 12 * np.ravel_multi_index(anchor, (side, side, side))
    drop = np.asarray((block + 3, block + 10), dtype=int)
    scale = np.sqrt(size)
    psi_norm = max(float(np.linalg.norm(psi)), 1.0)
    chi_plane = chi.copy()
    chi_plane[..., 2] = 0
    chi_norm = max(float(np.linalg.norm(chi_plane)), 1.0)

    def unpack(parameters):
        qp, qc = native.unpack(parameters[:size], side)
        return qp, qc, float(parameters[-2]), float(parameters[-1])

    def residual(parameters):
        qp, qc, wp, wc = unpack(parameters)
        fp, fc = branch.action_force(qp, qc, lam, coupling, 0.0)
        rp, rc = branch.phase_rates(wp, wc)
        stationary = np.delete(
            native.pack(fp + rp**2 * qp, fc + rc**2 * qc) / scale,
            drop)
        charge_error = ((branch.charges(qp, qc, wp, wc) - target)
                        / np.maximum(abs(target), 1.0))
        gauges = (np.vdot(psi, qp).imag / psi_norm,
                  np.vdot(chi_plane, qc).imag / chi_norm)
        return np.concatenate((stationary, charge_error, gauges))

    return initial, residual, drop


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--side", type=int, default=3)
    parser.add_argument("--lambda-self", type=float, default=0.25)
    parser.add_argument("--coupling", type=float, default=1.0)
    parser.add_argument("--amplitude-sq", type=float, default=250.0)
    parser.add_argument("--output", default="binding_solvability_rank.json")
    args = parser.parse_args()
    native.B = 1197.0
    initial, residual, drop = build_problem(
        args.side, args.lambda_self, args.coupling, args.amplitude_sq)
    baseline = residual(initial)
    jacobian = approx_derivative(
        residual, initial, method="3-point", rel_step=2e-6)
    u, singular, _ = np.linalg.svd(jacobian, full_matrices=False)
    # There is a clear ten-order spectral gap after the last physical singular
    # value.  A relative 1e-9 cutoff rejects finite-difference noise while
    # retaining the complete physical range.
    tolerance = singular[0] * 1e-9
    rank = int(np.sum(singular > tolerance))
    column_basis = u[:, :rank]
    rows = []
    all_names = []
    all_responses = []
    all_obstructions = []
    power_responses = []
    power_obstructions = []
    size = 12 * args.side**3
    kinds = (["uniform", "chi_equal_action", "psi_uniform",
              "two_wave_equal_action", "two_wave_equal_speed",
              "joint_projector6", "joint_density6"]
             + [f"inverse_mean_{power}" for power in range(1, 9)]
             + ["inverse_product", "normalized_joint"])
    for kind in kinds:
        force = action_force_vector(initial[:size], args.side, kind)
        action_residual = np.concatenate((
            np.delete(force / np.sqrt(size), drop), np.zeros(4)))
        projection = column_basis @ (column_basis.T @ action_residual)
        obstruction = action_residual - projection
        all_names.append(kind)
        all_responses.append(action_residual)
        all_obstructions.append(obstruction)
        if kind == "uniform" or kind.startswith("inverse_mean_"):
            power_responses.append(action_residual)
            power_obstructions.append(obstruction)
        rows.append({
            "action": kind,
            "action_response_norm": float(np.linalg.norm(action_residual)),
            "cokernel_obstruction_norm": float(np.linalg.norm(obstruction)),
            "cokernel_fraction": float(
                np.linalg.norm(obstruction)
                / max(np.linalg.norm(action_residual), 1e-30)),
        })
    obstruction_matrix = np.column_stack(power_obstructions)
    response_matrix = np.column_stack(power_responses)
    powers = np.arange(obstruction_matrix.shape[1])
    z_grid = np.concatenate((np.linspace(0, 4, 161),
                             np.geomspace(4.1, 1e4, 120)))
    h_grid = (1 + z_grid[:, None]) ** (-powers[None, :])

    def objective(coefficients):
        value = obstruction_matrix @ coefficients
        return float(value @ value)

    constraints = [
        {"type": "eq", "fun": lambda c: float(np.sum(c) - 1)},
        {"type": "ineq", "fun": lambda c: h_grid @ c},
        {"type": "ineq", "fun": lambda c: np.asarray((c[0],))},
    ]
    initial_coefficients = np.zeros(len(powers))
    initial_coefficients[0] = 1
    combination = minimize(
        objective, initial_coefficients, method="SLSQP",
        bounds=[(-100.0, 100.0)] * len(powers), constraints=constraints,
        options={"ftol": 1e-24, "maxiter": 3000})
    coefficients = combination.x
    combined_obstruction = obstruction_matrix @ coefficients
    combined_response = response_matrix @ coefficients
    combination_result = {
        "solver_success": bool(combination.success),
        "solver_message": str(combination.message),
        "coefficients_by_inverse_power": [float(x) for x in coefficients],
        "vacuum_normalization": float(np.sum(coefficients)),
        "minimum_sampled_metric": float(np.min(h_grid @ coefficients)),
        "asymptotic_metric": float(coefficients[0]),
        "response_norm": float(np.linalg.norm(combined_response)),
        "cokernel_obstruction_norm": float(np.linalg.norm(
            combined_obstruction)),
        "cokernel_fraction": float(np.linalg.norm(combined_obstruction)
                                   / max(np.linalg.norm(combined_response),
                                         1e-30)),
    }
    # Test whether a nonnegative sum of the simplest complete positive action
    # energies can meet the Fredholm condition.  This is a cone feasibility
    # test, not coefficient fitting to a physical observable.
    cone_names = ["chi_equal_action", "psi_uniform", "joint_projector6",
                  "joint_density6", "inverse_mean_1", "inverse_mean_2"]
    cone_indices = [all_names.index(name) for name in cone_names]
    cone_o = np.column_stack([all_obstructions[i] for i in cone_indices])
    cone_r = np.column_stack([all_responses[i] for i in cone_indices])
    cone_initial = np.full(len(cone_names), 1 / len(cone_names))
    cone = minimize(
        lambda c: float(np.linalg.norm(cone_o @ c)**2), cone_initial,
        method="SLSQP", bounds=[(0.0, 1.0)] * len(cone_names),
        constraints=[{"type": "eq", "fun": lambda c: float(
            np.sum(c) - 1)}], options={"ftol": 1e-24, "maxiter": 3000})
    cone_obstruction = cone_o @ cone.x
    cone_response = cone_r @ cone.x
    cone_result = {
        "solver_success": bool(cone.success),
        "solver_message": str(cone.message),
        "coefficients": {name: float(value)
                         for name, value in zip(cone_names, cone.x)},
        "obstruction_norm": float(np.linalg.norm(cone_obstruction)),
        "cokernel_fraction": float(np.linalg.norm(cone_obstruction)
                                   / max(np.linalg.norm(cone_response), 1e-30)),
    }
    result = {
        "status": "V4_FIXED_CHARGE_ACTION_SOLVABILITY_RANK",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "baseline_residual_norm": float(np.linalg.norm(baseline)),
        "jacobian_shape": list(jacobian.shape),
        "jacobian_rank": rank,
        "jacobian_nullity": int(jacobian.shape[1] - rank),
        "rank_relative_tolerance": 1e-9,
        "smallest_retained_singular_value": float(singular[rank - 1]),
        "rows": rows,
        "positive_normalized_inverse_metric_combination": combination_result,
        "positive_simple_action_cone": cone_result,
        "decision_rule": (
            "Only an action with cokernel fraction numerically zero admits a "
            "smooth fixed-charge branch from this exact carrier."),
    }
    (HERE / args.output).write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
