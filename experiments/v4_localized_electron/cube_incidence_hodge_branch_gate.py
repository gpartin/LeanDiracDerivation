"""Implicit-function gate for a deformed rest orbit with C19 Hodge coupling.

The exact old carrier is not stationary after any of the four complete Hodge
one-forms is added.  That is admissible only if the *same native action* has a
nearby freely solved rotating orbit.  At first order the branch tangent dq
satisfies

    K dq = f_H / dt,

where K is the full research-action rotating-frame stiffness and f_H is the
Hodge DEL force on the old orbit.  This script checks the Fredholm condition,
solves that equation without fitting a Dirac target, and verifies directly
that the nonlinear residual falls from O(epsilon) to O(epsilon^2).

No Hodge coefficient is selected or admitted here.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

import canonical_noncoplanar_bifurcation_audit as native_time
import canonical_transfer_oscillator_spectrum as native_spectrum
import cube_incidence_hodge_rest_gate as hodge
import cube_incidence_memory_channel_ablation as ablation
import cube_incidence_nonlinear_action as action
import personal_touch_unit_geometric_canonical as canonical


HERE = Path(__file__).resolve().parent
EPSILONS = (1e-3, 3e-4, 1e-4, 3e-5)


def orbit_layers(raw: np.ndarray, rotation: np.ndarray):
    return tuple(torch.tensor(value, dtype=torch.float64, requires_grad=True)
                 for value in (rotation.T@raw, raw, rotation@raw))


def total_residual(raw: np.ndarray, rotation: np.ndarray, coefficient: float,
                   hodge_index: int) -> np.ndarray:
    layers = orbit_layers(raw, rotation)
    hodge_coefficients = torch.zeros(
        len(hodge.HODGE_BASIS), dtype=torch.float64)
    hodge_coefficients[hodge_index] = coefficient
    value = action.del_residual(*layers)
    value = value+hodge_del(layers, hodge_coefficients)
    return value.detach().numpy()


def hodge_del(layers, coefficients):
    return hodge.hodge_del(layers, coefficients)


def main() -> None:
    bare = np.load(canonical.CACHE)
    rest = bare["rest"]
    dt = native_time.DT
    theta = float(np.arccos(1-dt**2*native_time.CHI0**2/2))
    rotation = native_spectrum.rotation_matrix(rest.size, theta)
    layers = orbit_layers(rest, rotation)
    forces = np.column_stack([
        hodge.hodge_force_column(layers, index)
        for index in range(len(hodge.HODGE_BASIS))])

    stiffness = ablation.blocks()["complete_research_action"][2]
    stiffness = (stiffness+stiffness.T)/2
    eigenvalues, eigenvectors = np.linalg.eigh(stiffness)
    scale = max(float(np.max(abs(eigenvalues))), 1e-30)
    threshold = 1e-10*scale
    resolved = abs(eigenvalues) > threshold
    kernel = eigenvectors[:, ~resolved]
    inverse = eigenvectors[:, resolved] / eigenvalues[resolved]

    rows = []
    for index, name in enumerate(hodge.HODGE_BASIS):
        force = forces[:, index]
        kernel_component = (kernel@(kernel.T@force)
                            if kernel.shape[1] else np.zeros_like(force))
        tangent = inverse@(eigenvectors[:, resolved].T@(force/dt))
        linear_residual = -dt*stiffness@tangent+force
        trials = []
        for epsilon in EPSILONS:
            unresolved = total_residual(
                rest, rotation, epsilon, index)
            corrected = total_residual(
                rest+epsilon*tangent, rotation, epsilon, index)
            trials.append({
                "epsilon": epsilon,
                "unresolved_residual_norm": float(np.linalg.norm(unresolved)),
                "corrected_residual_norm": float(np.linalg.norm(corrected)),
                "corrected_over_unresolved": float(
                    np.linalg.norm(corrected)
                    / max(np.linalg.norm(unresolved), 1e-30)),
                "corrected_over_epsilon_squared": float(
                    np.linalg.norm(corrected)/epsilon**2),
            })
        log_epsilon = np.log(np.asarray(EPSILONS))
        log_corrected = np.log(np.asarray(
            [row["corrected_residual_norm"] for row in trials]))
        order = float(np.polyfit(log_epsilon, log_corrected, 1)[0])
        rows.append({
            "hodge_term": name,
            "force_norm": float(np.linalg.norm(force)),
            "kernel_projection_norm": float(np.linalg.norm(kernel_component)),
            "kernel_projection_relative": float(
                np.linalg.norm(kernel_component)
                / max(np.linalg.norm(force), 1e-30)),
            "linear_branch_equation_relative_residual": float(
                np.linalg.norm(linear_residual)
                / max(np.linalg.norm(force), 1e-30)),
            "branch_tangent_norm": float(np.linalg.norm(tangent)),
            "branch_tangent_over_rest_norm": float(
                np.linalg.norm(tangent)/np.linalg.norm(rest)),
            "nonlinear_residual_convergence_order": order,
            "trials": trials,
        })

    compatible = [row for row in rows
                  if row["kernel_projection_relative"] < 1e-8
                  and row["linear_branch_equation_relative_residual"] < 1e-8
                  and row["nonlinear_residual_convergence_order"] > 1.8]
    result = {
        "status": "V4_C19_HODGE_BRANCH_GATE",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "stiffness_dimension": int(stiffness.shape[0]),
        "stiffness_scale": scale,
        "fredholm_threshold": threshold,
        "stiffness_kernel_dimension": int(kernel.shape[1]),
        "rows": rows,
        "compatible_branch_count": len(compatible),
        "compatible_branches": [row["hodge_term"] for row in compatible],
        "verdict": (
            "HODGE_DEFORMED_REST_BRANCH_EXISTS_TO_FIRST_ORDER"
            if compatible else "HODGE_BRANCH_FREDHOLM_OBSTRUCTED"),
        "claim_boundary": (
            "A compatible first-order tangent with quadratic residual proves "
            "a local implicit branch only when the quotient Hessian is "
            "invertible; it does not select a coefficient, prove a finite "
            "branch, motion, stability, or V4-006."),
    }
    (HERE/"cube_incidence_hodge_branch_gate.json").write_text(
        json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps({
        "kernel_dimension": result["stiffness_kernel_dimension"],
        "compatible_branches": result["compatible_branches"],
        "rows": [{
            "hodge_term": row["hodge_term"],
            "kernel_projection_relative": row[
                "kernel_projection_relative"],
            "linear_relative_residual": row[
                "linear_branch_equation_relative_residual"],
            "tangent_over_rest": row["branch_tangent_over_rest_norm"],
            "nonlinear_order": row["nonlinear_residual_convergence_order"],
            "smallest_corrected_over_unresolved": row["trials"][-1][
                "corrected_over_unresolved"],
        } for row in rows],
        "verdict": result["verdict"],
    }, indent=2))


if __name__ == "__main__":
    main()
