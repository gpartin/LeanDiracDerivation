"""Rank and damped-Newton diagnostic for the direct-chi-bond rest branch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import direct_chi_bond_rest_continuation as continuation
import edge_snap_action_rank as rank


HERE = Path(__file__).resolve().parent


def main(iterations: int = 3) -> None:
    rank.lock_inertia()
    data = np.load(HERE / "direct_chi_bond_rest_newton.npz")
    approximate = torch.tensor(data["current"], dtype=torch.float64)
    layers, _ = continuation.transfer_rest.rest.orbit_layers()
    reference = layers[1].detach()
    shape = (continuation.SIDE,) * 3 + (4, 3)
    fixed = np.asarray([
        np.ravel_multi_index((0, 0, 0, 1, 0), shape),
        np.ravel_multi_index((2, 2, 2, 3, 2), shape),
    ])
    free_mask = np.ones(approximate.numel(), dtype=bool)
    free_mask[fixed] = False
    free = torch.tensor(np.flatnonzero(free_mask), dtype=torch.long)
    base = approximate.clone()
    base[torch.tensor(fixed)] = reference[torch.tensor(fixed)]
    values = np.concatenate((
        base[free].numpy(),
        np.asarray([float(data["theta_psi"]), float(data["theta_chi"])])))

    def residual_tensor(parameters: torch.Tensor) -> torch.Tensor:
        current = base.clone()
        current[free] = parameters[:-2]
        return continuation.del_residual(
            current, parameters[-2], parameters[-1], 1.0) / rank.R

    rows = []
    for iteration in range(iterations + 1):
        parameters = torch.tensor(
            values, dtype=torch.float64, requires_grad=True)
        residual = residual_tensor(parameters)
        jacobian = torch.autograd.functional.jacobian(
            residual_tensor, parameters, vectorize=True)
        r = residual.detach().numpy()
        j = jacobian.detach().numpy()
        singular = np.linalg.svd(j, compute_uv=False)
        norm = float(np.linalg.norm(r))
        row = {
            "iteration": iteration,
            "residual_norm": norm,
            "largest_singular_value": float(singular[0]),
            "smallest_singular_value": float(singular[-1]),
            "condition_number": float(singular[0] / singular[-1]),
            "rank_rcond_1e_12": int(np.sum(singular > singular[0] * 1e-12)),
        }
        if iteration == iterations:
            rows.append(row)
            break
        delta, _, _, _ = np.linalg.lstsq(j, -r, rcond=1e-12)
        row["newton_step_norm"] = float(np.linalg.norm(delta))
        trials = []
        best_values = values
        best_norm = norm
        for exponent in range(13):
            alpha = 2.0 ** (-exponent)
            trial_values = values + alpha * delta
            trial = residual_tensor(torch.tensor(
                trial_values, dtype=torch.float64,
                requires_grad=True)).detach().numpy()
            trial_norm = float(np.linalg.norm(trial))
            trials.append({"alpha": alpha, "residual_norm": trial_norm})
            if trial_norm < best_norm:
                best_norm = trial_norm
                best_values = trial_values
        row["line_search"] = trials
        row["accepted_residual_norm"] = best_norm
        row["accepted"] = bool(best_norm < norm)
        rows.append(row)
        if best_norm >= norm:
            break
        values = best_values

    final = base.clone()
    final[free] = torch.tensor(values[:-2], dtype=torch.float64)
    final_metrics = continuation.metrics(
        final, reference,
        torch.tensor(values[-2], dtype=torch.float64),
        torch.tensor(values[-1], dtype=torch.float64), 1.0)
    improved = rows[-1]["residual_norm"] < rows[0]["residual_norm"]
    result = {
        "status": "V4_DIRECT_CHI_BOND_REST_NEWTON_DIAGNOSTIC",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "rows": rows,
        "final": final_metrics,
        "newton_direction_improved": improved,
        "verdict": (
            "LOCAL_ROOT_ACCESSIBLE_BY_NEWTON" if improved else
            "NO_DESCENT_ALONG_EXACT_NEWTON_DIRECTION"),
        "claim_boundary": (
            "This diagnoses the local nonlinear solve only; it is not a "
            "proof that no remote branch exists and not an action pass."),
    }
    (HERE / "direct_chi_bond_rest_newton_diagnostic.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=3)
    args = parser.parse_args()
    main(args.iterations)
