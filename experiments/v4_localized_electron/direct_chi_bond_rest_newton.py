"""Exact-Jacobian correction of the direct-chi-bond rest continuation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
import torch

import direct_chi_bond_rest_continuation as continuation
import edge_snap_action_rank as rank


HERE = Path(__file__).resolve().parent


def main(max_nfev: int = 20) -> None:
    rank.lock_inertia()
    data = np.load(HERE / "direct_chi_bond_rest_continuation.npz")
    approximate = torch.tensor(data["current"], dtype=torch.float64)
    layers, _ = continuation.transfer_rest.rest.orbit_layers()
    reference = layers[1].detach()
    shape = (continuation.SIDE,) * 3 + (4, 3)
    # Fix one Psi and one exterior-chi imaginary coordinate.  Both vanish on
    # the reference branch and remove only the two global U(1) phase gauges.
    fixed = np.asarray([
        np.ravel_multi_index((0, 0, 0, 1, 0), shape),
        np.ravel_multi_index((2, 2, 2, 3, 2), shape),
    ])
    free_mask = np.ones(approximate.numel(), dtype=bool)
    free_mask[fixed] = False
    free = torch.tensor(np.flatnonzero(free_mask), dtype=torch.long)
    base = approximate.clone()
    base[torch.tensor(fixed)] = reference[torch.tensor(fixed)]
    initial = torch.cat((
        base[free],
        torch.tensor([float(data["theta_psi"]), float(data["theta_chi"])],
                     dtype=torch.float64)))
    cache = {"x": None, "residual": None, "jacobian": None}

    def residual_tensor(parameters: torch.Tensor) -> torch.Tensor:
        current = base.clone()
        current[free] = parameters[:-2]
        return continuation.del_residual(
            current, parameters[-2], parameters[-1], 1.0) / rank.R

    def evaluate(values: np.ndarray, need_jacobian: bool):
        if cache["x"] is not None and np.array_equal(cache["x"], values):
            if not need_jacobian or cache["jacobian"] is not None:
                return (cache["jacobian"] if need_jacobian
                        else cache["residual"])
        parameters = torch.tensor(
            values, dtype=torch.float64, requires_grad=True)
        if need_jacobian:
            jacobian = torch.autograd.functional.jacobian(
                residual_tensor, parameters, vectorize=True)
            residual = residual_tensor(parameters)
            cache.update({
                "x": values.copy(),
                "residual": residual.detach().numpy(),
                "jacobian": jacobian.detach().numpy(),
            })
            return cache["jacobian"]
        residual = residual_tensor(parameters).detach().numpy()
        cache.update({
            "x": values.copy(), "residual": residual, "jacobian": None})
        return residual

    solution = least_squares(
        lambda values: evaluate(values, False), initial.numpy(),
        jac=lambda values: evaluate(values, True), method="lm",
        x_scale="jac", max_nfev=max_nfev, xtol=1e-12, ftol=1e-12,
        gtol=1e-12, verbose=2)
    final = base.clone()
    final[free] = torch.tensor(solution.x[:-2], dtype=torch.float64)
    theta_psi = torch.tensor(solution.x[-2], dtype=torch.float64)
    theta_chi = torch.tensor(solution.x[-1], dtype=torch.float64)
    row = continuation.metrics(
        final, reference, theta_psi, theta_chi, 1.0)
    exact = row["normalized_del_rms"] < 1e-8
    identity = (row["psi_carrier_overlap_sq"] > 0.95
                and row["psi_exterior_fraction"] < 0.1)
    result = {
        "status": "V4_DIRECT_CHI_BOND_REST_NEWTON",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "solver_success": bool(solution.success),
        "solver_status": int(solution.status),
        "solver_message": solution.message,
        "function_evaluations": int(solution.nfev),
        "jacobian_evaluations": int(solution.njev),
        "cost": float(solution.cost),
        "optimality": float(solution.optimality),
        "fixed_phase_gauge_coordinates": fixed.tolist(),
        "final": row,
        "exact_rest_orbit_found": exact,
        "carrier_identity_retained": identity,
        "verdict": (
            "ADVANCE_TO_FULL_REST_SPECTRUM" if exact and identity else
            "DIRECT_CHI_BOND_REST_BRANCH_NOT_CLOSED"),
        "claim_boundary": (
            "This is a gauge-fixed nonlinear rest solve. It is not a "
            "stability, movement, or action-admission result."),
    }
    (HERE / "direct_chi_bond_rest_newton.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    np.savez_compressed(
        HERE / "direct_chi_bond_rest_newton.npz",
        current=final.numpy(), theta_psi=float(theta_psi),
        theta_chi=float(theta_chi))
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-nfev", type=int, default=20)
    args = parser.parse_args()
    main(args.max_nfev)
