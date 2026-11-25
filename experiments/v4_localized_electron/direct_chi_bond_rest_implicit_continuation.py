"""Exact-Jacobian continuation of the direct-chi-bond rest branch."""

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


def main(target: float = 0.1, steps: int = 10, max_nfev: int = 20) -> None:
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
    values = np.concatenate((
        base[free].numpy(), np.asarray([float(theta), 0.0])))
    rows = []

    def residual_tensor(parameters: torch.Tensor,
                        strength: float) -> torch.Tensor:
        current = base.clone()
        current[free] = parameters[:-2]
        return continuation.del_residual(
            current, parameters[-2], parameters[-1], strength) / rank.R

    for index in range(steps + 1):
        strength = target * index / steps
        cache = {"x": None, "r": None, "j": None}

        def evaluate(x: np.ndarray, jacobian: bool):
            if cache["x"] is not None and np.array_equal(cache["x"], x):
                if not jacobian or cache["j"] is not None:
                    return cache["j"] if jacobian else cache["r"]
            parameters = torch.tensor(x, dtype=torch.float64,
                                      requires_grad=True)
            r = residual_tensor(parameters, strength)
            j = None
            if jacobian:
                j = torch.autograd.functional.jacobian(
                    lambda p: residual_tensor(p, strength), parameters,
                    vectorize=True)
            cache.update({
                "x": x.copy(), "r": r.detach().numpy(),
                "j": None if j is None else j.detach().numpy(),
            })
            return cache["j"] if jacobian else cache["r"]

        if index:
            solution = least_squares(
                lambda x: evaluate(x, False), values,
                jac=lambda x: evaluate(x, True), method="trf",
                x_scale="jac", max_nfev=max_nfev,
                xtol=1e-13, ftol=1e-13, gtol=1e-13)
            values = solution.x
        else:
            solution = None
        r = evaluate(values, False)
        j = evaluate(values, True)
        singular = np.linalg.svd(j, compute_uv=False)
        current = base.clone()
        current[free] = torch.tensor(values[:-2], dtype=torch.float64)
        metric = continuation.metrics(
            current, reference,
            torch.tensor(values[-2], dtype=torch.float64),
            torch.tensor(values[-1], dtype=torch.float64), strength)
        rows.append({
            "strength": strength,
            "residual_norm_scaled_by_R": float(np.linalg.norm(r)),
            "smallest_singular_value": float(singular[-1]),
            "condition_number": float(singular[0] / singular[-1]),
            "solver_success": True if solution is None else bool(solution.success),
            "solver_message": "exact baseline" if solution is None else solution.message,
            "function_evaluations": 0 if solution is None else int(solution.nfev),
            **metric,
        })
        print(json.dumps(rows[-1], allow_nan=False), flush=True)

    final = rows[-1]
    exact = final["normalized_del_rms"] < 1e-8
    identity = (final["psi_carrier_overlap_sq"] > 0.95
                and final["psi_exterior_fraction"] < 0.1)
    result = {
        "status": "V4_DIRECT_CHI_BOND_IMPLICIT_CONTINUATION",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "target_strength": target,
        "steps": steps,
        "rows": rows,
        "exact_rest_orbit_found": exact,
        "carrier_identity_retained": identity,
        "verdict": (
            "LOCAL_BRANCH_REACHES_TARGET" if exact and identity else
            "LOCAL_BRANCH_NOT_CLOSED_AT_TARGET"),
        "claim_boundary": (
            "This follows a gauge-fixed local rest branch. A passing result "
            "still requires the full spectrum and movement gates."),
    }
    stem = f"direct_chi_bond_rest_implicit_{target:g}"
    (HERE / f"{stem}.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    np.savez_compressed(
        HERE / f"{stem}.npz", current=current.numpy(),
        theta_psi=values[-2], theta_chi=values[-1])
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=float, default=0.1)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--max-nfev", type=int, default=20)
    args = parser.parse_args()
    main(args.target, args.steps, args.max_nfev)
