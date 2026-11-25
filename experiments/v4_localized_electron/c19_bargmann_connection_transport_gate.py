"""Fixed-charge transport screen for the rest-preserving joint connection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.linalg import lu_factor, lu_solve
from torch.func import grad

import plaquette_factorization_completion_probe as base
import c19_bargmann_joint_connection_gate as joint


HERE = Path(__file__).resolve().parent
STRENGTH = 1.0


def interval_action(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return (joint.canonical_interval_action(left, right)
            + STRENGTH * joint.connection_term(left, right))


def three_layer_action(previous: torch.Tensor, current: torch.Tensor,
                       following: torch.Tensor) -> torch.Tensor:
    return interval_action(previous, current) + interval_action(
        current, following)


def main() -> None:
    global STRENGTH
    parser = argparse.ArgumentParser()
    parser.add_argument("--strength", type=float, default=1.0)
    args = parser.parse_args()
    STRENGTH = args.strength

    base.inverse.rank.lock_inertia()
    base.inverse.rest.SIDE = base.inverse.SIDE
    base.inverse.rest.DT = base.inverse.DT
    previous, current, following, theta = base.inverse.rest_layers()
    layers = tuple(torch.tensor(value, dtype=torch.float64)
                   for value in (previous, current, following))
    del_residual = grad(three_layer_action, argnums=1)
    right_momentum = grad(interval_action, argnums=1)
    rest_del = del_residual(*layers).detach().numpy()

    saved_del = base.del_residual
    saved_momentum = base.right_momentum
    base.del_residual = del_residual
    base.right_momentum = right_momentum
    try:
        dimension = previous.size
        previous_block = base.batched_jvp(
            "previous", layers, np.eye(dimension))
        following_blocks = []
        factor = lu_factor(previous_block)
        rows = []
        for axis in range(3):
            labels, targets = base.component.component_targets(
                axis, np.exp(1j * theta))
            images = base.batched_jvp("following", layers, targets)
            following_blocks.append(images)
            launches = lu_solve(factor, -images)
            responses = np.column_stack([
                base.response(previous, current, following,
                              launches[:, column], targets[:, column])
                for column in range(targets.shape[1])
            ])
            charge_rank = np.linalg.matrix_rank(responses[:2], tol=1e-8)
            nullspace = np.linalg.svd(
                responses[:2], full_matrices=True)[2][charge_rank:].T
            fixed_center = responses[2:] @ nullspace
            singular = np.linalg.svd(fixed_center, compute_uv=False)
            threshold = 1e-8 * max(singular[0], 1e-30)
            rows.append({
                "axis": axis,
                "fixed_charge_center_rank": int(np.sum(singular > threshold)),
                "singular_values": [float(value) for value in singular],
                "relative_third_singular_value": float(
                    singular[2] / max(singular[0], 1e-30)),
                "component_labels": labels,
            })
    finally:
        base.del_residual = saved_del
        base.right_momentum = saved_momentum

    rest_rms = float(np.linalg.norm(rest_del) / np.sqrt(rest_del.size))
    rank_three = all(
        row["fixed_charge_center_rank"] == 3
        and row["relative_third_singular_value"] > 1e-6
        for row in rows)
    rest_preserved = rest_rms < 1e-10
    result = {
        "status": "V4_C19_BARGMANN_CONNECTION_TRANSPORT_GATE",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "strength": STRENGTH,
        "coefficient_provenance": (
            "response screen only; no Dirac speed or target was fitted"),
        "candidate_rest_DEL_rms": rest_rms,
        "rest_preserved": rest_preserved,
        "axis_rows": rows,
        "all_axes_robust_rank_three": rank_three,
        "verdict": (
            "BARGMANN_CONNECTION_CLOSES_LINEAR_TRANSPORT_GATE"
            if rest_preserved and rank_three else
            "BARGMANN_CONNECTION_DOES_NOT_CLOSE_LINEAR_TRANSPORT_GATE"),
        "claim_boundary": (
            "Rest and linear fixed-charge tangent screen only. A pass does "
            "not establish coefficient provenance, symplectic "
            "nondegeneracy, nonlinear translation, stability, or "
            "same-state Dirac current."),
    }
    suffix = f"{STRENGTH:g}".replace("-", "m").replace(".", "p")
    output = HERE / f"c19_bargmann_connection_transport_gate_{suffix}.json"
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n",
                      encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
