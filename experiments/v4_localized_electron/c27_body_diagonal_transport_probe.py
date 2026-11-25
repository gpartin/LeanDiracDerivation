"""Ablate the eight C27 body diagonals against the missing transport rank.

The native area action uses one shared C19 stencil, so saying that Psi and chi
use different stencils is not literally defined for that term.  The closest
real reciprocal action tests are to add the eight body-diagonal Dirichlet
bonds to (a) W=chi wedge Psi, (b) Psi alone, or (c) chi alone.  Variation then
produces the corresponding reciprocal field equations without changing any
runtime rule.

Each mode is a unit response-rank screen.  It is not an admitted C27 action;
C27 remains an ablation and any candidate would still need a native
coefficient, rest solve, spectrum, and nonlinear motion gates.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import torch
from scipy.linalg import lu_factor, lu_solve
from torch.func import grad

import plaquette_factorization_completion_probe as base


HERE = Path(__file__).resolve().parent
MODE = "joint_W"
BODY_OFFSETS = tuple(itertools.product((-1, 1), repeat=3))


def squared_norm(value: torch.Tensor) -> torch.Tensor:
    return torch.real(torch.conj(value) * value)


def body_potential(raw: torch.Tensor) -> torch.Tensor:
    psi, chi = base.inverse.current_audit.mold.unpack_tensor(
        raw, base.inverse.SIDE)
    if MODE == "joint_W":
        field = torch.linalg.cross(chi, psi, dim=-1) / base.inverse.rank.R
    elif MODE == "Psi":
        field = psi
    elif MODE == "chi":
        field = chi
    else:
        raise ValueError(MODE)
    result = torch.zeros((), dtype=torch.float64)
    for offset in BODY_OFFSETS:
        difference = torch.roll(
            field, shifts=offset, dims=(0, 1, 2)) - field
        result = result + torch.sum(squared_norm(difference))
    # Directed offsets count each undirected body-diagonal bond twice.
    return result / 4


def interval_action(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return (
        base.inverse.interval_action(left, right)
        - base.inverse.DT * (body_potential(left) + body_potential(right)) / 2
    )


def three_layer_action(previous: torch.Tensor, current: torch.Tensor,
                       following: torch.Tensor) -> torch.Tensor:
    return interval_action(previous, current) + interval_action(
        current, following)


def main() -> None:
    global MODE
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("joint_W", "Psi", "chi"),
                        required=True)
    args = parser.parse_args()
    MODE = args.mode

    # Reuse the independently audited fixed-charge response machinery with
    # this action's DEL and canonical momenta.
    base.del_residual = grad(three_layer_action, argnums=1)
    base.right_momentum = grad(interval_action, argnums=1)
    base.inverse.rank.lock_inertia()
    base.inverse.rest.SIDE = base.inverse.SIDE
    base.inverse.rest.DT = base.inverse.DT
    previous, current, following, theta = base.inverse.rest_layers()
    tensors = tuple(torch.tensor(value, dtype=torch.float64)
                    for value in (previous, current, following))
    rest_energy = float(body_potential(tensors[1]).detach())
    variable = tensors[1].detach().requires_grad_(True)
    first_variation = torch.autograd.grad(body_potential(variable), variable)[0]
    rest_del = base.del_residual(*tensors).detach().numpy()

    dimension = previous.size
    previous_block = base.batched_jvp("previous", tensors, np.eye(dimension))
    factor = lu_factor(previous_block)
    rows = []
    for axis in range(3):
        labels, targets = base.component.component_targets(
            axis, np.exp(1j * theta))
        images = base.batched_jvp("following", tensors, targets)
        launches = lu_solve(factor, -images)
        responses = np.column_stack([
            base.response(previous, current, following,
                          launches[:, column], targets[:, column])
            for column in range(targets.shape[1])
        ])
        charge_rank = np.linalg.matrix_rank(responses[:2], tol=1e-8)
        null = np.linalg.svd(
            responses[:2], full_matrices=True)[2][charge_rank:].T
        fixed_center = responses[2:] @ null
        left, singular, _ = np.linalg.svd(fixed_center, full_matrices=True)
        threshold = 1e-8 * max(singular[0], 1e-30)
        rows.append({
            "phase_ramp_axis": axis,
            "component_labels": labels,
            "fixed_charge_center_rank": int(np.sum(singular > threshold)),
            "fixed_charge_center_singular_values": [
                float(value) for value in singular],
            "relative_third_singular_value": float(
                singular[2] / max(singular[0], 1e-30)),
            "missing_physical_direction": [
                float(value) for value in left[:, -1]],
        })

    robust = all(
        row["fixed_charge_center_rank"] == 3
        and row["relative_third_singular_value"] > 1e-6
        for row in rows)
    result = {
        "status": "V4_C27_BODY_DIAGONAL_TRANSPORT_ABLATION",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "mode": MODE,
        "action_delta": (
            "positive Dirichlet energy on all eight body-diagonal offsets "
            f"for {MODE}"),
        "interpretation": {
            "joint_W": "C27 extension of the shared native area stencil",
            "Psi": "Psi receives body-diagonal direct bonds; chi area remains C19",
            "chi": "chi receives body-diagonal direct bonds; Psi area remains C19",
        }[MODE],
        "rest_body_energy": rest_energy,
        "rest_body_first_variation_norm": float(
            torch.linalg.vector_norm(first_variation).detach()),
        "candidate_rest_DEL_rms": float(
            np.linalg.norm(rest_del) / np.sqrt(rest_del.size)),
        "axis_rows": rows,
        "all_axes_robust_rank_three": robust,
        "verdict": (
            "C27_BODY_DIAGONALS_SUPPLY_THIRD_TRANSPORT_CHANNEL" if robust
            else "C27_BODY_DIAGONALS_DO_NOT_CLOSE_TRANSPORT"),
        "claim_boundary": (
            "Unit linear response ablation only. It does not promote C27, "
            "define separate fundamental stencils, prove an exact rest orbit, "
            "or establish nonlinear carrier movement."),
    }
    output = HERE / f"c27_body_diagonal_transport_probe_{MODE}.json"
    output.write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    print(json.dumps({
        "mode": MODE,
        "verdict": result["verdict"],
        "rest_body_first_variation_norm": result[
            "rest_body_first_variation_norm"],
        "candidate_rest_DEL_rms": result["candidate_rest_DEL_rms"],
        "ranks": [row["fixed_charge_center_rank"] for row in rows],
        "relative_third_singular_values": [
            row["relative_third_singular_value"] for row in rows],
        "output": str(output),
    }, indent=2))


if __name__ == "__main__":
    main()
