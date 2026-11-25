"""Algebraic gate for a native C19 Bargmann-triangle potential.

The candidate uses only native complex Psi/chi fields and the face--edge
triangles already present in C19 geometry.  It is deliberately screened
before any time evolution.  The screen asks whether the positive potential

    sum (Im ((chi_i^H Psi_j)(Psi_j^H Psi_k)(Psi_k^H chi_i)))^2 / chi0^8

has a nonzero chi response at a quiet receiver, preserves the proved V4 rest
orbit to first order, and supplies the missing fixed-charge transport rank.
Unit strength is only a response-basis test; it is not an admitted action
coefficient.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from scipy.linalg import lu_factor, lu_solve
from torch.func import grad

import plaquette_factorization_completion_probe as base
import c19_bargmann_joint_connection_gate as joint


HERE = Path(__file__).resolve().parent
CHI0 = 19.0
STRENGTH = 1.0


def inner(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return torch.sum(torch.conj(left) * right, dim=-1)


def shift(field: torch.Tensor, axis: int, amount: int) -> torch.Tensor:
    offsets = tuple(amount if index == axis else 0 for index in range(3))
    return torch.roll(field, shifts=offsets, dims=(0, 1, 2))


def bargmann_triangle_potential(raw: torch.Tensor) -> torch.Tensor:
    psi, chi = base.inverse.current_audit.mold.unpack_tensor(
        raw, base.inverse.SIDE)
    result = torch.zeros((), dtype=torch.float64)
    # For every coordinate plane and both edge signs, use the receiver i,
    # face neighbor j=i+s e_a, and edge neighbor k=i+s e_a+t e_b.  All
    # three pairwise separations are C19 face/edge bonds.  Squaring Im(B)
    # removes the arbitrary orientation of the triangle.
    for first, second in ((0, 1), (0, 2), (1, 2)):
        for first_sign in (-1, 1):
            for second_sign in (-1, 1):
                psi_j = shift(psi, first, first_sign)
                psi_k = shift(psi_j, second, second_sign)
                loop = (inner(chi, psi_j) * inner(psi_j, psi_k)
                        * inner(psi_k, chi))
                result = result + torch.sum(torch.imag(loop) ** 2)
    return result / CHI0**8


def interval_action(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return (joint.canonical_interval_action(left, right)
            - base.inverse.DT * STRENGTH
            * (bargmann_triangle_potential(left)
               + bargmann_triangle_potential(right)) / 2)


def three_layer_action(previous: torch.Tensor, current: torch.Tensor,
                       following: torch.Tensor) -> torch.Tensor:
    return interval_action(previous, current) + interval_action(
        current, following)


def quiet_receiver_test() -> dict:
    side = base.inverse.SIDE
    shape = (side, side, side, 3)
    psi = torch.zeros(shape, dtype=torch.complex128)
    chi = torch.zeros(shape, dtype=torch.complex128)
    receiver = (side // 2, side // 2, side // 2)
    sender_j = ((receiver[0] + 1) % side, receiver[1], receiver[2])
    sender_k = (sender_j[0], (sender_j[1] + 1) % side, sender_j[2])
    # A deterministic non-coplanar choice with Im(B) != 0.
    chi[receiver] = torch.tensor((1.0 + 0.0j, 0.0 + 1.0j, 0.5 + 0.0j))
    psi[sender_j] = torch.tensor((1.0 + 0.0j, 1.0 + 1.0j, 0.0 + 0.0j))
    psi[sender_k] = torch.tensor((0.0 + 1.0j, 1.0 + 0.0j, 1.0 + 0.0j))
    raw = base.inverse.current_audit.pack_tensor(psi, chi).detach()
    raw.requires_grad_(True)
    value = bargmann_triangle_potential(raw)
    derivative = torch.autograd.grad(value, raw)[0]
    _, chi_gradient = base.inverse.current_audit.mold.unpack_tensor(
        derivative, side)
    return {
        "receiver_psi_norm": float(torch.linalg.vector_norm(
            psi[receiver]).detach()),
        "potential": float(value.detach()),
        "receiver_chi_gradient_norm": float(torch.linalg.vector_norm(
            chi_gradient[receiver]).detach()),
    }


def fixed_charge_transport(base_layers: tuple[torch.Tensor, ...],
                           del_residual) -> list[dict]:
    previous, current, following = base_layers
    dimension = previous.numel()
    saved = base.del_residual
    base.del_residual = del_residual
    try:
        previous_block = base.batched_jvp(
            "previous", base_layers, np.eye(dimension))
        factor = lu_factor(previous_block)
        rows = []
        for axis in range(3):
            labels, targets = base.component.component_targets(
                axis, np.exp(1j * base.inverse.rest_layers()[3]))
            images = base.batched_jvp("following", base_layers, targets)
            launches = lu_solve(factor, -images)
            responses = np.column_stack([
                base.response(
                    previous.detach().numpy(), current.detach().numpy(),
                    following.detach().numpy(), launches[:, column],
                    targets[:, column])
                for column in range(targets.shape[1])
            ])
            charge_rank = np.linalg.matrix_rank(responses[:2], tol=1e-8)
            nullspace = np.linalg.svd(
                responses[:2], full_matrices=True)[2][charge_rank:].T
            fixed_center = responses[2:] @ nullspace
            singular = np.linalg.svd(
                fixed_center, compute_uv=False, full_matrices=True)
            threshold = 1e-8 * max(singular[0], 1e-30)
            rows.append({
                "axis": axis,
                "fixed_charge_center_rank": int(np.sum(singular > threshold)),
                "singular_values": [float(value) for value in singular],
                "relative_third_singular_value": float(
                    singular[2] / max(singular[0], 1e-30)),
                "component_labels": labels,
            })
        return rows
    finally:
        base.del_residual = saved


def main() -> None:
    base.inverse.rank.lock_inertia()
    base.inverse.rest.SIDE = base.inverse.SIDE
    base.inverse.rest.DT = base.inverse.DT
    previous, current, following, _theta = base.inverse.rest_layers()
    layers = tuple(torch.tensor(value, dtype=torch.float64)
                   for value in (previous, current, following))

    potential_at_rest = float(
        bargmann_triangle_potential(layers[1]).detach())
    variable = layers[1].detach().requires_grad_(True)
    rest_gradient = torch.autograd.grad(
        bargmann_triangle_potential(variable), variable)[0]
    del_residual = grad(three_layer_action, argnums=1)
    rest_del = del_residual(*layers).detach().numpy()
    rows = fixed_charge_transport(layers, del_residual)
    quiet = quiet_receiver_test()

    rest_preserved = (
        float(torch.linalg.vector_norm(rest_gradient).detach()) < 1e-10
        and np.linalg.norm(rest_del) / np.sqrt(rest_del.size) < 1e-10)
    receiver_channel = quiet["receiver_chi_gradient_norm"] > 1e-14
    rank_three = all(
        row["fixed_charge_center_rank"] == 3
        and row["relative_third_singular_value"] > 1e-6
        for row in rows)
    if rest_preserved and receiver_channel and rank_three:
        verdict = "BARGMANN_TRIANGLE_PASSES_ALGEBRAIC_TRANSPORT_GATE"
    elif not receiver_channel:
        verdict = "BARGMANN_TRIANGLE_HAS_NO_QUIET_RECEIVER_CHANNEL"
    elif not rest_preserved:
        verdict = "BARGMANN_TRIANGLE_TORQUES_PROVED_REST_ORBIT"
    else:
        verdict = "BARGMANN_TRIANGLE_DOES_NOT_CLOSE_TRANSPORT_RANK"

    result = {
        "status": "V4_C19_BARGMANN_TRIANGLE_GATE",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "candidate": "positive C19 Bargmann-loop phase potential",
        "coefficient": STRENGTH,
        "coefficient_provenance": (
            "unit response-basis normalization only; not selected"),
        "rest_potential": potential_at_rest,
        "rest_first_variation_norm": float(
            torch.linalg.vector_norm(rest_gradient).detach()),
        "candidate_rest_DEL_rms": float(
            np.linalg.norm(rest_del) / np.sqrt(rest_del.size)),
        "quiet_receiver": quiet,
        "axis_rows": rows,
        "verdict": verdict,
        "claim_boundary": (
            "Algebraic first-variation and fixed-charge tangent screen only. "
            "No coefficient is admitted and no nonlinear translation, "
            "stability, Dirac-current closure, or electron claim follows."),
    }
    output = HERE / "c19_bargmann_triangle_gate.json"
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n",
                      encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
