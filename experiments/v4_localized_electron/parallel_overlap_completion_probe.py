"""Test the complementary C19 overlap channel for missing z transport.

The incumbent spatial action transports only ``W = chi wedge Psi``.  The
complex-vector completeness identity also contains the parallel scalar
``S = chi^dagger Psi``.  This probe adds the positive reciprocal bond energy

    U_parallel = sum_(i,n) w_n |S_(i+n)-S_i|^2 / (2 chi0^2)

with the same normalization as the area channel.  Its coefficient is one by
completion, not fit to a transport target.  Since S vanishes on the exact rest
carrier and vacuum, the action delta and its first variation vanish there.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from scipy.linalg import lu_factor, lu_solve
from torch.func import grad, jvp, vmap

import localized_carrier_contrast_component_phase_ramp as component
import localized_carrier_contrast_inverse_launch as inverse


HERE = Path(__file__).resolve().parent


def parallel_potential(raw: torch.Tensor) -> torch.Tensor:
    psi, chi = inverse.current_audit.mold.unpack_tensor(raw, inverse.SIDE)
    overlap = torch.sum(torch.conj(chi)*psi, dim=-1)
    bond = torch.zeros((), dtype=torch.float64)
    for offset, weight in inverse.rank.native.DIRECTIONS:
        difference = overlap-torch.roll(
            overlap, shifts=offset, dims=(0, 1, 2))
        bond = bond+weight*torch.sum(torch.real(torch.conj(difference)*difference))
    return bond/(2*inverse.rank.R**2)


def interval_action(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return (inverse.interval_action(left, right)
            - inverse.DT*(parallel_potential(left)+parallel_potential(right))/2)


def three_layer_action(previous: torch.Tensor, current: torch.Tensor,
                       following: torch.Tensor) -> torch.Tensor:
    return interval_action(previous, current)+interval_action(current, following)


del_residual = grad(three_layer_action, argnums=1)
right_momentum = grad(interval_action, argnums=1)


def batched_jvp(argument: str, base: tuple[torch.Tensor, ...],
                vectors: np.ndarray) -> np.ndarray:
    previous, current, following = base
    vector_tensor = torch.tensor(vectors.T, dtype=torch.float64)
    if argument == "previous":
        function = lambda value: del_residual(value, current, following)
        point = previous
    else:
        function = lambda value: del_residual(previous, current, value)
        point = following
    images = vmap(
        lambda vector: jvp(function, (point,), (vector,))[1],
        chunk_size=8)(vector_tensor)
    return images.detach().numpy().T


def density_tensor(left: torch.Tensor, right: torch.Tensor,
                   mode: str) -> torch.Tensor:
    momentum = right_momentum(left, right)
    psi, chi = inverse.current_audit.mold.unpack_tensor(right, inverse.SIDE)
    zero_psi = torch.zeros_like(psi)
    zero_chi = torch.zeros_like(chi)
    generator = inverse.current_audit.pack_tensor(
        1j*psi if mode == "Psi" else zero_psi,
        1j*chi if mode == "chi" else zero_chi)
    return (momentum*generator).reshape(
        inverse.SIDE, inverse.SIDE, inverse.SIDE, 4, 3).sum(dim=(-1, -2))


def density_and_derivative(left: np.ndarray, right: np.ndarray,
                           dleft: np.ndarray | None,
                           dright: np.ndarray | None,
                           mode: str) -> tuple[np.ndarray, np.ndarray]:
    l = torch.tensor(left, dtype=torch.float64)
    r = torch.tensor(right, dtype=torch.float64)
    dl = torch.zeros_like(l) if dleft is None else torch.tensor(
        dleft, dtype=torch.float64)
    dr = torch.zeros_like(r) if dright is None else torch.tensor(
        dright, dtype=torch.float64)
    function = lambda a, b: density_tensor(a, b, mode)
    value, derivative = jvp(function, (l, r), (dl, dr))
    return value.detach().numpy(), derivative.detach().numpy()


def center_derivative(density: np.ndarray, derivative: np.ndarray) -> np.ndarray:
    total = float(np.sum(density))
    dtotal = float(np.sum(derivative))
    coordinates = np.indices(density.shape)
    result = []
    for axis in range(3):
        moment = float(np.sum(coordinates[axis]*density))
        dmoment = float(np.sum(coordinates[axis]*derivative))
        result.append((dmoment*total-moment*dtotal)/(total**2))
    return np.asarray(result)


def response(previous: np.ndarray, current: np.ndarray,
             following: np.ndarray, launch: np.ndarray,
             target: np.ndarray) -> np.ndarray:
    current_psi, dcurrent_psi = density_and_derivative(
        previous, current, launch, None, "Psi")
    following_psi, dfollowing_psi = density_and_derivative(
        current, following, None, target, "Psi")
    _, dcurrent_chi = density_and_derivative(
        previous, current, launch, None, "chi")
    _, dfollowing_chi = density_and_derivative(
        current, following, None, target, "chi")
    charges = np.asarray((
        (np.sum(dcurrent_psi)+np.sum(dfollowing_psi))/2,
        (np.sum(dcurrent_chi)+np.sum(dfollowing_chi))/2,
    ))
    displacement = (
        center_derivative(following_psi, dfollowing_psi)
        - center_derivative(current_psi, dcurrent_psi))
    return np.concatenate((charges, displacement))


def main() -> None:
    inverse.rank.lock_inertia()
    inverse.rest.SIDE = inverse.SIDE
    inverse.rest.DT = inverse.DT
    previous, current, following, theta = inverse.rest_layers()
    base = tuple(torch.tensor(value, dtype=torch.float64)
                 for value in (previous, current, following))
    rest_action_delta = float(parallel_potential(base[1]).detach())
    rest_variable = base[1].detach().requires_grad_(True)
    rest_first_variation = torch.autograd.grad(
        parallel_potential(rest_variable), rest_variable)[0]
    rest_del = del_residual(*base).detach().numpy()

    dimension = previous.size
    previous_block = batched_jvp("previous", base, np.eye(dimension))
    factor = lu_factor(previous_block)
    rows = []
    for axis in range(3):
        labels, targets = component.component_targets(axis, np.exp(1j*theta))
        images = batched_jvp("following", base, targets)
        launches = lu_solve(factor, -images)
        responses = np.column_stack([
            response(previous, current, following,
                     launches[:, column], targets[:, column])
            for column in range(targets.shape[1])
        ])
        charge_rank = np.linalg.matrix_rank(responses[:2], tol=1e-8)
        null = np.linalg.svd(responses[:2], full_matrices=True)[2][charge_rank:].T
        fixed_center = responses[2:]@null
        singular = np.linalg.svd(fixed_center, compute_uv=False)
        rank_value = int(np.sum(singular > 1e-9*max(singular[0], 1e-30)))
        rows.append({
            "phase_ramp_axis": axis,
            "component_labels": labels,
            "fixed_charge_center_rank": rank_value,
            "fixed_charge_center_singular_values": [float(value)
                                                     for value in singular],
            "response_matrix": responses.tolist(),
        })
    incumbent = json.loads((HERE /
        "localized_carrier_contrast_component_phase_ramp.json").read_text())
    incumbent_singular = [row["fixed_charge_center_singular_values"]
                          for row in incumbent["axis_rows"]]
    candidate_singular = [row["fixed_charge_center_singular_values"]
                          for row in rows]
    z_gain = float(candidate_singular[2][2]
                   / max(incumbent_singular[2][2], 1e-300))
    result = {
        "status": "V4_PARALLEL_OVERLAP_COMPLETION_RESPONSE_AUDIT",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "action_delta": (
            "positive C19 bond energy of S=chi^dagger Psi with the same "
            "1/(2 chi0^2) normalization as W=chi wedge Psi"),
        "coefficient_provenance": (
            "unit coefficient from the complementary complex-vector norm "
            "decomposition; not fitted to a Dirac or movement target"),
        "rest_parallel_energy": rest_action_delta,
        "rest_parallel_first_variation_norm": float(
            torch.linalg.vector_norm(rest_first_variation).detach()),
        "candidate_rest_DEL_rms": float(
            np.linalg.norm(rest_del)/np.sqrt(rest_del.size)),
        "incumbent_fixed_charge_center_singular_values": incumbent_singular,
        "candidate_rows": rows,
        "z_weak_channel_gain": z_gain,
        "verdict": (
            "PARALLEL_COMPLETION_SUPPLIES_THIRD_TRANSPORT_CHANNEL" if
            all(row["fixed_charge_center_rank"] == 3 for row in rows) and
            candidate_singular[2][2] > 1e-8 else
            "PARALLEL_COMPLETION_DOES_NOT_CLOSE_TRANSPORT"),
        "claim_boundary": (
            "This is a linear response screen. A favorable result still "
            "requires exact rest spectrum, bounded-energy, nonlinear motion, "
            "current, and Dirac gates."),
    }
    (HERE / "parallel_overlap_completion_probe.json").write_text(
        json.dumps(result, indent=2, allow_nan=False)+"\n",
        encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
