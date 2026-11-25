"""Screen the missing parallel-overlap symplectic transport basis.

The exact fixed-charge audit finds only two independent carrier-center
responses under all component phase ramps.  A positive position energy made
from ``S = chi^dagger Psi`` preserves the rest orbit but does not supply the
third response.  This file tests the two lowest velocity-linear one-forms in
that same *native* overlap channel:

    local:  Im(S_i^* Sdot_i)
    bond:   Im(S_i^* Sdot_j + S_j^* Sdot_i)

The bond expression is symmetric under exchanging the endpoints, so it does
not orient a C19 bond or select a direction.  The corresponding endpoint-
antisymmetric expression is the total time derivative of
``Im(S_i^* S_j)`` and is therefore excluded before numerical work.

Both tested terms vanish with their first variation on the exact carrier,
where S is zero.  Unit strength is used only for a response-rank screen; no
coefficient is admitted or fitted here.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.linalg import lu_factor, lu_solve
from torch.func import grad, jvp, vmap

import localized_carrier_contrast_component_phase_ramp as component
import localized_carrier_contrast_inverse_launch as inverse


HERE = Path(__file__).resolve().parent


def overlap_and_rate(left: torch.Tensor, right: torch.Tensor
                     ) -> tuple[torch.Tensor, torch.Tensor]:
    """Return midpoint S and its exact path derivative on one interval."""
    psi_left, chi_left = inverse.current_audit.mold.unpack_tensor(
        left, inverse.SIDE)
    psi_right, chi_right = inverse.current_audit.mold.unpack_tensor(
        right, inverse.SIDE)
    psi = (psi_left+psi_right)/2
    chi = (chi_left+chi_right)/2
    psi_rate = (psi_right-psi_left)/inverse.DT
    chi_rate = (chi_right-chi_left)/inverse.DT
    overlap = torch.sum(torch.conj(chi)*psi, dim=-1)
    overlap_rate = torch.sum(
        torch.conj(chi_rate)*psi+torch.conj(chi)*psi_rate, dim=-1)
    return overlap, overlap_rate


def local_one_form(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    overlap, rate = overlap_and_rate(left, right)
    return torch.sum(torch.imag(torch.conj(overlap)*rate))/inverse.rank.R**2


def bond_one_form(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    overlap, rate = overlap_and_rate(left, right)
    result = torch.zeros((), dtype=torch.float64)
    for offset, weight in inverse.rank.native.DIRECTIONS:
        neighbor_overlap = torch.roll(overlap, shifts=offset, dims=(0, 1, 2))
        neighbor_rate = torch.roll(rate, shifts=offset, dims=(0, 1, 2))
        result = result+weight*torch.sum(torch.imag(
            torch.conj(overlap)*neighbor_rate
            + torch.conj(neighbor_overlap)*rate))
    # DIRECTIONS contains both orientations of every undirected C19 bond.
    return result/(2*inverse.rank.R**2)


def area_gradient_kinetic(left: torch.Tensor,
                          right: torch.Tensor) -> torch.Tensor:
    """Positive C19 kinetic square of the native area velocity."""
    psi_left, chi_left = inverse.current_audit.mold.unpack_tensor(
        left, inverse.SIDE)
    psi_right, chi_right = inverse.current_audit.mold.unpack_tensor(
        right, inverse.SIDE)
    psi = (psi_left+psi_right)/2
    chi = (chi_left+chi_right)/2
    psi_rate = (psi_right-psi_left)/inverse.DT
    chi_rate = (chi_right-chi_left)/inverse.DT
    area_rate = (torch.linalg.cross(chi_rate, psi, dim=-1)
                 + torch.linalg.cross(chi, psi_rate, dim=-1))
    result = torch.zeros((), dtype=torch.float64)
    for offset, weight in inverse.rank.native.DIRECTIONS:
        difference = area_rate-torch.roll(
            area_rate, shifts=offset, dims=(0, 1, 2))
        result = result+weight*torch.sum(
            torch.real(torch.conj(difference)*difference))
    # 24 is the C19 weighted degree; the directed-bond 1/2 matches the
    # canonical area-gradient convention.
    return result/(2*24*inverse.rank.R**2)


def make_interval_action(mode: str):
    if mode == "local":
        one_form = local_one_form
    elif mode == "bond":
        one_form = bond_one_form
    else:
        one_form = area_gradient_kinetic

    def interval_action(left: torch.Tensor,
                        right: torch.Tensor) -> torch.Tensor:
        return inverse.interval_action(left, right)+inverse.DT*one_form(
            left, right)

    return interval_action


def run(mode: str) -> dict:
    inverse.rank.lock_inertia()
    inverse.rest.SIDE = inverse.SIDE
    inverse.rest.DT = inverse.DT
    previous, current, following, theta = inverse.rest_layers()
    base = tuple(torch.tensor(value, dtype=torch.float64)
                 for value in (previous, current, following))
    interval_action = make_interval_action(mode)
    three_layer_action = lambda a, b, c: (
        interval_action(a, b)+interval_action(b, c))
    del_residual = grad(three_layer_action, argnums=1)
    right_momentum = grad(interval_action, argnums=1)
    if mode == "local":
        one_form = local_one_form
    elif mode == "bond":
        one_form = bond_one_form
    else:
        one_form = area_gradient_kinetic

    rest_value = float(one_form(base[0], base[1]).detach())
    rest_left = base[0].detach().requires_grad_(True)
    rest_right = base[1].detach().requires_grad_(True)
    rest_gradient = torch.autograd.grad(
        one_form(rest_left, rest_right), (rest_left, rest_right))
    rest_first_variation = float(torch.sqrt(sum(
        torch.sum(value*value) for value in rest_gradient)).detach())
    rest_del = del_residual(*base).detach().numpy()

    def batched_jvp(argument: str, vectors: np.ndarray) -> np.ndarray:
        vector_tensor = torch.tensor(vectors.T, dtype=torch.float64)
        if argument == "previous":
            function = lambda value: del_residual(value, base[1], base[2])
            point = base[0]
        else:
            function = lambda value: del_residual(base[0], base[1], value)
            point = base[2]
        images = vmap(
            lambda vector: jvp(function, (point,), (vector,))[1],
            chunk_size=8)(vector_tensor)
        return images.detach().numpy().T

    def density_tensor(left: torch.Tensor, right: torch.Tensor,
                       charge_mode: str) -> torch.Tensor:
        momentum = right_momentum(left, right)
        psi, chi = inverse.current_audit.mold.unpack_tensor(
            right, inverse.SIDE)
        zero_psi = torch.zeros_like(psi)
        zero_chi = torch.zeros_like(chi)
        generator = inverse.current_audit.pack_tensor(
            1j*psi if charge_mode == "Psi" else zero_psi,
            1j*chi if charge_mode == "chi" else zero_chi)
        return (momentum*generator).reshape(
            inverse.SIDE, inverse.SIDE, inverse.SIDE, 4, 3).sum(dim=(-1, -2))

    def density_and_derivative(left: np.ndarray, right: np.ndarray,
                               dleft: np.ndarray | None,
                               dright: np.ndarray | None,
                               charge_mode: str
                               ) -> tuple[np.ndarray, np.ndarray]:
        l = torch.tensor(left, dtype=torch.float64)
        r = torch.tensor(right, dtype=torch.float64)
        dl = torch.zeros_like(l) if dleft is None else torch.tensor(
            dleft, dtype=torch.float64)
        dr = torch.zeros_like(r) if dright is None else torch.tensor(
            dright, dtype=torch.float64)
        function = lambda a, b: density_tensor(a, b, charge_mode)
        value, derivative = jvp(function, (l, r), (dl, dr))
        return value.detach().numpy(), derivative.detach().numpy()

    def center_derivative(density: np.ndarray,
                          derivative: np.ndarray) -> np.ndarray:
        total = float(np.sum(density))
        dtotal = float(np.sum(derivative))
        coordinates = np.indices(density.shape)
        values = []
        for axis in range(3):
            moment = float(np.sum(coordinates[axis]*density))
            dmoment = float(np.sum(coordinates[axis]*derivative))
            values.append((dmoment*total-moment*dtotal)/(total**2))
        return np.asarray(values)

    def response(launch: np.ndarray, target: np.ndarray) -> np.ndarray:
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

    dimension = previous.size
    previous_block = batched_jvp("previous", np.eye(dimension))
    factor = lu_factor(previous_block)
    rows = []
    for axis in range(3):
        labels, targets = component.component_targets(axis, np.exp(1j*theta))
        images = batched_jvp("following", targets)
        launches = lu_solve(factor, -images)
        responses = np.column_stack([
            response(launches[:, column], targets[:, column])
            for column in range(targets.shape[1])
        ])
        charge_rank = np.linalg.matrix_rank(responses[:2], tol=1e-8)
        null = np.linalg.svd(
            responses[:2], full_matrices=True)[2][charge_rank:].T
        fixed_center = responses[2:]@null
        singular = np.linalg.svd(fixed_center, compute_uv=False)
        threshold = 1e-9*max(singular[0], 1e-30)
        rows.append({
            "phase_ramp_axis": axis,
            "component_labels": labels,
            "fixed_charge_center_rank": int(np.sum(singular > threshold)),
            "fixed_charge_center_singular_values": [float(value)
                                                     for value in singular],
            "response_matrix": responses.tolist(),
        })

    result = {
        "status": "V4_OVERLAP_SYMPLECTIC_COMPLETION_RESPONSE_AUDIT",
        "mode": mode,
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "action_delta": (
            "unit-strength midpoint discretization of "
            + ({
                "local": "sum Im(conj(S_i) Sdot_i), S=chi^dagger Psi",
                "bond": ("sum_C19 w Im(conj(S_i) Sdot_j + "
                         "conj(S_j) Sdot_i)/2, S=chi^dagger Psi"),
                "area_gradient": ("sum_C19 w |D_t(chi wedge Psi)_j-"
                                  "D_t(chi wedge Psi)_i|^2/(2*24)"),
            }[mode]) + ", normalized by chi0^2"),
        "coefficient_provenance": (
            "unit normalization is a rank screen only; no coefficient has "
            "been selected or admitted"),
        "excluded_total_derivative": (
            "sum_C19 w Im(conj(S_i) Sdot_j - conj(S_j) Sdot_i) "
            "= d/dt sum_C19 w Im(conj(S_i) S_j)"),
        "rest_one_form_value": rest_value,
        "rest_one_form_first_variation_norm": rest_first_variation,
        "candidate_rest_DEL_rms": float(
            np.linalg.norm(rest_del)/np.sqrt(rest_del.size)),
        "candidate_rows": rows,
        "verdict": (
            "OVERLAP_SYMPLECTIC_BASIS_SUPPLIES_THIRD_TRANSPORT_CHANNEL"
            if all(row["fixed_charge_center_rank"] == 3 for row in rows)
            and rows[2]["fixed_charge_center_singular_values"][2] > 1e-8
            else "OVERLAP_SYMPLECTIC_BASIS_DOES_NOT_CLOSE_TRANSPORT"),
        "claim_boundary": (
            "This is a linear response-rank screen at unit strength. It is "
            "not coefficient provenance, a bounded Hamiltonian proof, rest "
            "stability, nonlinear movement, or a Dirac pass."),
    }
    output = HERE / f"overlap_symplectic_{mode}_probe.json"
    output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n",
                      encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("local", "bond", "area_gradient"),
                        required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.mode), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
