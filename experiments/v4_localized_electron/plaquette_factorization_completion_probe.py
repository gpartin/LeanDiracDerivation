"""Screen a native square-face interaction for the missing edge channel.

At the exact carrier, the six cube faces obey the product identity

    W_i tensor W_(i+a+b) = W_(i+a) tensor W_(i+b),

where ``W = chi wedge Psi`` and ``a,b`` are distinct positive lattice axes.
This motivates the positive, local and cubic-symmetric face energy

    U_face = 1/2 sum_(i,a<b) ||
        (W_i/R) tensor (W_(i+a+b)/R)
        - (W_(i+a)/R) tensor (W_(i+b)/R) ||_F^2.

It contains no corner label, direction selector, spinor matrix or target
track.  It vanishes with its first variation on every factorized face.  This
probe asks only whether its curvature supplies the absent third fixed-charge
transport direction.  Unit normalization is a basis-rank screen, not an
admitted coefficient.
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
STRENGTH = 1.0


def plaquette_potential(raw: torch.Tensor) -> torch.Tensor:
    psi, chi = inverse.current_audit.mold.unpack_tensor(raw, inverse.SIDE)
    w = torch.linalg.cross(chi, psi, dim=-1)/inverse.rank.R
    result = torch.zeros((), dtype=torch.float64)
    for first, second in ((0, 1), (0, 2), (1, 2)):
        shift_first = tuple(1 if axis == first else 0 for axis in range(3))
        shift_second = tuple(1 if axis == second else 0 for axis in range(3))
        wa = torch.roll(w, shifts=shift_first, dims=(0, 1, 2))
        wb = torch.roll(w, shifts=shift_second, dims=(0, 1, 2))
        wab = torch.roll(
            w, shifts=tuple(shift_first[axis]+shift_second[axis]
                            for axis in range(3)), dims=(0, 1, 2))
        defect = (w[..., :, None]*wab[..., None, :]
                  - wa[..., :, None]*wb[..., None, :])
        result = result+torch.sum(torch.real(torch.conj(defect)*defect))/2
    return result


def interval_action(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return (inverse.interval_action(left, right)
            - inverse.DT*STRENGTH*(plaquette_potential(left)
                          + plaquette_potential(right))/2)


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
    return vmap(
        lambda vector: jvp(function, (point,), (vector,))[1],
        chunk_size=8)(vector_tensor).detach().numpy().T


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


def center_derivative(density: np.ndarray,
                      derivative: np.ndarray) -> np.ndarray:
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
    global STRENGTH
    parser = argparse.ArgumentParser()
    parser.add_argument("--strength", type=float, default=1.0)
    args = parser.parse_args()
    STRENGTH = args.strength
    inverse.rank.lock_inertia()
    inverse.rest.SIDE = inverse.SIDE
    inverse.rest.DT = inverse.DT
    previous, current, following, theta = inverse.rest_layers()
    base = tuple(torch.tensor(value, dtype=torch.float64)
                 for value in (previous, current, following))
    rest_energy = float(plaquette_potential(base[1]).detach())
    rest_variable = base[1].detach().requires_grad_(True)
    rest_gradient = torch.autograd.grad(
        plaquette_potential(rest_variable), rest_variable)[0]
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
        null = np.linalg.svd(
            responses[:2], full_matrices=True)[2][charge_rank:].T
        fixed_center = responses[2:]@null
        left, singular, _ = np.linalg.svd(fixed_center, full_matrices=True)
        threshold = 1e-8*max(singular[0], 1e-30)
        rows.append({
            "phase_ramp_axis": axis,
            "component_labels": labels,
            "fixed_charge_center_rank": int(np.sum(singular > threshold)),
            "fixed_charge_center_singular_values": [float(value)
                                                     for value in singular],
            "relative_third_singular_value": float(
                singular[2]/max(singular[0], 1e-30)),
            "missing_physical_direction": [float(value)
                                            for value in left[:, -1]],
            "response_matrix": responses.tolist(),
        })

    robust = all(row["fixed_charge_center_rank"] == 3
                 and row["relative_third_singular_value"] > 1e-6
                 for row in rows)
    result = {
        "status": "V4_PLAQUETTE_FACTORIZATION_TRANSPORT_AUDIT",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "action_delta": (
            "positive square-face factorization energy of opposite-corner "
            "W tensor products, summed over all three cubic face planes"),
        "screen_strength": STRENGTH,
        "coefficient_provenance": (
            "unit normalization of W/chi0 is a response-rank screen only"),
        "rest_plaquette_energy": rest_energy,
        "rest_plaquette_first_variation_norm": float(
            torch.linalg.vector_norm(rest_gradient).detach()),
        "candidate_rest_DEL_rms": float(
            np.linalg.norm(rest_del)/np.sqrt(rest_del.size)),
        "axis_rows": rows,
        "all_axes_robust_rank_three": robust,
        "verdict": (
            "PLAQUETTE_FACTORIZATION_SUPPLIES_THIRD_TRANSPORT_CHANNEL"
            if robust else
            "PLAQUETTE_FACTORIZATION_DOES_NOT_CLOSE_TRANSPORT"),
        "claim_boundary": (
            "This is a linear response-rank screen. A pass still requires "
            "coefficient derivation, full rest spectrum, nonlinear movement, "
            "current matching, and all Dirac gates."),
    }
    suffix = ("unit" if STRENGTH == 1 else
              f"strength_{STRENGTH:g}".replace(".", "p"))
    (HERE / f"plaquette_factorization_completion_probe_{suffix}.json").write_text(
        json.dumps(result, indent=2, allow_nan=False)+"\n",
        encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
