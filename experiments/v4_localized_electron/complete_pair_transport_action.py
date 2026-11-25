"""Derive and screen the missing complement to the native wedge transport.

The canonical bond field ``chi wedge Psi`` is the antisymmetric part of the
local bilinear pair.  This calculation tests the complementary native data
without introducing a direction, corner label, spinor matrix, or target
track.  Every candidate is a positive C19 sum and is varied with respect to
both complex fields.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from scipy.linalg import lu_factor, lu_solve, null_space
from scipy.optimize import lsq_linear

import canonical_noncoplanar_bifurcation_audit as native_time
import canonical_noncoplanar_box_transfer as native_box
import current_action_two_layer_spectrum as spectrum
import localized_stability as native
import personal_touch_unit_geometric_canonical as canonical


HERE = Path(__file__).resolve().parent
SIDE = 5
NAMES = (
    "bilinear_pair_tensor",
    "hermitian_pair_tensor",
    "parallel_scalar",
    "symmetric_bilinear_tensor",
    "total_density_mold",
)


def norm2(value: torch.Tensor) -> torch.Tensor:
    return torch.real(torch.sum(torch.conj(value)*value, dim=(-2, -1)))


def scalar_norm2(value: torch.Tensor) -> torch.Tensor:
    return torch.real(torch.conj(value)*value)


def candidate_energies(raw: torch.Tensor) -> tuple[torch.Tensor, ...]:
    psi, chi = native_box.unpack(raw, SIDE)
    bilinear = chi[..., :, None]*psi[..., None, :]
    hermitian = chi[..., :, None]*torch.conj(psi[..., None, :])
    parallel = torch.sum(torch.conj(chi)*psi, dim=-1)
    symmetric = (chi[..., :, None]*psi[..., None, :]
                 + psi[..., :, None]*chi[..., None, :])/np.sqrt(2)
    rho_psi = torch.real(torch.sum(torch.conj(psi)*psi, dim=-1))
    rho_chi = torch.real(torch.sum(torch.conj(chi)*chi, dim=-1))
    values = [raw.new_zeros(()) for _ in NAMES]
    for offset, weight in native.DIRECTIONS:
        db = bilinear-torch.roll(bilinear, shifts=offset, dims=(0, 1, 2))
        dh = hermitian-torch.roll(hermitian, shifts=offset, dims=(0, 1, 2))
        dp = parallel-torch.roll(parallel, shifts=offset, dims=(0, 1, 2))
        ds = symmetric-torch.roll(symmetric, shifts=offset, dims=(0, 1, 2))
        values[0] += weight*torch.sum(norm2(db))/(2*native_time.CHI0**2)
        values[1] += weight*torch.sum(norm2(dh))/(2*native_time.CHI0**2)
        values[2] += weight*torch.sum(scalar_norm2(dp))/(2*native_time.CHI0**2)
        values[3] += weight*torch.sum(norm2(ds))/(2*native_time.CHI0**2)
    values[4] = torch.sum(
        (rho_chi+rho_psi-native_time.CHI0**2)**2)/4
    return tuple(values)


def force_and_products(state: torch.Tensor, directions: list[np.ndarray]):
    forces, products = [], [[] for _ in directions]
    for index in range(len(NAMES)):
        function = lambda raw, selected=index: candidate_energies(raw)[selected]
        force = torch.func.grad(function)(state)
        forces.append(force.detach().numpy())
        for direction_index, direction in enumerate(directions):
            vector = torch.tensor(direction, dtype=torch.float64)
            _value, product = torch.autograd.functional.hvp(
                function, state, vector, create_graph=False, strict=True)
            products[direction_index].append(product.detach().numpy())
    return np.column_stack(forces), products


def main() -> None:
    bare = np.load(canonical.CACHE)
    additions = np.load(HERE/"personal_touch_activated_bond_basis_v2_cache.npz")
    current_stiffness = (
        bare["stiffness"]
        + additions["hessian_carrier_carrier_chi"]
        + additions["hessian_carrier_carrier_Psi_tight_frame"])
    state = torch.tensor(bare["rest"], dtype=torch.float64)
    mode = np.load(HERE/"current_action_fastest_mode.npz")["current"]
    psi, chi = native_box.unpack(state, SIDE)
    translations = native.translation_vectors(
        psi.detach().numpy(), chi.detach().numpy())
    directions = [mode.real, mode.imag]
    directions.extend(translations[:, axis] for axis in range(3))
    forces, products = force_and_products(state, directions)

    mass, gyro = bare["mass"], bare["gyro"]
    denominator = float(np.vdot(mode, mass@mode).real)
    curvatures = np.asarray([
        (np.dot(mode.real, products[0][index])
         + np.dot(mode.imag, products[1][index]))/denominator
        for index in range(len(NAMES))])

    dt = native_time.DT
    factor = lu_factor(mass/dt+gyro/2)
    backward = mass/dt-gyro/2
    zero = dt*current_stiffness-2*mass/dt
    base_blocks, response_columns = [], [[] for _ in NAMES]
    for axis in range(3):
        vector = translations[:, axis]
        following = -lu_solve(factor, zero@vector+backward@vector)
        base_blocks.append(following-vector)
        for index in range(len(NAMES)):
            response_columns[index].append(
                -lu_solve(factor, dt*products[2+axis][index]))
    target = -np.concatenate(base_blocks)
    response = np.column_stack([
        np.concatenate(blocks) for blocks in response_columns])

    force_scale = max(float(np.linalg.norm(forces, axis=0).max()), 1.0)
    force_rank = int(np.linalg.matrix_rank(forces, tol=1e-10*force_scale))
    kernel = null_space(forces, rcond=1e-10)
    if kernel.shape[1]:
        reduced_response = response@kernel
        reduced_solution, *_ = np.linalg.lstsq(
            reduced_response, target, rcond=1e-12)
        coefficients = kernel@reduced_solution
        residual = reduced_response@reduced_solution-target
        rest_preserving_residual = float(
            np.linalg.norm(residual)/np.linalg.norm(target))
    else:
        coefficients = np.zeros(len(NAMES))
        rest_preserving_residual = 1.0

    positive = lsq_linear(
        np.vstack((response, forces/force_scale)),
        np.concatenate((target, np.zeros(forces.shape[0]))),
        bounds=(0, np.inf), tol=1e-13, lsmr_tol=1e-13, max_iter=1000)
    positive_transport_residual = float(
        np.linalg.norm(response@positive.x-target)/np.linalg.norm(target))
    positive_rest_force = float(np.linalg.norm(forces@positive.x))

    energy_values = [float(value.detach()) for value in candidate_energies(state)]
    force_norms = np.linalg.norm(forces, axis=0)
    result = {
        "status": "V4_COMPLETE_PAIR_TRANSPORT_ACTION_SCREEN",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "derivation": {
            "native_existing_channel": "antisymmetric bilinear chi wedge Psi",
            "tested_complement": (
                "complete bilinear tensor, its Hermitian phase channel, "
                "the parallel scalar, and the symmetric tensor complement"),
            "all_terms": (
                "real positive reciprocal C19 bond sums using only native "
                "complex Psi and chi"),
        },
        "candidates": {
            name: {
                "rest_energy": energy_values[index],
                "rest_force_norm": float(force_norms[index]),
                "fastest_mode_generalized_curvature": float(curvatures[index]),
            } for index, name in enumerate(NAMES)
        },
        "rest_force_matrix_rank": force_rank,
        "rest_preserving_coefficient_kernel_dimension": int(kernel.shape[1]),
        "rest_preserving_unconstrained_coefficients": {
            name: float(value) for name, value in zip(
                NAMES, coefficients, strict=True)},
        "rest_preserving_translation_relative_residual": rest_preserving_residual,
        "positive_joint_fit": {
            "coefficients": {name: float(value) for name, value in zip(
                NAMES, positive.x, strict=True)},
            "translation_relative_residual": positive_transport_residual,
            "rest_force_norm": positive_rest_force,
        },
        "existing_fastest_mode_stiffness": -1.0807423181893132,
        "verdict": (
            "PAIR_COMPLEMENT_HAS_REST_PRESERVING_TRANSPORT_RESPONSE"
            if rest_preserving_residual < 1e-8
            else "PAIR_COMPLEMENT_ALONE_DOES_NOT_CLOSE_TRANSPORT"),
        "claim_boundary": (
            "This screens the mathematically complete two-site pair channels "
            "at the exact rest orbit. It neither installs an action nor proves "
            "nonlinear motion. A failed screen is scoped to these two-site "
            "position terms and does not exclude triangle or symplectic terms."),
    }
    (HERE/"complete_pair_transport_action.json").write_text(
        json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
