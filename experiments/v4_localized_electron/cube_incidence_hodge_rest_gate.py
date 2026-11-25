"""Exact rest gate for the complete C19 scalar--vector Hodge connection.

The native complex pair supplies a scalar overlap and an oriented-area vector,

    tau = chi . psi / chi0,       W = chi x psi / chi0.

The unique rotationally covariant first-order scalar/vector operator assembled
from the established C19 first moments is

    D(tau, W) = (-div_19 W, grad_19 tau + curl_19 W).

It obeys D^2 = -sum_a G_a^2 on periodic fields.  This script tests four real
interval one-forms Im <F, D dF> (common/relative phase and unactivated/carrier-
activated) together with the eight previously derived cube-incidence Berry
forms.  A Hodge repair is admissible at the present carrier only if the exact
DEL-force nullspace contains a direction with nonzero Hodge content.

This is an inverse/action-space gate.  It does not change or admit an action.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from scipy.linalg import null_space

import canonical_noncoplanar_bifurcation_audit as native_time
import canonical_noncoplanar_box_transfer as native_box
import canonical_transfer_oscillator_spectrum as native_spectrum
import cube_incidence_symplectic_basis as old
import localized_stability as native
import personal_touch_unit_geometric_canonical as canonical


HERE = Path(__file__).resolve().parent
SIDE = 5
HODGE_BASIS = (
    "hodge_common",
    "hodge_common_activated",
    "hodge_relative",
    "hodge_relative_activated",
)
BASIS = old.BASIS + HODGE_BASIS


def derivative19(field: torch.Tensor, axis: int) -> torch.Tensor:
    result = torch.zeros_like(field)
    for offset, weight in native.DIRECTIONS:
        if offset[axis] != 0:
            result = result + weight*offset[axis]*torch.roll(
                field, shifts=offset, dims=(0, 1, 2))
    return result/12


def hodge19(tau: torch.Tensor, area: torch.Tensor):
    grad_tau = torch.stack(
        tuple(derivative19(tau, axis) for axis in range(3)), dim=-1)
    div_area = sum(derivative19(area[..., axis], axis)
                   for axis in range(3))
    curl_area = torch.stack((
        derivative19(area[..., 2], 1)-derivative19(area[..., 1], 2),
        derivative19(area[..., 0], 2)-derivative19(area[..., 2], 0),
        derivative19(area[..., 1], 0)-derivative19(area[..., 0], 1),
    ), dim=-1)
    return -div_area, grad_tau+curl_area


def fields(raw: torch.Tensor, relative: bool, activated: bool):
    psi, chi = native_box.unpack(raw, SIDE)
    if relative:
        psi = torch.conj(psi)
    tau = torch.sum(chi*psi, dim=-1)/native_time.CHI0
    area = torch.linalg.cross(chi, psi, dim=-1)/native_time.CHI0
    if activated:
        density = torch.real(torch.sum(
            torch.conj(psi)*psi, dim=-1))/native_time.CHI0**2
        tau = density*tau
        area = density[..., None]*area
    return tau, area


def hodge_one_form(left: torch.Tensor, right: torch.Tensor,
                   relative: bool, activated: bool):
    midpoint = (left+right)/2
    tau, area = fields(midpoint, relative, activated)
    left_tau, left_area = fields(left, relative, activated)
    right_tau, right_area = fields(right, relative, activated)
    d_tau = right_tau-left_tau
    d_area = right_area-left_area
    scalar_image, vector_image = hodge19(d_tau, d_area)
    return torch.sum(torch.imag(
        torch.conj(tau)*scalar_image
        + torch.sum(torch.conj(area)*vector_image, dim=-1)))


def hodge_interval(left: torch.Tensor, right: torch.Tensor,
                   coefficients: torch.Tensor):
    terms = (
        hodge_one_form(left, right, False, False),
        hodge_one_form(left, right, False, True),
        hodge_one_form(left, right, True, False),
        hodge_one_form(left, right, True, True),
    )
    return sum(coefficients[index]*term for index, term in enumerate(terms))


def hodge_del(layers, coefficients: torch.Tensor):
    value = hodge_interval(layers[0], layers[1], coefficients)
    value = value+hodge_interval(layers[1], layers[2], coefficients)
    return torch.autograd.grad(value, layers[1], create_graph=True)[0]


def hodge_force_column(layers, index: int):
    coefficients = torch.zeros(len(HODGE_BASIS), dtype=torch.float64)
    coefficients[index] = 1
    return hodge_del(layers, coefficients).detach().numpy()


def square_identity_error() -> dict:
    generator = torch.Generator().manual_seed(4006)
    tau = torch.randn((SIDE, SIDE, SIDE), dtype=torch.complex128,
                      generator=generator)
    area = torch.randn((SIDE, SIDE, SIDE, 3), dtype=torch.complex128,
                       generator=generator)
    first_tau, first_area = hodge19(tau, area)
    second_tau, second_area = hodge19(first_tau, first_area)
    lap_tau = sum(derivative19(derivative19(tau, axis), axis)
                  for axis in range(3))
    lap_area = sum(derivative19(derivative19(area, axis), axis)
                   for axis in range(3))
    return {
        "scalar_relative_error": float(
            torch.linalg.norm(second_tau+lap_tau)
            / torch.linalg.norm(lap_tau)),
        "vector_relative_error": float(
            torch.linalg.norm(second_area+lap_area)
            / torch.linalg.norm(lap_area)),
    }


def main() -> None:
    bare = np.load(canonical.CACHE)
    dt = native_time.DT
    theta = float(np.arccos(1-dt**2*native_time.CHI0**2/2))
    rotation = native_spectrum.rotation_matrix(bare["rest"].size, theta)
    layers = tuple(torch.tensor(value, dtype=torch.float64,
                                requires_grad=True)
                   for value in (rotation.T@bare["rest"], bare["rest"],
                                 rotation@bare["rest"]))

    old_forces = np.column_stack(
        [old.force_column(layers, index) for index in range(len(old.BASIS))])
    hodge_forces = np.column_stack([
        hodge_force_column(layers, index)
        for index in range(len(HODGE_BASIS))])
    forces = np.column_stack((old_forces, hodge_forces))
    norms = np.linalg.norm(forces, axis=0)
    normalized = forces/np.maximum(norms, 1e-30)
    singular = np.linalg.svd(normalized, compute_uv=False)
    kernel = null_space(normalized, rcond=1e-10)

    rows = []
    hodge_directions = 0
    maximum_hodge_fraction = 0.0
    for column in range(kernel.shape[1]):
        physical = kernel[:, column]/np.maximum(norms, 1e-30)
        physical /= max(np.linalg.norm(physical), 1e-30)
        hodge_fraction = float(np.linalg.norm(physical[len(old.BASIS):]))
        maximum_hodge_fraction = max(maximum_hodge_fraction, hodge_fraction)
        if hodge_fraction > 1e-8:
            hodge_directions += 1
        rows.append({
            "hodge_fraction": hodge_fraction,
            "rest_DEL_norm": float(np.linalg.norm(forces@physical)),
            "coefficients": {
                name: float(value) for name, value in zip(
                    BASIS, physical, strict=True)
                if abs(value) > 1e-8},
        })

    # Determine whether any Hodge force can be cancelled by the old basis,
    # independently of the arbitrary numerical nullspace basis orientation.
    cancellation = []
    for index, name in enumerate(HODGE_BASIS):
        solved, *_ = np.linalg.lstsq(
            old_forces, -hodge_forces[:, index], rcond=1e-12)
        residual = old_forces@solved+hodge_forces[:, index]
        cancellation.append({
            "hodge_term": name,
            "raw_rest_DEL_norm": float(norms[len(old.BASIS)+index]),
            "best_cancelled_rest_DEL_norm": float(np.linalg.norm(residual)),
            "best_cancelled_relative_residual": float(
                np.linalg.norm(residual)/max(
                    norms[len(old.BASIS)+index], 1e-30)),
        })

    exact_hodge_null = hodge_directions > 0
    result = {
        "status": "V4_C19_HODGE_REST_GATE",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "operator": "D(tau,W)=(-div19 W, grad19 tau+curl19 W)",
        "native_fields": (
            "tau=(chi dot psi)/chi0, W=(chi cross psi)/chi0; "
            "common/relative and unactivated/activated variants"),
        "square_identity": square_identity_error(),
        "basis": list(BASIS),
        "rest_DEL_norms": {
            name: float(value) for name, value in zip(
                BASIS, norms, strict=True)},
        "normalized_force_rank": int(np.linalg.matrix_rank(
            normalized, tol=1e-10)),
        "nullity": int(kernel.shape[1]),
        "normalized_singular_values": [float(value) for value in singular],
        "nullspace_directions": rows,
        "hodge_nullspace_directions": hodge_directions,
        "maximum_hodge_fraction_in_reported_null_basis": (
            maximum_hodge_fraction),
        "individual_hodge_cancellation_by_old_basis": cancellation,
        "exact_rest_preserving_hodge_direction_found": exact_hodge_null,
        "verdict": (
            "REST_PRESERVING_HODGE_DIRECTION_FOUND"
            if exact_hodge_null else
            "HODGE_CONNECTION_REQUIRES_RESOLVED_REST_ORBIT"),
        "interpretation": (
            "The complete first-order C19 square root exists geometrically, "
            "but it may enter the action only if its rest force cancels "
            "exactly or the carrier is re-solved as a deformed native orbit."),
    }
    (HERE/"cube_incidence_hodge_rest_gate.json").write_text(
        json.dumps(result, indent=2)+"\n")
    print(json.dumps({
        "square_identity": result["square_identity"],
        "rank": result["normalized_force_rank"],
        "nullity": result["nullity"],
        "hodge_nullspace_directions": hodge_directions,
        "cancellation": cancellation,
        "verdict": result["verdict"],
    }, indent=2))


if __name__ == "__main__":
    main()
