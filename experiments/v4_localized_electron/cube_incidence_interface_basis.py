"""Derive carrier--vacuum interface versions of the cube Berry action.

The product activation a_i a_j used by the stable rest candidate vanishes on
an occupied--quiet bond.  This screen replaces only that scalar prefactor by
the symmetric endpoint mean (a_i+a_j)/2.  The construction remains local,
reciprocal, C19-cubic and free of carrier labels.  Coefficients are restricted
to the exact rest-force nullspace before receiver response is inspected.
"""

from __future__ import annotations

import fractions
import json
from pathlib import Path

import numpy as np
import torch
from scipy.linalg import null_space

import canonical_noncoplanar_bifurcation_audit as native_time
import canonical_transfer_oscillator_spectrum as native_spectrum
import cube_incidence_nonlinear_action as action
import cube_incidence_symplectic_basis as basis
import localized_stability as native


HERE = Path(__file__).resolve().parent
SIDE = 5
NAMES = tuple(f"{name}_C19_interface_mean"
              for name in basis.REPRESENTATIONS)


def one_form(left: torch.Tensor, right: torch.Tensor, name: str):
    midpoint = (left+right)/2
    field = basis.representation(midpoint, name)
    dfield = (basis.representation(right, name)
              - basis.representation(left, name))
    active = basis.activation(midpoint)
    result = left.new_zeros((), dtype=torch.float64)
    for offset, weight in native.DIRECTIONS:
        neighbor = torch.roll(field, shifts=offset, dims=(0, 1, 2))
        dneighbor = torch.roll(dfield, shifts=offset, dims=(0, 1, 2))
        neighbor_active = torch.roll(active, shifts=offset, dims=(0, 1, 2))
        difference = neighbor-field
        ddifference = dneighbor-dfield
        pairing = torch.sum(torch.conj(difference)*ddifference,
                            dim=tuple(range(3, difference.ndim)))
        result = result+0.5*weight*torch.sum(
            0.5*(active+neighbor_active)*torch.imag(pairing))
    return result


def interval(left, right, coefficients):
    return sum(coefficients[index]*one_form(left, right, name)
               for index, name in enumerate(basis.REPRESENTATIONS))


def residual(left, middle, right, coefficients):
    value = interval(left, middle, coefficients)+interval(
        middle, right, coefficients)
    return torch.autograd.grad(value, middle, create_graph=True)[0]


def phase_tangent(raw: np.ndarray, axis: int) -> np.ndarray:
    tensor = torch.tensor(raw, dtype=torch.float64)
    psi, chi = basis.native_box.unpack(tensor, SIDE)
    coordinate = np.indices((SIDE, SIDE, SIDE))[axis]-(SIDE-1)/2
    coordinate = torch.tensor(coordinate, dtype=torch.float64)[..., None]
    dpsi = -1j*coordinate*psi
    dchi = -1j*coordinate*chi
    return torch.stack((dpsi.real, dpsi.imag, dchi.real, dchi.imag),
                       dim=-2).reshape(-1).numpy()


def receiver_selector(axis: int, sign: int) -> np.ndarray:
    coordinate = np.indices((SIDE, SIDE, SIDE))[axis]
    sites = coordinate >= 3 if sign > 0 else coordinate <= 0
    # Select both native fields and every complex/C3 coordinate.
    return np.repeat(sites[..., None, None], 4, axis=3).repeat(3, axis=4).ravel()


def rational(value: float) -> str:
    return str(fractions.Fraction(float(value)).limit_denominator(64))


def main() -> None:
    bare, _matrices = action.expected_blocks()
    theta = float(np.arccos(
        1-native_time.DT**2*native_time.CHI0**2/2))
    rotation = native_spectrum.rotation_matrix(bare["rest"].size, theta)
    layers = tuple(torch.tensor(value, dtype=torch.float64,
                                requires_grad=True)
                   for value in (rotation.T@bare["rest"], bare["rest"],
                                 rotation@bare["rest"]))
    old_columns = []
    old_receiver_columns = {(axis, sign): [] for axis in range(3)
                            for sign in (-1, 1)}
    for index in range(len(basis.BASIS)):
        coefficients = torch.zeros(len(basis.BASIS), dtype=torch.float64)
        coefficients[index] = 1
        force = basis.del_residual(*layers, coefficients)
        old_columns.append(force.detach().numpy())
        for axis in range(3):
            direction = torch.tensor(
                phase_tangent(layers[0].detach().numpy(), axis),
                dtype=torch.float64)

            def old_function(value):
                return basis.del_residual(
                    value, layers[1], layers[2], coefficients)

            _value, image = torch.autograd.functional.jvp(
                old_function, layers[0], direction, create_graph=False,
                strict=False)
            data = image.detach().numpy()
            for sign in (-1, 1):
                old_receiver_columns[(axis, sign)].append(
                    data[receiver_selector(axis, sign)])

    columns = []
    receiver_columns = {(axis, sign): [] for axis in range(3)
                        for sign in (-1, 1)}
    for index in range(len(NAMES)):
        coefficients = torch.zeros(len(NAMES), dtype=torch.float64)
        coefficients[index] = 1
        force = residual(*layers, coefficients)
        columns.append(force.detach().numpy())
        for axis in range(3):
            direction = torch.tensor(
                phase_tangent(layers[0].detach().numpy(), axis),
                dtype=torch.float64)

            def function(value):
                return residual(value, layers[1], layers[2], coefficients)

            _value, image = torch.autograd.functional.jvp(
                function, layers[0], direction, create_graph=False,
                strict=False)
            data = image.detach().numpy()
            for sign in (-1, 1):
                selector = receiver_selector(axis, sign)
                receiver_columns[(axis, sign)].append(data[selector])

    forces = np.column_stack(columns)
    old_forces = np.column_stack(old_columns)
    scale = np.maximum(np.linalg.norm(forces, axis=0), 1e-30)
    kernel_normalized = null_space(forces/scale, rcond=1e-10)
    kernel = (kernel_normalized/scale[:, None]
              if kernel_normalized.size else np.zeros((len(NAMES), 0)))
    # Normalize each exact-rest direction for readable reporting only.
    rows = []
    for column in range(kernel.shape[1]):
        vector = kernel[:, column]
        vector /= max(np.max(abs(vector)), 1e-30)
        responses = {}
        for key, values in receiver_columns.items():
            responses[f"axis_{key[0]}_sign_{key[1]}"] = float(
                np.linalg.norm(np.column_stack(values)@vector))
        rows.append({
            "coefficients": {name: float(value) for name, value in zip(
                NAMES, vector, strict=True)},
            "rationalized_coefficients": {name: rational(value)
                                           for name, value in zip(
                                               NAMES, vector, strict=True)},
            "rest_DEL_norm": float(np.linalg.norm(forces@vector)),
            "receiver_phase_response_norms": responses,
            "minimum_receiver_phase_response": min(responses.values()),
        })
    best = max(rows, key=lambda row: row[
        "minimum_receiver_phase_response"], default=None)

    # Each interface term is now completed by the least-norm combination of
    # the already-enumerated local/product-activated cube actions that cancels
    # its rest force.  This is an inverse rest equation, not a transport fit.
    completion_rows = []
    old_pinv = np.linalg.pinv(old_forces, rcond=1e-10)
    for index, name in enumerate(NAMES):
        old_coefficients = -old_pinv@forces[:, index]
        residual_force = old_forces@old_coefficients+forces[:, index]
        responses = {}
        for key in receiver_columns:
            image = (np.column_stack(old_receiver_columns[key])
                     @old_coefficients
                     + np.column_stack(receiver_columns[key])[:, index])
            responses[f"axis_{key[0]}_sign_{key[1]}"] = float(
                np.linalg.norm(image))
        completion_rows.append({
            "interface_term": name,
            "interface_coefficient": 1.0,
            "old_basis_coefficients": {
                old_name: float(value) for old_name, value in zip(
                    basis.BASIS, old_coefficients, strict=True)},
            "old_basis_rationalized": {
                old_name: rational(value) for old_name, value in zip(
                    basis.BASIS, old_coefficients, strict=True)},
            "rest_DEL_norm": float(np.linalg.norm(residual_force)),
            "receiver_phase_response_norms": responses,
            "minimum_receiver_phase_response": min(responses.values()),
        })
    completed = [row for row in completion_rows
                 if row["rest_DEL_norm"] < 1e-8]
    best_completion = max(
        completed, key=lambda row: row["minimum_receiver_phase_response"],
        default=None)
    result = {
        "status": "V4_CUBE_INCIDENCE_INTERFACE_BASIS",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "basis": NAMES,
        "basis_rest_force_norms": np.linalg.norm(forces, axis=0).tolist(),
        "rest_force_rank": int(np.linalg.matrix_rank(
            forces, tol=max(np.linalg.svd(forces, compute_uv=False)[0], 1e-30)
            *1e-10)),
        "rest_nullity": int(kernel.shape[1]),
        "rest_preserving_rows": rows,
        "best_rest_preserving_row": best,
        "rest_completed_interface_rows": completion_rows,
        "best_rest_completed_interface_row": best_completion,
        "verdict": (
            "REST_COMPLETED_INTERFACE_RESPONSE_EXISTS"
            if best_completion is not None
            and best_completion["minimum_receiver_phase_response"] > 1e-12
            else "INTERFACE_RESPONSE_REST_CONFLICT"),
        "claim_boundary": (
            "Rest-force nullspace and first receiver response only.  No "
            "coefficient is admitted; spectrum, nonlinear movement and "
            "Dirac-current closure remain separate gates."),
    }
    (HERE/"cube_incidence_interface_basis.json").write_text(
        json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
