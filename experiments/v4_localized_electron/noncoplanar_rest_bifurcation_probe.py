"""Screen the exact rest orbit for a nearby non-coplanar carrier branch.

The fixed-charge transport obstruction is caused by the present carrier's
common complex two-plane.  Before changing the action, test whether the same
unrestricted rest equations have a local branch that gives the eight occupied
corners third-component Psi/chi structure.

The probe constructs the relative-orbit action

    S_orbit(q) = S(R_- q, q) + S(q, R_+ q)

using the current action and exact Psi phase step.  It projects the exact
Hessian onto all real and imaginary third-component directions of Psi and chi
at the eight occupied corners (32 real directions).  A robust zero eigenvalue
is necessary for a carrier-only branch to bifurcate continuously from this
rest state.  Absence of such a zero is only a scoped local result: a branch
involving the halo/exterior or a disconnected finite deformation remains
possible.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import torch
from torch.func import grad, jvp, vmap

import localized_carrier_contrast_inverse_launch as inverse


HERE = Path(__file__).resolve().parent


def phase_rotate(raw: torch.Tensor, phase: complex) -> torch.Tensor:
    psi, chi = inverse.current_audit.mold.unpack_tensor(raw, inverse.SIDE)
    phase_tensor = torch.tensor(phase, dtype=torch.complex128)
    return inverse.current_audit.pack_tensor(phase_tensor * psi, chi)


def orbit_action(raw: torch.Tensor, phase: complex) -> torch.Tensor:
    previous = phase_rotate(raw, np.conjugate(phase))
    following = phase_rotate(raw, phase)
    return (
        inverse.interval_action(previous, raw)
        + inverse.interval_action(raw, following)
    )


def normal_basis() -> tuple[list[str], np.ndarray]:
    shape = (inverse.SIDE, inverse.SIDE, inverse.SIDE, 4, 3)
    center = inverse.ORIGIN
    labels: list[str] = []
    columns: list[np.ndarray] = []
    field_rows = (("Psi_re", 0), ("Psi_im", 1),
                  ("chi_re", 2), ("chi_im", 3))
    for corner in itertools.product((0, 1), repeat=3):
        site = tuple(center + value for value in corner)
        for field, row in field_rows:
            value = np.zeros(shape, dtype=np.float64)
            value[site + (row, 2)] = 1.0
            labels.append(f"{field}_{corner[0]}{corner[1]}{corner[2]}")
            columns.append(value.reshape(-1))
    return labels, np.column_stack(columns)


def main() -> None:
    inverse.rank.lock_inertia()
    inverse.rest.SIDE = inverse.SIDE
    inverse.rest.DT = inverse.DT
    _, current, _, theta = inverse.rest_layers()
    phase = np.exp(1j * theta)
    base = torch.tensor(current, dtype=torch.float64)
    residual_function = grad(lambda raw: orbit_action(raw, phase))
    residual = residual_function(base)
    labels, basis = normal_basis()
    vectors = torch.tensor(basis.T, dtype=torch.float64)
    hessian_columns = vmap(
        lambda vector: jvp(residual_function, (base,), (vector,))[1],
        chunk_size=8)(vectors).detach().numpy().T
    projected = basis.T @ hessian_columns
    symmetric = (projected + projected.T) / 2
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    scale = max(float(np.max(np.abs(eigenvalues))), 1e-30)
    relative = np.abs(eigenvalues) / scale
    zero_threshold = 1e-8
    zero_indices = np.where(relative < zero_threshold)[0]
    leading_rows = []
    for index in np.argsort(np.abs(eigenvalues))[:8]:
        vector = eigenvectors[:, index]
        order = np.argsort(np.abs(vector))[::-1][:6]
        leading_rows.append({
            "eigenvalue": float(eigenvalues[index]),
            "relative_abs_eigenvalue": float(relative[index]),
            "largest_components": [
                {"label": labels[column], "coefficient": float(vector[column])}
                for column in order
            ],
        })

    result = {
        "status": "V4_NONCOPLANAR_REST_BIFURCATION_SCREEN",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "action": "unchanged current reciprocal research action",
        "subspace": (
            "real/imaginary third components of Psi and chi on all eight "
            "occupied carrier corners"),
        "subspace_dimension": int(basis.shape[1]),
        "rest_orbit_gradient_rms": float(
            torch.linalg.vector_norm(residual).detach()
            / np.sqrt(residual.numel())),
        "projected_hessian_symmetry_defect": float(
            np.linalg.norm(projected - projected.T)
            / max(np.linalg.norm(projected), 1e-30)),
        "eigenvalue_minimum": float(np.min(eigenvalues)),
        "eigenvalue_maximum": float(np.max(eigenvalues)),
        "smallest_absolute_eigenvalue": float(
            np.min(np.abs(eigenvalues))),
        "smallest_relative_absolute_eigenvalue": float(np.min(relative)),
        "zero_threshold_relative": zero_threshold,
        "zero_mode_count": int(zero_indices.size),
        "smallest_modes": leading_rows,
        "verdict": (
            "LOCAL_NONCOPLANAR_CARRIER_BIFURCATION_DIRECTION_FOUND"
            if zero_indices.size else
            "NO_LOCAL_CARRIER_ONLY_NONCOPLANAR_BIFURCATION"),
        "claim_boundary": (
            "Necessary local carrier-only bifurcation screen. A zero mode is "
            "not a solved branch. No zero excludes only a continuously nearby "
            "branch confined to these 32 corner-normal directions; halo, "
            "exterior, mixed in-plane, and disconnected branches remain open."),
    }
    output = HERE / "noncoplanar_rest_bifurcation_probe.json"
    output.write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    print(json.dumps({
        "verdict": result["verdict"],
        "rest_orbit_gradient_rms": result["rest_orbit_gradient_rms"],
        "projected_hessian_symmetry_defect": result[
            "projected_hessian_symmetry_defect"],
        "zero_mode_count": result["zero_mode_count"],
        "smallest_relative_absolute_eigenvalue": result[
            "smallest_relative_absolute_eigenvalue"],
        "output": str(output),
    }, indent=2))


if __name__ == "__main__":
    main()
