"""Screen a native cube-flavor confinement invariant.

The cubical incidence algebra has a central orientation involution ``F``.
Its two eigenspaces are the two four-complex-dimensional carrier copies, but
the action need not select either copy.  For every elementary cube this file
forms the corner covariance of the native complex fields and tests the
positive, orientation-blind invariant

    U_F = 1/2 sum_c ||[F, C_c]||_F^2.

Because the commutator vanishes for either eigenspace, this is not a fixed
copy projector.  ``F`` is generated from oriented cube incidence, and the
square is unchanged when orientation reverses (F -> -F).  The screen checks
whether the invariant is already rest-null before it is allowed into a
movement action.  No Dirac comparator is used.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.func import grad, jvp

import canonical_noncoplanar_bifurcation_audit as native_time
import canonical_transfer_oscillator_spectrum as native_spectrum
import cubical_edge_shift_channels as cubical
import cube_incidence_nonlinear_action as native_action
import density_mold_connection_probe as local


HERE = Path(__file__).resolve().parent
SIDE = 5
CORNER_OFFSETS = tuple(
    (x, y, z) for x in (0, 1) for y in (0, 1) for z in (0, 1))


def cube_values(field: torch.Tensor) -> torch.Tensor:
    """Return [..., corner, native-component] for every cube origin."""
    return torch.stack([
        torch.roll(field, shifts=tuple(-value for value in offset),
                   dims=(0, 1, 2))
        for offset in CORNER_OFFSETS
    ], dim=-2)


def covariance(field: torch.Tensor) -> torch.Tensor:
    values = cube_values(field)
    return values @ torch.conj(values).transpose(-1, -2)


def commutator_energy(raw: torch.Tensor, channel: str) -> torch.Tensor:
    psi, chi = local.unpack_tensor(raw, SIDE)
    scale = native_time.CHI0**2
    if channel == "psi":
        matrix = covariance(psi)/scale
    elif channel == "chi_contrast":
        # A uniform chi vacuum has identical corner values.  Remove its
        # cube average before forming the covariance so vacuum contributes
        # exactly zero without prescribing an orientation.
        values = cube_values(chi)
        values = values-torch.mean(values, dim=-2, keepdim=True)
        matrix = (values @ torch.conj(values).transpose(-1, -2))/scale
    elif channel == "joint":
        matrix = covariance(psi)/scale
        values = cube_values(chi)
        values = values-torch.mean(values, dim=-2, keepdim=True)
        matrix = matrix+(values @ torch.conj(values).transpose(-1, -2))/scale
    else:
        raise ValueError(channel)
    flavor = torch.tensor(cubical.flavor(), dtype=torch.complex128)
    commutator = flavor@matrix-matrix@flavor
    return torch.sum(torch.real(torch.conj(commutator)*commutator))/2


def response(function, state: torch.Tensor, vector: np.ndarray) -> dict:
    tangent = torch.tensor(vector, dtype=torch.float64)
    _value, image = jvp(grad(function), (state,), (tangent,))
    image_np = image.detach().numpy()
    return {
        "curvature": float(np.dot(vector, image_np)
                           / max(np.dot(vector, vector), 1e-300)),
        "image_norm_per_vector_norm": float(
            np.linalg.norm(image_np)/max(np.linalg.norm(vector), 1e-300)),
    }


def main() -> None:
    bare, _matrices = native_action.expected_blocks()
    rest = bare["rest"]
    state = torch.tensor(rest, dtype=torch.float64)
    dt = native_time.DT
    theta = float(np.arccos(1-dt**2*native_time.CHI0**2/2))
    rotation = native_spectrum.rotation_matrix(rest.size, theta)
    phase_tangent = (rotation@rest-rotation.T@rest)/(2*dt)

    rows = []
    for channel in ("psi", "chi_contrast", "joint"):
        function = lambda value, selected=channel: commutator_energy(
            value, selected)
        value = function(state)
        force = grad(function)(state)
        rows.append({
            "channel": channel,
            "rest_energy": float(value.detach()),
            "rest_force_norm": float(torch.linalg.vector_norm(force).detach()),
            "rest_force_rms": float(
                torch.linalg.vector_norm(force).detach()/np.sqrt(rest.size)),
            "phase_tangent_response": response(function, state, phase_tangent),
        })

    result = {
        "status": "V4_CUBE_FLAVOR_COMMUTATOR_SCREEN",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "action_invariant": (
            "one half the squared commutator of the native cube-corner "
            "covariance with the cubical orientation/flavor involution"),
        "geometry": {
            "cube_corner_count": 8,
            "orientation_reversal_invariant": True,
            "selected_flavor_copy": False,
            "Dirac_object_used": False,
        },
        "rows": rows,
        "verdict": (
            "CUBE_FLAVOR_INVARIANT_REST_NULL"
            if any(row["rest_force_rms"] < 1e-12 for row in rows)
            else "CUBE_FLAVOR_INVARIANT_DEFORMS_REST"),
        "claim_boundary": (
            "Rest force and one tangent response only. A rest-null channel "
            "would still require a full cross-block, spectrum, movement, "
            "and nonlinear test before action admission."),
    }
    output = HERE/"cube_flavor_commutator_screen.json"
    output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n",
                      encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
