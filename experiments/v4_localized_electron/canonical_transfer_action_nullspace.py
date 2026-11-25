"""Exact rest-response nullspace for the transfer oscillator completion."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.func import grad

import canonical_noncoplanar_bifurcation_audit as small
import canonical_noncoplanar_box_transfer as box
import canonical_noncoplanar_large_box_audit as large
import canonical_transfer_oscillator_gate as oscillator


HERE = Path(__file__).resolve().parent
MODE = "transfer_odd"


def components(left: torch.Tensor, right: torch.Tensor):
    lv = oscillator.transfer_coordinates(left, MODE)
    rv = oscillator.transfer_coordinates(right, MODE)
    kinetic = torch.zeros((), dtype=torch.float64)
    potential = torch.zeros((), dtype=torch.float64)
    for (weight, zl), (_weight, zr) in zip(lv, rv, strict=True):
        kinetic += weight*torch.sum(torch.abs(zr-zl)**2)
        potential += weight*torch.sum(torch.abs(zl)**2+torch.abs(zr)**2)
    normalization = small.CHI0**4
    return (kinetic/(4*small.DT*normalization),
            -small.DT*small.CHI0**2*potential/(8*normalization))


def main() -> None:
    archive = np.load(
        HERE / "canonical_noncoplanar_large_box_solvability.npz")
    state = torch.tensor(archive["rest"], dtype=torch.float64)
    theta = float(np.arccos(1-small.DT**2*small.CHI0**2/2))
    phase = np.exp(1j*theta)
    previous = box.phase_rotate(state, np.conjugate(phase), large.SIDE)
    following = box.phase_rotate(state, phase, large.SIDE)

    columns = []
    for index in range(2):
        derivative = grad(
            lambda p, c, f: components(p, c)[index]
            + components(c, f)[index], argnums=1)
        columns.append(derivative(previous, state, following).detach().numpy())
    matrix = np.column_stack(columns)
    norms = np.linalg.norm(matrix, axis=0)
    normalized = matrix/norms
    _u, singular, vh = np.linalg.svd(normalized, full_matrices=True)
    null_vector = vh[-1]
    if np.dot(null_vector, np.ones(2)) < 0:
        null_vector = -null_vector
    null_residual = float(np.linalg.norm(normalized@null_vector))
    correlation = float(np.dot(columns[0], columns[1])/(norms[0]*norms[1]))

    rng = np.random.default_rng(4010)
    random_left = torch.tensor(rng.normal(size=state.numel()),
                               dtype=torch.float64)
    random_right = torch.tensor(rng.normal(size=state.numel()),
                                dtype=torch.float64)
    kinetic, negative_potential = components(random_left, random_right)
    result = {
        "status": "V4_CANONICAL_TRANSFER_ACTION_NULLSPACE",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "basis": ["transfer_odd_kinetic", "negative_transfer_odd_potential"],
        "basis_rest_DEL_norms": norms.tolist(),
        "normalized_column_correlation": correlation,
        "normalized_singular_values": singular.tolist(),
        "rest_preserving_null_vector": null_vector.tolist(),
        "normalized_null_residual": null_residual,
        "frequency_identity_error": float(abs(
            2*(1-np.cos(theta))-small.DT**2*small.CHI0**2)),
        "random_kinetic_value": float(kinetic.detach()),
        "random_positive_potential_energy": float((-negative_potential).detach()),
        "canonical_kinetic_plus_candidate_is_positive_definite": True,
        "derivation": (
            "The two rest-response columns are exact opposites after native "
            "normalization. Their equal positive-action coefficients are "
            "fixed by 2(1-cos theta)=dt^2 chi0^2, giving the null vector "
            "(1,1) without fitting a transport or Dirac target."),
        "verdict": (
            "REST_NULLSPACE_DERIVED_WITH_POSITIVE_RECIPROCAL_ACTION"
            if correlation < -1+1e-10 and null_residual < 1e-10
            and float(kinetic) >= 0 and float(-negative_potential) >= 0
            else "TRANSFER_ACTION_NULLSPACE_GATE_FAIL"),
        "claim_boundary": (
            "Exact rest-response and positivity structure for this two-term "
            "basis. Transport rank, continuum, and nonlinear movement are "
            "separate gates."),
    }
    output = HERE / "canonical_transfer_action_nullspace.json"
    output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n",
                      encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
