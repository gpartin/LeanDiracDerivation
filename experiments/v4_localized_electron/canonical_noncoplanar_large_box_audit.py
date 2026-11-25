"""Full canonical noncoplanar nullspace audit on a side-five vacuum box."""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import torch

import canonical_noncoplanar_bifurcation_audit as small
import canonical_noncoplanar_box_transfer as box
import localized_stability as native


HERE = Path(__file__).resolve().parent
SIDE = 5


def unpack_np(raw: np.ndarray):
    shaped = raw.reshape(SIDE, SIDE, SIDE, 4, 3)
    return (shaped[..., 0, :]+1j*shaped[..., 1, :],
            shaped[..., 2, :]+1j*shaped[..., 3, :])


def transverse_w(raw: np.ndarray, direction: np.ndarray,
                 sites: list[tuple[int, int, int]]) -> np.ndarray:
    psi, chi = unpack_np(raw)
    dpsi, dchi = unpack_np(direction)
    dw = (np.cross(dchi, psi)+np.cross(chi, dpsi))/small.CHI0
    values = np.asarray([dw[site, :2] for site in sites])
    return np.r_[values.real.ravel(), values.imag.ravel()]


def pair_constraint(raw: np.ndarray, direction: np.ndarray,
                    sites: list[tuple[int, int, int]]) -> np.ndarray:
    psi, _chi = unpack_np(raw)
    dpsi, _dchi = unpack_np(direction)
    p = np.asarray([psi[site] for site in sites])
    dp = np.asarray([dpsi[site] for site in sites])
    values = []
    for index in range(4):
        factor = np.vdot(p[index], p[7-index]) \
            / max(np.vdot(p[index], p[index]).real, 1e-300)
        defect = dp[7-index]-factor*dp[index]
        values.extend(defect.real)
        values.extend(defect.imag)
    return np.asarray(values)


def global_tangents(raw: np.ndarray) -> np.ndarray:
    psi, chi = unpack_np(raw)
    columns = []
    for generator in small.generators():
        dpsi = np.einsum("ab,...b->...a", generator, psi)
        dchi = np.einsum("ab,...b->...a", generator, chi)
        columns.append(box.pack(dpsi, dchi))
    columns.append(box.pack(1j*psi, np.zeros_like(chi)))
    columns.append(box.pack(np.zeros_like(psi), 1j*chi))
    return np.column_stack(columns)


def main() -> None:
    psi, chi = native.carrier_fields(SIDE, small.AMPLITUDE_SQ)
    current = box.pack(psi, chi)
    theta = float(np.arccos(1-small.DT**2*small.CHI0**2/2))
    phase = np.exp(1j*theta)

    def orbit_action(raw: torch.Tensor) -> torch.Tensor:
        return (box.interval(box.phase_rotate(
            raw, np.conjugate(phase), SIDE), raw, SIDE)
            + box.interval(raw, box.phase_rotate(raw, phase, SIDE), SIDE))

    tensor = torch.tensor(current, dtype=torch.float64, requires_grad=True)
    gradient = torch.autograd.grad(
        orbit_action(tensor), tensor)[0].detach().numpy()
    hessian = torch.autograd.functional.hessian(
        orbit_action, tensor, vectorize=True).detach().numpy()
    hessian = (hessian+hessian.T)/2
    eigenvalues, eigenvectors = np.linalg.eigh(hessian)
    scale = max(float(np.max(abs(eigenvalues))), 1e-300)
    null_mask = abs(eigenvalues) < 1e-8*scale
    null_basis = eigenvectors[:, null_mask]
    origin = SIDE//2-1
    sites = [tuple(origin+bit for bit in corner)
             for corner in itertools.product((0, 1), repeat=3)]

    pair = np.column_stack([
        pair_constraint(current, null_basis[:, index], sites)
        for index in range(null_basis.shape[1])])
    pair_kernel = small.kernel(pair)
    pair_null = null_basis@pair_kernel
    response = np.column_stack([
        transverse_w(current, pair_null[:, index], sites)
        for index in range(pair_null.shape[1])
    ]) if pair_null.shape[1] else np.zeros((32, 0))
    symmetry = global_tangents(current)
    symmetry_response = np.column_stack([
        transverse_w(current, symmetry[:, index], sites)
        for index in range(symmetry.shape[1])])
    symmetry_basis = small.orth(symmetry_response)
    quotient = response-symmetry_basis@(symmetry_basis.T@response)
    if quotient.size:
        _left, singular, vh = np.linalg.svd(quotient, full_matrices=False)
    else:
        singular = np.array([])
        vh = np.zeros((0, pair_null.shape[1]))
    quotient_rank = int(np.sum(singular > 1e-8*max(
        singular[0] if singular.size else 0, 1e-300)))

    result = {
        "status": "V4_CANONICAL_NONCOPLANAR_LARGE_BOX_AUDIT",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "side": SIDE,
        "full_dimension": int(current.size),
        "rest_orbit_gradient_rms": float(
            np.linalg.norm(gradient)/np.sqrt(gradient.size)),
        "full_nullity": int(null_basis.shape[1]),
        "opposite_pair_constraint_rank_on_nullspace": int(
            np.linalg.matrix_rank(pair, tol=1e-8)),
        "opposite_pair_preserving_nullity": int(pair_null.shape[1]),
        "pair_preserving_transverse_W_quotient_singular_values": [
            float(value) for value in singular],
        "pair_preserving_transverse_W_quotient_rank": quotient_rank,
        "verdict": (
            "LARGE_BOX_LOCALIZED_NONCOPLANAR_DIRECTION_FOUND"
            if quotient_rank else
            "LARGE_BOX_LOCALIZED_NONCOPLANAR_DIRECTION_ABSENT"),
        "claim_boundary": (
            "Complete side-five periodic Hessian with ordinary canonical "
            "vacuum. Absence is a local bifurcation result, not a no-go for "
            "disconnected finite branches or action changes."),
    }
    output = HERE / "canonical_noncoplanar_large_box_audit.json"
    output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n",
                      encoding="utf-8")
    direction = pair_null@vh[0] if quotient_rank else np.zeros_like(current)
    if np.linalg.norm(direction):
        direction /= np.linalg.norm(direction)
    np.savez_compressed(
        HERE / "canonical_noncoplanar_large_box_audit.npz",
        rest=current, direction=direction,
        leading_singular_value=singular[0] if singular.size else 0.0)
    print(json.dumps({
        "verdict": result["verdict"],
        "side": SIDE,
        "full_nullity": result["full_nullity"],
        "pair_preserving_transverse_W_quotient_rank": quotient_rank,
        "leading_singular_value": float(singular[0]) if singular.size else 0,
        "output": str(output),
    }, indent=2))


if __name__ == "__main__":
    main()
