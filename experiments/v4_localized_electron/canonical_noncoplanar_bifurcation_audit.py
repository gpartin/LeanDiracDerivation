"""Full-lattice noncoplanar bifurcation audit for canonical V40.

This repeats the C19-triangle carrier-W quotient audit using only the locked
V40 action.  It is intentionally side-three so the complete 324-real-variable
orbit Hessian can be diagonalized without projecting away the exterior.
Global U(3) rotations and the independent global field phases are quotiented
before a carrier-shape direction is counted.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import torch

import localized_stability as native


HERE = Path(__file__).resolve().parent
SIDE = 3
DT = 1.0 / 256.0
CHI0 = 19.0
B = 1197.0
AMPLITUDE_SQ = CHI0**2 / (2 + 54 / CHI0**2)


def pack(psi: np.ndarray, chi: np.ndarray) -> np.ndarray:
    return np.stack((psi.real, psi.imag, chi.real, chi.imag), axis=-2).ravel()


def unpack(raw: np.ndarray | torch.Tensor):
    shaped = raw.reshape(SIDE, SIDE, SIDE, 4, 3)
    if isinstance(raw, torch.Tensor):
        return (torch.complex(shaped[..., 0, :], shaped[..., 1, :]),
                torch.complex(shaped[..., 2, :], shaped[..., 3, :]))
    return (shaped[..., 0, :] + 1j * shaped[..., 1, :],
            shaped[..., 2, :] + 1j * shaped[..., 3, :])


def squared_norm(value: torch.Tensor) -> torch.Tensor:
    return torch.real(torch.sum(torch.conj(value) * value, dim=-1))


def potential(raw: torch.Tensor) -> torch.Tensor:
    psi, chi = unpack(raw)
    rho_psi = squared_norm(psi)
    rho_chi = squared_norm(chi)
    local = torch.sum(0.25 * (rho_chi - CHI0**2)**2
                      + 0.5 * rho_chi * rho_psi
                      + 0.25 * rho_psi**2)
    area = torch.linalg.cross(chi, psi, dim=-1)
    bond = torch.zeros((), dtype=torch.float64)
    for offset, weight in native.DIRECTIONS:
        difference = area - torch.roll(
            area, shifts=offset, dims=(0, 1, 2))
        # DIRECTIONS contains both orientations.  This is the canonical
        # directed C19 sum as stated by the framework action.
        bond = bond + weight * torch.sum(squared_norm(difference))
    return local + bond / (2 * CHI0**2)


def interval_action(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    left_psi, left_chi = unpack(left)
    right_psi, right_chi = unpack(right)
    psi_dot = (right_psi - left_psi) / DT
    chi_dot = (right_chi - left_chi) / DT
    kinetic = 0.5 * torch.sum(squared_norm(psi_dot))
    kinetic = kinetic + B / 2 * torch.sum(squared_norm(chi_dot))
    return DT * (kinetic - (potential(left) + potential(right)) / 2)


def phase_rotate(raw: torch.Tensor, phase: complex) -> torch.Tensor:
    psi, chi = unpack(raw)
    phase_tensor = torch.tensor(phase, dtype=torch.complex128)
    shaped = torch.stack((torch.real(phase_tensor * psi),
                          torch.imag(phase_tensor * psi),
                          torch.real(chi), torch.imag(chi)), dim=-2)
    return shaped.reshape(-1)


def orth(matrix: np.ndarray, tolerance: float = 1e-10) -> np.ndarray:
    if matrix.size == 0:
        return np.zeros((matrix.shape[0], 0))
    u, singular, _ = np.linalg.svd(matrix, full_matrices=False)
    threshold = tolerance * max(singular[0], 1e-300)
    return u[:, singular > threshold]


def kernel(matrix: np.ndarray, tolerance: float = 1e-10) -> np.ndarray:
    """Right nullspace with a scale-relative SVD threshold."""
    _u, singular, vh = np.linalg.svd(matrix, full_matrices=True)
    threshold = tolerance * max(singular[0] if singular.size else 0, 1e-300)
    rank = int(np.sum(singular > threshold))
    return vh[rank:].T


def generators() -> list[np.ndarray]:
    values: list[np.ndarray] = []
    for index in range(3):
        value = np.zeros((3, 3), dtype=complex)
        value[index, index] = 1j
        values.append(value)
    for first in range(3):
        for second in range(first + 1, 3):
            real = np.zeros((3, 3), dtype=complex)
            real[first, second] = 1
            real[second, first] = -1
            values.append(real)
            imaginary = np.zeros((3, 3), dtype=complex)
            imaginary[first, second] = 1j
            imaginary[second, first] = 1j
            values.append(imaginary)
    return values


def global_tangents(raw: np.ndarray) -> np.ndarray:
    psi, chi = unpack(raw)
    columns = []
    for generator in generators():
        dpsi = np.einsum("ab,...b->...a", generator, psi)
        dchi = np.einsum("ab,...b->...a", generator, chi)
        columns.append(pack(dpsi, dchi))
    columns.append(pack(1j * psi, np.zeros_like(chi)))
    columns.append(pack(np.zeros_like(psi), 1j * chi))
    return np.column_stack(columns)


def carrier_w_tangent(raw: np.ndarray, direction: np.ndarray,
                      occupied: np.ndarray) -> np.ndarray:
    psi, chi = unpack(raw)
    dpsi, dchi = unpack(direction)
    dw = (np.cross(dchi, psi) + np.cross(chi, dpsi)) / CHI0
    selected = dw[occupied]
    return np.r_[selected.real.ravel(), selected.imag.ravel()]


def carrier_transverse_w_tangent(raw: np.ndarray, direction: np.ndarray,
                                 occupied: np.ndarray) -> np.ndarray:
    """W response orthogonal to the locked carrier's third-component axis."""
    psi, chi = unpack(raw)
    dpsi, dchi = unpack(direction)
    dw = (np.cross(dchi, psi) + np.cross(chi, dpsi)) / CHI0
    selected = dw[occupied][..., :2]
    return np.r_[selected.real.ravel(), selected.imag.ravel()]


def opposite_pair_constraint(raw: np.ndarray, direction: np.ndarray,
                             occupied: np.ndarray) -> np.ndarray:
    """Linearized preservation of the four locked opposite-corner maps."""
    psi, _chi = unpack(raw)
    dpsi, _dchi = unpack(direction)
    p = np.asarray([psi[tuple(site)] for site in np.argwhere(occupied)])
    dp = np.asarray([dpsi[tuple(site)] for site in np.argwhere(occupied)])
    values = []
    for index in range(4):
        factor = np.vdot(p[index], p[7-index]) \
            / max(np.vdot(p[index], p[index]).real, 1e-300)
        defect = dp[7-index] - factor * dp[index]
        values.extend(defect.real)
        values.extend(defect.imag)
    return np.asarray(values)


def main() -> None:
    psi, chi = native.carrier_fields(SIDE, AMPLITUDE_SQ)
    current = pack(psi, chi)
    theta = float(np.arccos(1 - DT**2 * CHI0**2 / 2))
    phase = np.exp(1j * theta)

    def orbit_action(raw: torch.Tensor) -> torch.Tensor:
        return (interval_action(phase_rotate(raw, np.conjugate(phase)), raw)
                + interval_action(raw, phase_rotate(raw, phase)))

    base = torch.tensor(current, dtype=torch.float64, requires_grad=True)
    residual = torch.autograd.grad(
        orbit_action(base), base, create_graph=False)[0].detach().numpy()
    hessian = torch.autograd.functional.hessian(
        orbit_action, base, vectorize=True).detach().numpy()
    hessian = (hessian + hessian.T) / 2
    eigenvalues, eigenvectors = np.linalg.eigh(hessian)
    scale = max(float(np.max(np.abs(eigenvalues))), 1e-300)
    threshold = 1e-8 * scale
    null_indices = np.where(np.abs(eigenvalues) < threshold)[0]
    null_basis = eigenvectors[:, null_indices]

    occupied = np.sum(np.abs(psi)**2, axis=-1) > 1e-12
    w_null = np.column_stack([
        carrier_w_tangent(current, null_basis[:, index], occupied)
        for index in range(null_basis.shape[1])
    ]) if null_basis.shape[1] else np.zeros((48, 0))
    symmetries = global_tangents(current)
    w_symmetry = np.column_stack([
        carrier_w_tangent(current, symmetries[:, index], occupied)
        for index in range(symmetries.shape[1])
    ])
    w_symmetry_basis = orth(w_symmetry)
    quotient = w_null - w_symmetry_basis @ (w_symmetry_basis.T @ w_null)
    singular = np.linalg.svd(quotient, compute_uv=False) \
        if quotient.size else np.array([])
    quotient_threshold = 1e-8 * max(
        singular[0] if singular.size else 0, 1e-300)
    quotient_rank = int(np.sum(singular > quotient_threshold))

    transverse_null = np.column_stack([
        carrier_transverse_w_tangent(
            current, null_basis[:, index], occupied)
        for index in range(null_basis.shape[1])
    ]) if null_basis.shape[1] else np.zeros((32, 0))
    transverse_symmetry = np.column_stack([
        carrier_transverse_w_tangent(
            current, symmetries[:, index], occupied)
        for index in range(symmetries.shape[1])
    ])
    transverse_symmetry_basis = orth(transverse_symmetry)
    transverse_quotient = transverse_null - transverse_symmetry_basis @ (
        transverse_symmetry_basis.T @ transverse_null)
    transverse_singular = np.linalg.svd(
        transverse_quotient, compute_uv=False) \
        if transverse_quotient.size else np.array([])
    transverse_threshold = 1e-8 * max(
        transverse_singular[0] if transverse_singular.size else 0, 1e-300)
    transverse_rank = int(np.sum(transverse_singular > transverse_threshold))

    pair_constraint = np.column_stack([
        opposite_pair_constraint(
            current, null_basis[:, index], occupied)
        for index in range(null_basis.shape[1])
    ])
    pair_kernel = kernel(pair_constraint)
    pair_preserving_null = null_basis @ pair_kernel
    pair_transverse = np.column_stack([
        carrier_transverse_w_tangent(
            current, pair_preserving_null[:, index], occupied)
        for index in range(pair_preserving_null.shape[1])
    ]) if pair_preserving_null.shape[1] else np.zeros((32, 0))
    pair_transverse_quotient = pair_transverse - transverse_symmetry_basis @ (
        transverse_symmetry_basis.T @ pair_transverse)
    pair_transverse_singular = np.linalg.svd(
        pair_transverse_quotient, compute_uv=False) \
        if pair_transverse_quotient.size else np.array([])
    pair_transverse_threshold = 1e-8 * max(
        pair_transverse_singular[0] if pair_transverse_singular.size else 0,
        1e-300)
    pair_transverse_rank = int(np.sum(
        pair_transverse_singular > pair_transverse_threshold))

    # Project the historical 32 corner-normal directions and measure whether
    # their projected zeros are true zeros of the complete Hessian.
    columns = []
    center = SIDE // 2 - 1
    for corner in itertools.product((0, 1), repeat=3):
        site = tuple(center + value for value in corner)
        for row in (0, 1, 2, 3):
            value = np.zeros((SIDE, SIDE, SIDE, 4, 3), dtype=float)
            value[site + (row, 2)] = 1
            columns.append(value.ravel())
    corner_normal = np.column_stack(columns)
    projected = corner_normal.T @ hessian @ corner_normal
    projected_values, projected_vectors = np.linalg.eigh(
        (projected + projected.T) / 2)
    projected_scale = max(float(np.max(np.abs(projected_values))), 1e-300)
    projected_zero = np.where(
        np.abs(projected_values) < 1e-8 * projected_scale)[0]
    leakage = []
    for index in projected_zero:
        direction = corner_normal @ projected_vectors[:, index]
        leakage.append(float(np.linalg.norm(hessian @ direction)
                             / max(scale * np.linalg.norm(direction), 1e-300)))

    result = {
        "status": "V4_CANONICAL_NONCOPLANAR_BIFURCATION_AUDIT",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "action": "locked canonical V40 U_local plus U_area",
        "side": SIDE,
        "dt": DT,
        "rest_orbit_gradient_rms": float(
            np.linalg.norm(residual) / np.sqrt(residual.size)),
        "full_dimension": int(hessian.shape[0]),
        "hessian_scale": scale,
        "null_threshold": threshold,
        "full_nullity": int(null_basis.shape[1]),
        "carrier_W_rank_from_nullspace": int(
            np.linalg.matrix_rank(w_null, tol=1e-8)) if w_null.size else 0,
        "global_U3_and_phase_W_rank": int(w_symmetry_basis.shape[1]),
        "carrier_W_quotient_singular_values": [float(x) for x in singular],
        "carrier_W_quotient_rank": quotient_rank,
        "transverse_W_quotient_singular_values": [
            float(x) for x in transverse_singular],
        "transverse_W_quotient_rank": transverse_rank,
        "opposite_pair_constraint_rank_on_nullspace": int(
            np.linalg.matrix_rank(pair_constraint, tol=1e-8)),
        "opposite_pair_preserving_nullity": int(
            pair_preserving_null.shape[1]),
        "pair_preserving_transverse_W_quotient_singular_values": [
            float(x) for x in pair_transverse_singular],
        "pair_preserving_transverse_W_quotient_rank": pair_transverse_rank,
        "projected_corner_normal_zero_count": int(projected_zero.size),
        "projected_zero_full_leakage": leakage,
        "smallest_absolute_eigenvalues": [
            float(x) for x in eigenvalues[np.argsort(np.abs(eigenvalues))[:24]]],
        "verdict": (
            "CANONICAL_NONCOPLANAR_SHAPE_DIRECTION_FOUND"
            if transverse_rank else
            "CANONICAL_NONCOPLANAR_SHAPE_DIRECTION_ABSENT"),
        "claim_boundary": (
            "Complete side-three orbit Hessian at the locked rest carrier. "
            "A quotient direction is necessary, not sufficient, for a "
            "nearby nonlinear branch. Absence excludes only a continuous "
            "bifurcation at this orbit, not disconnected finite branches."),
    }
    output = HERE / "canonical_noncoplanar_bifurcation_audit.json"
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n",
                      encoding="utf-8")
    print(json.dumps({
        "verdict": result["verdict"],
        "rest_orbit_gradient_rms": result["rest_orbit_gradient_rms"],
        "full_nullity": result["full_nullity"],
        "carrier_W_quotient_rank": result["carrier_W_quotient_rank"],
        "transverse_W_quotient_rank": result[
            "transverse_W_quotient_rank"],
        "pair_preserving_transverse_W_quotient_rank": result[
            "pair_preserving_transverse_W_quotient_rank"],
        "projected_corner_normal_zero_count": result[
            "projected_corner_normal_zero_count"],
        "output": str(output),
    }, indent=2))


if __name__ == "__main__":
    main()
