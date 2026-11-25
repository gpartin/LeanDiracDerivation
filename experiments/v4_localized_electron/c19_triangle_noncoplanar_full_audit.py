"""Full-lattice noncoplanar bifurcation audit of the C19 triangle action.

The historical carrier-only Hessian found 16 projected zero modes, but a zero
of B^T H B need not be a zero of H B.  This audit uses the complete side-three
orbit Hessian, identifies its true nullspace, and asks whether any null vector
changes the occupied carrier W=chi x Psi geometry after quotienting global
U(3) rotations.  Exterior chi orientation gauges automatically map to zero W
and therefore do not create a false carrier branch.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

import c19_triangle_gyro_candidate_equivalence as triangle
import c27_body_gyro_basis_probe as direct
import c27_body_gyro_rest_spectrum as spectrum
import relative_kinetic_probe as relative


HERE = Path(__file__).resolve().parent


def orth(matrix: np.ndarray, tolerance: float = 1e-10) -> np.ndarray:
    if matrix.size == 0:
        return np.zeros((matrix.shape[0], 0))
    u, singular, _ = np.linalg.svd(matrix, full_matrices=False)
    threshold = tolerance * max(singular[0], 1e-300)
    return u[:, singular > threshold]


def global_generators() -> list[np.ndarray]:
    generators: list[np.ndarray] = []
    for index in range(3):
        value = np.zeros((3, 3), dtype=complex)
        value[index, index] = 1j
        generators.append(value)
    for first in range(3):
        for second in range(first + 1, 3):
            real = np.zeros((3, 3), dtype=complex)
            real[first, second] = 1
            real[second, first] = -1
            generators.append(real)
            imaginary = np.zeros((3, 3), dtype=complex)
            imaginary[first, second] = 1j
            imaginary[second, first] = 1j
            generators.append(imaginary)
    return generators


def carrier_w_tangent(raw: np.ndarray, direction: np.ndarray,
                      occupied: np.ndarray) -> np.ndarray:
    psi, chi = direct.base.inverse.rank.native.unpack(raw, spectrum.SIDE)
    dpsi, dchi = direct.base.inverse.rank.native.unpack(
        direction, spectrum.SIDE)
    dw = (np.cross(dchi, psi) + np.cross(chi, dpsi)) \
        / direct.base.inverse.rank.R
    selected = dw[occupied]
    return np.r_[selected.real.ravel(), selected.imag.ravel()]


def global_tangents(raw: np.ndarray) -> np.ndarray:
    psi, chi = direct.base.inverse.rank.native.unpack(raw, spectrum.SIDE)
    columns = []
    for generator in global_generators():
        dpsi = np.einsum("ab,...b->...a", generator, psi)
        dchi = np.einsum("ab,...b->...a", generator, chi)
        columns.append(direct.base.inverse.rank.pack(dpsi, dchi))
    # The action also has independent global Psi and chi phases.
    columns.append(direct.base.inverse.rank.pack(1j * psi, np.zeros_like(chi)))
    columns.append(direct.base.inverse.rank.pack(np.zeros_like(psi), 1j * chi))
    return np.column_stack(columns)


def main() -> None:
    spectrum.configure()
    direct.base.inverse.rank.lock_inertia()
    _previous, current, _following, theta = direct.base.inverse.rest_layers()
    values = triangle.coefficients(theta)
    rotation = torch.tensor(relative.raw_rotation_matrix(
        spectrum.SIDE, theta, 0.0, 1), dtype=torch.float64)
    inverse_rotation = rotation.T

    def orbit_action(raw: torch.Tensor) -> torch.Tensor:
        return (triangle.triangle_interval(inverse_rotation @ raw, raw, values)
                + triangle.triangle_interval(raw, rotation @ raw, values))

    base = torch.tensor(current, dtype=torch.float64, requires_grad=True)
    residual = torch.autograd.grad(
        orbit_action(base), base, create_graph=False)[0].detach().numpy()
    hessian = torch.autograd.functional.hessian(
        orbit_action, base, vectorize=True).detach().numpy()
    hessian = (hessian + hessian.T) / 2
    eigenvalues, eigenvectors = np.linalg.eigh(hessian)
    spectral_scale = max(float(np.max(abs(eigenvalues))), 1e-300)
    threshold = 1e-8 * spectral_scale
    null_indices = np.where(abs(eigenvalues) < threshold)[0]
    null_basis = eigenvectors[:, null_indices]

    psi, _chi = direct.base.inverse.rank.native.unpack(
        current, spectrum.SIDE)
    occupied = np.sum(abs(psi)**2, axis=-1) > 1e-12
    w_from_null = np.column_stack([
        carrier_w_tangent(current, null_basis[:, index], occupied)
        for index in range(null_basis.shape[1])
    ]) if null_basis.shape[1] else np.zeros((48, 0))
    symmetries = global_tangents(current)
    symmetry_w = np.column_stack([
        carrier_w_tangent(current, symmetries[:, index], occupied)
        for index in range(symmetries.shape[1])
    ])
    symmetry_w_basis = orth(symmetry_w)
    quotient_w = w_from_null - symmetry_w_basis @ (
        symmetry_w_basis.T @ w_from_null)
    quotient_singular = np.linalg.svd(
        quotient_w, compute_uv=False) if quotient_w.size else np.array([])
    quotient_rank = int(np.sum(
        quotient_singular > 1e-8 * max(
            quotient_singular[0] if quotient_singular.size else 0, 1e-300)))

    # Re-evaluate the historical 32 carrier-normal projected modes and their
    # leakage into the complete Hessian.
    columns = []
    center = spectrum.SIDE // 2 - 1
    for corner in np.ndindex(2, 2, 2):
        site = tuple(center + value for value in corner)
        for row in (0, 1, 2, 3):
            value = np.zeros((spectrum.SIDE, spectrum.SIDE, spectrum.SIDE,
                              4, 3), dtype=float)
            value[site + (row, 2)] = 1
            columns.append(value.reshape(-1))
    carrier_normal = np.column_stack(columns)
    projected = carrier_normal.T @ hessian @ carrier_normal
    projected_values, projected_vectors = np.linalg.eigh(
        (projected + projected.T) / 2)
    projected_scale = max(float(np.max(abs(projected_values))), 1e-300)
    projected_zero = np.where(abs(projected_values)
                              < 1e-8 * projected_scale)[0]
    leakage = []
    for index in projected_zero:
        direction = carrier_normal @ projected_vectors[:, index]
        leakage.append(float(
            np.linalg.norm(hessian @ direction)
            / max(np.linalg.norm(hessian, 2) * np.linalg.norm(direction),
                  1e-300)))

    result = {
        "status": "V4_C19_TRIANGLE_NONCOPLANAR_FULL_AUDIT",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "action": "exact C19 triangle/gyro research candidate",
        "rest_orbit_gradient_rms": float(
            np.linalg.norm(residual) / np.sqrt(residual.size)),
        "full_dimension": int(hessian.shape[0]),
        "full_hessian_symmetry_defect": float(
            np.linalg.norm(hessian - hessian.T)
            / max(np.linalg.norm(hessian), 1e-300)),
        "full_null_threshold": threshold,
        "full_nullity": int(null_basis.shape[1]),
        "carrier_W_rank_from_full_nullspace": int(np.linalg.matrix_rank(
            w_from_null, tol=1e-8)) if w_from_null.size else 0,
        "global_U3_and_phase_W_rank": int(symmetry_w_basis.shape[1]),
        "carrier_W_quotient_singular_values": [
            float(value) for value in quotient_singular],
        "carrier_W_quotient_rank": quotient_rank,
        "historical_carrier_normal_projected_zero_count": int(
            projected_zero.size),
        "historical_projected_zero_full_leakage": leakage,
        "verdict": (
            "FULL_NONCOPLANAR_SHAPE_BIFURCATION_DIRECTION_FOUND"
            if quotient_rank else
            "PROJECTED_NONCOPLANAR_ZEROS_ARE_SYMMETRY_OR_GAUGE"
        ),
        "claim_boundary": (
            "Full-lattice Hessian and carrier-W quotient at one rest orbit. "
            "A surviving quotient direction is necessary but not sufficient "
            "for a nonlinear noncoplanar branch."
        ),
    }
    output = HERE / "c19_triangle_noncoplanar_full_audit.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
