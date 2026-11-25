"""Curvature screen for the smallest complete-frame position actions.

The candidates contain no prepared ray, corner label, projector onto a target
sector, Dirac matrix, or selected direction.  This script is an inverse
diagnostic only: a favorable curvature does not admit a coefficient.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.func import grad, jvp

import canonical_noncoplanar_bifurcation_audit as small
import canonical_noncoplanar_large_box_audit as large
import canonical_noncoplanar_box_transfer as box
import canonical_transfer_oscillator_gate as oscillator


HERE = Path(__file__).resolve().parent
SIDE = large.SIDE


def frobenius_sq(value: torch.Tensor) -> torch.Tensor:
    return torch.real(torch.sum(torch.conj(value) * value, dim=(-2, -1)))


def dyad(field: torch.Tensor) -> torch.Tensor:
    return field[..., :, None] * torch.conj(field[..., None, :])


def cross_dyad(psi: torch.Tensor, chi: torch.Tensor) -> torch.Tensor:
    return psi[..., :, None] * torch.conj(chi[..., None, :])


def local_parallel(raw: torch.Tensor) -> torch.Tensor:
    psi, chi = box.unpack(raw, SIDE)
    overlap = torch.sum(torch.conj(chi) * psi, dim=-1)
    return torch.sum(torch.real(torch.conj(overlap) * overlap))


def local_area(raw: torch.Tensor) -> torch.Tensor:
    psi, chi = box.unpack(raw, SIDE)
    area = torch.linalg.cross(chi, psi, dim=-1)
    return torch.sum(torch.real(torch.sum(torch.conj(area) * area, dim=-1)))


def bond_difference(field: torch.Tensor) -> torch.Tensor:
    value = torch.zeros((), dtype=torch.float64)
    for offset, weight in oscillator.native.DIRECTIONS:
        delta = field - torch.roll(field, shifts=offset, dims=(0, 1, 2))
        value += weight * torch.sum(frobenius_sq(delta))
    # The direction list contains both orientations.
    return value / (4 * small.CHI0**2)


def ordered_dyad_bonds(raw: torch.Tensor) -> torch.Tensor:
    psi, chi = box.unpack(raw, SIDE)
    return bond_difference(dyad(psi)) + bond_difference(dyad(chi))


def cross_dyad_bonds(raw: torch.Tensor) -> torch.Tensor:
    psi, chi = box.unpack(raw, SIDE)
    return bond_difference(cross_dyad(psi, chi))


def complete_frame_bonds(raw: torch.Tensor) -> torch.Tensor:
    return ordered_dyad_bonds(raw) + 2 * cross_dyad_bonds(raw)


CANDIDATES = {
    "local_parallel_overlap": local_parallel,
    "local_Gram_area": local_area,
    "C19_ordered_Psi_chi_dyads": ordered_dyad_bonds,
    "C19_cross_dyad": cross_dyad_bonds,
    "C19_complete_ordered_frame": complete_frame_bonds,
}


def mode_curvature(function, state: torch.Tensor,
                   mode: np.ndarray) -> dict[str, float]:
    gradient = grad(function)
    vr = torch.tensor(np.real(mode), dtype=torch.float64)
    vi = torch.tensor(np.imag(mode), dtype=torch.float64)
    _gr, hvr = jvp(gradient, (state,), (vr,))
    _gi, hvi = jvp(gradient, (state,), (vi,))
    norm_sq = float(np.vdot(mode, mode).real)
    qr = float(torch.dot(vr, hvr).detach())
    qi = float(torch.dot(vi, hvi).detach())
    return {
        "quadrature_averaged_Hessian_curvature": (qr + qi) / norm_sq,
        "real_quadrature_curvature": qr / max(float(torch.dot(vr, vr)), 1e-300),
        "imag_quadrature_curvature": qi / max(float(torch.dot(vi, vi)), 1e-300),
    }


def translation_response(function, state: torch.Tensor) -> dict:
    shaped = state.detach().numpy().reshape(SIDE, SIDE, SIDE, 4, 3)
    directions = [((np.roll(shaped, 1, axis=axis)
                    - np.roll(shaped, -1, axis=axis)) / 2).reshape(-1)
                  for axis in range(3)]
    gradient = grad(function)
    columns = []
    for direction in directions:
        vector = torch.tensor(direction, dtype=torch.float64)
        _g, image = jvp(gradient, (state,), (vector,))
        columns.append(image.detach().numpy())
    matrix = np.column_stack(columns)
    singular = np.linalg.svd(matrix, compute_uv=False)
    tolerance = max(singular[0], 1.0) * 1e-10
    return {
        "Hessian_translation_rank": int(np.sum(singular > tolerance)),
        "Hessian_translation_singular_values": singular.tolist(),
    }


def main() -> None:
    state = torch.tensor(np.load(
        HERE / "canonical_noncoplanar_large_box_solvability.npz")["rest"],
        dtype=torch.float64)
    canonical = np.load(
        HERE / "canonical_transfer_oscillator_spectrum_0_leading_mode.npz")
    repaired = np.load(HERE /
        "canonical_transfer_oscillator_spectrum_0_even_2_density_0p5_leading_mode.npz")
    loop_repaired = np.load(HERE /
        "canonical_transfer_oscillator_spectrum_0_even_2_density_0p5_loop_curvature_0p3_leading_mode.npz")
    dimension = state.numel()
    modes = {
        "canonical": canonical["phase_space_vector"][:dimension],
        "best_exact_rest_diagnostic": repaired["phase_space_vector"][:dimension],
        "post_loop_exterior_diagnostic": loop_repaired["phase_space_vector"][:dimension],
    }
    rows = []
    for name, function in CANDIDATES.items():
        value = function(state)
        gradient = grad(function)(state)
        row = {
            "candidate": name,
            "positive_square_or_Gram_invariant": True,
            "energy_at_exact_rest": float(value.detach()),
            "rest_force_norm": float(torch.linalg.vector_norm(gradient).detach()),
            "rest_force_rms": float(torch.linalg.vector_norm(gradient).detach()
                                    / np.sqrt(state.numel())),
            "soft_mode_curvatures": {
                label: mode_curvature(function, state, mode)
                for label, mode in modes.items()
            },
        }
        row.update(translation_response(function, state))
        rows.append(row)
    result = {
        "status": "V4_PERSONAL_TOUCH_POSITION_ACTION_BASIS_SCREEN",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "normalization": (
            "C19 dyad terms use the native bidirectional bond normalization "
            "1/(4 chi0^2); coefficients remain unselected."),
        "rows": rows,
        "verdict": "POSITION_ACTION_CURVATURES_MEASURED",
        "claim_boundary": (
            "Rest force, soft-mode Hessian curvature, and translation-response "
            "rank only. Nonzero curvature is not coefficient provenance, a "
            "re-solved orbit, spectral stability, or movement."),
    }
    output = HERE / "personal_touch_action_basis_screen.json"
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n",
                      encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
