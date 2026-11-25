"""Screen a native bond-current material connection for complex chi.

The density-mold connection knows how much Psi density is changing but not
where its phase-bearing charge is going. This probe adds the lowest-order C19
material derivative built from the native U(1)-invariant Psi bond current.
It contains no carrier labels, selected direction, corner map, or Dirac data.

For a smooth plane wave the C19 normalizations used below give

    j_Psi = rho_Psi grad(theta) + O(dx^2),
    A_v chi = (v . grad) chi + O(dx^2).

The material coefficient is therefore one. The variants distinguish standard
scalar/vector transport conventions; they are not fitted to a target speed.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

import density_mold_connection_probe as local
import density_mold_tensor_probe as tensor
import edge_snap_action_rank as rank


HERE = Path(__file__).resolve().parent
SIDE = 5
SECOND_MOMENT = 12.0
MODES = (
    "density_tensor",
    "density_tensor_plus_advection",
    "density_tensor_plus_vector_lie",
    "density_tensor_plus_covector_lie",
    "advection_only",
)


def neighbor(field: torch.Tensor, offset: tuple[int, int, int]) -> torch.Tensor:
    """Return the field at i+n using the native shift convention."""
    return torch.roll(field, shifts=tuple(-x for x in offset), dims=(0, 1, 2))


def psi_bond_current(psi: torch.Tensor) -> torch.Tensor:
    """C19 first moment of phase flow, normalized by sum w n n^T."""
    result = torch.zeros(psi.shape[:-1] + (3,), dtype=torch.float64)
    for offset, weight in rank.native.DIRECTIONS:
        direction = torch.tensor(offset, dtype=torch.float64)
        phase_flux = torch.imag(torch.sum(
            torch.conj(psi) * neighbor(psi, offset), dim=-1))
        result = result + weight * phase_flux[..., None] * direction
    return result / SECOND_MOMENT


def velocity_gradient(velocity: torch.Tensor) -> torch.Tensor:
    """C19 gradient with derivative index first and vector index second."""
    result = torch.zeros(velocity.shape[:-1] + (3, 3), dtype=torch.float64)
    for offset, weight in rank.native.DIRECTIONS:
        direction = torch.tensor(offset, dtype=torch.float64)
        difference = neighbor(velocity, offset) - velocity
        result = result + weight * torch.einsum(
            "a,...b->...ab", direction, difference)
    return result / SECOND_MOMENT


def advect_chi(chi: torch.Tensor, velocity: torch.Tensor) -> torch.Tensor:
    """Centered C19 approximation to (v . grad) chi."""
    result = torch.zeros_like(chi)
    for offset, weight in rank.native.DIRECTIONS:
        direction = torch.tensor(offset, dtype=torch.float64)
        velocity_mid = (velocity + neighbor(velocity, offset)) / 2
        along = torch.einsum("...a,a->...", velocity_mid, direction)
        result = result + weight * along[..., None] * (
            neighbor(chi, offset) - chi)
    return result / SECOND_MOMENT


def material_response(psi: torch.Tensor, chi: torch.Tensor,
                      mode: str) -> torch.Tensor:
    rho_psi = torch.sum(local.squared_norm(psi), dim=-1)
    denominator = rho_psi + rank.KAPPA * rank.R**2
    velocity = psi_bond_current(psi) / denominator[..., None]
    advection = advect_chi(chi, velocity)
    gradient = velocity_gradient(velocity)
    symmetric = (gradient + gradient.transpose(-1, -2)) / 2
    trace = torch.diagonal(symmetric, dim1=-2, dim2=-1).sum(-1)
    identity = torch.eye(3, dtype=torch.float64)
    strain = symmetric - trace[..., None, None] * identity / 3
    strain_chi = torch.einsum(
        "...ab,...b->...a", strain.to(torch.complex128), chi)
    if mode == "density_tensor":
        return torch.zeros_like(chi)
    if mode in ("density_tensor_plus_advection", "advection_only"):
        return advection
    if mode == "density_tensor_plus_vector_lie":
        return advection - strain_chi
    if mode == "density_tensor_plus_covector_lie":
        return advection + strain_chi
    raise ValueError(mode)


def interval_extra_action(left_raw: torch.Tensor, right_raw: torch.Tensor,
                          dt: float, mode: str) -> torch.Tensor:
    left_psi, left_chi = local.unpack_tensor(left_raw, SIDE)
    right_psi, right_chi = local.unpack_tensor(right_raw, SIDE)
    psi = (left_psi + right_psi) / 2
    chi = (left_chi + right_chi) / 2
    psi_dot = (right_psi - left_psi) / dt
    chi_dot = (right_chi - left_chi) / dt

    if mode == "advection_only":
        base_response = torch.zeros_like(chi)
    else:
        rho_dot = 2 * torch.real(torch.sum(
            torch.conj(psi) * psi_dot, dim=-1))
        curvature = tensor.density_curvature(rho_dot)
        trace = torch.diagonal(curvature, dim1=-2, dim2=-1).sum(-1)
        identity = torch.eye(3, dtype=torch.float64)
        traceless = (curvature
                     - trace[..., None, None] * identity / 3).to(
                         torch.complex128)
        base_response = (rho_dot[..., None] * chi
                         + torch.einsum("...ab,...b->...a", traceless, chi))

    response = base_response + material_response(psi, chi, mode)
    rho_chi = torch.sum(local.squared_norm(chi), dim=-1)
    denominator = rho_chi + rank.KAPPA * rank.R**2
    covariant_velocity = (
        chi_dot + local.ALPHA * base_response / (2 * denominator[..., None])
        + material_response(psi, chi, mode))
    averaged = local.adjacency_average(covariant_velocity)
    candidate = rank.KAPPA * torch.sum(local.squared_norm(chi_dot))
    candidate = candidate + torch.sum(local.squared_norm(averaged))
    canonical = torch.sum(local.squared_norm(chi_dot))
    return dt * rank.B * (candidate - canonical) / 2


def action_delta(previous, current, following, dt: float,
                 mode: str) -> np.ndarray:
    raws = [torch.tensor(rank.pack(*layer), dtype=torch.float64,
                         requires_grad=True)
            for layer in (previous, current, following)]
    action = interval_extra_action(raws[0], raws[1], dt, mode)
    action = action + interval_extra_action(raws[1], raws[2], dt, mode)
    return torch.autograd.grad(action, raws[1])[0].detach().numpy()


def classify(mode: str) -> dict:
    evidence = json.loads(
        (HERE / "edge_snap_verlet_gate.json").read_text(encoding="utf-8"))
    amplitude_sq = rank.R**2 / (2 + 54 / rank.R**2)
    psi, chi = rank.native.carrier_fields(SIDE, amplitude_sq)
    force_psi, force_chi = rank.native.force(psi, chi)
    numerator = denominator = 0.0
    rows = []
    for row in evidence["rows"]:
        dt = row["best_dt"]
        layers = rank.translated_layers(row, psi, chi)
        baseline = rank.canonical_residual(
            *layers, dt, force_psi, force_chi)
        residual = baseline + action_delta(*layers, dt, mode)
        base_sq = float(np.vdot(baseline, baseline).real)
        residual_sq = float(np.vdot(residual, residual).real)
        numerator += residual_sq
        denominator += base_sq
        rows.append({
            "axis": row["axis"],
            "sign": row["sign"],
            "relative_residual": float(np.sqrt(residual_sq / base_sq)),
        })
    return {
        "mode": mode,
        "combined_relative_residual": float(np.sqrt(numerator / denominator)),
        "rows": rows,
    }


def main() -> None:
    rank.lock_inertia()
    rows = [classify(mode) for mode in MODES]
    incumbent = next(row for row in rows if row["mode"] == "density_tensor")
    best = min(rows, key=lambda row: row["combined_relative_residual"])
    result = {
        "status": "V4_BOND_CURRENT_MATERIAL_CONNECTION_SCREEN",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "target_pattern": (
            "same-assignment eight-site carrier translated by one C19 face "
            "edge in both signs of all three axes"),
        "candidate_action": (
            "Tchi=(B/2)[kappa|chiDot|^2+|A19 D_t chi/24|^2], "
            "where D_t contains the density-mold connection and the C19 "
            "material derivative built from jPsi"),
        "bond_current": (
            "jPsi=sum_C19 w n Im(Psi_i^dagger Psi_(i+n))/12"),
        "coefficient_provenance": (
            "sum_C19 w n_a n_b=12 delta_ab; unit material-derivative "
            "coefficient from the continuum normalization"),
        "real_action": True,
        "positive_kinetic_square": True,
        "phase_invariant": True,
        "cubic_covariant": True,
        "locality": "one C19 hop inside the connection; two after variation",
        "modes": rows,
        "incumbent_relative_residual": incumbent["combined_relative_residual"],
        "best_mode": best["mode"],
        "best_relative_residual": best["combined_relative_residual"],
        "relative_improvement": float(
            (incumbent["combined_relative_residual"]
             - best["combined_relative_residual"])
            / incumbent["combined_relative_residual"]),
        "verdict": (
            "MATERIAL_CURRENT_CHANNEL_IMPROVES_EXACT_HANDOFF"
            if best["combined_relative_residual"]
            < incumbent["combined_relative_residual"] else
            "MATERIAL_CURRENT_CHANNEL_RETIRED_AT_CHEAP_GATE"),
        "claim_boundary": (
            "This is a six-direction discrete-action response screen, not "
            "a solved orbit, rest-spectrum pass, action admission, or Dirac "
            "movement proof."),
    }
    (HERE / "bond_current_material_connection_probe.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
