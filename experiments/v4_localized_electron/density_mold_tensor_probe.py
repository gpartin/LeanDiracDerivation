"""Screen the C19 tensor completion of the density-mold connection.

The scalar connection can only multiply chi by a local density rate.  The
next cubic-covariant response is the weighted C19 second-moment tensor of that
rate acting on chi.  It has no preferred axis: under a cubic rotation both the
bond tensor and chi rotate together.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

import density_mold_connection_probe as local
import edge_snap_action_rank as rank


HERE = Path(__file__).resolve().parent
SIDE = 5
MODES = ("local", "local_plus_full", "local_plus_traceless",
         "local_plus_trace", "tensor_only")
SECOND_MOMENT = 12.0


def density_curvature(value: torch.Tensor) -> torch.Tensor:
    shape = value.shape + (3, 3)
    result = torch.zeros(shape, dtype=torch.float64)
    for offset, weight in rank.native.DIRECTIONS:
        direction = torch.tensor(offset, dtype=torch.float64)
        dyad = direction[:, None] * direction[None, :]
        difference = torch.roll(
            value, shifts=offset, dims=(0, 1, 2)) - value
        result = result + weight * difference[..., None, None] * dyad
    return result / SECOND_MOMENT


def interval_extra_action(left_raw: torch.Tensor, right_raw: torch.Tensor,
                          dt: float, mode: str,
                          tensor_strength: float = 1.0) -> torch.Tensor:
    left_psi, left_chi = local.unpack_tensor(left_raw, SIDE)
    right_psi, right_chi = local.unpack_tensor(right_raw, SIDE)
    psi = (left_psi + right_psi) / 2
    chi = (left_chi + right_chi) / 2
    psi_dot = (right_psi - left_psi) / dt
    chi_dot = (right_chi - left_chi) / dt
    rho_dot = 2 * torch.real(torch.sum(
        torch.conj(psi) * psi_dot, dim=-1))
    curvature = density_curvature(rho_dot)
    trace = torch.diagonal(curvature, dim1=-2, dim2=-1).sum(-1)
    identity = torch.eye(3, dtype=torch.float64)
    traceless = curvature - trace[..., None, None] * identity / 3
    curvature_complex = curvature.to(torch.complex128)
    traceless_complex = traceless.to(torch.complex128)
    if mode == "local":
        response = rho_dot[..., None] * chi
    elif mode == "local_plus_full":
        response = (rho_dot[..., None] * chi
                    + tensor_strength * torch.einsum(
                        "...ab,...b->...a", curvature_complex, chi))
    elif mode == "local_plus_traceless":
        response = (rho_dot[..., None] * chi
                    + tensor_strength * torch.einsum(
                        "...ab,...b->...a", traceless_complex, chi))
    elif mode == "local_plus_trace":
        response = (rho_dot + tensor_strength * trace / 3)[..., None] * chi
    elif mode == "tensor_only":
        response = tensor_strength * torch.einsum(
            "...ab,...b->...a", curvature_complex, chi)
    else:
        raise ValueError(mode)
    rho_chi = torch.sum(local.squared_norm(chi), dim=-1)
    denominator = rho_chi + rank.KAPPA * rank.R**2
    velocity = (chi_dot + local.ALPHA * response
                / (2 * denominator[..., None]))
    averaged = local.adjacency_average(velocity)
    candidate = rank.KAPPA * torch.sum(local.squared_norm(chi_dot))
    candidate = candidate + torch.sum(local.squared_norm(averaged))
    canonical = torch.sum(local.squared_norm(chi_dot))
    return dt * rank.B * (candidate - canonical) / 2


def action_delta(previous, current, following, dt: float,
                 mode: str, tensor_strength: float = 1.0) -> np.ndarray:
    raws = [torch.tensor(rank.pack(*layer), dtype=torch.float64,
                          requires_grad=True)
            for layer in (previous, current, following)]
    action = interval_extra_action(
        raws[0], raws[1], dt, mode, tensor_strength)
    action = action + interval_extra_action(
        raws[1], raws[2], dt, mode, tensor_strength)
    return torch.autograd.grad(action, raws[1])[0].detach().numpy()


def classify(mode: str, tensor_strength: float = 1.0):
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
        residual = baseline + action_delta(
            *layers, dt, mode, tensor_strength)
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
        "tensor_strength": tensor_strength,
        "combined_relative_residual": float(np.sqrt(numerator / denominator)),
        "rows": rows,
    }


def main() -> None:
    rank.lock_inertia()
    rows = [classify(mode) for mode in MODES]
    incumbent = rows[0]["combined_relative_residual"]
    best = min(rows, key=lambda row: row["combined_relative_residual"])
    result = {
        "status": "V4_DENSITY_MOLD_TENSOR_RESPONSE_SCREEN",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "action_family": (
            "T_chi=(B/2)[kappa|chiDot|^2+|A19 Vchi/24|^2], "
            "Vchi=chiDot+alpha [rhoPsiDot I + H19(rhoPsiDot)]chi/"
            "[2(rhoChi+kappa chi0^2)]"),
        "second_moment_normalization": SECOND_MOMENT,
        "coefficient_provenance": (
            "sum_C19 w n_a n_b=12 delta_ab; alpha=1+54/chi0^2"),
        "phase_covariant": True,
        "cubic_covariant": True,
        "positive_kinetic_square": True,
        "modes": rows,
        "best_mode": best["mode"],
        "best_relative_residual": best["combined_relative_residual"],
        "relative_improvement_over_local": float(
            (incumbent - best["combined_relative_residual"]) / incumbent),
        "verdict": (
            "TENSOR_COMPLETION_IMPROVES_HANDOFF" if best["mode"] != "local"
            else "SCALAR_LOCAL_CONNECTION_REMAINS_BEST"),
        "claim_boundary": (
            "This is a six-direction inverse response test, not a rest "
            "spectrum, free orbit, action admission, or Dirac pass."),
    }
    (HERE / "density_mold_tensor_probe.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
