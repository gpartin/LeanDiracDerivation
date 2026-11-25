"""Screen a positive transfer-activated C19 χ kinetic filter.

The residual-action projection identifies a negative χ-gradient kinetic
response as the strongest missing direction.  Adding that term with a
negative coefficient would make the kinetic energy indefinite.  The smallest
positive completion instead changes the operator inside the existing square:

  Fχ = Pχ Vχ + s(a)(I-Pχ)Vχ
  Kχ± = A19(Fχ)/24 ± h(a) D19(Fχ)/24
  s(a) = (1 + 19 a²)^(-1/2)
  h(a) = g a²/(1 + g a²).

Then Tχ=B[κ|χdot|²+|Kχ±|²]/2 remains positive.  Both modifications vanish
above quadratic order at the rest orbit.  Expanding the square gives the
two possible A19--D19 cross-responses without a negative-energy term.  The
sign cannot be inferred from the separately added gradient square, so both
factorizations are screened.
No preferred direction, corner, copy, or Dirac data occur in the action.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
import torch

import density_mold_connection_probe as local
import density_mold_tensor_probe as tensor
import edge_snap_action_rank as rank


HERE = Path(__file__).resolve().parent
SIDE = 5
SOFTENING_FACTOR = 19.0


def c19_dirichlet(value: torch.Tensor) -> torch.Tensor:
    return value * rank.C19_WEIGHT_SUM - sum(
        weight * torch.roll(value, shifts=offset, dims=(0, 1, 2))
        for offset, weight in rank.native.DIRECTIONS)


def interval_extra_action(left_raw: torch.Tensor, right_raw: torch.Tensor,
                          dt: float, filter_factor: float,
                          polarity: int) -> torch.Tensor:
    left_psi, left_chi = local.unpack_tensor(left_raw, SIDE)
    right_psi, right_chi = local.unpack_tensor(right_raw, SIDE)
    psi = (left_psi + right_psi) / 2
    chi = (left_chi + right_chi) / 2
    psi_dot = (right_psi - left_psi) / dt
    chi_dot = (right_chi - left_chi) / dt
    rho_dot = 2 * torch.real(torch.sum(
        torch.conj(psi) * psi_dot, dim=-1))
    rho_psi = torch.sum(local.squared_norm(psi), dim=-1)
    curvature = tensor.density_curvature(rho_dot)
    trace = torch.diagonal(curvature, dim1=-2, dim2=-1).sum(-1)
    identity = torch.eye(3, dtype=torch.float64)
    traceless = (curvature - trace[..., None, None] * identity / 3).to(
        torch.complex128)
    response = (rho_dot[..., None] * chi
                + torch.einsum("...ab,...b->...a", traceless, chi))
    rho_chi = torch.sum(local.squared_norm(chi), dim=-1)
    chi_denominator = rho_chi + rank.KAPPA * rank.R**2
    material = chi_dot + local.ALPHA * response / (
        2 * chi_denominator[..., None])
    overlap = torch.sum(torch.conj(chi) * material, dim=-1)
    longitudinal = chi * overlap[..., None] / chi_denominator[..., None]
    transverse = material - longitudinal
    transfer = rho_dot / (
        2 * rank.R * (rho_psi + rank.KAPPA * rank.R**2))
    transverse_scale = torch.rsqrt(
        1 + SOFTENING_FACTOR * transfer**2)
    framed = longitudinal + transverse_scale[..., None] * transverse
    average = local.adjacency_average(framed)
    dirichlet = c19_dirichlet(framed) / rank.C19_WEIGHT_SUM
    activation = (filter_factor * transfer**2
                  / (1 + filter_factor * transfer**2))
    filtered = average + polarity * activation[..., None] * dirichlet
    candidate = rank.KAPPA * torch.sum(local.squared_norm(chi_dot))
    candidate = candidate + torch.sum(local.squared_norm(filtered))
    canonical = torch.sum(local.squared_norm(chi_dot))
    return dt * rank.B * (candidate - canonical) / 2


def action_delta(previous, current, following, dt: float,
                 filter_factor: float, polarity: int) -> np.ndarray:
    raws = [torch.tensor(rank.pack(*layer), dtype=torch.float64,
                        requires_grad=True)
            for layer in (previous, current, following)]
    action = interval_extra_action(
        raws[0], raws[1], dt, filter_factor, polarity)
    action = action + interval_extra_action(
        raws[1], raws[2], dt, filter_factor, polarity)
    return torch.autograd.grad(action, raws[1])[0].detach().numpy()


def evaluate(filter_factor: float, polarity: int, rows, psi, chi,
             force_psi, force_chi):
    numerator = denominator = 0.0
    direction_rows = []
    for row in rows:
        dt = row["best_dt"]
        layers = rank.translated_layers(row, psi, chi)
        baseline = rank.canonical_residual(
            *layers, dt, force_psi, force_chi)
        residual = baseline + action_delta(
            *layers, dt, filter_factor, polarity)
        baseline_sq = float(np.vdot(baseline, baseline).real)
        residual_sq = float(np.vdot(residual, residual).real)
        numerator += residual_sq
        denominator += baseline_sq
        direction_rows.append({
            "axis": row["axis"],
            "sign": row["sign"],
            "relative_residual": float(np.sqrt(residual_sq / baseline_sq)),
        })
    return float(np.sqrt(numerator / denominator)), direction_rows


def main() -> None:
    rank.lock_inertia()
    evidence = json.loads((HERE / "edge_snap_verlet_gate.json").read_text(
        encoding="utf-8"))
    rows = evidence["rows"]
    amplitude_sq = rank.R**2 / (2 + 54 / rank.R**2)
    psi, chi = rank.native.carrier_fields(SIDE, amplitude_sq)
    force_psi, force_chi = rank.native.force(psi, chi)
    factors = (0.0, 1/6, 1.0, 3.0, 6.0, 19.0, 24.0, 54.0, 63.0)
    scans = []
    optima = []
    for polarity in (-1, 1):
        scan = []
        for factor in factors:
            residual, direction_rows = evaluate(
                factor, polarity, rows, psi, chi, force_psi, force_chi)
            scan.append({
                "polarity": polarity,
                "filter_factor": factor,
                "combined_relative_residual": residual,
                "rows": direction_rows,
            })
        scans.extend(scan)
        optimum = minimize_scalar(
            lambda factor: evaluate(
                factor, polarity, rows, psi, chi, force_psi, force_chi)[0],
            bounds=(0.0, 100.0), method="bounded",
            options={"xatol": 1e-8})
        optima.append({
            "polarity": polarity,
            "filter_factor": float(optimum.x),
            "combined_relative_residual": float(optimum.fun),
            "coefficient_admitted": False,
        })
    incumbent = next(row for row in scans if row["filter_factor"] == 0.0)
    best_geometry = min(
        (row for row in scans if row["filter_factor"] > 0.0),
        key=lambda row: row["combined_relative_residual"])
    result = {
        "status": "V4_CHI_TRANSFER_FILTER_RESPONSE_SCREEN",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "positive_and_invertible": True,
        "phase_covariant": True,
        "cubic_covariant": True,
        "rest_quadratic_action_identical_to_incumbent": True,
        "softening_factor": SOFTENING_FACTOR,
        "filter_factor_provenance": (
            "1/6 is the equal-share normalization of the six C19 face "
            "transport directions; the diagnostic optimum is reported only "
            "as an independent comparison"),
        "scan": scans,
        "best_geometry": best_geometry,
        "diagnostic_optima": optima,
        "relative_improvement_over_unfiltered": float(
            (incumbent["combined_relative_residual"]
             - best_geometry["combined_relative_residual"])
            / incumbent["combined_relative_residual"]),
        "verdict": (
            "ADVANCE_FILTER_FAMILY" if
            best_geometry["combined_relative_residual"] <
            incumbent["combined_relative_residual"] else
            "RETIRE_FILTER_FAMILY"),
        "claim_boundary": (
            "This is a frozen finite-handoff response screen. No coefficient "
            "or action is admitted without the rest and free-orbit gates."),
    }
    (HERE / "chi_transfer_filter_probe.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
