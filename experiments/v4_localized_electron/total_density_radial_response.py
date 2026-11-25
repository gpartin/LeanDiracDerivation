"""Project the free handoff residual onto total-density radial inertia.

The canonical local potential has the exact identity

  Ulocal = (rhoChi + rhoPsi - chi0^2)^2/4 + chi0^2 rhoPsi/2.

Its native radial coordinate is therefore rhoChi+rhoPsi.  This diagnostic
tests the positive local kinetic invariant

  Delta T = c/8 sum [dot(rhoChi+rhoPsi)^2 /
                     (rhoChi+rhoPsi+kappa chi0^2)].

The target residual is used only to classify response and find the diagnostic
least-squares coefficient.  No fitted coefficient is admitted.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

import density_mold_connection_probe as local
import edge_snap_action_rank as rank
import edge_snap_residual_action_projection as projection


HERE = Path(__file__).resolve().parent
INPUT_STEM = "edge_snap_transfer_charge_four_update_orbit"
DT = 0.1


def interval_extra_action(left_raw: torch.Tensor,
                          right_raw: torch.Tensor) -> torch.Tensor:
    left_psi, left_chi = local.unpack_tensor(left_raw, rank.SIDE)
    right_psi, right_chi = local.unpack_tensor(right_raw, rank.SIDE)
    psi = (left_psi + right_psi) / 2
    chi = (left_chi + right_chi) / 2
    psi_dot = (right_psi - left_psi) / DT
    chi_dot = (right_chi - left_chi) / DT
    rho_psi_dot = 2 * torch.real(torch.sum(
        torch.conj(psi) * psi_dot, dim=-1))
    rho_chi_dot = 2 * torch.real(torch.sum(
        torch.conj(chi) * chi_dot, dim=-1))
    rho_total = (torch.sum(local.squared_norm(psi), dim=-1)
                 + torch.sum(local.squared_norm(chi), dim=-1))
    denominator = rho_total + rank.KAPPA * rank.R**2
    return DT * torch.sum(
        (rho_psi_dot + rho_chi_dot)**2 / denominator) / 8


def action_delta(previous, current, following) -> np.ndarray:
    raws = [torch.tensor(rank.pack(*layer), dtype=torch.float64,
                        requires_grad=True)
            for layer in (previous, current, following)]
    action = interval_extra_action(raws[0], raws[1])
    action = action + interval_extra_action(raws[1], raws[2])
    return torch.autograd.grad(action, raws[1])[0].detach().numpy()


def main() -> None:
    rank.lock_inertia()
    triples, data = projection.layers_from_artifact(INPUT_STEM)
    baseline = data["residuals"].reshape(-1)
    column = np.concatenate([action_delta(*triple) for triple in triples])
    target = -baseline
    coefficient = float(np.dot(column, target) / np.dot(column, column))
    correlation = float(np.dot(column, target)
                        / (np.linalg.norm(column) * np.linalg.norm(target)))
    controls = []
    for name, value in (("zero", 0.0), ("unit", 1.0),
                        ("chi0", rank.R),
                        ("inverse_kappa", 1/rank.KAPPA),
                        ("B", rank.B),
                        ("diagnostic", max(coefficient, 0.0))):
        residual = baseline + value * column
        controls.append({
            "name": name,
            "coefficient": value,
            "relative_residual": float(
                np.linalg.norm(residual) / np.linalg.norm(baseline)),
        })
    best_native = min(controls[1:-1], key=lambda row:
                      row["relative_residual"])
    result = {
        "status": "V4_TOTAL_DENSITY_RADIAL_RESPONSE_SCREEN",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "input_stem": INPUT_STEM,
        "local_potential_identity": (
            "Ulocal=(rhoChi+rhoPsi-chi0^2)^2/4+chi0^2 rhoPsi/2"),
        "candidate": (
            "DeltaT=c/8 sum dot(rhoChi+rhoPsi)^2/"
            "(rhoChi+rhoPsi+kappa chi0^2)"),
        "real_positive_local_action": True,
        "response_correlation_with_needed_repair": correlation,
        "diagnostic_optimum_coefficient": coefficient,
        "diagnostic_coefficient_admitted": False,
        "controls": controls,
        "best_native_control": best_native,
        "verdict": (
            "ADVANCE_RADIAL_INERTIA" if
            coefficient > 0 and best_native["relative_residual"] < 1 else
            "RETIRE_RADIAL_INERTIA"),
        "claim_boundary": (
            "Response projection on one freely optimized handoff. It is not "
            "an admitted coefficient, rest pass, or moving orbit."),
    }
    (HERE / "total_density_radial_response.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
