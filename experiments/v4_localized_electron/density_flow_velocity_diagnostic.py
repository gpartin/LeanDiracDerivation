"""Diagnose the local continuity velocity on the locked exact translation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

import density_mold_connection_probe as local
import edge_snap_action_rank as rank
import exact_translation_density_flow_prong_gate as flow
import exact_translation_precursor_linear_gate as precursor


HERE = Path(__file__).resolve().parent


def main() -> None:
    rank.lock_inertia()
    evidence = json.loads((HERE / "edge_snap_verlet_gate.json").read_text())
    amplitude_sq = rank.R**2 / (2 + 54 / rank.R**2)
    psi, chi = rank.native.carrier_fields(precursor.SIDE, amplitude_sq)
    q0 = torch.tensor(rank.pack(psi, chi), dtype=torch.float64)
    rows = []
    for source in evidence["rows"]:
        axis, sign, dt = source["axis"], source["sign"], source["best_dt"]
        q1 = precursor.translate(
            q0, axis, sign, source["best_phase_psi"],
            source["best_phase_chi"])
        psi0, _ = local.unpack_tensor(q0, precursor.SIDE)
        psi1, _ = local.unpack_tensor(q1, precursor.SIDE)
        midpoint = (psi0 + psi1) / 2
        derivative = (psi1 - psi0) / dt
        rho_dot = 2 * torch.real(torch.sum(
            torch.conj(midpoint) * derivative, dim=-1))
        velocity = flow.continuity_velocity(midpoint, derivative)
        weights = torch.abs(rho_dot)
        weights = weights / torch.sum(weights)
        mean = torch.sum(velocity * weights[..., None], dim=(0, 1, 2))
        expected = torch.zeros(3, dtype=torch.float64)
        expected[axis] = sign / dt
        projection = mean[axis] * sign
        transverse = torch.linalg.vector_norm(
            mean - mean[axis] * torch.eye(3, dtype=torch.float64)[axis])
        local_projection = velocity[..., axis] * sign
        local_transverse_sq = torch.sum(velocity**2, dim=-1) - velocity[..., axis]**2
        rows.append({
            "axis": axis, "sign": sign,
            "expected_cell_velocity": float(1 / dt),
            "weighted_mean_velocity": mean.tolist(),
            "mean_forward_fraction": float(projection / (1 / dt)),
            "mean_transverse_fraction": float(transverse / (1 / dt)),
            "weighted_rms_forward_fraction": float(torch.sqrt(torch.sum(
                weights * local_projection**2)) / (1 / dt)),
            "weighted_rms_transverse_fraction": float(torch.sqrt(torch.sum(
                weights * local_transverse_sq.clamp_min(0))) / (1 / dt)),
        })
    result = {
        "status": "V4_DENSITY_FLOW_VELOCITY_DIAGNOSTIC",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "rows": rows,
        "claim_boundary": (
            "Velocity diagnostic on prescribed exact-shift intervals only; "
            "not a freely solved trajectory or action pass."),
    }
    (HERE / "density_flow_velocity_diagnostic.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
