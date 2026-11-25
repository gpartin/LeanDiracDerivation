"""Rest-spectrum gate for the transfer-activated native bond transporter."""

from __future__ import annotations

import json
from pathlib import Path

import torch

import chi_bond_frame_transport_probe as probe
import density_mold_connection_probe as local
import density_mold_rest_spectrum as rest
import edge_snap_action_rank as rank


HERE = Path(__file__).resolve().parent


def interval_action(left: torch.Tensor, right: torch.Tensor,
                    factor: float) -> torch.Tensor:
    psi_left, chi_left = local.unpack_tensor(left, rest.SIDE)
    psi_right, chi_right = local.unpack_tensor(right, rest.SIDE)
    psi_dot = (psi_right - psi_left) / rest.DT
    chi_dot = (chi_right - chi_left) / rest.DT
    canonical_kinetic = 0.5 * torch.sum(local.squared_norm(psi_dot))
    canonical_kinetic = canonical_kinetic + rank.B / 2 * torch.sum(
        local.squared_norm(chi_dot))
    canonical = rest.DT * (
        canonical_kinetic - (rest.potential(left) + rest.potential(right)) / 2)
    return canonical + probe.interval_extra_action(
        left, right, rest.DT, factor)


def classify(factor: float):
    original_interval = rest.interval_action
    original_side = probe.SIDE
    rest.interval_action = interval_action
    probe.SIDE = rest.SIDE
    try:
        return rest.classify(factor)
    finally:
        rest.interval_action = original_interval
        probe.SIDE = original_side


def main() -> None:
    rank.lock_inertia()
    rows = []
    for name, factor in (("raw_average", 0.0),
                         ("six_face_bond_transport", 6.0)):
        row = classify(factor)
        row["name"] = name
        row["factor"] = factor
        rows.append(row)
    incumbent, candidate = rows
    growth_defect = abs(
        candidate["growth_rate_per_unit_time"]
        - incumbent["growth_rate_per_unit_time"])
    radius_defect = abs(
        candidate["spectral_radius"] - incumbent["spectral_radius"])
    passed = growth_defect < 1e-7 and radius_defect < 1e-10
    result = {
        "status": (
            "V4_CHI_BOND_FRAME_REST_GATE_PASS" if passed else
            "V4_CHI_BOND_FRAME_REST_GATE_FAIL"),
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "side": rest.SIDE,
        "dt": rest.DT,
        "rows": rows,
        "growth_rate_defect": float(growth_defect),
        "spectral_radius_defect": float(radius_defect),
        "coefficient_provenance": (
            "factor 6 from the six C19 face transport directions"),
        "verdict": (
            "ADVANCE_TO_FREE_ORBIT_GATE" if passed else
            "RETIRE_BOND_FRAME_AT_REST_GATE"),
        "claim_boundary": (
            "Complete side-three two-layer rest-spectrum gate only; not "
            "nonlinear stability or movement."),
    }
    (HERE / "chi_bond_frame_transport_rest_spectrum.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
