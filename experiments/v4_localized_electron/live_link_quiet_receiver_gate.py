"""Does the live-link action create a native force at a quiet receiver?

The source carrier receives a small signed phase launch only in its previous
layer.  The current receiver has exactly zero Psi and ordinary nonzero chi.
We measure the additional reciprocal action residual on exterior Psi sites.
No receiver preload, target field, projection, or Dirac force is used.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

import canonical_noncoplanar_bifurcation_audit as native_time
import canonical_noncoplanar_box_transfer as native_box
import canonical_transfer_oscillator_spectrum as native_spectrum
import cube_incidence_noether_launch as launch
import cube_incidence_nonlinear_action as native_action
import live_clifford_bond_action as candidate


HERE = Path(__file__).resolve().parent
SIDE = 5


def psi_field(raw: np.ndarray) -> np.ndarray:
    psi, _chi = native_box.unpack(torch.tensor(raw, dtype=torch.float64), SIDE)
    return psi.detach().numpy()


def exterior_mask(axis: int, sign: int) -> np.ndarray:
    coordinate = np.indices((SIDE, SIDE, SIDE))[axis]
    return coordinate >= 3 if sign > 0 else coordinate <= 0


def main() -> None:
    rest = native_action.expected_blocks()[0]["rest"]
    dt = native_time.DT
    theta = float(np.arccos(1-dt**2*native_time.CHI0**2/2))
    rotation = native_spectrum.rotation_matrix(rest.size, theta)
    previous0 = torch.tensor(rotation.T@rest, dtype=torch.float64)
    current = torch.tensor(rest, dtype=torch.float64)
    following = torch.tensor(rotation@rest, dtype=torch.float64)
    rows = []
    for axis in range(3):
        for sign in (-1, 1):
            target = exterior_mask(axis, sign)
            opposite = exterior_mask(axis, -sign)
            for mode in ("Psi", "common"):
                previous = launch.launch_previous(
                    previous0, axis, sign, mode, 0.1)
                full = candidate.del_residual(
                    previous, current, following, dt)
                baseline = candidate.incumbent.del_residual(
                    previous, current, following, dt)
                response = psi_field((full-baseline).detach().numpy())
                target_norm = float(np.linalg.norm(response[target]))
                opposite_norm = float(np.linalg.norm(response[opposite]))
                rows.append({
                    "axis": axis,
                    "sign": sign,
                    "launch_mode": mode,
                    "target_receiver_force_norm": target_norm,
                    "opposite_receiver_force_norm": opposite_norm,
                    "signed_receiver_contrast": target_norm-opposite_norm,
                })
    target_norms = [row["target_receiver_force_norm"] for row in rows]
    result = {
        "status": "V4_LIVE_LINK_QUIET_RECEIVER_GATE",
        "link_source": candidate.LINK_SOURCE,
        "minimum_target_receiver_force_norm": min(target_norms),
        "maximum_target_receiver_force_norm": max(target_norms),
        "rows": rows,
        "verdict": (
            "LIVE_LINK_CAN_SEED_QUIET_RECEIVER"
            if min(target_norms) > 1e-12 else
            "LIVE_LINK_COMPACT_SUPPORT_OBSTRUCTION_REMAINS"),
        "claim_boundary": (
            "First reciprocal action residual at an exactly quiet receiver. "
            "It is not a solved update or a complete handoff."),
    }
    suffix = candidate.LINK_SOURCE.replace(".", "p")
    (HERE/f"live_link_quiet_receiver_gate_{suffix}.json").write_text(
        json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items()
                      if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
