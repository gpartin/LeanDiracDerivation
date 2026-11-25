"""Derive the full interval-action phase flux of a moving carrier launch.

The position-only area current vanishes on a mere displaced profile.  Motion
in a second-order native system is phase-space data, so this audit perturbs the
previous layer toward the neighboring cell while keeping the current carrier
fixed.  A local phase twist of the complete discrete action then yields the
spatial flux through the receiving cut.  Psi-only, chi-only, and common-phase
symmetries are reported separately.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

import density_mold_connection_probe as mold
import edge_snap_action_rank as rank
import localized_embedding_two_layer_closure as two_layer
import transfer_neighbor_reflection_rest_spectrum as reflection


HERE = Path(__file__).resolve().parent
SIDE = 5
DT = 1.0 / 256.0
ORIGIN = SIDE // 2 - 1
MODES = ("Psi", "chi", "common")


def pack_tensor(psi: torch.Tensor, chi: torch.Tensor) -> torch.Tensor:
    return torch.stack(
        (psi.real, psi.imag, chi.real, chi.imag), dim=-2).reshape(-1)


def twist(raw: torch.Tensor, mask: torch.Tensor,
          epsilon: torch.Tensor, mode: str) -> torch.Tensor:
    psi, chi = mold.unpack_tensor(raw, SIDE)
    phase = torch.exp(1j * epsilon * mask)
    if mode in ("Psi", "common"):
        psi = phase[..., None] * psi
    if mode in ("chi", "common"):
        chi = phase[..., None] * chi
    return pack_tensor(psi, chi)


def cut_mask(axis: int, sign: int) -> np.ndarray:
    coordinate = np.indices((SIDE, SIDE, SIDE))[axis]
    if sign > 0:
        return coordinate >= ORIGIN + 2
    return coordinate <= ORIGIN - 1


def interval_flux(left: np.ndarray, right: np.ndarray,
                  mask: np.ndarray, mode: str) -> float:
    epsilon = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    mask_tensor = torch.tensor(mask, dtype=torch.float64)
    left_raw = torch.tensor(left, dtype=torch.float64)
    right_raw = torch.tensor(right, dtype=torch.float64)
    action = reflection.interval_action(
        twist(left_raw, mask_tensor, epsilon, mode),
        twist(right_raw, mask_tensor, epsilon, mode),
        reflection.COEFFICIENT)
    derivative = torch.autograd.grad(action, epsilon)[0]
    return float(-derivative.detach() / DT)


def rest_layers() -> tuple[np.ndarray, np.ndarray]:
    amplitude_sq = rank.R**2 / (2 + 54/rank.R**2)
    psi, chi = rank.native.carrier_fields(SIDE, amplitude_sq)
    theta = float(np.arccos(1-DT**2*rank.R**2/2))
    return rank.pack(np.exp(-1j*theta)*psi, chi), rank.pack(psi, chi)


def shifted_previous(previous: np.ndarray, axis: int,
                     sign: int) -> np.ndarray:
    psi, chi = rank.native.unpack(previous, SIDE)
    return rank.pack(
        np.roll(psi, -sign, axis=axis),
        np.roll(chi, -sign, axis=axis))


def launch_flux_response(axis: int, sign: int, mode: str,
                         step: float = 1e-5) -> dict:
    previous, current = rest_layers()
    direction = shifted_previous(previous, axis, sign)-previous
    mask = cut_mask(axis, sign)
    baseline = interval_flux(previous, current, mask, mode)
    left_flux = interval_flux(previous-step*direction, current, mask, mode)
    right_flux = interval_flux(previous+step*direction, current, mask, mode)
    derivative = (right_flux-left_flux)/(2*step)
    # Positive means flux into the destination half-space.  Multiplying by
    # the lattice direction gives the signed Cartesian current.
    return {
        "axis": axis,
        "sign": sign,
        "phase_mode": mode,
        "rest_cut_flux": baseline,
        "destination_inflow_response": float(derivative),
        "signed_axis_current_response": float(sign*derivative),
        "nonzero": bool(abs(derivative) > 1e-9),
        "flows_into_receiver": bool(derivative > 0),
    }


def global_phase_controls() -> dict:
    previous, current = rest_layers()
    mask = np.ones((SIDE, SIDE, SIDE), dtype=float)
    return {
        mode: interval_flux(previous, current, mask, mode)
        for mode in MODES
    }


def main() -> None:
    rank.lock_inertia()
    two_layer.rest.SIDE = SIDE
    two_layer.rest.DT = DT
    rows = [
        launch_flux_response(axis, sign, mode)
        for axis in range(3) for sign in (-1, 1) for mode in MODES
    ]
    common = [row for row in rows if row["phase_mode"] == "common"]
    psi = [row for row in rows if row["phase_mode"] == "Psi"]
    chi = [row for row in rows if row["phase_mode"] == "chi"]
    controls = global_phase_controls()
    result = {
        "status": "V4_LOCALIZED_CARRIER_CONTRAST_FULL_CURRENT_AUDIT",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "action": (
            "complete transfer-softened interval action plus the half-weight "
            "transfer-activated chi bond and 1/(2 chi0) frame reflection"),
        "method": (
            "differentiate the complete discrete interval action under an "
            "infinitesimal local phase twist across the receiving cut; launch "
            "direction is encoded only by the previous committed layer"),
        "global_phase_flux_controls": controls,
        "rows": rows,
        "common_all_nonzero": all(row["nonzero"] for row in common),
        "common_all_flow_into_receiver": all(
            row["flows_into_receiver"] for row in common),
        "Psi_all_nonzero": all(row["nonzero"] for row in psi),
        "chi_all_nonzero": all(row["nonzero"] for row in chi),
        "verdict": (
            "FULL_NATIVE_PHASE_SPACE_LAUNCH_HAS_RECEIVER_CURRENT"
            if (all(row["nonzero"] for row in common)
                and all(row["flows_into_receiver"] for row in common)
                and max(abs(value) for value in controls.values()) < 1e-8)
            else "FULL_NATIVE_CURRENT_BRIDGE_REQUIRES_RCA"),
        "claim_boundary": (
            "This is an infinitesimal launch-current response of the complete "
            "discrete action.  It does not establish continuity on a finite "
            "trajectory, equality with the Dirac current, nonlinear handoff, "
            "or long-time stability."),
        "corrective_action": (
            "On pass, derive the matching discrete phase density and verify "
            "the cut continuity identity.  On failure, identify whether the "
            "Psi, chi, or common symmetry carries the wrong receiver sign."),
    }
    output = HERE / "localized_carrier_contrast_full_current.json"
    output.write_text(
        json.dumps(result, indent=2, allow_nan=False)+"\n",
        encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
