"""Derive signed launch currents from the unit cube-incidence action."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

import canonical_noncoplanar_bifurcation_audit as native_time
import canonical_noncoplanar_box_transfer as native_box
import canonical_transfer_oscillator_spectrum as native_spectrum
import cube_incidence_nonlinear_action as action


HERE = Path(__file__).resolve().parent
SIDE = 5
MODES = ("Psi", "chi", "common", "relative")


def pack(psi, chi):
    return torch.stack((psi.real, psi.imag, chi.real, chi.imag),
                       dim=-2).reshape(-1)


def transform(raw, phase, mode):
    psi, chi = native_box.unpack(raw, SIDE)
    if mode in ("Psi", "common", "relative"):
        psi = phase[..., None]*psi
    if mode in ("chi", "common"):
        chi = phase[..., None]*chi
    elif mode == "relative":
        chi = torch.conj(phase)[..., None]*chi
    return pack(psi, chi)


def launch_previous(previous, axis, sign, mode, momentum):
    coordinate = np.indices((SIDE, SIDE, SIDE))[axis].astype(float)
    center = (SIDE-1)/2
    phase = torch.tensor(
        np.exp(-1j*sign*momentum*(coordinate-center)),
        dtype=torch.complex128)
    return transform(previous, phase, mode)


def cut_mask(axis, sign):
    coordinate = np.indices((SIDE, SIDE, SIDE))[axis]
    # The carrier occupies coordinates 1 and 2 on a side-five box.  A mask
    # containing coordinate 1 or 2 cuts *through* the carrier and measures its
    # internal circulation, not transport into the quiet receiving cell.
    # Translation by one lattice cell crosses 2--3 in the positive direction
    # and 1--0 in the negative direction, so only the exterior half-space is
    # phased here.
    return coordinate >= SIDE//2+1 if sign > 0 else coordinate <= SIDE//2-2


def flux(previous, current, mask, symmetry):
    epsilon = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    phase = torch.exp(1j*epsilon*torch.tensor(mask, dtype=torch.float64))
    value = action.interval_action(
        transform(previous, phase, symmetry),
        transform(current, phase, symmetry))
    return float(-torch.autograd.grad(value, epsilon)[0].detach()
                 / native_time.DT)


def main() -> None:
    bare, _matrices = action.expected_blocks()
    theta = float(np.arccos(
        1-native_time.DT**2*native_time.CHI0**2/2))
    rotation = native_spectrum.rotation_matrix(bare["rest"].size, theta)
    previous0 = torch.tensor(rotation.T@bare["rest"], dtype=torch.float64)
    current = torch.tensor(bare["rest"], dtype=torch.float64)
    momentum = 0.1
    rows = []
    for axis in range(3):
        for sign in (-1, 1):
            mask = cut_mask(axis, sign)
            for launch_mode in MODES:
                previous = launch_previous(
                    previous0, axis, sign, launch_mode, momentum)
                fluxes = {symmetry: flux(previous, current, mask, symmetry)
                          for symmetry in ("Psi", "chi", "common")}
                rows.append({
                    "axis": axis,
                    "sign": sign,
                    "launch_mode": launch_mode,
                    "fluxes": fluxes,
                    "signed_common_current": sign*fluxes["common"],
                })
    global_mask = np.ones((SIDE, SIDE, SIDE), dtype=bool)
    controls = {symmetry: flux(previous0, current, global_mask, symmetry)
                for symmetry in ("Psi", "chi", "common")}
    per_mode = {}
    for mode in MODES:
        selected = [row for row in rows if row["launch_mode"] == mode]
        currents = [row["signed_common_current"] for row in selected]
        per_mode[mode] = {
            "minimum_signed_common_current": min(currents),
            "maximum_signed_common_current": max(currents),
            "all_six_directions_positive": all(value > 0 for value in currents),
            "axis_sign_spread": max(currents)-min(currents),
        }
    candidates = [mode for mode, row in per_mode.items()
                  if row["all_six_directions_positive"]]
    selected_mode = (max(candidates, key=lambda mode: per_mode[mode][
        "minimum_signed_common_current"]) if candidates else None)
    result = {
        "status": "V4_CUBE_INCIDENCE_NOETHER_LAUNCH",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "momentum": momentum,
        "global_flux_controls": controls,
        "per_launch_mode": per_mode,
        "selected_native_launch_mode": selected_mode,
        "rows": rows,
        "verdict": (
            "NATIVE_NOETHER_LAUNCH_HAS_ALL_AXIS_CURRENT"
            if selected_mode is not None
            and max(abs(value) for value in controls.values()) < 1e-8
            else "NATIVE_NOETHER_LAUNCH_CURRENT_FAIL"),
        "claim_boundary": (
            "Action-derived one-interval current and sign gate. It does not "
            "prove finite translation, Dirac-current normalization, two "
            "handoffs, or V4-006."),
    }
    (HERE/"cube_incidence_noether_launch.json").write_text(
        json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items()
                      if key != "rows"}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
