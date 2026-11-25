"""Exact one-cell relative-periodic solvability screen for the current action.

The state at the current layer is the exact localized rest carrier.  The
previous and following layers are its literal one-cell translations with the
native rest phase in opposite directions.  We evaluate the full nonlinear
discrete Euler--Lagrange residual for both signs of every axis and several
translation times.  No force, projection, or target is added to the action.

Failure rules out only a rigid one-update-per-cell orbit.  It is the cheap
screen required before an unrestricted multi-layer boundary-value solve.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

import canonical_noncoplanar_bifurcation_audit as native_time
import canonical_noncoplanar_box_transfer as native_box
import cube_incidence_nonlinear_action as action
import cubical_edge_shift_channels as cubical


HERE = Path(__file__).resolve().parent
SIDE = 5
DT_VALUES = (1/512, 1/128, 1/32, 1/16, 0.08, 0.10)


def translate(raw: np.ndarray, axis: int, amount: int,
              psi_phase: complex) -> np.ndarray:
    tensor = torch.tensor(raw, dtype=torch.float64)
    psi, chi = native_box.unpack(tensor, SIDE)
    shifted_psi = torch.roll(psi, amount, dims=axis) * complex(psi_phase)
    shifted_chi = torch.roll(chi, amount, dims=axis)
    return np.asarray(native_box.pack(shifted_psi, shifted_chi))


def carrier_data(rest: np.ndarray):
    tensor = torch.tensor(rest, dtype=torch.float64)
    psi, chi = native_box.unpack(tensor, SIDE)
    psi_np, chi_np = np.asarray(psi), np.asarray(chi)
    sites = [tuple(int(value) for value in site)
             for site in np.argwhere(np.sum(abs(psi_np)**2, axis=-1) > 1e-12)]
    sites.sort()
    psi_frame = np.stack([psi_np[site] for site in sites])
    chi_frame = np.stack([chi_np[site] for site in sites])
    psi_frame /= np.linalg.norm(psi_frame, axis=1)[:, None]
    chi_frame /= np.linalg.norm(chi_frame, axis=1)[:, None]
    return sites, psi_frame, chi_frame


def projected_channels(residual: np.ndarray, sites, psi_frame, chi_frame):
    psi, chi = native_box.unpack(torch.tensor(residual), SIDE)
    psi_np, chi_np = np.asarray(psi), np.asarray(chi)
    psi_amp = np.asarray([
        np.vdot(psi_frame[index], psi_np[site])
        for index, site in enumerate(sites)])
    chi_amp = np.asarray([
        np.vdot(chi_frame[index], chi_np[site])
        for index, site in enumerate(sites)])
    beta = cubical.beta()
    identity = np.eye(8, dtype=complex)
    flavor = cubical.flavor()
    parity = {
        "even": (identity+beta)/2,
        "odd": (identity-beta)/2,
    }
    copies = {
        "plus": (identity+flavor)/2,
        "minus": (identity-flavor)/2,
    }

    def fractions(amplitude):
        total = max(float(np.vdot(amplitude, amplitude).real), 1e-30)
        return {
            "beta_even_fraction": float(
                np.vdot(parity["even"]@amplitude,
                         parity["even"]@amplitude).real/total),
            "beta_odd_fraction": float(
                np.vdot(parity["odd"]@amplitude,
                         parity["odd"]@amplitude).real/total),
            "flavor_plus_fraction": float(
                np.vdot(copies["plus"]@amplitude,
                         copies["plus"]@amplitude).real/total),
            "flavor_minus_fraction": float(
                np.vdot(copies["minus"]@amplitude,
                         copies["minus"]@amplitude).real/total),
        }

    full_power = float(np.dot(residual, residual))
    carrier_mask = np.zeros((SIDE, SIDE, SIDE), dtype=bool)
    for site in sites:
        carrier_mask[site] = True
    carrier_power = float(np.sum(abs(psi_np[carrier_mask])**2)
                          + np.sum(abs(chi_np[carrier_mask])**2))
    return {
        "Psi": fractions(psi_amp),
        "chi": fractions(chi_amp),
        "carrier_site_power_fraction": carrier_power/max(full_power, 1e-30),
        "Psi_power_fraction": float(np.sum(abs(psi_np)**2)
                                    / max(full_power, 1e-30)),
    }


def main() -> None:
    original_dt = native_time.DT
    bare, _ = action.expected_blocks()
    rest = bare["rest"]
    sites, psi_frame, chi_frame = carrier_data(rest)
    rows = []
    try:
        for dt in DT_VALUES:
            native_time.DT = dt
            theta = float(np.arccos(1-dt**2*native_time.CHI0**2/2))
            for axis in range(3):
                for sign in (-1, 1):
                    previous = translate(
                        rest, axis, -sign, np.exp(-1j*theta))
                    following = translate(
                        rest, axis, sign, np.exp(1j*theta))
                    residual = np.asarray(action.del_residual(
                        torch.tensor(previous, dtype=torch.float64),
                        torch.tensor(rest, dtype=torch.float64,
                                     requires_grad=True),
                        torch.tensor(following, dtype=torch.float64)
                    ).detach())
                    rows.append({
                        "dt": dt,
                        "translation_time": dt,
                        "axis": axis,
                        "sign": sign,
                        "rest_phase": theta,
                        "DEL_RMS": float(np.sqrt(np.mean(residual**2))),
                        "DEL_norm_over_state_norm": float(
                            np.linalg.norm(residual)/np.linalg.norm(rest)),
                        "residual_channels": projected_channels(
                            residual, sites, psi_frame, chi_frame),
                    })
    finally:
        native_time.DT = original_dt

    best = min(rows, key=lambda row: row["DEL_norm_over_state_norm"])
    result = {
        "status": "V4_COMPLETE_COCHAIN_RIGID_RELATIVE_ORBIT_SCREEN",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "tested_orbit": (
            "literal one-cell translation per update of the complete joint "
            "Psi-chi localized rest carrier with native rest phase"),
        "rows": rows,
        "best_row": best,
        "verdict": (
            "EXACT_RIGID_ONE_UPDATE_CELL_ORBIT_FOUND"
            if best["DEL_norm_over_state_norm"] < 1e-10 else
            "RIGID_ONE_UPDATE_CELL_ORBIT_OBSTRUCTED"),
        "claim_boundary": (
            "Exact nonlinear DEL screen of a rigid one-update orbit only. "
            "Failure does not rule out a multi-update deformed relative-"
            "periodic orbit or an action-derived traveling tail."),
    }
    output = HERE/"cube_incidence_complete_cochain_rigid_orbit_screen.json"
    output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n",
                      encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items()
                      if key != "rows"}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
