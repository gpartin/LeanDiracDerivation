"""Screen a smooth multi-update cell translation under the unchanged action.

The exact rest carrier is shifted fractionally by the periodic Fourier
translation operator.  After N updates the state is exactly the integer-cell
translation of the start (up to the native rest phase).  This is a trajectory
diagnostic, not an imposed runtime rule: the full nonlinear DEL residual says
whether the action would execute that path freely.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

import canonical_noncoplanar_bifurcation_audit as native_time
import canonical_noncoplanar_box_transfer as native_box
import cube_incidence_complete_cochain_rigid_orbit_screen as rigid
import cube_incidence_nonlinear_action as action


HERE = Path(__file__).resolve().parent
SIDE = 5
ROWS = (
    (16, 1/16),
    (32, 1/32),
    (32, 1/16),
    (64, 1/64),
    (64, 1/32),
    (64, 1/16),
)


def fractional_shift(field: np.ndarray, axis: int, amount: float) -> np.ndarray:
    frequencies = np.fft.fftfreq(SIDE)
    shape = [1]*field.ndim
    shape[axis] = SIDE
    multiplier = np.exp(-2j*np.pi*frequencies*amount).reshape(shape)
    return np.fft.ifft(
        np.fft.fft(field, axis=axis)*multiplier, axis=axis)


def state(rest: np.ndarray, axis: int, sign: int, fraction: float,
          phase: float) -> np.ndarray:
    psi, chi = native_box.unpack(torch.tensor(rest), SIDE)
    psi_np, chi_np = np.asarray(psi), np.asarray(chi)
    shifted_psi = fractional_shift(psi_np, axis, sign*fraction)
    shifted_chi = fractional_shift(chi_np, axis, sign*fraction)
    shifted_psi *= np.exp(1j*phase)
    return np.asarray(native_box.pack(shifted_psi, shifted_chi))


def main() -> None:
    original_dt = native_time.DT
    bare, _ = action.expected_blocks()
    rest = bare["rest"]
    sites, psi_frame, chi_frame = rigid.carrier_data(rest)

    # Exact integer endpoint control for the fractional translation operator.
    psi, chi = native_box.unpack(torch.tensor(rest), SIDE)
    endpoint_defects = []
    for axis in range(3):
        for sign in (-1, 1):
            fp = fractional_shift(np.asarray(psi), axis, sign)
            fc = fractional_shift(np.asarray(chi), axis, sign)
            endpoint_defects.append(float(max(
                np.linalg.norm(fp-np.roll(np.asarray(psi), sign, axis=axis))
                / max(np.linalg.norm(psi), 1e-30),
                np.linalg.norm(fc-np.roll(np.asarray(chi), sign, axis=axis))
                / max(np.linalg.norm(chi), 1e-30))))

    rows = []
    try:
        for updates, dt in ROWS:
            native_time.DT = dt
            theta = float(np.arccos(1-dt**2*native_time.CHI0**2/2))
            sample_indices = sorted(set(
                int(round(value)) % updates
                for value in np.linspace(0, updates-1, min(updates, 8))))
            speed = 1/(updates*dt)
            for axis in range(3):
                for sign in (-1, 1):
                    samples = []
                    for index in sample_indices:
                        previous = state(
                            rest, axis, sign, (index-1)/updates,
                            (index-1)*theta)
                        current = state(
                            rest, axis, sign, index/updates, index*theta)
                        following = state(
                            rest, axis, sign, (index+1)/updates,
                            (index+1)*theta)
                        residual = np.asarray(action.del_residual(
                            torch.tensor(previous, dtype=torch.float64),
                            torch.tensor(current, dtype=torch.float64,
                                         requires_grad=True),
                            torch.tensor(following, dtype=torch.float64)
                        ).detach())
                        samples.append({
                            "layer": index,
                            "fraction": index/updates,
                            "DEL_norm_over_state_norm": float(
                                np.linalg.norm(residual)/np.linalg.norm(current)),
                            "DEL_RMS": float(np.sqrt(np.mean(residual**2))),
                            "residual_channels": rigid.projected_channels(
                                residual, sites, psi_frame, chi_frame),
                        })
                    rows.append({
                        "updates_per_cell": updates,
                        "dt": dt,
                        "translation_time": updates*dt,
                        "cell_speed": speed,
                        "axis": axis,
                        "sign": sign,
                        "maximum_DEL_norm_over_state_norm": max(
                            row["DEL_norm_over_state_norm"] for row in samples),
                        "RMS_DEL_norm_over_state_norm": float(np.sqrt(np.mean([
                            row["DEL_norm_over_state_norm"]**2
                            for row in samples]))),
                        "samples": samples,
                    })
    finally:
        native_time.DT = original_dt

    best = min(rows, key=lambda row: row["maximum_DEL_norm_over_state_norm"])
    result = {
        "status": "V4_COMPLETE_COCHAIN_FRACTIONAL_RELATIVE_ORBIT_SCREEN",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "endpoint_translation_maximum_relative_defect": max(endpoint_defects),
        "trajectory": (
            "periodic spectral fractional translation of the complete joint "
            "Psi-chi rest carrier; exact integer translation after N layers"),
        "rows": rows,
        "best_row": {key: value for key, value in best.items()
                     if key != "samples"},
        "verdict": (
            "FRACTIONAL_TRANSLATION_IS_NATIVE_RELATIVE_ORBIT"
            if best["maximum_DEL_norm_over_state_norm"] < 1e-8 else
            "FRACTIONAL_TRANSLATION_REQUIRES_NATIVE_DEFORMATION"),
        "claim_boundary": (
            "The path is an exact endpoint-compatible diagnostic ansatz, not "
            "a force or runtime projection. Failure requires an unrestricted "
            "deformed path solve before any action no-go can be stated."),
    }
    output = HERE/"cube_incidence_complete_cochain_fractional_orbit_screen.json"
    output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n",
                      encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items()
                      if key != "rows"}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
