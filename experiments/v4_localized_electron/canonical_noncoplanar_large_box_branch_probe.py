"""Nonlinear screen of the unique side-five noncoplanar null direction."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import canonical_noncoplanar_bifurcation_audit as small
import canonical_noncoplanar_box_transfer as box
import canonical_noncoplanar_large_box_audit as large


HERE = Path(__file__).resolve().parent
FRACTIONS = (1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2)


def main() -> None:
    data = np.load(HERE / "canonical_noncoplanar_large_box_audit.npz")
    rest = data["rest"]
    direction = data["direction"]
    theta = float(np.arccos(1-small.DT**2*small.CHI0**2/2))
    phase = np.exp(1j*theta)
    rest_norm = np.linalg.norm(rest)
    psi0, _chi0 = large.unpack_np(rest)
    core = np.sum(abs(psi0)**2, axis=-1) > 1e-12
    rows = []
    for fraction in FRACTIONS:
        epsilon = fraction*rest_norm
        state = rest+epsilon*direction
        value = box.residual(state, large.SIDE, phase)
        psi, chi = large.unpack_np(state)
        psi_power = np.sum(abs(psi)**2)
        halo_psi = float(np.sum(abs(psi[~core])**2)/max(psi_power, 1e-300))
        vacuum = np.asarray((0, 0, small.CHI0), dtype=complex)
        halo_chi = float(np.linalg.norm(chi[~core]-vacuum))
        rows.append({
            "fraction_of_rest_norm": fraction,
            "epsilon": epsilon,
            "orbit_residual_rms": float(
                np.linalg.norm(value)/np.sqrt(value.size)),
            "residual_over_epsilon_squared": float(
                np.linalg.norm(value)/max(epsilon**2, 1e-300)),
            "residual_over_epsilon_cubed": float(
                np.linalg.norm(value)/max(epsilon**3, 1e-300)),
            "halo_psi_power_fraction": halo_psi,
            "halo_chi_deviation_norm": halo_chi,
        })
    slope = float(np.polyfit(
        np.log([row["epsilon"] for row in rows[-5:]]),
        np.log([row["orbit_residual_rms"] for row in rows[-5:]]), 1)[0])
    result = {
        "status": "V4_CANONICAL_NONCOPLANAR_LARGE_BOX_BRANCH_PROBE",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "rows": rows,
        "large_amplitude_log_slope": slope,
        "verdict": (
            "UNIQUE_LARGE_BOX_DIRECTION_IS_EXACT_NONLINEAR_BRANCH"
            if max(row["orbit_residual_rms"] for row in rows[-3:]) < 1e-10
            else "UNIQUE_LARGE_BOX_DIRECTION_REQUIRES_NONLINEAR_CORRECTION"),
        "claim_boundary": (
            "Uncorrected continuation along the exact Hessian direction. "
            "A residual does not exclude a range-corrected branch."),
    }
    output = HERE / "canonical_noncoplanar_large_box_branch_probe.json"
    output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n",
                      encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
