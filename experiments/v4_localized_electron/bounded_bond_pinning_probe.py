"""Compare the one-edge pinning barrier of canonical and bounded bond actions.

The bounded research potential uses the previously audited coefficients
q=-0.8 and s=0.03 in the C19 area-difference polynomial

  Vbond = sum w/(2 chi0^2) [d + q d^2/chi0^4 + s d^3/chi0^8],
  d = |W_i-W_j|^2.

Each action uses its own freely solved exact rest amplitude.  The path is a
charge-normalized interpolation between one carrier and its one-face
translation.  This is a PN diagnostic, not a moving solution.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

import density_mold_connection_probe as local
import edge_snap_action_rank as rank
import edge_snap_four_update_orbit as four
import localized_stability as stability


HERE = Path(__file__).resolve().parent
SIDE = 5
QUARTIC = -0.8
SEXTIC = 0.03


def potential(raw: torch.Tensor, quartic: float,
              sextic: float) -> torch.Tensor:
    psi, chi = local.unpack_tensor(raw, SIDE)
    rho_psi = torch.sum(local.squared_norm(psi), dim=-1)
    rho_chi = torch.sum(local.squared_norm(chi), dim=-1)
    value = torch.sum(
        0.25 * (rho_chi - rank.R**2)**2
        + 0.5 * rho_chi * rho_psi + 0.25 * rho_psi**2)
    area = torch.linalg.cross(chi, psi, dim=-1)
    for offset, weight in rank.native.DIRECTIONS:
        difference = area - torch.roll(
            area, shifts=offset, dims=(0, 1, 2))
        size = torch.sum(local.squared_norm(difference), dim=-1)
        polynomial = (size + quartic * size**2 / rank.R**4
                      + sextic * size**3 / rank.R**8)
        value = value + weight * torch.sum(polynomial) / (2 * rank.R**2)
    return value


def exact_rest(quartic: float, sextic: float):
    if quartic == 0 and sextic == 0:
        amplitude_sq = rank.R**2 / (2 + 54 / rank.R**2)
        return rank.native.carrier_fields(SIDE, amplitude_sq), amplitude_sq
    psi, chi, _, _, _ = stability.rest_state(
        SIDE, four.DT, 1.0, quartic, sextic,
        1.0, 1.0, 0.0, 0.0, 0.0, 0.0)
    active = np.sum(abs(psi)**2, axis=-1) > 1e-12
    amplitude_sq = float(np.mean(np.sum(abs(psi[active])**2, axis=-1)))
    return (psi, chi), amplitude_sq


def profile(name: str, quartic: float, sextic: float) -> dict:
    (psi, chi), amplitude_sq = exact_rest(quartic, sextic)
    q0 = torch.tensor(rank.pack(psi, chi), dtype=torch.float64)
    qn = four.orbit.translate(
        q0, four.orbit.SIGN, four.PHASE_PSI, four.PHASE_CHI).detach()
    q0_charge = float(torch.dot(q0, q0))
    rows = []
    for fraction in np.linspace(0, 1, 41):
        angle = np.pi * fraction / 2
        state = np.cos(angle) * q0 + np.sin(angle) * qn
        state = state * np.sqrt(q0_charge / float(torch.dot(state, state)))
        rows.append({
            "fraction": float(fraction),
            "potential": float(potential(state, quartic, sextic)),
        })
    endpoints = (rows[0]["potential"] + rows[-1]["potential"]) / 2
    barrier = max(row["potential"] for row in rows) - endpoints
    trough = endpoints - min(row["potential"] for row in rows)
    return {"name": name, "quartic": quartic, "sextic": sextic,
            "amplitude_sq": amplitude_sq, "positive_barrier": barrier,
            "intermediate_trough_depth": trough,
            "rows": rows}


def main() -> None:
    rank.lock_inertia()
    canonical = profile("canonical", 0.0, 0.0)
    bounded = profile("bounded_polynomial", QUARTIC, SEXTIC)
    tolerance = 1e-10 * max(
        canonical["rows"][0]["potential"],
        bounded["rows"][0]["potential"])
    positive_barrier_absent = (
        canonical["positive_barrier"] < tolerance
        and bounded["positive_barrier"] < tolerance)
    trough_change = (
        bounded["intermediate_trough_depth"]
        / canonical["intermediate_trough_depth"] - 1)
    result = {
        "status": "V4_BOUNDED_BOND_PINNING_SCREEN",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "profiles": [canonical, bounded],
        "numerical_barrier_tolerance": float(tolerance),
        "positive_barrier_absent": positive_barrier_absent,
        "relative_trough_depth_change": float(trough_change),
        "bounded_below": True,
        "coefficient_status": (
            "previous best bounded rest-spectrum research coefficients; "
            "not yet derived constants"),
        "verdict": (
            "RETIRE_BOUNDED_BOND_FOR_MOVEMENT: deeper intermediate trough"
            if trough_change > 0 else
            "ADVANCE_BOUNDED_MOVEMENT_COMBINATION"),
        "claim_boundary": (
            "Charge-normalized interpolation barrier only; not an exact "
            "minimum-energy path, moving orbit, or action admission."),
    }
    (HERE / "bounded_bond_pinning_probe.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
