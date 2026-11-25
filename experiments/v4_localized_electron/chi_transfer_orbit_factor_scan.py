"""Scan the nonlinear χ-transfer normalization on one optimized orbit.

The saved four-layer state is held fixed.  Only the native geometric factor
in ``s(a)=(1+g a^2)^(-1/2)`` changes.  This distinguishes a wrong coefficient
from a residual outside the response span of the entire scalar transfer
channel.  A diagnostic optimum is not admitted as coefficient provenance.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
import torch

import edge_snap_action_rank as rank
import edge_snap_four_update_orbit as four
import edge_snap_transfer_four_update_orbit as transfer


HERE = Path(__file__).resolve().parent
SOURCE_STEM = "edge_snap_transfer_charge_four_update_orbit"


def load_states():
    data = np.load(HERE / f"{SOURCE_STEM}.npz")
    q0 = torch.tensor(data["q0"], dtype=torch.float64)
    middle = [torch.tensor(value, dtype=torch.float64, requires_grad=True)
              for value in data["middle"]]
    qn = torch.tensor(data["qn"], dtype=torch.float64)
    return q0, middle, qn


def evaluate(factor: float):
    transfer.GEOMETRIC_FACTOR = factor
    q0, middle, qn = load_states()
    qm1 = four.orbit.translate(
        middle[-1], -four.orbit.SIGN,
        -four.PHASE_PSI, -four.PHASE_CHI)
    states = [q0] + middle + [qn]
    q0_variable = q0.detach().clone().requires_grad_(True)
    residuals = [four.del_at(qm1, q0_variable, middle[0])]
    for index in range(1, four.UPDATES):
        residuals.append(four.del_at(
            states[index - 1], middle[index - 1], states[index + 1]))
    scale = rank.B * rank.R / four.DT
    scaled_del = float(torch.sqrt(sum(
        torch.mean((row / scale)**2) for row in residuals)).detach())
    charges = four.charge_rows(states, create_graph=False)
    charge_scales = {
        field: max(max(abs(float(value.detach())) for value in values), 1.0)
        for field, values in charges.items()
    }
    defects = {
        field: float(max(
            abs(float(value.detach() - values[0].detach()))
            / charge_scales[field] for value in values[1:]))
        for field, values in charges.items()
    }
    return {
        "geometric_factor": float(factor),
        "scaled_del_rms": scaled_del,
        "normalized_charge_defects": defects,
        "combined": float(np.sqrt(
            scaled_del**2 + sum(value**2 for value in defects.values()))),
    }


def main() -> None:
    rank.lock_inertia()
    original = four.orbit.interval_action
    four.orbit.interval_action = transfer.interval_action
    try:
        factors = (0.0, 1.0, 3.0, 6.0, 12.0, 19.0, 24.0,
                   27.0, 54.0, 63.0, 361.0)
        rows = [evaluate(factor) for factor in factors]
        optimum = minimize_scalar(
            lambda factor: evaluate(factor)["scaled_del_rms"],
            bounds=(0.0, 100.0), method="bounded",
            options={"xatol": 1e-7})
        optimum_row = evaluate(float(optimum.x))
    finally:
        four.orbit.interval_action = original
    incumbent = next(row for row in rows if row["geometric_factor"] == 19.0)
    result = {
        "status": "V4_CHI_TRANSFER_ORBIT_FACTOR_SCAN",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "source_orbit": SOURCE_STEM,
        "rows": rows,
        "diagnostic_optimum": {
            **optimum_row,
            "coefficient_admitted": False,
        },
        "relative_improvement_over_g19_at_fixed_state": float(
            (incumbent["scaled_del_rms"] - optimum_row["scaled_del_rms"])
            / incumbent["scaled_del_rms"]),
        "verdict": "COEFFICIENT_ONLY_DIAGNOSTIC",
        "claim_boundary": (
            "The orbit is not re-solved at each factor. This scan determines "
            "local coefficient sensitivity only and cannot admit an action."),
    }
    (HERE / "chi_transfer_orbit_factor_scan.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
