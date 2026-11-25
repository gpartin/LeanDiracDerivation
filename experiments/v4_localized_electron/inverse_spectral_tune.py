"""Mode-driven coordinate repair of the admitted gyroscopic action family.

This is a diagnostic continuation, not coefficient admission.  Each trial
re-solves the exact rotating rest orbit and evaluates the complete two-layer
Floquet spectrum.  A coefficient is retained only when the maximum finite
growth rate decreases relative to the incumbent.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import field_berry_probe as field


HERE = Path(__file__).resolve().parent


def starting_coefficients() -> list[float]:
    values = [0.0] * 38
    values[0:11] = (-0.011, 0.017, -0.0165, -1.8, 0.1, 0.015, 0.0,
                    0.007, 0.0, 3.1, -3.1)
    # Best complete overlap-basis response.  These remain diagnostic until a
    # native degeneracy/critical-point condition fixes them.
    values[30] = 20.0
    values[32] = 20.0
    values[33] = 10.0
    return values


COORDINATES = (
    (0, "relative_local_gamma", 0.002),
    (1, "relative_bond_gamma", 0.003),
    (2, "common_gamma", 0.003),
    (3, "psi_berry", 0.2),
    (4, "chi_berry", 0.05),
    (5, "psi_density_common", 0.005),
    (7, "psi_berry_rho_psi", 0.003),
    (9, "chi_berry_rho_psi", 0.4),
    (10, "chi_berry_rho_chi", 0.4),
    (30, "overlap_relative_berry", 5.0),
    (32, "overlap_bond_common_berry", 5.0),
    (33, "overlap_bond_relative_berry", 5.0),
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--passes", type=int, default=1)
    parser.add_argument("--initial-json")
    parser.add_argument("--step-scale", type=float, default=1.0)
    parser.add_argument("--coordinates", nargs="+", type=int)
    parser.add_argument("--side", type=int, default=3)
    parser.add_argument("--rate", type=int, default=256)
    parser.add_argument("--output", default="inverse_spectral_tune.json")
    args = parser.parse_args()

    field.set_chi_inertia(229.0)
    dt = 1 / args.rate
    psi, _, qp, qc, _ = field.prior.bond.local.common.base.rest_state(
        args.side, dt, 1.0, -0.8, 0.03, 1.0, 1.0,
        0.0, 0.0, 0.0, 0.0)
    guess = np.asarray((
        np.log(field.prior.bond.local.common.base.norm2(psi).max()),
        np.angle(qp), np.angle(qc)))
    coefficients = starting_coefficients()
    if args.initial_json:
        loaded = json.loads(Path(args.initial_json).read_text(encoding="utf-8"))
        coefficients = [float(value) for value in loaded["best_coefficients"]]
    records: list[dict] = []

    def evaluate(label: str, trial: list[float]) -> dict:
        row = field.classify(args.side, dt, tuple(trial), guess.copy())
        growth = row.get("leading_growth_rate_per_unit_time")
        record = {
            "label": label,
            "growth": growth,
            "rest_residual": row["rest_del_rms_residual"],
            "coefficients": trial.copy(),
        }
        records.append(record)
        Path(HERE / args.output).write_text(json.dumps({
            "status": "INVERSE_SPECTRAL_DIAGNOSTIC",
            "records": records,
        }, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(f"{label}: residual={record['rest_residual']:.3e} "
              f"growth={growth}", flush=True)
        return record

    incumbent = evaluate("incumbent", coefficients)
    best_growth = float(incumbent["growth"])
    for pass_index in range(args.passes):
        improved_in_pass = False
        selected = (COORDINATES if args.coordinates is None else
                    tuple(item for item in COORDINATES
                          if item[0] in set(args.coordinates)))
        for index, name, initial_step in selected:
            step = args.step_scale * initial_step / (2**pass_index)
            candidates = []
            for sign in (-1.0, 1.0):
                trial = coefficients.copy()
                trial[index] += sign * step
                result = evaluate(
                    f"pass{pass_index + 1}:{name}:{sign:+.0f}", trial)
                if result["growth"] is not None:
                    candidates.append(result)
            if not candidates:
                continue
            winner = min(candidates, key=lambda item: item["growth"])
            if winner["growth"] < best_growth - 1e-8:
                coefficients = winner["coefficients"].copy()
                best_growth = float(winner["growth"])
                improved_in_pass = True
                print(f"RETAIN {name}: growth={best_growth:.12g}", flush=True)
        if not improved_in_pass:
            break

    result = {
        "status": "INVERSE_SPECTRAL_DIAGNOSTIC",
        "canonical_action_changed": False,
        "coefficient_admission": False,
        "method": "mode-driven coordinate descent with exact-rest re-solve",
        "best_growth": best_growth,
        "best_coefficients": coefficients,
        "records": records,
    }
    (HERE / args.output).write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    print(json.dumps({
        "best_growth": best_growth,
        "best_coefficients": coefficients,
        "evaluations": len(records),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
