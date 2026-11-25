"""Exact discrete Noether charge/face-cut continuity for the live action."""

from __future__ import annotations

import json
import argparse
from pathlib import Path

import numpy as np
import torch

import canonical_noncoplanar_bifurcation_audit as native_time
import live_clifford_bond_action as action


HERE = Path(__file__).resolve().parent
SIDE = 5


def generator(raw: torch.Tensor, field: str,
              mask: torch.Tensor | None = None) -> torch.Tensor:
    shaped = raw.reshape(SIDE, SIDE, SIDE, 12)
    result = torch.zeros_like(shaped)
    if field in ("Psi", "common"):
        result[..., 0:3] = -shaped[..., 3:6]
        result[..., 3:6] = shaped[..., 0:3]
    if field in ("chi", "common"):
        result[..., 6:9] = -shaped[..., 9:12]
        result[..., 9:12] = shaped[..., 6:9]
    if mask is not None:
        result = result*mask[..., None]
    return result.reshape(-1)


def interval_gradients(left: torch.Tensor, right: torch.Tensor):
    return torch.func.grad(action.interval, argnums=(0, 1))(
        left, right, native_time.DT)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", default="live_face_noether_trajectory.npz")
    parser.add_argument("--output", default="live_face_noether_continuity_gate.json")
    parser.add_argument(
        "--charge-mode", choices=("full", "perturbation"), default="full",
        help=("Conserve the full native state charge for stability, or "
              "subtract the rotating rest orbit for the later small-signal "
              "Dirac-current audit."))
    args = parser.parse_args()
    archive = np.load(HERE/args.archive)
    states = [torch.tensor(value, dtype=torch.float64)
              for value in archive["states"]]
    rest = archive["rest"]
    rotation = archive["rotation"]
    rest_states = [torch.tensor(
        np.linalg.matrix_power(rotation, index-1)@rest,
        dtype=torch.float64) for index in range(len(states))]
    masks = {"global": torch.ones((SIDE, SIDE, SIDE), dtype=torch.float64)}
    coordinates = torch.arange(SIDE)
    for axis in range(3):
        shape = [1, 1, 1]
        shape[axis] = SIDE
        masks[f"{('x','y','z')[axis]}_half"] = (
            coordinates.reshape(shape) <= SIDE//2).to(torch.float64).expand(
                SIDE, SIDE, SIDE)

    rows = []
    for field in ("Psi", "chi", "common"):
        for name, mask in masks.items():
            charges = []
            interval_variations = []
            continuity_defects = []
            for index in range(1, len(states)):
                left, right = states[index-1], states[index]
                grad_left, grad_right = interval_gradients(left, right)
                rest_left, rest_right = rest_states[index-1], rest_states[index]
                rest_grad_left, rest_grad_right = interval_gradients(
                    rest_left, rest_right)
                charge = torch.dot(grad_right, generator(
                    right, field, mask))
                rest_charge = torch.dot(rest_grad_right, generator(
                    rest_right, field, mask))
                charges.append(float(
                    charge-rest_charge
                    if args.charge_mode == "perturbation" else charge))
                variation = (
                    torch.dot(grad_left, generator(left, field, mask))
                    + torch.dot(grad_right, generator(right, field, mask)))
                rest_variation = (
                    torch.dot(rest_grad_left,
                              generator(rest_left, field, mask))
                    + torch.dot(rest_grad_right,
                                generator(rest_right, field, mask)))
                interval_variations.append(float(
                    variation-rest_variation
                    if args.charge_mode == "perturbation" else variation))
                if index >= 2:
                    continuity_defects.append(
                        charges[-1]-charges[-2]-interval_variations[-1])
            scale = max(max(abs(value) for value in charges), 1e-30)
            flux_scale = max(max(abs(value) for value in interval_variations),
                             scale, 1e-30)
            rows.append({
                "field_phase": field,
                "region": name,
                "charges": charges,
                "interval_cut_fluxes": interval_variations,
                "maximum_relative_charge_drift": float(
                    max(abs(value-charges[0]) for value in charges)/scale),
                "maximum_relative_continuity_defect": float(
                    max((abs(value) for value in continuity_defects),
                        default=0.0)/flux_scale),
                "signed_mean_cut_flux": float(np.mean(interval_variations)),
            })
    # Psi phase and common phase are exact declared symmetries of the action.
    # The chi-only row is retained as an ablation because some inherited
    # material terms exchange chi phase with Psi rather than conserve it
    # independently.
    symmetry_rows = [row for row in rows
                     if row["field_phase"] in ("Psi", "common")]
    global_rows = [row for row in symmetry_rows if row["region"] == "global"]
    local_rows = [row for row in rows if row["region"] != "global"]
    maximum_global = max(row["maximum_relative_charge_drift"]
                         for row in global_rows)
    maximum_continuity = max(row["maximum_relative_continuity_defect"]
                             for row in local_rows
                             if row["field_phase"] in ("Psi", "common"))
    result = {
        "status": "V4_LIVE_FACE_NOETHER_CONTINUITY_GATE",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "link_source": action.LINK_SOURCE,
        "action_scale": action.SCALE,
        "trajectory_updates": len(states)-2,
        "charge_mode": args.charge_mode,
        "rows": rows,
        "maximum_global_charge_relative_drift": maximum_global,
        "maximum_local_continuity_relative_defect": maximum_continuity,
        "verdict": (
            "LIVE_FACE_NOETHER_CONTINUITY_PASS"
            if maximum_global < 1e-8 and maximum_continuity < 1e-8
            else "LIVE_FACE_NOETHER_CONTINUITY_OPEN"),
        "claim_boundary": (
            "Exact discrete phase-charge and face-cut Ward identity on one "
            "four-update nonlinear trajectory. Full-state charge is the "
            "V4-002 conservation check; rest-subtracted small-signal charge "
            "and equality with the four-component Dirac probability/current "
            "remain later mapping gates."),
    }
    path = HERE/args.output
    path.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n",
                    encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items()
                      if key != "rows"}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
