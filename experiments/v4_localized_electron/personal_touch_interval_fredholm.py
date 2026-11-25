"""Fredholm branch test for personal-touch discrete one-forms/metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import canonical_noncoplanar_bifurcation_audit as small
import canonical_noncoplanar_box_transfer as box
import personal_touch_complete_frame_branch as branch


HERE = Path(__file__).resolve().parent
SIDE = 3


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", choices=(
        "plane_kinetic", "area_berry", "cross_berry",
        "cross_bond_berry"), required=True)
    args = parser.parse_args()
    candidate = {
        "plane_kinetic": branch.plane_kinetic_interval,
        "area_berry": branch.area_berry_interval,
        "cross_berry": branch.cross_berry_interval,
        "cross_bond_berry": branch.cross_bond_berry_interval,
    }[args.candidate]
    reference = np.load(
        HERE / "canonical_noncoplanar_solvability_search.npz")["rest"]
    reference_psi, _ = small.unpack(reference)
    theta = float(np.arccos(1-small.DT**2*small.CHI0**2/2))
    parameters = torch.tensor(np.r_[reference, theta, 0.0],
                              dtype=torch.float64, requires_grad=True)
    anchor, exterior = (0, 0, 0), (2, 2, 2)

    def layers(value: torch.Tensor):
        raw, tp, tc = value[:-2], value[-2], value[-1]
        return (branch.phase_rotate(raw, tp, tc, -1), raw,
                branch.phase_rotate(raw, tp, tc, 1))

    def augmented_residual(value: torch.Tensor) -> torch.Tensor:
        previous, raw, following = layers(value)
        action = box.interval(previous, raw, SIDE) + box.interval(
            raw, following, SIDE)
        dynamic = torch.autograd.grad(action, raw, create_graph=True)[0]
        psi, chi = box.unpack(raw, SIDE)
        gauges = torch.stack((torch.imag(torch.sum(
            torch.conj(torch.tensor(reference_psi[anchor]))*psi[anchor])),
            torch.imag(chi[exterior][2])))
        return torch.cat((dynamic, gauges))

    jacobian = torch.autograd.functional.jacobian(
        augmented_residual, parameters, vectorize=True).detach().numpy()
    previous, raw, following = layers(parameters)
    candidate_action = candidate(previous, raw) + candidate(raw, following)
    source_dynamic = torch.autograd.grad(
        candidate_action, raw, create_graph=False)[0].detach().numpy()
    source = np.r_[source_dynamic, 0.0, 0.0]
    solution, _, rank, singular = np.linalg.lstsq(
        jacobian, -source, rcond=1e-10)
    obstruction = source+jacobian@solution
    source_norm = float(np.linalg.norm(source))
    obstruction_norm = float(np.linalg.norm(obstruction))
    relative = obstruction_norm/max(source_norm, 1e-300)
    result = {
        "status": "V4_PERSONAL_TOUCH_INTERVAL_FREDHOLM",
        "candidate": args.candidate,
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "augmented_dimension": int(jacobian.shape[0]),
        "augmented_rank": int(rank),
        "augmented_nullity": int(jacobian.shape[1]-rank),
        "unit_strength_source_norm": source_norm,
        "unit_strength_range_obstruction_norm": obstruction_norm,
        "relative_range_obstruction": relative,
        "predicted_DEL_rms_at_strength_0p001": float(
            0.001*np.linalg.norm(obstruction[:-2])
            / np.sqrt(source_dynamic.size)),
        "verdict": (
            "SOURCE_ZERO_ON_REFERENCE_ORBIT" if source_norm < 1e-12
            else "SMOOTH_BRANCH_FREDHOLM_OBSTRUCTED" if relative > 1e-8
            else "SMOOTH_BRANCH_FIRST_ORDER_SOLVABLE"),
        "claim_boundary": (
            "First-order continuation from the canonical exact orbit with "
            "free fields and two global phase advances only."),
    }
    output = HERE / f"personal_touch_{args.candidate}_fredholm.json"
    output.write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
