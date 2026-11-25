"""Fredholm gate for the residual loop-curvature action response."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

import canonical_noncoplanar_bifurcation_audit as small
import canonical_noncoplanar_box_transfer as box
import personal_touch_complete_frame_branch as branch


HERE = Path(__file__).resolve().parent
SIDE = 3


def dyad(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return left[..., :, None]*torch.conj(right[..., None, :])


def norm_sq(value: torch.Tensor) -> torch.Tensor:
    return torch.real(torch.sum(torch.conj(value)*value, dim=(-2, -1)))


def shift(value: torch.Tensor, axis: int) -> torch.Tensor:
    offset = tuple(1 if entry == axis else 0 for entry in range(3))
    return torch.roll(value, shifts=offset, dims=(0, 1, 2))


def curvature(field: torch.Tensor) -> torch.Tensor:
    result = torch.zeros((), dtype=torch.float64)
    for first, second in ((0, 1), (0, 2), (1, 2)):
        da, db = shift(field, first)-field, shift(field, second)-field
        commutator = da@db-db@da
        result += torch.sum(norm_sq(commutator))/2
    return result/small.CHI0**4


def combined(raw: torch.Tensor) -> torch.Tensor:
    psi, chi = box.unpack(raw, SIDE)
    return curvature(dyad(psi, psi)+dyad(chi, chi))


def separated(raw: torch.Tensor) -> torch.Tensor:
    psi, chi = box.unpack(raw, SIDE)
    return curvature(dyad(psi, psi))+curvature(dyad(chi, chi))


def main() -> None:
    reference = np.load(
        HERE / "canonical_noncoplanar_solvability_search.npz")["rest"]
    reference_psi, _ = small.unpack(reference)
    theta = float(np.arccos(1-small.DT**2*small.CHI0**2/2))
    parameters = torch.tensor(np.r_[reference, theta, 0.0],
                              dtype=torch.float64, requires_grad=True)
    anchor, exterior = (0, 0, 0), (2, 2, 2)

    def augmented_residual(value: torch.Tensor) -> torch.Tensor:
        raw, tp, tc = value[:-2], value[-2], value[-1]
        previous = branch.phase_rotate(raw, tp, tc, -1)
        following = branch.phase_rotate(raw, tp, tc, 1)
        action = (box.interval(previous, raw, SIDE)
                  + box.interval(raw, following, SIDE))
        dynamic = torch.autograd.grad(action, raw, create_graph=True)[0]
        psi, chi = box.unpack(raw, SIDE)
        gauge_psi = torch.imag(torch.sum(
            torch.conj(torch.tensor(reference_psi[anchor]))*psi[anchor]))
        gauge_chi = torch.imag(chi[exterior][2])
        return torch.cat((dynamic, gauge_psi[None], gauge_chi[None]))

    jacobian = torch.autograd.functional.jacobian(
        augmented_residual, parameters, vectorize=True).detach().numpy()
    projector = jacobian@np.linalg.pinv(jacobian, rcond=1e-10)
    identity = np.eye(jacobian.shape[0])
    raw = parameters[:-2]
    rows = []
    for name, function in (("combined_ordered_frame", combined),
                           ("separated_ordered_frame", separated)):
        source_dynamic = (-2*small.DT*torch.func.grad(function)(raw)) \
            .detach().numpy()
        source = np.r_[source_dynamic, 0.0, 0.0]
        obstruction = (identity-projector)@source
        rows.append({
            "candidate": name,
            "source_norm": float(np.linalg.norm(source)),
            "obstruction_norm": float(np.linalg.norm(obstruction)),
            "relative_range_obstruction": float(
                np.linalg.norm(obstruction)/max(np.linalg.norm(source), 1e-300)),
        })
    result = {
        "status": "V4_LOOP_CURVATURE_FREDHOLM_GATE",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "rows": rows,
        "verdict": (
            "SEPARATED_LOOP_HAS_SMOOTH_REST_BRANCH"
            if rows[1]["relative_range_obstruction"] < 1e-8 else
            "SEPARATED_LOOP_SMOOTH_BRANCH_OBSTRUCTED"),
        "claim_boundary": (
            "First-order smooth continuation at side three. A pass does not "
            "prove a finite-strength side-five branch or spectral stability."),
    }
    (HERE / "personal_touch_loop_fredholm.json").write_text(
        json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
