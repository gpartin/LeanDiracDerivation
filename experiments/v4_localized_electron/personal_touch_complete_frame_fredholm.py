"""First-order branch-solvability test for the complete-frame action."""

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


def main() -> None:
    reference = np.load(
        HERE / "canonical_noncoplanar_solvability_search.npz")["rest"]
    reference_psi, _reference_chi = small.unpack(reference)
    theta = float(np.arccos(1-small.DT**2*small.CHI0**2/2))
    parameters = torch.tensor(np.r_[reference, theta, 0.0],
                              dtype=torch.float64, requires_grad=True)
    anchor, exterior = (0, 0, 0), (2, 2, 2)

    def augmented_residual(value: torch.Tensor) -> torch.Tensor:
        raw, tp, tc = value[:-2], value[-2], value[-1]
        previous = branch.phase_rotate(raw, tp, tc, -1)
        following = branch.phase_rotate(raw, tp, tc, 1)
        action = box.interval(previous, raw, SIDE) + box.interval(
            raw, following, SIDE)
        dynamic = torch.autograd.grad(action, raw, create_graph=True)[0]
        psi, chi = box.unpack(raw, SIDE)
        gauge_psi = torch.imag(torch.sum(
            torch.conj(torch.tensor(reference_psi[anchor]))*psi[anchor]))
        gauge_chi = torch.imag(chi[exterior][2])
        return torch.cat((dynamic, gauge_psi[None], gauge_chi[None]))

    jacobian = torch.autograd.functional.jacobian(
        augmented_residual, parameters, vectorize=True).detach().numpy()
    raw = parameters[:-2]
    # Two adjacent trapezoidal intervals contribute -2 dt U_frame.
    source_dynamic = (-2*small.DT*torch.func.grad(
        branch.complete_frame_energy)(raw)).detach().numpy()
    source = np.r_[source_dynamic, 0.0, 0.0]
    solution, _residuals, rank, singular = np.linalg.lstsq(
        jacobian, -source, rcond=1e-10)
    obstruction = source + jacobian@solution
    dynamic = obstruction[:-2].reshape(SIDE, SIDE, SIDE, 4, 3)
    psi_power = np.sum(dynamic[..., :2, :]**2, axis=(-2, -1))
    chi_power = np.sum(dynamic[..., 2:, :]**2, axis=(-2, -1))
    total = float(np.sum(psi_power)+np.sum(chi_power))
    occupied = np.sum(abs(reference_psi)**2, axis=-1) > 1e-12
    result = {
        "status": "V4_PERSONAL_TOUCH_COMPLETE_FRAME_FREDHOLM",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "augmented_dimension": int(jacobian.shape[0]),
        "augmented_rank": int(rank),
        "augmented_nullity": int(jacobian.shape[1]-rank),
        "largest_singular_value": float(singular[0]),
        "smallest_retained_singular_value": float(singular[rank-1]),
        "unit_strength_source_norm": float(np.linalg.norm(source)),
        "unit_strength_range_obstruction_norm": float(
            np.linalg.norm(obstruction)),
        "relative_range_obstruction": float(
            np.linalg.norm(obstruction)/np.linalg.norm(source)),
        "predicted_DEL_rms_at_strength_0p001": float(
            0.001*np.linalg.norm(obstruction[:-2])
            / np.sqrt(source_dynamic.size)),
        "obstruction_Psi_fraction": float(np.sum(psi_power)/total),
        "obstruction_chi_fraction": float(np.sum(chi_power)/total),
        "obstruction_carrier_fraction": float(
            (np.sum(psi_power[occupied])+np.sum(chi_power[occupied]))/total),
        "verdict": (
            "COMPLETE_FRAME_SMOOTH_BRANCH_FREDHOLM_OBSTRUCTED"
            if np.linalg.norm(obstruction)/np.linalg.norm(source) > 1e-8
            else "COMPLETE_FRAME_SMOOTH_BRANCH_FIRST_ORDER_SOLVABLE"),
        "claim_boundary": (
            "First-order smooth continuation from the exact canonical orbit "
            "with free fields and two global phase advances. It does not "
            "exclude a disconnected branch or a more general multi-frequency "
            "orbit."),
    }
    output = HERE / "personal_touch_complete_frame_fredholm.json"
    output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n",
                      encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
