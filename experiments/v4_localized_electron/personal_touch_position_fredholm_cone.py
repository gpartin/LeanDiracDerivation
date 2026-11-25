"""Positive-cone solvability of the complete-frame position-action basis."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import minimize

import canonical_noncoplanar_bifurcation_audit as small
import canonical_noncoplanar_box_transfer as box
import personal_touch_action_basis_screen as screen
import personal_touch_complete_frame_branch as branch


HERE = Path(__file__).resolve().parent
SIDE = 3


def side3_local_area(raw: torch.Tensor) -> torch.Tensor:
    psi, chi = box.unpack(raw, SIDE)
    area = torch.linalg.cross(chi, psi, dim=-1)
    return torch.sum(torch.real(torch.sum(torch.conj(area)*area, dim=-1)))


def side3_bond_difference(field: torch.Tensor) -> torch.Tensor:
    value = torch.zeros((), dtype=torch.float64)
    import localized_stability as native
    for offset, weight in native.DIRECTIONS:
        delta = field-torch.roll(field, shifts=offset, dims=(0, 1, 2))
        value += weight*torch.sum(torch.real(torch.sum(
            torch.conj(delta)*delta, dim=(-2, -1))))
    return value/(4*small.CHI0**2)


def side3_ordered(raw: torch.Tensor) -> torch.Tensor:
    psi, chi = box.unpack(raw, SIDE)
    return (side3_bond_difference(screen.dyad(psi))
            + side3_bond_difference(screen.dyad(chi)))


def side3_chi_orientation(raw: torch.Tensor) -> torch.Tensor:
    _psi, chi = box.unpack(raw, SIDE)
    return side3_bond_difference(screen.dyad(chi))


def side3_cross(raw: torch.Tensor) -> torch.Tensor:
    psi, chi = box.unpack(raw, SIDE)
    return side3_bond_difference(screen.cross_dyad(psi, chi))


CANDIDATES = {
    "local_Gram_area": side3_local_area,
    "C19_ordered_Psi_chi_dyads": side3_ordered,
    "C19_chi_orientation_dyad": side3_chi_orientation,
    "C19_cross_dyad": side3_cross,
}


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
    projector = jacobian @ np.linalg.pinv(jacobian, rcond=1e-10)
    identity = np.eye(jacobian.shape[0])
    sources, obstructions, rows = [], [], []
    for name, function in CANDIDATES.items():
        source_dynamic = (-2*small.DT*torch.func.grad(function)(raw)) \
            .detach().numpy()
        source = np.r_[source_dynamic, 0.0, 0.0]
        obstruction = (identity-projector)@source
        sources.append(source)
        obstructions.append(obstruction)
        rows.append({
            "candidate": name,
            "source_norm": float(np.linalg.norm(source)),
            "obstruction_norm": float(np.linalg.norm(obstruction)),
            "relative_obstruction": float(
                np.linalg.norm(obstruction)/max(np.linalg.norm(source), 1e-300)),
        })
    matrix = np.column_stack(obstructions)
    norms = np.linalg.norm(matrix, axis=0)
    normalized = matrix/np.where(norms > 1e-300, norms, 1)

    def objective(coefficients):
        return float(np.linalg.norm(normalized@coefficients)**2)

    constraints = ({"type": "eq", "fun": lambda x: np.sum(x)-1},)
    best = None
    for seed in np.eye(len(CANDIDATES)):
        solution = minimize(objective, seed, method="SLSQP",
                            bounds=[(0, 1)]*len(CANDIDATES),
                            constraints=constraints,
                            options={"ftol": 1e-18, "maxiter": 2000})
        if best is None or solution.fun < best.fun:
            best = solution
    normalized_coefficients = np.maximum(best.x, 0)
    physical_coefficients = normalized_coefficients/np.where(
        norms > 1e-300, norms, 1)
    physical_coefficients /= max(np.max(physical_coefficients), 1e-300)
    combined_source = np.column_stack(sources)@physical_coefficients
    combined_obstruction = (identity-projector)@combined_source
    result = {
        "status": "V4_PERSONAL_TOUCH_POSITION_FREDHOLM_CONE",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "candidates": list(CANDIDATES),
        "rows": rows,
        "obstruction_singular_values": np.linalg.svd(
            normalized, compute_uv=False).tolist(),
        "best_positive_normalized_weights": normalized_coefficients.tolist(),
        "corresponding_relative_action_coefficients": physical_coefficients.tolist(),
        "best_normalized_cone_residual": float(np.sqrt(best.fun)),
        "combined_relative_range_obstruction": float(
            np.linalg.norm(combined_obstruction)
            / max(np.linalg.norm(combined_source), 1e-300)),
        "verdict": (
            "POSITIVE_POSITION_FRAME_COMBINATION_FIRST_ORDER_SOLVABLE"
            if np.linalg.norm(combined_obstruction)
            / max(np.linalg.norm(combined_source), 1e-300) < 1e-8
            else "POSITIVE_POSITION_FRAME_CONE_FREDHOLM_OBSTRUCTED"),
        "claim_boundary": (
            "Positive cone of the four declared lowest-degree position-only "
            "frame energies at the canonical orbit. This does not exclude "
            "higher-degree, multi-site, or velocity-dependent actions."),
    }
    output = HERE / "personal_touch_position_fredholm_cone.json"
    output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n",
                      encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
