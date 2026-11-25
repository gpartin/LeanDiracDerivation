"""Inverse response-rank test for native translation under geometric actions."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.linalg import lu_factor, lu_solve
from scipy.optimize import lsq_linear
import torch

import canonical_noncoplanar_bifurcation_audit as small
import canonical_noncoplanar_box_transfer as box
import localized_stability as native
import personal_touch_activated_bond_exact_scan as bond_names
import personal_touch_face_psd_cone as cone
import personal_touch_face_shift_invert_scan as face_names
import personal_touch_unit_geometric_canonical as canonical


HERE = Path(__file__).resolve().parent


def main() -> None:
    bare = np.load(canonical.CACHE)
    faces = np.load(cone.CACHE)
    bonds = np.load(HERE/"personal_touch_activated_bond_basis_v2_cache.npz")
    names = ([f"face:{name}" for name in face_names.NAMES]
             + [f"bond:{name}" for name in bond_names.ACTIVATED_NAMES])
    hessians = ([faces[f"hessian_{name}"] for name in face_names.NAMES]
                + [bonds[f"hessian_{name}"]
                   for name in bond_names.ACTIVATED_NAMES])
    mass, gyro, stiffness = bare["mass"], bare["gyro"], bare["stiffness"]
    dt = small.DT
    left = mass/dt+gyro/2
    backward = mass/dt-gyro/2
    zero = dt*stiffness-2*mass/dt
    factor = lu_factor(left)
    psi, chi = box.unpack(torch.tensor(bare["rest"]), 5)
    translations = native.translation_vectors(
        psi.detach().numpy(), chi.detach().numpy())
    base_blocks = []
    response_columns = [[] for _ in hessians]
    for axis in range(3):
        vector = translations[:, axis]
        following = -lu_solve(factor, zero@vector+backward@vector)
        base_blocks.append(following-vector)
        for index, hessian in enumerate(hessians):
            response_columns[index].append(
                -lu_solve(factor, dt*hessian@vector))
    target = -np.concatenate(base_blocks)
    response = np.column_stack(
        [np.concatenate(blocks) for blocks in response_columns])
    positive = lsq_linear(
        response, target, bounds=(0, np.inf), tol=1e-13,
        max_iter=500, lsmr_tol=1e-13)
    free, *_ = np.linalg.lstsq(response, target, rcond=1e-12)

    def row(label: str, coefficients: np.ndarray) -> dict:
        residual = response@coefficients-target
        return {
            "fit": label,
            "coefficients": {
                name: float(value) for name, value in zip(
                    names, coefficients, strict=True) if abs(value) > 1e-12},
            "relative_translation_response_residual": float(
                np.linalg.norm(residual)/np.linalg.norm(target)),
            "absolute_translation_response_residual": float(
                np.linalg.norm(residual)),
        }

    positive_row = row("nonnegative", positive.x)
    free_row = row("unconstrained", free)
    result = {
        "status": "V4_TRANSLATION_INVERSE_ACTION_RANK",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "basis": names,
        "response_rank": int(np.linalg.matrix_rank(response, tol=1e-10)),
        "target_norm": float(np.linalg.norm(target)),
        "positive_solver_success": bool(positive.success),
        "rows": [positive_row, free_row],
        "verdict": (
            "POSITIVE_POSITION_ACTION_TRANSLATION_RESPONSE_EXISTS"
            if positive_row["relative_translation_response_residual"] < 1e-8
            else "POSITIVE_POSITION_ACTION_BASIS_DEFICIENT"),
        "claim_boundary": (
            "Linear one-update inverse response on three centered-translation "
            "directions. Deficiency retires this nine-term positive position "
            "basis for neutral translation; it does not exclude higher-site "
            "or velocity-dependent reciprocal actions or finite branches."),
    }
    (HERE/"personal_touch_translation_inverse_action.json").write_text(
        json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
