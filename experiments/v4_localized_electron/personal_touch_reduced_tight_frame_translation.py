"""Native translation-tangent gate for the reduced tight-frame action."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.linalg import lu_factor, lu_solve
import torch

import canonical_noncoplanar_bifurcation_audit as small
import canonical_noncoplanar_box_transfer as box
import localized_stability as native
import personal_touch_face_psd_cone as cone
import personal_touch_face_shift_invert_scan as face_names
import personal_touch_unit_geometric_canonical as canonical


HERE = Path(__file__).resolve().parent
TIGHT_TERMS = ("carrier_carrier_chi", "carrier_carrier_Psi_tight_frame")


def main() -> None:
    bare = np.load(canonical.CACHE)
    additions = np.load(
        HERE/"personal_touch_activated_bond_basis_v2_cache.npz")
    mass = bare["mass"]
    gyro = bare["gyro"]
    faces = np.load(cone.CACHE)
    dt = small.DT
    rest = bare["rest"]
    psi, chi = box.unpack(torch.tensor(rest), 5)
    translations = native.translation_vectors(
        psi.detach().numpy(), chi.detach().numpy())
    candidates = {
        "tight_frames": list(TIGHT_TERMS),
        "tight_frames_plus_interface": [
            *TIGHT_TERMS, "carrier_vacuum_chi"],
        "all_activated_bonds": [
            "vacuum_vacuum_chi", "carrier_vacuum_chi", *TIGHT_TERMS],
    }
    candidate_rows = []
    for name, terms in candidates.items():
        stiffness = bare["stiffness"]+sum(
            additions[f"hessian_{term}"] for term in terms)
        if name == "all_activated_bonds":
            stiffness += sum(
                faces[f"hessian_{term}"] for term in face_names.NAMES)
        left = mass/dt+gyro/2
        backward = mass/dt-gyro/2
        zero = dt*stiffness-2*mass/dt
        factor = lu_factor(left)
        rows = []
        images = []
        for axis in range(3):
            vector = translations[:, axis]
            following = -lu_solve(factor, zero@vector+backward@vector)
            image = np.r_[following, vector]
            target = np.r_[vector, vector]
            multiplier = float(target@image/(target@target))
            residual = float(
                np.linalg.norm(image-target)/np.linalg.norm(target))
            scalar_residual = float(
                np.linalg.norm(image-multiplier*target)/np.linalg.norm(target))
            stiffness_rayleigh = float(
                vector@stiffness@vector/(vector@mass@vector))
            rows.append({
                "axis": axis,
                "neutral_update_relative_residual": residual,
                "best_real_multiplier": multiplier,
                "best_scalar_relative_residual": scalar_residual,
                "generalized_stiffness_rayleigh": stiffness_rayleigh,
            })
            images.append(image)
        phase_space = np.column_stack(
            [np.r_[translations[:, axis], translations[:, axis]]
             for axis in range(3)])
        phase_space, _ = np.linalg.qr(phase_space)
        response = np.column_stack(images)
        leakage = response-phase_space@(phase_space.T@response)
        candidate_rows.append({
            "candidate": name,
            "terms": terms,
            "rows": rows,
            "translation_subspace_cross_block": float(
                np.linalg.norm(leakage)/np.linalg.norm(response)),
        })
    best = min(candidate_rows, key=lambda item: max(
        row["neutral_update_relative_residual"] for row in item["rows"]))
    passed = max(row["neutral_update_relative_residual"]
                 for row in best["rows"]) < 1e-8
    result = {
        "status": "V4_REDUCED_TIGHT_FRAME_TRANSLATION_TANGENT",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "candidates": candidate_rows,
        "best_candidate": best["candidate"],
        "verdict": (
            "NATIVE_NEUTRAL_TRANSLATION_MODE_PASS" if passed
            else "NATIVE_NEUTRAL_TRANSLATION_MODE_ABSENT"),
        "claim_boundary": (
            "Linear co-rotating translation tangent of the exact localized "
            "rest orbit. Absence establishes lattice pinning of this orbit, "
            "not a no-go for every finite relative-periodic branch."),
    }
    (HERE/"personal_touch_reduced_tight_frame_translation.json").write_text(
        json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
