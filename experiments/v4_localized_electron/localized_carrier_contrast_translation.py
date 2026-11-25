"""Test the carrier-versus-vacuum profile as the edge-snap embedding.

The carrier replaces ordinary vacuum chi = chi0 e3 by an in-plane companion
at each occupied corner.  Moving the carrier therefore transports not only
Psi and the companion part of chi, but also the missing vacuum-normal chi.
This script uses that exact native contrast as one indivisible local profile;
it does not add an independent field or fit a response coefficient.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np

import edge_snap_action_rank as rank
import localized_embedding_native_closure as local
import localized_embedding_translation_tangent as tangent


HERE = Path(__file__).resolve().parent
SIDE = tangent.SIDE
ORIGIN = tangent.ORIGIN


def carrier_contrast() -> tuple[np.ndarray, np.ndarray]:
    amplitude_sq = rank.R**2 / (2 + 54 / rank.R**2)
    psi, chi = rank.native.carrier_fields(3, amplitude_sq)
    sites = local.core_sites()
    psi_profile = np.stack([psi[site] for site in sites])
    vacuum = np.zeros(3, dtype=complex)
    vacuum[2] = rank.R
    chi_profile = np.stack([chi[site] - vacuum for site in sites])
    return psi_profile, chi_profile


def profile_columns(
        basis: np.ndarray, psi_profile: np.ndarray,
        chi_profile: np.ndarray,
        offset: tuple[int, int, int]) -> list[np.ndarray]:
    columns: list[np.ndarray] = []
    for component in range(basis.shape[1]):
        dpsi = np.zeros((SIDE, SIDE, SIDE, 3), dtype=complex)
        dchi = np.zeros_like(dpsi)
        for corner, bits in enumerate(itertools.product((0, 1), repeat=3)):
            site = tuple((ORIGIN + offset[axis] + bits[axis]) % SIDE
                         for axis in range(3))
            amplitude = basis[corner, component]
            dpsi[site] += amplitude * psi_profile[corner]
            dchi[site] += amplitude * chi_profile[corner]
        columns.extend((
            rank.pack(dpsi, dchi),
            rank.pack(1j * dpsi, 1j * dchi),
        ))
    return columns


def classify(
        bases: list[np.ndarray], psi_profile: np.ndarray,
        chi_profile: np.ndarray, centers: list[tuple[int, int, int]],
        translations: np.ndarray) -> dict:
    raw = np.column_stack([
        column
        for basis in bases
        for center in centers
        for column in profile_columns(
            basis, psi_profile, chi_profile, center)
    ])
    q = tangent.orth(raw)
    residual = translations - q @ (q.T @ translations)
    defects = [float(np.linalg.norm(residual[:, axis]))
               for axis in range(3)]
    return {
        "center_count": len(centers),
        "span_real_rank": int(q.shape[1]),
        "axis_relative_defects": defects,
        "worst_axis_relative_defect": float(max(defects)),
        "mean_axis_squared_defect": float(np.mean(np.square(defects))),
    }


def main() -> None:
    rank.lock_inertia()
    psi_profile, chi_profile = carrier_contrast()
    translations = tangent.carrier_translation_tangents()
    sets = tangent.center_sets()
    flavors = local.flavor_bases()
    families = {
        "plus_carrier_contrast": [flavors["plus"]],
        "minus_carrier_contrast": [flavors["minus"]],
        "both_copies_carrier_contrast": [
            flavors["plus"], flavors["minus"]],
        "full_corner_carrier_contrast": [np.eye(8, dtype=complex)],
    }
    rows: dict[str, list[dict]] = {}
    for name, bases in families.items():
        cases = []
        for set_name, centers in sets.items():
            row = classify(
                bases, psi_profile, chi_profile, centers, translations)
            row["center_set"] = set_name
            cases.append(row)
        rows[name] = cases
    face_rows = {
        name: next(row for row in cases
                   if row["center_set"] == "source_plus_faces")
        for name, cases in rows.items()
    }
    best_name = min(
        face_rows,
        key=lambda name: face_rows[name]["mean_axis_squared_defect"])
    best = face_rows[best_name]
    result = {
        "status": "V4_LOCALIZED_CARRIER_CONTRAST_TRANSLATION_AUDIT",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "profile": (
            "exact localized carrier Psi and chi minus ordinary native "
            "vacuum chi0*e3 on its eight sites"),
        "coefficient_provenance": (
            "all relative amplitudes are read from the freely prepared "
            "localized carrier and chi0=19 vacuum; none are fitted"),
        "rows": rows,
        "best_face_neighbor_family": best_name,
        "best_face_neighbor_worst_axis_defect": float(
            best["worst_axis_relative_defect"]),
        "verdict": (
            "CARRIER_CONTRAST_ENVELOPE_SPANS_TRANSLATION"
            if best["worst_axis_relative_defect"] < 1e-10 else
            "CARRIER_CONTRAST_ENVELOPE_INCOMPLETE"),
        "claim_boundary": (
            "This establishes only the kinematic tangent span of the exact "
            "native carrier contrast.  It does not prove that the action "
            "generates the hop, closes the current, or stabilizes the rest "
            "orbit."),
        "corrective_action": (
            "If exact, derive the native two-layer invariant graph and bond "
            "current on this same contrast envelope, separately for each "
            "copy and jointly.  Otherwise retain an independently moving "
            "chi-normal response in the phase-space graph."),
    }
    output = HERE / "localized_carrier_contrast_translation.json"
    output.write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
