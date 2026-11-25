"""Test whether independent native Psi and chi amplitudes span translation.

The preceding localized envelope tied the chi perturbation to the same
four-component amplitude as Psi.  That is appropriate for a rest-sector
graph, but it is not a valid assumption for physical center motion: the
substrate can turn and move independently while the carrier is handed across
a shared face.

This audit keeps the carrier-derived Psi and chi frames, but gives their
amplitudes independent real coordinates.  It compares one flavor copy, both
copies, and the complete eight-corner scalar amplitude space.  No action,
Dirac matrix, target force, or evolution rule is introduced.
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


def independent_local_columns(
        basis: np.ndarray, psi_frame: np.ndarray, chi_frame: np.ndarray,
        offset: tuple[int, int, int], *, include_chi_normal: bool = False,
        full_native_frame: bool = False) -> list[np.ndarray]:
    """Return independent real/imag Psi and chi columns at one cube center."""
    columns: list[np.ndarray] = []
    for component in range(basis.shape[1]):
        dpsi = np.zeros((SIDE, SIDE, SIDE, 3), dtype=complex)
        dchi = np.zeros_like(dpsi)
        for corner, bits in enumerate(itertools.product((0, 1), repeat=3)):
            site = tuple((ORIGIN + offset[axis] + bits[axis]) % SIDE
                         for axis in range(3))
            amplitude = basis[corner, component]
            dpsi[site] += amplitude * psi_frame[corner]
            dchi[site] += amplitude * chi_frame[corner]
        zero_psi = np.zeros_like(dpsi)
        zero_chi = np.zeros_like(dchi)
        columns.extend((
            rank.pack(dpsi, zero_chi),
            rank.pack(1j * dpsi, zero_chi),
            rank.pack(zero_psi, dchi),
            rank.pack(zero_psi, 1j * dchi),
        ))
        if include_chi_normal or full_native_frame:
            dchi_normal = np.zeros_like(dchi)
            for corner, bits in enumerate(
                    itertools.product((0, 1), repeat=3)):
                site = tuple((ORIGIN + offset[axis] + bits[axis]) % SIDE
                             for axis in range(3))
                dchi_normal[site][2] += basis[corner, component]
            columns.extend((
                rank.pack(zero_psi, dchi_normal),
                rank.pack(zero_psi, 1j * dchi_normal),
            ))
        if full_native_frame:
            dpsi_companion = np.zeros_like(dpsi)
            dpsi_normal = np.zeros_like(dpsi)
            dchi_ray = np.zeros_like(dchi)
            for corner, bits in enumerate(
                    itertools.product((0, 1), repeat=3)):
                site = tuple((ORIGIN + offset[axis] + bits[axis]) % SIDE
                             for axis in range(3))
                amplitude = basis[corner, component]
                dpsi_companion[site] += amplitude * chi_frame[corner]
                dpsi_normal[site][2] += amplitude
                dchi_ray[site] += amplitude * psi_frame[corner]
            columns.extend((
                rank.pack(dpsi_companion, zero_chi),
                rank.pack(1j * dpsi_companion, zero_chi),
                rank.pack(dpsi_normal, zero_chi),
                rank.pack(1j * dpsi_normal, zero_chi),
                rank.pack(zero_psi, dchi_ray),
                rank.pack(zero_psi, 1j * dchi_ray),
            ))
    return columns


def classify(
        bases: list[np.ndarray], psi_frame: np.ndarray,
        chi_frame: np.ndarray, centers: list[tuple[int, int, int]],
        translations: np.ndarray, *, include_chi_normal: bool = False,
        full_native_frame: bool = False) -> dict:
    raw = np.column_stack([
        column
        for basis in bases
        for center in centers
        for column in independent_local_columns(
            basis, psi_frame, chi_frame, center,
            include_chi_normal=include_chi_normal,
            full_native_frame=full_native_frame)
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
    _, _, psi_frame, chi_frame = local.carrier_frames()
    translations = tangent.carrier_translation_tangents()
    center_sets = tangent.center_sets()
    flavors = local.flavor_bases()
    families = {
        "plus_independent_Psi_chi": {
            "bases": [flavors["plus"]]},
        "minus_independent_Psi_chi": {
            "bases": [flavors["minus"]]},
        "both_copies_independent_Psi_chi": {
            "bases": [flavors["plus"], flavors["minus"]]},
        "full_corner_independent_Psi_chi": {
            "bases": [np.eye(8, dtype=complex)]},
        "full_corner_plus_chi_vacuum_normal": {
            "bases": [np.eye(8, dtype=complex)],
            "include_chi_normal": True},
        "full_corner_full_native_C3": {
            "bases": [np.eye(8, dtype=complex)],
            "full_native_frame": True},
    }
    rows: dict[str, list[dict]] = {}
    for family_name, specification in families.items():
        cases = []
        for set_name, centers in center_sets.items():
            row = classify(
                specification["bases"], psi_frame, chi_frame, centers,
                translations,
                include_chi_normal=specification.get(
                    "include_chi_normal", False),
                full_native_frame=specification.get(
                    "full_native_frame", False))
            row["center_set"] = set_name
            cases.append(row)
        rows[family_name] = cases

    face_rows = {
        name: next(row for row in cases
                   if row["center_set"] == "source_plus_faces")
        for name, cases in rows.items()
    }
    best_name = min(
        face_rows,
        key=lambda name: face_rows[name]["mean_axis_squared_defect"])
    best = face_rows[best_name]
    exact = best["worst_axis_relative_defect"] < 1e-10
    result = {
        "status": "V4_LOCALIZED_INDEPENDENT_FIELD_TRANSLATION_TANGENT_AUDIT",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "target_pattern": (
            "physical center-translation tangents of the same eight-site "
            "joint Psi-chi carrier in all three axes"),
        "rows": rows,
        "best_face_neighbor_family": best_name,
        "best_face_neighbor_worst_axis_defect": float(
            best["worst_axis_relative_defect"]),
        "verdict": (
            "INDEPENDENT_NATIVE_FIELDS_SPAN_TRANSLATION_TANGENTS"
            if exact else
            "CARRIER_FRAME_OR_PHASE_SPACE_EXTENSION_REQUIRED"),
        "interpretation": (
            "This removes the unproved assumption that chi must use the "
            "same amplitude as Psi during a handoff.  An exact result shows "
            "that no extra spatial field is needed to represent infinitesimal "
            "center motion; it does not show that the action generates it."),
        "claim_boundary": (
            "This is a configuration-space tangent-span audit.  It contains "
            "no time evolution, bond-current closure, stability result, or "
            "finite translation."),
        "corrective_action": (
            "If exact, derive the native two-layer graph and current on this "
            "independent-field envelope.  If nonzero even for the full-corner "
            "control, add the carrier-derived frame/bond direction before "
            "testing another action."),
    }
    output = HERE / "localized_embedding_independent_fields_tangent.json"
    output.write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
