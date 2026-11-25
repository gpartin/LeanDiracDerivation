"""Test whether translated localized embeddings span carrier center motion.

An internal four-component readout is relevant to movement only if changing
its envelope across neighboring cell centers can reproduce the native
translation tangents of the same eight-site Psi/chi carrier.  This audit
places the carrier-derived flavor embedding at the source center and nearby
centers, orthonormalizes the resulting real native-field span, and projects
the x/y/z central differences of the complete carrier onto that span.

No action or Dirac target enters this geometric compatibility test.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
from scipy.optimize import differential_evolution, minimize

import edge_snap_action_rank as rank
import localized_embedding_native_closure as local


HERE = Path(__file__).resolve().parent
SIDE = 5
ORIGIN = SIDE // 2 - 1


def local_columns(basis: np.ndarray, psi_frame: np.ndarray,
                  chi_frame: np.ndarray, chi_scale: complex,
                  offset: tuple[int, int, int]) -> list[np.ndarray]:
    columns = []
    for component in range(basis.shape[1]):
        dpsi = np.zeros((SIDE, SIDE, SIDE, 3), dtype=complex)
        dchi = np.zeros_like(dpsi)
        for corner, bits in enumerate(itertools.product((0, 1), repeat=3)):
            site = tuple((ORIGIN + offset[axis] + bits[axis]) % SIDE
                         for axis in range(3))
            amplitude = basis[corner, component]
            dpsi[site] += amplitude * psi_frame[corner]
            dchi[site] += chi_scale * amplitude * chi_frame[corner]
        columns.append(rank.pack(dpsi, dchi))
        columns.append(rank.pack(1j * dpsi, 1j * dchi))
    return columns


def orth(matrix: np.ndarray, tolerance: float = 1e-11) -> np.ndarray:
    u, singular, _ = np.linalg.svd(matrix, full_matrices=False)
    return u[:, singular > tolerance * max(float(singular[0]), 1.0)]


def center_sets() -> dict[str, list[tuple[int, int, int]]]:
    source = [(0, 0, 0)]
    faces = source + [tuple(sign if axis == index else 0 for index in range(3))
                      for axis in range(3) for sign in (-1, 1)]
    cube = list(itertools.product((-1, 0, 1), repeat=3))
    return {"source_only": source, "source_plus_faces": faces,
            "three_by_three_by_three": cube}


def carrier_translation_tangents() -> np.ndarray:
    amplitude_sq = rank.R**2 / (2 + 54 / rank.R**2)
    psi, chi = rank.native.carrier_fields(SIDE, amplitude_sq)
    columns = []
    for axis in range(3):
        dpsi = (np.roll(psi, 1, axis=axis)-np.roll(psi, -1, axis=axis))/2
        dchi = (np.roll(chi, 1, axis=axis)-np.roll(chi, -1, axis=axis))/2
        column = rank.pack(dpsi, dchi)
        columns.append(column / np.linalg.norm(column))
    return np.column_stack(columns)


def classify(basis: np.ndarray, psi_frame: np.ndarray, chi_frame: np.ndarray,
             chi_scale: complex, centers: list[tuple[int, int, int]],
             translations: np.ndarray) -> dict:
    raw = np.column_stack([
        column for center in centers
        for column in local_columns(
            basis, psi_frame, chi_frame, chi_scale, center)
    ])
    q = orth(raw)
    residual = translations - q @ (q.T @ translations)
    defects = [float(np.linalg.norm(residual[:, axis]))
               for axis in range(3)]
    return {
        "center_count": len(centers),
        "span_real_rank": int(q.shape[1]),
        "axis_relative_defects": defects,
        "worst_axis_relative_defect": max(defects),
        "mean_axis_squared_defect": float(np.mean(np.square(defects))),
    }


def classify_two_copy(bases: dict[str, np.ndarray],
                      psi_frame: np.ndarray, chi_frame: np.ndarray,
                      scales: dict[str, complex],
                      centers: list[tuple[int, int, int]],
                      translations: np.ndarray) -> dict:
    raw = np.column_stack([
        column
        for name in ("plus", "minus")
        for center in centers
        for column in local_columns(
            bases[name], psi_frame, chi_frame, scales[name], center)
    ])
    q = orth(raw)
    residual = translations - q @ (q.T @ translations)
    defects = [float(np.linalg.norm(residual[:, axis]))
               for axis in range(3)]
    return {
        "center_count": len(centers),
        "span_real_rank": int(q.shape[1]),
        "axis_relative_defects": defects,
        "worst_axis_relative_defect": max(defects),
        "mean_axis_squared_defect": float(np.mean(np.square(defects))),
    }


def optimize_scale(basis: np.ndarray, psi_frame: np.ndarray,
                   chi_frame: np.ndarray,
                   centers: list[tuple[int, int, int]],
                   translations: np.ndarray) -> complex:
    def objective(value):
        row = classify(
            basis, psi_frame, chi_frame, complex(value[0], value[1]),
            centers, translations)
        return row["mean_axis_squared_defect"]
    global_fit = differential_evolution(
        objective, [(-4, 4), (-4, 4)], seed=40613,
        maxiter=100, popsize=12, tol=1e-10, polish=False)
    local_fit = minimize(
        objective, global_fit.x, method="Nelder-Mead",
        options={"maxiter": 1500, "xatol": 1e-12, "fatol": 1e-12})
    return complex(local_fit.x[0], local_fit.x[1])


def optimize_two_copy_scales(bases: dict[str, np.ndarray],
                             psi_frame: np.ndarray, chi_frame: np.ndarray,
                             centers: list[tuple[int, int, int]],
                             translations: np.ndarray) -> dict[str, complex]:
    def objective(value):
        row = classify_two_copy(
            bases, psi_frame, chi_frame,
            {"plus": complex(value[0], value[1]),
             "minus": complex(value[2], value[3])},
            centers, translations)
        return row["mean_axis_squared_defect"]
    global_fit = differential_evolution(
        objective, [(-4, 4)] * 4, seed=40617,
        maxiter=160, popsize=16, tol=1e-10, polish=False)
    local_fit = minimize(
        objective, global_fit.x, method="Nelder-Mead",
        options={"maxiter": 3000, "xatol": 1e-12, "fatol": 1e-12})
    return {"plus": complex(local_fit.x[0], local_fit.x[1]),
            "minus": complex(local_fit.x[2], local_fit.x[3])}


def main() -> None:
    rank.lock_inertia()
    _, _, psi_frame, chi_frame = local.carrier_frames()
    translations = carrier_translation_tangents()
    sets = center_sets()
    rows = []
    bases = local.flavor_bases()
    for flavor, basis in bases.items():
        selected = optimize_scale(
            basis, psi_frame, chi_frame,
            sets["source_plus_faces"], translations)
        cases = []
        for set_name, centers in sets.items():
            row = classify(
                basis, psi_frame, chi_frame, selected, centers, translations)
            row["center_set"] = set_name
            cases.append(row)
        rows.append({
            "flavor": flavor,
            "native_geometry_selected_chi_scale_real": float(selected.real),
            "native_geometry_selected_chi_scale_imag": float(selected.imag),
            "cases": cases,
        })
    scale_controls = {
        "same_joint": {"plus": 1 + 0j, "minus": 1 + 0j},
        "opposite_joint": {"plus": 1 + 0j, "minus": -1 + 0j},
        "psi_vs_joint": {"plus": 0 + 0j, "minus": 1 + 0j},
        "two_layer_native_control": {
            "plus": complex(-0.8396226481, -0.0038803773),
            "minus": complex(-0.8523803568, -0.0012860747),
        },
    }
    scale_control_rows = {}
    for control_name, control_scales in scale_controls.items():
        scale_control_rows[control_name] = classify_two_copy(
            bases, psi_frame, chi_frame, control_scales,
            sets["source_plus_faces"], translations)
    best_control = min(
        scale_control_rows,
        key=lambda name: scale_control_rows[name]["mean_axis_squared_defect"])
    two_scales = scale_controls[best_control]
    two_cases = []
    for set_name, centers in sets.items():
        row = classify_two_copy(
            bases, psi_frame, chi_frame, two_scales, centers, translations)
        row["center_set"] = set_name
        two_cases.append(row)
    two_copy = {
        "selected_control": best_control,
        "scale_control_face_rows": scale_control_rows,
        "scales": {
            name: {"real": float(value.real), "imag": float(value.imag)}
            for name, value in two_scales.items()
        },
        "cases": two_cases,
    }
    best_face = min(
        case["worst_axis_relative_defect"]
        for row in rows for case in row["cases"]
        if case["center_set"] == "source_plus_faces")
    two_face = next(case for case in two_cases
                    if case["center_set"] == "source_plus_faces")
    result = {
        "status": "V4_LOCALIZED_EMBEDDING_TRANSLATION_TANGENT_AUDIT",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "rows": rows,
        "two_copy_joint_envelope": two_copy,
        "best_face_neighbor_worst_axis_defect": float(best_face),
        "two_copy_face_neighbor_worst_axis_defect": float(
            two_face["worst_axis_relative_defect"]),
        "verdict": (
            "TWO_COPY_NEAREST_CENTER_ENVELOPE_SPANS_TRANSLATION_TANGENTS"
            if two_face["worst_axis_relative_defect"] < 1e-8 else
            ("SINGLE_COPY_NEAREST_CENTER_ENVELOPE_SPANS_TRANSLATION_TANGENTS"
             if best_face < 1e-8 else
             "LOCALIZED_EMBEDDING_TRANSLATION_TANGENT_MISMATCH")),
        "interpretation": (
            "A pass means the carrier's infinitesimal physical center motion "
            "can be represented by redistributing the retained joint Psi/chi "
            "envelope among adjacent cube centers. The two-copy row directly "
            "tests whether distinct native chi responses make the algebraic "
            "copies complementary for motion. A pass does not "
            "mean the nonlinear action executes that redistribution."),
        "claim_boundary": (
            "This is a geometric tangent-span test. It contains no native "
            "time evolution, current, stability, or finite translation."),
        "corrective_action": (
            "On pass, compress the native two-layer update and bond current "
            "on the neighboring-center envelope. On failure, enlarge the "
            "carrier-derived spatial basis before changing the action."),
    }
    (HERE / "localized_embedding_translation_tangent.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
