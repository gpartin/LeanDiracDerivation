"""Derive center-to-center carrier transport from native mode overlaps.

For each four-component copy, the kinetic inner product between the native
Psi--chi carrier modes at neighboring centers defines a normalized overlap.
Its polar factor is a native parallel-transport matrix.  Dirac and movement
targets do not enter this construction.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.linalg import polar

import canonical_noncoplanar_bifurcation_audit as native_time
import cube_incidence_extended_packet_gate as gate
import cube_incidence_nonlinear_action as action
import current_action_localized_envelope_closure as envelope
import localized_stability as native


HERE = Path(__file__).resolve().parent
SIDE = 5


def mode_matrix(basis, psi_frame, chi_frame, sites, offset, chi_scale):
    columns = []
    for component in range(4):
        dpsi = np.zeros((SIDE, SIDE, SIDE, 3), dtype=complex)
        dchi = np.zeros_like(dpsi)
        for corner, site in enumerate(sites):
            target = tuple((site[axis] + offset[axis]) % SIDE
                           for axis in range(3))
            amplitude = basis[corner, component]
            dpsi[target] += amplitude * psi_frame[corner]
            dchi[target] += chi_scale * amplitude * chi_frame[corner]
        columns.append(np.concatenate((
            dpsi.reshape(-1),
            np.sqrt(native_time.B) * dchi.reshape(-1))))
    return np.column_stack(columns)


def inverse_sqrt(matrix):
    values, vectors = np.linalg.eigh((matrix + matrix.conj().T) / 2)
    return (vectors @ np.diag(1 / np.sqrt(np.maximum(values, 1e-300)))
            @ vectors.conj().T)


def main() -> None:
    bare, _ = action.expected_blocks()
    psi_frame, chi_frame, sites = envelope.carrier_frames(bare["rest"])
    prior = json.loads(
        (HERE / "cube_incidence_dirac_envelope_gate.json").read_text())
    scales = {}
    for row in prior["rows"]:
        case = next(item for item in row["cases"]
                    if item["envelope"] == "all_125_centers")
        scales[row["flavor"]] = complex(
            case["chi_scale_real"], case["chi_scale_imag"])

    result_rows = []
    saved = {}
    for flavor, basis in gate.flavor_bases().items():
        alpha, _beta = gate.compressed_clifford(basis)
        center = mode_matrix(
            basis, psi_frame, chi_frame, sites, (0, 0, 0), scales[flavor])
        metric = center.conj().T @ center
        whiten = inverse_sqrt(metric)
        rows = []
        for offset, weight in native.DIRECTIONS:
            neighbor = mode_matrix(
                basis, psi_frame, chi_frame, sites, offset, scales[flavor])
            normalized = whiten @ (center.conj().T @ neighbor) @ whiten
            unitary, _positive = polar(normalized)
            comparisons = {
                "identity": float(np.linalg.norm(unitary - np.eye(4))),
                "minus_identity": float(np.linalg.norm(unitary + np.eye(4))),
            }
            for axis in range(3):
                comparisons[f"alpha_{axis}"] = float(np.linalg.norm(
                    unitary - alpha[axis]))
                comparisons[f"minus_alpha_{axis}"] = float(np.linalg.norm(
                    unitary + alpha[axis]))
            rows.append({
                "offset": list(offset),
                "weight": weight,
                "normalized_overlap_singular_values": [
                    float(value) for value in np.linalg.svd(
                        normalized, compute_uv=False)],
                "polar_unitarity_defect": float(np.linalg.norm(
                    unitary.conj().T @ unitary - np.eye(4))),
                "nearest_reference": min(comparisons, key=comparisons.get),
                "nearest_reference_defect": min(comparisons.values()),
                "polar_unitary_real": unitary.real.tolist(),
                "polar_unitary_imag": unitary.imag.tolist(),
            })
            key = "_".join(str(value).replace("-", "m") for value in offset)
            saved[f"{flavor}_{key}"] = unitary

        face_units = []
        for axis in range(3):
            offset = [0, 0, 0]
            offset[axis] = 1
            key = "_".join(str(value) for value in offset)
            face_units.append(saved[f"{flavor}_{key}"])
        holonomy = []
        for first in range(3):
            for second in range(first + 1, 3):
                holonomy.append({
                    "axes": [first, second],
                    "commutator_norm": float(np.linalg.norm(
                        face_units[first] @ face_units[second]
                        - face_units[second] @ face_units[first])),
                    "loop_from_identity_norm": float(np.linalg.norm(
                        face_units[first] @ face_units[second]
                        @ face_units[first].conj().T
                        @ face_units[second].conj().T - np.eye(4))),
                })
        result_rows.append({
            "flavor": flavor,
            "kinetic_gram_condition": float(np.linalg.cond(metric)),
            "neighbor_rows": rows,
            "face_connection_holonomy": holonomy,
        })

    np.savez(HERE / "cube_incidence_native_frame_connection.npz", **saved)
    result = {
        "status": "V4_NATIVE_CARRIER_FRAME_CONNECTION",
        "canonical_action_changed": False,
        "research_action_changed": False,
        "inner_product": "native kinetic metric: Psi + B chi",
        "rows": result_rows,
        "verdict": "NATIVE_FRAME_CONNECTION_DERIVED",
        "claim_boundary": (
            "Mode-overlap polar connection only. It is not a force, action "
            "change, invariant-sector proof, or movement pass."),
    }
    (HERE / "cube_incidence_native_frame_connection.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    summary = []
    for flavor in result_rows:
        summary.append({
            "flavor": flavor["flavor"],
            "kinetic_gram_condition": flavor["kinetic_gram_condition"],
            "face_neighbors": [row for row in flavor["neighbor_rows"]
                               if sum(abs(x) for x in row["offset"]) == 1],
            "holonomy": flavor["face_connection_holonomy"],
        })
    print(json.dumps({"status": result["status"], "summary": summary},
                     indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
