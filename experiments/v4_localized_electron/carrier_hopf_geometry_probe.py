"""Audit the basis-free orientation geometry latent in the eight corner rays.

This is a kinematic diagnostic, not an action candidate.  The prepared rays
live in a common complex two-plane.  A normalized ray z determines the
rank-one Hermitian projector z z^dagger; subtracting half the identity gives
a traceless Hermitian operator, whose three real coordinates are the usual
Hopf/Bloch orientation.  Pairwise Hilbert--Schmidt products are basis
independent, so the audit does not depend on a chosen set of Pauli matrices.

The result is useful only if it diagnoses the existing transport obstruction.
It must not be installed as a force, shift, projector, or runtime readout.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "carrier_hopf_geometry_probe.json"
TOL = 2.0e-12


def prepared_rays() -> np.ndarray:
    omega = np.exp(2j * np.pi / 3)
    u0 = np.array([1.0, 0.0], dtype=np.complex128)
    u1 = np.array([1 / np.sqrt(3), np.sqrt(2 / 3)], dtype=np.complex128)
    u2 = np.array(
        [1 / np.sqrt(3), np.sqrt(2 / 3) * omega], dtype=np.complex128
    )
    u3 = np.array(
        [1 / np.sqrt(3), np.sqrt(2 / 3) * np.conjugate(omega)],
        dtype=np.complex128,
    )
    return np.stack([u0, u1, u2, u3, -u3, -u2, -u1, -u0])


def projector(z: np.ndarray) -> np.ndarray:
    z = z / np.linalg.norm(z)
    return np.outer(z, np.conjugate(z))


def bloch(z: np.ndarray) -> np.ndarray:
    """Coordinates used only for the reported rank/frame diagnostic."""
    a, b = z / np.linalg.norm(z)
    return np.array(
        [
            2 * np.real(np.conjugate(a) * b),
            2 * np.imag(np.conjugate(a) * b),
            np.abs(a) ** 2 - np.abs(b) ** 2,
        ],
        dtype=np.float64,
    )


def projector_orientation_inner(pi: np.ndarray, pj: np.ndarray) -> float:
    """Bloch-vector dot product evaluated without selecting Pauli matrices."""
    # For normalized C2 rays, n_i.n_j = 2 Tr(P_i P_j) - 1.
    return float(np.real(2 * np.trace(pi @ pj) - 1))


def main() -> None:
    rays = prepared_rays()
    projectors = np.stack([projector(z) for z in rays])
    orientations = np.stack([bloch(z) for z in rays])
    corners = list(itertools.product((0, 1), repeat=3))

    by_distance: dict[str, dict[str, object]] = {}
    all_pairs: list[dict[str, object]] = []
    for distance in (1, 2, 3):
        values: list[float] = []
        for i in range(8):
            for j in range(i + 1, 8):
                hamming = sum(a != b for a, b in zip(corners[i], corners[j]))
                if hamming != distance:
                    continue
                value = projector_orientation_inner(projectors[i], projectors[j])
                values.append(value)
                all_pairs.append(
                    {
                        "i": i,
                        "j": j,
                        "hamming_distance": distance,
                        "orientation_inner": value,
                    }
                )
        target = -1 / 3 if distance in (1, 2) else 1.0
        by_distance[str(distance)] = {
            "pair_count": len(values),
            "target": target,
            "minimum": min(values),
            "maximum": max(values),
            "max_abs_error": max(abs(v - target) for v in values),
        }

    sign_pair_defects = [
        float(np.linalg.norm(projectors[i] - projectors[7 - i]))
        for i in range(4)
    ]
    frame = orientations.T @ orientations
    frame_target = (8 / 3) * np.eye(3)
    c19_pair_errors = [
        abs(row["orientation_inner"] + 1 / 3)
        for row in all_pairs
        if row["hamming_distance"] in (1, 2)
    ]
    body_pair_errors = [
        abs(row["orientation_inner"] - 1.0)
        for row in all_pairs
        if row["hamming_distance"] == 3
    ]

    checks = {
        "all_ray_norms_unity": float(
            np.max(np.abs(np.sum(np.abs(rays) ** 2, axis=1) - 1))
        )
        < TOL,
        "sign_pairs_identical_as_projectors": max(sign_pair_defects) < TOL,
        "orientation_span_is_three": int(np.linalg.matrix_rank(orientations, TOL))
        == 3,
        "orientation_sum_is_zero": float(np.linalg.norm(np.sum(orientations, axis=0)))
        < TOL,
        "orientation_frame_is_isotropic": float(np.linalg.norm(frame - frame_target))
        < TOL,
        "all_c19_corner_pairs_are_tetrahedral": max(c19_pair_errors) < TOL,
        "all_excluded_body_diagonals_identify_same_orientation": max(
            body_pair_errors
        )
        < TOL,
    }

    result = {
        "schema": "lfm.v4.carrier_hopf_geometry_probe.v1",
        "scope": "kinematic geometry only; no action or transport claim",
        "construction": {
            "basis_free_object": "P(z)=z z^dagger for each normalized C2 ray",
            "orientation_inner": "2 Tr(P_i P_j)-1",
            "corner_order": [list(c) for c in corners],
            "c19_corner_adjacency": "Hamming distance 1 (faces) or 2 (edges)",
            "excluded_corner_adjacency": "Hamming distance 3 (body diagonal)",
        },
        "by_hamming_distance": by_distance,
        "sign_pair_projector_defects": sign_pair_defects,
        "orientation_rank": int(np.linalg.matrix_rank(orientations, TOL)),
        "orientation_sum": np.sum(orientations, axis=0).tolist(),
        "orientation_frame": frame.tolist(),
        "orientation_frame_target": frame_target.tolist(),
        "checks": checks,
        "verdict": (
            "C19_TETRAHEDRAL_ORIENTATION_GEOMETRY_PASS"
            if all(checks.values())
            else "C19_TETRAHEDRAL_ORIENTATION_GEOMETRY_FAIL"
        ),
        "claim_boundary": {
            "proved_by_this_probe": [
                "the signed eight-ray preparation is a two-to-one lift of four orientations",
                "those four orientations span three real dimensions as an isotropic tight frame",
                "every C19 face/edge corner pair has orientation inner product -1/3",
                "every excluded body-diagonal pair has identical orientation",
            ],
            "not_proved": [
                "that the native action derives this orientation as a dynamical field",
                "that the localized rest orbit is stable",
                "that phase charge moves in all three directions",
                "that either Clifford copy is a leading or trailing physical particle",
                "that the carrier completes an unrestricted edge-to-edge handoff",
            ],
            "admissible_use": (
                "screen basis-free reciprocal actions built from native projectors or "
                "frame connections; never install a Pauli map or conditional shift"
            ),
        },
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(result["verdict"])
    print(json.dumps(by_distance, indent=2))


if __name__ == "__main__":
    main()
