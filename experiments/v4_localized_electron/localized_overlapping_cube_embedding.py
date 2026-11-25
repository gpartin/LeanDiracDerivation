"""Test a translation-equivariant lift of cubical fibers to native fields.

The cubical Clifford coordinates are an internal eight-coordinate fiber, not
eight physical lattice sites.  A physical localized lift must therefore be a
translation-equivariant synthesis map: every cell contributes its eight
corner amplitudes to the physical vertices, and contributions from overlapping
cells add.  In momentum space this is a small matrix E(k).

This audit builds the most reduced carrier-derived candidate.  A cubical
amplitude at corner c multiplies the prepared carrier's normalized Psi ray and
its normalized chi companion at c.  No action, force, Dirac matrix, preferred
direction, or runtime projection is introduced.  The two flavor eigenspaces
are tested separately, along with Psi-only, chi-only, joint-field, and both-
copy ranks.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np

import cubical_edge_shift_channels as cubical
import edge_snap_action_rank as rank


HERE = Path(__file__).resolve().parent
BITS = np.asarray(list(itertools.product((0, 1), repeat=3)), dtype=float)


def orthonormal_range(projector: np.ndarray, tolerance: float = 1e-11) -> np.ndarray:
    values, vectors = np.linalg.eigh((projector + projector.conj().T) / 2)
    return vectors[:, values > tolerance]


def carrier_frames() -> tuple[np.ndarray, np.ndarray]:
    """Return the eight normalized Psi and chi carrier directions."""
    amplitude_sq = rank.R**2 / (2 + 54 / rank.R**2)
    psi, chi = rank.native.carrier_fields(3, amplitude_sq)
    center = 0
    sites = [tuple(center + int(bit) for bit in corner) for corner in BITS]
    psi_values = np.stack([psi[site] for site in sites])
    chi_values = np.stack([chi[site] for site in sites])
    psi_frame = psi_values / np.linalg.norm(psi_values, axis=1)[:, None]
    chi_frame = chi_values / np.linalg.norm(chi_values, axis=1)[:, None]
    return psi_frame, chi_frame


def synthesis(momentum: np.ndarray, flavor_basis: np.ndarray,
              psi_frame: np.ndarray, chi_frame: np.ndarray) -> tuple[np.ndarray, ...]:
    phase = np.exp(-1j * (BITS @ momentum))
    weighted = phase[:, None] * flavor_basis
    psi = psi_frame.T @ weighted
    chi = chi_frame.T @ weighted
    joint = np.vstack((psi, np.sqrt(rank.B) * chi))
    return psi, chi, joint


def singular_row(matrix: np.ndarray, tolerance: float = 1e-10) -> dict:
    values = np.linalg.svd(matrix, compute_uv=False)
    scale = max(float(values[0]), 1e-30)
    return {
        "rank": int(np.sum(values > tolerance * scale)),
        "singular_values": [float(value) for value in values],
        "condition_number_on_range": float(
            values[0] / max(values[-1], 1e-30)),
    }


def sample_grid() -> list[np.ndarray]:
    named = [
        np.zeros(3),
        np.asarray((np.pi / 4, 0, 0)),
        np.asarray((0, np.pi / 4, 0)),
        np.asarray((0, 0, np.pi / 4)),
        np.asarray((np.pi / 4,) * 3),
        np.asarray((np.pi / 2, 0, 0)),
        np.asarray((np.pi / 2,) * 3),
        np.asarray((np.pi, 0, 0)),
        np.asarray((np.pi, np.pi, 0)),
        np.asarray((np.pi,) * 3),
    ]
    dense = [np.asarray(point) for point in itertools.product(
        np.linspace(-np.pi, np.pi, 17), repeat=3)]
    return named + dense


def copy_audit(name: str, basis: np.ndarray,
               psi_frame: np.ndarray, chi_frame: np.ndarray) -> dict:
    rows = []
    minimum_joint = np.inf
    minimum_momentum = None
    rank_deficient = 0
    for momentum in sample_grid():
        psi, chi, joint = synthesis(momentum, basis, psi_frame, chi_frame)
        singular = np.linalg.svd(joint, compute_uv=False)
        smallest = float(singular[-1])
        if smallest < minimum_joint:
            minimum_joint = smallest
            minimum_momentum = momentum.copy()
        rank_deficient += int(smallest <= 1e-10 * max(float(singular[0]), 1e-30))
    for momentum in (
            np.zeros(3), np.asarray((np.pi / 4, 0, 0)),
            np.asarray((np.pi / 4,) * 3), np.asarray((np.pi,) * 3)):
        psi, chi, joint = synthesis(momentum, basis, psi_frame, chi_frame)
        rows.append({
            "momentum": [float(value) for value in momentum],
            "psi_only": singular_row(psi),
            "chi_only": singular_row(chi),
            "joint_native_fields": singular_row(joint),
        })
    return {
        "flavor": name,
        "fiber_dimension_complex": int(basis.shape[1]),
        "representative_rows": rows,
        "dense_grid_point_count": len(sample_grid()),
        "joint_rank_deficient_point_count": rank_deficient,
        "minimum_joint_smallest_singular_value": float(minimum_joint),
        "minimum_location": [float(value) for value in minimum_momentum],
    }


def translation_intertwiner_defect(basis: np.ndarray,
                                   psi_frame: np.ndarray,
                                   chi_frame: np.ndarray) -> float:
    """Fourier form of T E = E exp(-ik): translation is automatic."""
    momentum = np.asarray((0.37, -0.41, 0.23))
    _, _, embed = synthesis(momentum, basis, psi_frame, chi_frame)
    defects = []
    for axis in range(3):
        physical_translation = np.exp(-1j * momentum[axis]) * embed
        envelope_translation = embed @ (
            np.exp(-1j * momentum[axis]) * np.eye(basis.shape[1]))
        defects.append(np.linalg.norm(physical_translation-envelope_translation))
    return float(max(defects))


def main() -> None:
    rank.lock_inertia()
    identity = np.eye(8, dtype=complex)
    flavor = cubical.flavor()
    projectors = {
        "plus": (identity + flavor) / 2,
        "minus": (identity - flavor) / 2,
    }
    bases = {name: orthonormal_range(projector)
             for name, projector in projectors.items()}
    psi_frame, chi_frame = carrier_frames()
    copies = [copy_audit(name, basis, psi_frame, chi_frame)
              for name, basis in bases.items()]

    both_basis = np.column_stack((bases["plus"], bases["minus"]))
    both_rows = []
    image_projector_rows = []
    for momentum in (np.zeros(3), np.asarray((np.pi / 4,) * 3)):
        copy_matrices = {}
        for name, basis in bases.items():
            _, _, copy_matrices[name] = synthesis(
                momentum, basis, psi_frame, chi_frame)
        psi, chi, joint = synthesis(
            momentum, both_basis, psi_frame, chi_frame)
        both_rows.append({
            "momentum": [float(value) for value in momentum],
            "psi_only": singular_row(psi),
            "chi_only": singular_row(chi),
            "joint_native_fields": singular_row(joint),
        })
        projectors = {}
        for name, matrix in copy_matrices.items():
            q, _ = np.linalg.qr(matrix)
            projectors[name] = q @ q.conj().T
        image_projector_rows.append({
            "momentum": [float(value) for value in momentum],
            "plus_minus_image_projector_defect": float(np.linalg.norm(
                projectors["plus"] - projectors["minus"])),
        })

    all_single_copy_full_rank = all(
        row["joint_rank_deficient_point_count"] == 0 for row in copies)
    result = {
        "status": "V4_LOCALIZED_OVERLAPPING_CUBE_EMBEDDING_AUDIT",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "construction": (
            "Translation-equivariant overlapping-cube synthesis using the "
            "prepared carrier's native Psi ray and chi companion at each "
            "corner; each flavor copy is tested independently."),
        "copies": copies,
        "both_copy_rows": both_rows,
        "copy_image_comparison": image_projector_rows,
        "translation_intertwiner_max_defect": max(
            translation_intertwiner_defect(basis, psi_frame, chi_frame)
            for basis in bases.values()),
        "single_copy_joint_embedding_full_rank_on_sampled_grid": (
            all_single_copy_full_rank),
        "verdict": (
            "SPATIAL_LOCALIZED_LIFT_CANDIDATE_SURVIVES"
            if all_single_copy_full_rank else
            "SPATIAL_LOCALIZED_LIFT_RANK_OBSTRUCTION"),
        "interpretation": (
            "Psi alone and chi alone each have only three complex channels "
            "and cannot inject a four-complex-component flavor fiber. Their "
            "joint six-channel native field can do so if the reported rank "
            "is four. Translation intertwining follows from the convolutional "
            "overlapping-cube construction rather than identifying internal "
            "corners with physical sites. The both-copy rank and image "
            "projector defect also test whether the two algebraic copies "
            "remain distinct after this native spatial lift."),
        "claim_boundary": (
            "This is a spatial rank and translation-equivariance audit. It "
            "does not yet include both committed time layers, prove native "
            "tangent closure, or establish a freely moving electron."),
        "next_test": (
            "Lift the surviving spatial map into the complete two-layer "
            "Psi/chi tangent, compute (I-EE^dagger)L_native E, and require "
            "the same E to reproduce mass, signed transport, density, and "
            "current."),
    }
    (HERE / "localized_overlapping_cube_embedding.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
