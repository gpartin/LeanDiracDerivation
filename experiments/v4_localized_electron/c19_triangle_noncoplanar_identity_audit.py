"""Check the locked eight-site identity along the noncoplanar rest family."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import c27_body_gyro_basis_probe as direct
import c27_body_gyro_rest_spectrum as spectrum


HERE = Path(__file__).resolve().parent
PAIRS = ((0, 7), (1, 6), (2, 5), (3, 4))
FACES = (
    (0, 2, 3, 1), (4, 6, 7, 5),
    (0, 4, 5, 1), (2, 6, 7, 3),
    (0, 4, 6, 2), (1, 5, 7, 3),
)


def metrics(raw: np.ndarray, fraction: float) -> dict:
    psi, chi = direct.base.inverse.rank.native.unpack(raw, spectrum.SIDE)
    occupied = np.sum(abs(psi)**2, axis=-1) > 1e-12
    coordinates = list(zip(*np.where(occupied)))
    coordinates.sort()
    rays = np.asarray([psi[site] for site in coordinates])
    chi_rays = np.asarray([chi[site] for site in coordinates])
    scale = max(float(np.linalg.norm(rays)), 1e-300)
    pair_error = max(
        float(np.linalg.norm(rays[first] + rays[second]) / scale)
        for first, second in PAIRS)
    chi_even_error = max(
        float(np.linalg.norm(chi_rays[first] - chi_rays[second])
              / max(np.linalg.norm(chi_rays), 1e-300))
        for first, second in PAIRS)
    face_errors = []
    overlap_floor = 1.0
    for face in FACES:
        product = 1 + 0j
        for start, end in zip(face, face[1:] + face[:1]):
            overlap = np.vdot(rays[start], rays[end])
            normalized = overlap / max(
                np.linalg.norm(rays[start]) * np.linalg.norm(rays[end]),
                1e-300)
            overlap_floor = min(overlap_floor, abs(normalized))
            product *= normalized / max(abs(normalized), 1e-300)
        face_errors.append(abs(product + 1))
    singular = np.linalg.svd(rays, compute_uv=False)
    exterior = ~occupied
    exterior_power = float(np.sum(abs(psi[exterior])**2))
    total_power = max(float(np.sum(abs(psi)**2)), 1e-300)
    return {
        "fraction_of_rest_norm": fraction,
        "occupied_site_count": len(coordinates),
        "occupied_coordinates": [
            [int(value) for value in site] for site in coordinates],
        "opposite_pair_error": pair_error,
        "face_product_error": float(max(face_errors)),
        "minimum_normalized_overlap": float(overlap_floor),
        "frame_singular_values": [float(value) for value in singular],
        "rank_three_to_rank_two_ratio": float(
            singular[2] / max(singular[1], 1e-300)),
        "chi_even_pair_error": chi_even_error,
        "exterior_psi_power_fraction": exterior_power / total_power,
    }


def main() -> None:
    spectrum.configure()
    direct.base.inverse.rank.lock_inertia()
    _previous, rest, _following, _theta = direct.base.inverse.rest_layers()
    archive = np.load(
        HERE / "c19_triangle_noncoplanar_branch_probe.npz")
    rows = [metrics(rest, 0.0)]
    for fraction, state in zip(archive["fractions"], archive["states"]):
        if float(fraction) in (1e-3, 1e-2, 3e-2):
            rows.append(metrics(state, float(fraction)))
    strict_identity = [row for row in rows
                       if row["opposite_pair_error"] < 1e-8
                       and row["face_product_error"] < 1e-8
                       and row["rank_three_to_rank_two_ratio"] < 1e-8
                       and row["exterior_psi_power_fraction"] < 1e-12]
    result = {
        "status": "V4_C19_TRIANGLE_NONCOPLANAR_IDENTITY_AUDIT",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "rows": rows,
        "strict_identity_fractions": [
            row["fraction_of_rest_norm"] for row in strict_identity],
        "verdict": (
            "NONCOPLANAR_BRANCH_RETAINS_LOCKED_EIGHT_SITE_IDENTITY"
            if len(strict_identity) == len(rows) else
            "NONCOPLANAR_BRANCH_BREAKS_LOCKED_CARRIER_IDENTITY"
        ),
        "claim_boundary": (
            "Locked opposite-pair, face-product, rank-two, chi-even, and "
            "localization diagnostics only. No evolution or spin claim."
        ),
    }
    output = HERE / "c19_triangle_noncoplanar_identity_audit.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
