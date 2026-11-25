"""Audit normalized chi--chi overlaps on the exact localized rest orbit."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import c27_body_gyro_basis_probe as candidate
import c27_body_gyro_rest_spectrum as spectrum


HERE = Path(__file__).resolve().parent


def main() -> None:
    spectrum.configure()
    candidate.base.inverse.rank.lock_inertia()
    _previous, current, _following, _theta = (
        candidate.base.inverse.rest_layers())
    psi, chi = candidate.base.inverse.rank.native.unpack(
        current, spectrum.SIDE)
    occupied = np.sum(abs(psi)**2, axis=-1) > 1e-12
    rows = []
    for offset, weight in candidate.base.inverse.rank.native.DIRECTIONS:
        neighbor = np.roll(chi, shift=offset, axis=(0, 1, 2))
        neighbor_occupied = np.roll(
            occupied, shift=offset, axis=(0, 1, 2))
        overlap = abs(np.sum(np.conj(chi) * neighbor, axis=-1))**2
        product = (np.sum(abs(chi)**2, axis=-1)
                   * np.sum(abs(neighbor)**2, axis=-1))
        ratio = overlap / np.maximum(product, 1e-300)
        for relation, mask in (
                ("carrier-carrier", occupied & neighbor_occupied),
                ("carrier-exterior", occupied ^ neighbor_occupied),
                ("exterior-exterior", ~occupied & ~neighbor_occupied)):
            if np.any(mask):
                values = ratio[mask]
                rows.append({
                    "offset": list(offset),
                    "weight": float(weight),
                    "relation": relation,
                    "count": int(np.sum(mask)),
                    "minimum": float(np.min(values)),
                    "maximum": float(np.max(values)),
                    "rounded_unique": sorted(set(
                        float(value) for value in np.round(values, 12))),
                })
    all_unique = sorted(set(
        value for row in rows for value in row["rounded_unique"]))
    result = {
        "status": "V4_C19_CHI_BOND_ANGLE_AUDIT",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "normalized_overlap_values": all_unique,
        "rows": rows,
        "interpretation": (
            "Any chi-only bond energy preserving this exact rest orbit must "
            "have zero first derivative at every listed overlap value; "
            "positivity and curvature then determine whether those values "
            "are simultaneous minima."
        ),
    }
    output = HERE / "c19_chi_bond_angle_audit.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "normalized_overlap_values": all_unique,
        "relations": sorted(set(row["relation"] for row in rows)),
        "output": str(output),
    }, indent=2))


if __name__ == "__main__":
    main()
