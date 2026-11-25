"""Superseded shared-face-continuity control for carrier handoff.

Adjacent 2x2x2 carriers share one four-site face.  This control required that
face to retain its old values and therefore reflected the destination's local
assignment.  The later literal edge-snap audit showed that this is not the
minimum-hop map: four sites remain fixed and four effectively cross two face
spacings.  A same-assignment shift moves every occupied site by one spacing
and permits the shared face to be replaced during the tick.

The Psi path uses cosine/sine weights and a global ``i`` on the destination.
This keeps the native Psi charge exactly constant: on the shared face the
field only rotates in phase, while the trailing and leading faces exchange
amplitude.  Chi is transferred locally with a linear seed; it is not held or
used as evidence of a solution.  The resulting potential profile is a direct
Peierls--Nabarro diagnostic of the unchanged action.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np

import localized_stability as native


HERE = Path(__file__).resolve().parent


def canonical_potential(psi: np.ndarray, chi: np.ndarray) -> float:
    p = native.norm2(psi)
    c = native.norm2(chi)
    value = float(np.sum((c-native.R**2)**2/4+c*p/2+p**2/4))
    area = np.cross(chi, psi)
    for offset, weight in native.DIRECTIONS:
        difference = native.shift(area, offset)-area
        value += weight*float(np.sum(abs(difference)**2))/(2*native.R**2)
    return value


def reflected_destination(psi: np.ndarray, chi: np.ndarray,
                          axis: int, sign: int):
    side = psi.shape[0]
    center = side//2-1
    vacuum = np.zeros_like(chi)
    vacuum[..., 2] = native.R
    destination_psi = np.zeros_like(psi)
    destination_chi = vacuum.copy()
    for local in itertools.product((0, 1), repeat=3):
        source_local = list(local)
        source_local[axis] = 1-local[axis]
        source_site = tuple(center+x for x in source_local)
        destination_site = list(center+x for x in local)
        destination_site[axis] += sign
        destination_site = tuple(destination_site)
        destination_psi[destination_site] = psi[source_site]
        destination_chi[destination_site] = chi[source_site]
    return destination_psi, destination_chi


def metrics_for_direction(axis: int, sign: int):
    side = 5
    psi0, chi0 = native.carrier_fields(side, 1.0)
    psi1, chi1 = reflected_destination(psi0, chi0, axis, sign)
    mask0 = native.norm2(psi0) > 0
    mask1 = native.norm2(psi1) > 0
    shared = mask0 & mask1
    union = mask0 | mask1
    vacuum = np.zeros_like(chi0)
    vacuum[..., 2] = native.R

    shared_psi_error = float(np.linalg.norm(psi0[shared]-psi1[shared]))
    shared_chi_error = float(np.linalg.norm(chi0[shared]-chi1[shared]))
    charge0 = float(np.sum(abs(psi0)**2))
    same_copy_psi = np.roll(psi0, sign, axis=axis)
    same_copy_chi = np.roll(chi0, sign, axis=axis)
    same_mask = mask0 & (native.norm2(same_copy_psi) > 0)

    def best_scalar_error(left, right):
        coefficient = np.vdot(right, left)/max(np.vdot(right, right).real, 1e-30)
        return float(np.linalg.norm(left-coefficient*right)
                     / max(np.linalg.norm(left), 1e-30))

    coordinates = np.indices((side, side, side)).reshape(3, -1).T
    rows = []
    for fraction in np.linspace(0, 1, 21):
        angle = np.pi*fraction/2
        left, right = np.cos(angle), np.sin(angle)
        psi = left*psi0+1j*right*psi1
        # This is only a local chi seed for measuring the barrier.  A true
        # moving branch must solve chi freely at every fraction.
        chi = (vacuum+(1-fraction)*(chi0-vacuum)
               + fraction*(chi1-vacuum))
        density = native.norm2(psi).reshape(-1)
        center_of_charge = (coordinates*density[:, None]).sum(0)/density.sum()
        rows.append({
            "fraction": float(fraction),
            "psi_charge": float(density.sum()),
            "psi_charge_relative_error": float(density.sum()/charge0-1),
            "center_of_charge": center_of_charge.tolist(),
            "support_site_count": int(np.sum(density > 1e-12)),
            "canonical_potential": canonical_potential(psi, chi),
        })
    endpoint = rows[0]["canonical_potential"]
    barrier = max(row["canonical_potential"] for row in rows)-endpoint
    return {
        "axis": axis,
        "sign": sign,
        "source_sites": int(mask0.sum()),
        "destination_sites": int(mask1.sum()),
        "shared_sites": int(shared.sum()),
        "union_sites": int(union.sum()),
        "shared_psi_error": shared_psi_error,
        "shared_chi_error": shared_chi_error,
        "same_copy_shared_psi_best_scalar_relative_error": best_scalar_error(
            psi0[same_mask], same_copy_psi[same_mask]),
        "same_copy_shared_chi_best_scalar_relative_error": best_scalar_error(
            chi0[same_mask], same_copy_chi[same_mask]),
        "copy_map_squared_is_identity": True,
        "maximum_charge_relative_error": float(max(
            abs(row["psi_charge_relative_error"]) for row in rows)),
        "potential_barrier": float(barrier),
        "rows": rows,
    }


def main():
    directions = [metrics_for_direction(axis, sign)
                  for axis in range(3) for sign in (-1, 1)]
    result = {
        "status": "V4_SUPERSEDED_REFLECTED_ASSIGNMENT_HANDOFF_CONTROL",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "directions": directions,
        "verdict": (
            "REFLECTED_ASSIGNMENT_CLOSES_SHARED_FACE_BUT_IS_NOT_MINIMUM_HOP"
        ),
        "superseded_by": "edge_snap_verlet_gate.json",
        "claim_boundary": (
            "This remains a charge-preserving reflected interpolation and PN "
            "negative control. It is not the literal nearest-neighbor shift, "
            "does not establish flavor exchange, and is not a freely evolving "
            "moving solution."
        ),
    }
    destination = HERE/"copy_alternating_handoff.json"
    destination.write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "verdict": result["verdict"],
        "summary": [{key: row[key] for key in (
            "axis", "sign", "shared_sites", "union_sites",
            "shared_psi_error", "shared_chi_error",
            "same_copy_shared_psi_best_scalar_relative_error",
            "same_copy_shared_chi_best_scalar_relative_error",
            "maximum_charge_relative_error", "potential_barrier")}
            for row in directions],
    }, indent=2))


if __name__ == "__main__":
    main()
