"""Pull the native C19 phase current through the carrier-contrast lift.

The spatial part of the native Psi U(1) Noether current follows directly from
the area bond energy.  For an oriented bond i -> j it is proportional to

    J_ij = (2 w_ij / chi0^2) Im(W_i^dagger W_j),
    W = chi wedge Psi.

The factor two reflects the directed C19 sum used by the action.  This audit
linearizes the flux into the receiving half-space along the exact physical
translation tangent and decomposes that tangent into the two cubical flavor
copies.  No Dirac current or fitted velocity enters the calculation.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np

import edge_snap_action_rank as rank
import localized_carrier_contrast_translation as contrast
import localized_embedding_native_closure as local
import localized_embedding_translation_tangent as tangent


HERE = Path(__file__).resolve().parent
SIDE = tangent.SIDE
ORIGIN = tangent.ORIGIN


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
        columns.extend((rank.pack(dpsi, dchi),
                        rank.pack(1j*dpsi, 1j*dchi)))
    return columns


def copy_envelope(basis: np.ndarray) -> np.ndarray:
    psi_profile, chi_profile = contrast.carrier_contrast()
    raw = np.column_stack([
        column
        for center in tangent.center_sets()["source_plus_faces"]
        for column in profile_columns(
            basis, psi_profile, chi_profile, center)
    ])
    return tangent.orth(raw)


def receiving_mask(axis: int, sign: int) -> np.ndarray:
    coordinate = np.indices((SIDE, SIDE, SIDE))[axis]
    if sign > 0:
        return coordinate >= ORIGIN + 2
    return coordinate <= ORIGIN - 1


def native_area_flux(
        psi: np.ndarray, chi: np.ndarray,
        mask: np.ndarray) -> float:
    area = np.cross(chi, psi)
    flux = 0.0
    for offset, weight in rank.native.DIRECTIONS:
        neighbor_area = rank.native.shift(area, offset)
        neighbor_mask = rank.native.shift(mask[..., None], offset)[..., 0]
        crosses_into = (~mask) & neighbor_mask
        phase = np.imag(np.sum(np.conj(area)*neighbor_area, axis=-1))
        flux += (2*weight/rank.R**2) * float(np.sum(phase[crosses_into]))
    return flux


def flux_derivative(
        base: np.ndarray, direction: np.ndarray,
        mask: np.ndarray, step: float = 1e-6) -> float:
    values = []
    for sign in (-1, 1):
        psi, chi = rank.native.unpack(base + sign*step*direction, SIDE)
        values.append(native_area_flux(psi, chi, mask))
    return float((values[1]-values[0])/(2*step))


def physical_handoff(axis: int, sign: int) -> tuple[np.ndarray, np.ndarray]:
    amplitude_sq = rank.R**2 / (2 + 54/rank.R**2)
    psi, chi = rank.native.carrier_fields(SIDE, amplitude_sq)
    shifted_psi = np.roll(psi, sign, axis=axis)
    shifted_chi = np.roll(chi, sign, axis=axis)
    return rank.pack(psi, chi), rank.pack(
        shifted_psi-psi, shifted_chi-chi)


def decompose(
        direction: np.ndarray, plus: np.ndarray,
        minus: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    joint = np.column_stack((plus, minus))
    coefficients, _, _, _ = np.linalg.lstsq(
        joint, direction, rcond=1e-12)
    split = plus.shape[1]
    plus_part = plus @ coefficients[:split]
    minus_part = minus @ coefficients[split:]
    residual = float(np.linalg.norm(
        direction-plus_part-minus_part)/max(np.linalg.norm(direction), 1e-30))
    return plus_part, minus_part, residual


def main() -> None:
    rank.lock_inertia()
    flavors = local.flavor_bases()
    plus = copy_envelope(flavors["plus"])
    minus = copy_envelope(flavors["minus"])
    rows = []
    for axis in range(3):
        for sign in (-1, 1):
            base, direction = physical_handoff(axis, sign)
            plus_part, minus_part, defect = decompose(
                direction, plus, minus)
            mask = receiving_mask(axis, sign)
            total = flux_derivative(base, direction, mask)
            plus_flux = flux_derivative(base, plus_part, mask)
            minus_flux = flux_derivative(base, minus_part, mask)
            rows.append({
                "axis": axis,
                "sign": sign,
                "two_copy_decomposition_relative_defect": defect,
                "native_flux_linear_response": total,
                "plus_copy_flux_contribution": plus_flux,
                "minus_copy_flux_contribution": minus_flux,
                "copy_sum_flux_defect": float(
                    abs(total-plus_flux-minus_flux)
                    / max(abs(total), 1e-30)),
                "flux_has_requested_sign": bool(sign*total > 0),
            })
    signed = [row["flux_has_requested_sign"] for row in rows]
    nonzero = [abs(row["native_flux_linear_response"]) > 1e-10
               for row in rows]
    result = {
        "status": "V4_LOCALIZED_CARRIER_CONTRAST_CURRENT_AUDIT",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "current": (
            "J_ij=(2 w_ij/chi0^2) Im[(chi_i wedge Psi_i)^dagger "
            "(chi_j wedge Psi_j)] from the directed native area-bond action"),
        "rows": rows,
        "verdict": (
            "TWO_COPY_CONTRAST_HAS_SIGNED_NATIVE_HANDOFF_CURRENT"
            if all(signed) and all(nonzero) else
            "TWO_COPY_CONTRAST_CURRENT_SIGN_OR_RANK_FAIL"),
        "interpretation": (
            "The derivative measures current into the newly occupied side of "
            "the receiving cube along the exact carrier handoff.  The copy "
            "columns are an oblique decomposition of the same physical "
            "tangent, so their currents must add to the total."),
        "claim_boundary": (
            "This is the linearized spatial current of the native area bond. "
            "It does not include a completed discrete-time density bridge, "
            "prove nonlinear motion, or establish current equality with the "
            "Dirac readout."),
        "corrective_action": (
            "On pass, derive the discrete phase density of the same interval "
            "action and verify continuity on the retained two-layer graph. "
            "On failure, decompose the missing flux by C19 bond orbit before "
            "changing the action."),
    }
    output = HERE / "localized_carrier_contrast_current.json"
    output.write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
