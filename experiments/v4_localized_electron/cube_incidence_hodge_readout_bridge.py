"""Test whether the native Hodge four-field is the historical corner readout.

For the carrier-activated relative field used by the compatible Hodge branch,

    tau = a (chi . conjugate(Psi))/chi0,
    W   = a (chi x conjugate(Psi))/chi0,
    a   = rho_Psi/chi0^2,

this script differentiates (tau,W) on the exact localized rest orbit and pulls
both historical four-component copies through that derivative.  It measures
rank, conditioning, copy overlap, and closure under the native C19 Hodge
operator.  The test decides whether the new action response was compared in
an equivalent coordinate system; it does not refit a readout or admit an
action.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import canonical_noncoplanar_bifurcation_audit as native_time
import cube_incidence_hodge_rest_gate as hodge
import personal_touch_unit_geometric_canonical as canonical


HERE = Path(__file__).resolve().parent
SIDE = 5


def native_variations(columns: np.ndarray):
    shaped = columns.T.reshape(-1, SIDE, SIDE, SIDE, 4, 3)
    dpsi = shaped[..., 0, :]+1j*shaped[..., 1, :]
    dchi = shaped[..., 2, :]+1j*shaped[..., 3, :]
    return dpsi, dchi


def rest_fields(rest: np.ndarray):
    shaped = rest.reshape(SIDE, SIDE, SIDE, 4, 3)
    psi = shaped[..., 0, :]+1j*shaped[..., 1, :]
    chi = shaped[..., 2, :]+1j*shaped[..., 3, :]
    return psi, chi


def composite_jacobian(rest: np.ndarray, columns: np.ndarray,
                       activated: bool):
    psi, chi = rest_fields(rest)
    dpsi, dchi = native_variations(columns)
    relative = np.conjugate(psi)
    drelative = np.conjugate(dpsi)
    density = np.sum(abs(psi)**2, axis=-1)/native_time.CHI0**2
    ddensity = 2*np.real(np.sum(
        np.conjugate(psi)[None, ...]*dpsi, axis=-1))/native_time.CHI0**2
    tau0 = np.sum(chi*relative, axis=-1)/native_time.CHI0
    area0 = np.cross(chi, relative)/native_time.CHI0
    dtau0 = (np.sum(dchi*relative[None, ...], axis=-1)
             + np.sum(chi[None, ...]*drelative, axis=-1))/native_time.CHI0
    darea0 = (np.cross(dchi, relative[None, ...])
              + np.cross(chi[None, ...], drelative))/native_time.CHI0
    if activated:
        dtau = ddensity*tau0[None, ...]+density[None, ...]*dtau0
        darea = (ddensity[..., None]*area0[None, ...]
                 + density[None, ..., None]*darea0)
    else:
        dtau, darea = dtau0, darea0
    field = np.concatenate((dtau[..., None], darea), axis=-1)
    return np.stack((field.real, field.imag), axis=-1).reshape(
        columns.shape[1], -1).T


def derivative19_numpy(field: np.ndarray, axis: int):
    result = np.zeros_like(field)
    for offset, weight in hodge.native.DIRECTIONS:
        if offset[axis] != 0:
            result += weight*offset[axis]*np.roll(
                field, shift=offset, axis=(0, 1, 2))
    return result/12


def hodge_on_columns(columns: np.ndarray):
    count = columns.shape[1]
    shaped = columns.T.reshape(count, SIDE, SIDE, SIDE, 4, 2)
    complex_field = shaped[..., 0]+1j*shaped[..., 1]
    output = []
    for field in complex_field:
        tau, area = field[..., 0], field[..., 1:]
        grad = np.stack(
            [derivative19_numpy(tau, axis) for axis in range(3)], axis=-1)
        divergence = sum(derivative19_numpy(area[..., axis], axis)
                         for axis in range(3))
        curl = np.stack((
            derivative19_numpy(area[..., 2], 1)
            - derivative19_numpy(area[..., 1], 2),
            derivative19_numpy(area[..., 0], 2)
            - derivative19_numpy(area[..., 2], 0),
            derivative19_numpy(area[..., 1], 0)
            - derivative19_numpy(area[..., 0], 1),
        ), axis=-1)
        transformed = np.concatenate(
            ((-divergence)[..., None], grad+curl), axis=-1)
        output.append(np.stack(
            (transformed.real, transformed.imag), axis=-1).reshape(-1))
    return np.column_stack(output)


def analyze(name: str, mapped: np.ndarray):
    u, singular, _vh = np.linalg.svd(mapped, full_matrices=False)
    threshold = 1e-10*max(float(singular[0]), 1e-30)
    rank = int(np.sum(singular > threshold))
    q = u[:, :rank]
    image = hodge_on_columns(mapped)
    leakage = image-q@(q.T@image)
    return {
        "copy": name,
        "input_real_dimension": int(mapped.shape[1]),
        "mapped_real_rank": rank,
        "smallest_retained_singular": float(singular[rank-1]) if rank else 0,
        "largest_singular": float(singular[0]),
        "condition_number": float(singular[0]/singular[rank-1])
        if rank else None,
        "hodge_relative_cross_block": float(
            np.linalg.norm(leakage)/max(np.linalg.norm(image), 1e-30)),
        "basis": q,
    }


def main() -> None:
    rest = np.load(canonical.CACHE)["rest"]
    historical = np.load(HERE/"cube_incidence_dirac_envelope_gate.npz")
    variants = []
    for activated in (True, False):
        mapped = {name: composite_jacobian(
            rest, historical[name], activated)
            for name in ("plus", "minus")}
        rows = {name: analyze(name, value) for name, value in mapped.items()}
        q_plus = rows["plus"].pop("basis")
        q_minus = rows["minus"].pop("basis")
        overlap = np.linalg.svd(q_plus.T@q_minus, compute_uv=False)
        full_rank = all(
            row["mapped_real_rank"] == row["input_real_dimension"]
            for row in rows.values())
        closed = all(row["hodge_relative_cross_block"] < 1e-8
                     for row in rows.values())
        variants.append({
            "variant": "activated_relative" if activated else "relative",
            "rows": list(rows.values()),
            "copy_subspace_overlap": {
                "count": int(overlap.size),
                "minimum_singular": float(np.min(overlap)),
                "maximum_singular": float(np.max(overlap)),
                "count_above_1_minus_1e_minus_10": int(np.sum(
                    overlap > 1-1e-10)),
            },
            "both_historical_copies_recoverable": full_rank,
            "both_copies_closed_under_hodge_operator": closed,
        })
    unactivated = variants[1]
    result = {
        "status": "V4_C19_HODGE_READOUT_BRIDGE",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "readout": (
            "d[(chi dot conjugate(Psi)), "
            "(chi cross conjugate(Psi))]/chi0 with and without "
            "rho_Psi/chi0^2 activation"),
        "variants": variants,
        "verdict": (
            "UNACTIVATED_HODGE_READOUT_EQUIVALENT_AND_CLOSED"
            if unactivated["both_historical_copies_recoverable"]
            and unactivated["both_copies_closed_under_hodge_operator"]
            else "HODGE_READOUT_NOT_EQUIVALENT_TO_HISTORICAL_EMBEDDING"),
        "claim_boundary": (
            "Linear differential at the exact localized rest orbit. Rank "
            "establishes recoverability, not mass/current equivalence or a "
            "nonlinear global coordinate chart."),
    }
    (HERE/"cube_incidence_hodge_readout_bridge.json").write_text(
        json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
