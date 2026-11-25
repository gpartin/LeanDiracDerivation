"""Moving phase-space closure gate for the live-Clifford action term.

This uses a geometry-only conditional edge wave.  The spatial generator and
the two rank-four sectors are both derived from the normalized live Psi links.
The resulting previous/current/following perturbations are passed through the
exact reciprocal action variation.  The measured cross block asks whether the
action response remains in the same all-center native carrier lift.

No Dirac matrix or target trajectory is supplied.
"""

from __future__ import annotations

import itertools
import json
import os
from pathlib import Path

import numpy as np
from scipy.linalg import expm
import torch
from torch.func import jvp

import canonical_noncoplanar_bifurcation_audit as native_time
import canonical_transfer_oscillator_spectrum as native_spectrum
import cube_incidence_extended_packet_gate as coordinates
import cube_incidence_nonlinear_action as native_action
import current_action_localized_envelope_closure as envelope
import live_clifford_bond_action as candidate
import live_clifford_link_geometry as geometry


HERE = Path(__file__).resolve().parent
SIDE = 5
OFFSETS = list(itertools.product((-2, -1, 0, 1, 2), repeat=3))
NEAREST_OFFSETS = list(itertools.product((-1, 0, 1), repeat=3))
GATE_CHI_SCALE = float(os.environ.get("LFM_LIVE_GATE_CHI_SCALE", "0"))
GATE_ENVELOPE = os.environ.get("LFM_LIVE_GATE_ENVELOPE", "nearest27")
GATE_TIME_SIGN = float(os.environ.get("LFM_LIVE_GATE_TIME_SIGN", "1"))
GATE_BASIS = os.environ.get("LFM_LIVE_GATE_BASIS", "canonical")


def orth(columns: np.ndarray) -> np.ndarray:
    u, singular, _ = np.linalg.svd(columns, full_matrices=False)
    tolerance = 1e-11*max(float(singular[0]), 1.0)
    return u[:, singular > tolerance]


def complex_field(component: np.ndarray, momentum: np.ndarray,
                  offsets) -> np.ndarray:
    xyz = np.asarray(offsets, dtype=float)
    phase = np.exp(1j*np.einsum("...a,a->...", xyz, momentum))
    return phase[..., None]*component


def joint_raw_columns(psi_basis: np.ndarray, chi_basis: np.ndarray,
                      psi_frame: np.ndarray, chi_frame: np.ndarray,
                      sites, chi_scale: complex,
                      conjugate_chi_amplitude: bool = True,
                      offsets=OFFSETS) -> np.ndarray:
    columns = []
    for offset in offsets:
        for component in range(psi_basis.shape[1]):
            dpsi = np.zeros((SIDE, SIDE, SIDE, 3), dtype=complex)
            dchi = np.zeros_like(dpsi)
            for corner, site in enumerate(sites):
                target = tuple((site[axis]+offset[axis]) % SIDE
                               for axis in range(3))
                dpsi[target] += psi_basis[corner, component]*psi_frame[corner]
                dchi[target] += (chi_scale*chi_basis[corner, component]
                                 * chi_frame[corner])
            imaginary_chi = -1j*dchi if conjugate_chi_amplitude else 1j*dchi
            columns.extend((envelope.rank.pack(dpsi, dchi),
                            envelope.rank.pack(1j*dpsi, imaginary_chi)))
    return np.column_stack(columns)


def main() -> None:
    bare, _ = native_action.expected_blocks()
    rest = bare["rest"]
    dt = native_time.DT
    theta = float(np.arccos(1-dt**2*native_time.CHI0**2/2))
    rotation = native_spectrum.rotation_matrix(rest.size, theta)
    layers = tuple(torch.tensor(value, dtype=torch.float64)
                   for value in (rotation.T@rest, rest, rotation@rest))
    psi_generators = [geometry.live_generator(
        geometry.occupied_rays(rest, "psi"), axis) for axis in range(3)]
    chi_generators = [geometry.live_generator(
        geometry.occupied_rays(rest, "chi"), axis) for axis in range(3)]
    intertwiner = geometry.diagonal_intertwiner(
        psi_generators, chi_generators)
    generators = psi_generators
    flavor = 1j*generators[0]@generators[1]@generators[2]
    identity = np.eye(8, dtype=complex)
    psi_frame, chi_frame, sites = envelope.carrier_frames(rest)

    def delta_residual(previous: torch.Tensor, current: torch.Tensor,
                       following: torch.Tensor) -> torch.Tensor:
        return candidate.del_residual(previous, current, following, dt) \
            - candidate.incumbent.del_residual(
                previous, current, following, dt)

    fundamental = 2*np.pi/SIDE
    offsets = NEAREST_OFFSETS if GATE_ENVELOPE == "nearest27" else OFFSETS
    rows = []
    arrays = {}
    if GATE_BASIS == "canonical":
        sector_bases = coordinates.flavor_bases().items()
    else:
        sector_bases = []
        for sector, projector in (("live_plus", (identity+flavor)/2),
                                  ("live_minus", (identity-flavor)/2)):
            values, vectors = np.linalg.eigh(
                (projector+projector.conj().T)/2)
            sector_bases.append((sector, vectors[:, values > 0.5]))
    for sector, basis in sector_bases:
        chi_basis = intertwiner@basis if GATE_BASIS == "live" else basis
        raw = joint_raw_columns(
            basis, chi_basis, psi_frame, chi_frame, sites,
            GATE_CHI_SCALE+0j, offsets=offsets)
        retained = orth(raw)
        psi_only = joint_raw_columns(
            basis, chi_basis, psi_frame, chi_frame, sites, 0j,
            offsets=offsets)
        retained_psi = orth(psi_only)
        chi_mask = np.zeros(rest.size, dtype=bool)
        for site_index in range(rest.size//12):
            chi_mask[site_index*12+6:site_index*12+12] = True
        for axis in range(3):
            if GATE_BASIS == "canonical":
                alpha, _beta = coordinates.compressed_clifford(basis)
                alpha = alpha[axis]
            else:
                alpha = basis.conj().T@generators[axis]@basis
            momentum = np.zeros(3)
            momentum[axis] = fundamental
            speed = np.sin(fundamental)
            forward = expm(-1j*GATE_TIME_SIGN*dt*speed*alpha)
            backward = expm(1j*GATE_TIME_SIGN*dt*speed*alpha)
            component = np.zeros(4, dtype=complex)
            component[0] = 1
            current_field = complex_field(component, momentum, offsets)
            previous_field = np.einsum(
                "ab,...b->...a", backward, current_field)
            following_field = np.einsum(
                "ab,...b->...a", forward, current_field)
            current = raw@coordinates.real_coordinates(current_field)
            previous = rotation.T@(
                raw@coordinates.real_coordinates(previous_field))
            following = rotation@(
                raw@coordinates.real_coordinates(following_field))
            directions = tuple(torch.tensor(value, dtype=torch.float64)
                               for value in (previous, current, following))
            _value, image = jvp(delta_residual, layers, directions)
            response = image.detach().numpy()
            projected = retained@(retained.T@response)
            projected_psi_with_free_chi = retained_psi@(
                retained_psi.T@response)
            projected_psi_with_free_chi[chi_mask] = response[chi_mask]
            key = f"{sector}_axis_{axis}"
            arrays[f"{key}_response"] = response
            arrays[f"{key}_projected"] = projected
            arrays[f"{key}_projected_psi_with_free_chi"] = (
                projected_psi_with_free_chi)
            rows.append({
                "sector": sector,
                "axis": axis,
                "retained_real_rank": int(retained.shape[1]),
                "response_norm": float(np.linalg.norm(response)),
                "relative_cross_block": float(
                    np.linalg.norm(response-projected)
                    / max(np.linalg.norm(response), 1e-30)),
                "psi_sector_cross_block_with_independent_chi": float(
                    np.linalg.norm(response-projected_psi_with_free_chi)
                    / max(np.linalg.norm(response), 1e-30)),
            })
    active = [row for row in rows if row["response_norm"] > 1e-10]
    passed = bool(active) and max(
        row["psi_sector_cross_block_with_independent_chi"]
        for row in active) < 1e-8
    result = {
        "status": "V4_LIVE_CLIFFORD_MOVING_SECTOR_GATE",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "link_source": candidate.LINK_SOURCE,
        "joint_gauge": (
            "Psi amplitudes use the live-Psi sector; chi amplitudes use the "
            "exact diagonal edge-link intertwiner into the live-chi sector; "
            "the chi coefficient is conjugated as required by the native "
            "companion map"),
        "chi_amplitude_map": "antilinear_conjugate",
        "input_chi_amplitude_scale": GATE_CHI_SCALE,
        "envelope": GATE_ENVELOPE,
        "transport_time_sign": GATE_TIME_SIGN,
        "amplitude_basis": GATE_BASIS,
        "closure_rule": (
            "Psi must stay in the live rank-four carrier lift; independent "
            "chi response is retained rather than counted as spinor leakage"),
        "intertwiner_defect": float(max(np.linalg.norm(
            chi_generators[axis]@intertwiner
            - intertwiner@psi_generators[axis]) for axis in range(3))),
        "input": (
            "fundamental periodic conditional edge wave derived only from "
            "the live-Psi link generators"),
        "rows": rows,
        "verdict": (
            "LIVE_CLIFFORD_ACTION_MOVING_SECTOR_CLOSED"
            if passed else "LIVE_CLIFFORD_ACTION_MOVING_SECTOR_OPEN"),
        "claim_boundary": (
            "Linear all-center phase-space action response. It is not a "
            "nonlinear carrier translation or V4-006 pass."),
    }
    suffix = candidate.LINK_SOURCE.replace(".", "p")
    output = HERE/f"live_clifford_moving_sector_gate_{suffix}.json"
    output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n",
                      encoding="utf-8")
    np.savez_compressed(
        HERE/f"live_clifford_moving_sector_gate_{suffix}.npz", **arrays)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
