"""Classify the fastest residual after the native-scale pair connection.

This is a diagnostic of the research action

  L_V40 + (B chi0) A_common.

It reconstructs the complete invariant two-layer hull, saves the fastest
mode, and decomposes it without projecting the dynamics during evolution.
No action coefficient is fitted here.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.linalg import eig, lu_factor, lu_solve

import canonical_noncoplanar_bifurcation_audit as native_time
import canonical_noncoplanar_box_transfer as native_box
import current_action_localized_envelope_closure as envelope
import current_action_two_layer_spectrum as current
import density_oriented_pair_connection as connection
import localized_stability as native
import personal_touch_unit_geometric_canonical as canonical


HERE = Path(__file__).resolve().parent
SIDE = 5


def projector_overlap(vector: np.ndarray, basis: np.ndarray) -> float:
    denominator = float(np.vdot(vector, vector).real)
    if denominator == 0 or basis.size == 0:
        return 0.0
    return float(np.sum(abs(basis.T.conj() @ vector)**2) / denominator)


def spatial_regions(rest: np.ndarray) -> dict[str, np.ndarray]:
    psi, _chi = native.unpack(rest, SIDE)
    carrier = native.norm2(psi) > 1e-12
    shell = np.zeros_like(carrier)
    for offset, _weight in native.DIRECTIONS:
        shell |= np.roll(carrier, shift=offset, axis=(0, 1, 2))
    shell &= ~carrier
    return {
        "carrier": carrier,
        "first_C19_shell": shell,
        "remaining_exterior": ~(carrier | shell),
    }


def region_power(shaped: np.ndarray, mask: np.ndarray) -> float:
    return float(np.sum(abs(shaped[mask])**2))


def fourier_rows(shaped: np.ndarray) -> list[dict]:
    transformed = np.fft.fftn(shaped, axes=(0, 1, 2), norm="ortho")
    power = np.sum(abs(transformed)**2, axis=(3, 4))
    rows = []
    for flat in np.argsort(power.ravel())[::-1][:12]:
        index = np.unravel_index(int(flat), power.shape)
        wave = tuple(float(2*np.pi*np.fft.fftfreq(SIDE)[entry])
                     for entry in index)
        rows.append({
            "index": [int(value) for value in index],
            "wavevector": list(wave),
            "power_fraction": float(power[index]/np.sum(power)),
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--factor", type=float, default=1.0)
    parser.add_argument("--carrier-vacuum", type=float, default=0.0)
    parser.add_argument("--vacuum-vacuum", type=float, default=0.0)
    args = parser.parse_args()
    bare = np.load(canonical.CACHE)
    additions = np.load(HERE/"personal_touch_activated_bond_basis_v2_cache.npz")
    cached = np.load(HERE/"density_oriented_pair_connection_matrices.npz")
    delta = {kind: cached[f"{kind}_common_pair_connection"]
             for kind in ("mass", "gyro", "stiffness")}
    relative = {kind: cached[f"{kind}_relative_pair_connection"]
                for kind in ("mass", "gyro", "stiffness")}
    coefficient = args.factor*native_time.B*native_time.CHI0
    base_stiffness = (
        bare["stiffness"]
        + additions["hessian_carrier_carrier_chi"]
        + additions["hessian_carrier_carrier_Psi_tight_frame"])
    mass = bare["mass"]+coefficient*delta["mass"]
    gyro = bare["gyro"]+coefficient*delta["gyro"]
    stiffness = (
        base_stiffness+coefficient*delta["stiffness"]
        + args.carrier_vacuum*additions["hessian_carrier_vacuum_chi"]
        + args.vacuum_vacuum*additions["hessian_vacuum_vacuum_chi"])
    dressed = current.dressed_configuration(stiffness, mass, bare["rest"])

    dt = native_time.DT
    factor = lu_factor(mass/dt+gyro/2)
    forward = -lu_solve(factor, (dt*stiffness-2*mass/dt)@dressed)
    backward = -lu_solve(factor, (mass/dt-gyro/2)@dressed)
    residual = np.column_stack((
        forward-dressed@(dressed.T@forward),
        backward-dressed@(dressed.T@backward)))
    u, singular, _ = np.linalg.svd(residual, full_matrices=False)
    extra = u[:, singular > 1e-10*max(float(singular[0]), 1.0)]
    hull = envelope.orth(np.column_stack((dressed, extra)))
    fh = -lu_solve(factor, (dt*stiffness-2*mass/dt)@hull)
    bh = -lu_solve(factor, (mass/dt-gyro/2)@hull)
    f, g = hull.T@fh, hull.T@bh
    identity = np.eye(hull.shape[1])
    monodromy = np.block([[f, g], [identity, np.zeros_like(identity)]])
    values, modes = eig(monodromy, check_finite=False)
    fastest = int(np.argmax(abs(values)))
    present = hull@modes[:hull.shape[1], fastest]
    previous = hull@modes[hull.shape[1]:, fastest]
    suffix = "" if (args.factor == 1.0 and args.carrier_vacuum == 0
                    and args.vacuum_vacuum == 0) else (
        f"_factor_{args.factor:g}_cv_{args.carrier_vacuum:g}"
        f"_vv_{args.vacuum_vacuum:g}".replace(".", "p"))
    np.savez_compressed(
        HERE/f"density_connection_fastest_residual_mode{suffix}.npz",
        current=present, previous=previous, multiplier=values[fastest],
        mass=mass, gyro=gyro, stiffness=stiffness)

    shaped = present.reshape(SIDE, SIDE, SIDE, 4, 3)
    total = float(np.sum(abs(shaped)**2))
    field_power = np.sum(abs(shaped)**2, axis=(0, 1, 2, 4))/total
    component_power = np.sum(abs(shaped)**2, axis=(0, 1, 2, 3))/total
    regions = spatial_regions(bare["rest"])

    rest_psi, rest_chi = native.unpack(bare["rest"], SIDE)
    translation = native.translation_vectors(rest_psi, rest_chi)
    global_modes, corner_modes, radial_modes, phase_modes = (
        native.internal_mode_bases(rest_psi, rest_chi))
    psi_frame, chi_frame, sites = envelope.carrier_frames(bare["rest"])
    offsets = list(itertools.product((-2, -1, 0, 1, 2), repeat=3))
    copy_bases = {
        name: envelope.orth(envelope.raw_columns(
            basis, psi_frame, chi_frame, sites, offsets,
            0.00104831247009457))
        for name, basis in envelope.flavor_bases().items()
    }

    denominator = float(np.vdot(present, mass@present).real)
    response = {}
    for name, blocks in (("common", delta), ("relative", relative)):
        response[name] = {
            kind: float(np.vdot(present, matrix@present).real/denominator)
            for kind, matrix in blocks.items()
        }
        response[name]["gyro_magnitude"] = float(
            abs(np.vdot(present, blocks["gyro"]@present))/denominator)

    result = {
        "status": "V4_DENSITY_CONNECTION_RESIDUAL_MODE",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "coefficient": {
            "common": coefficient,
            "relative": 0.0,
            "provenance": "B times chi0",
            "carrier_vacuum": args.carrier_vacuum,
            "vacuum_vacuum": args.vacuum_vacuum,
        },
        "multiplier": {
            "real": float(values[fastest].real),
            "imag": float(values[fastest].imag),
            "modulus": float(abs(values[fastest])),
            "growth_rate": float(np.log(abs(values[fastest]))/dt),
            "phase": float(np.angle(values[fastest])),
        },
        "field_power_fractions": {
            "Psi_real_coordinates": float(field_power[0]),
            "Psi_imag_coordinates": float(field_power[1]),
            "chi_real_coordinates": float(field_power[2]),
            "chi_imag_coordinates": float(field_power[3]),
        },
        "C3_component_power_fractions": component_power.tolist(),
        "spatial_region_power_fractions": {
            name: region_power(shaped, mask)/total
            for name, mask in regions.items()
        },
        "geometric_subspace_overlaps": {
            "translations": projector_overlap(present, translation),
            "global_phase_radial": projector_overlap(present, global_modes),
            "carrier_corner_all": projector_overlap(present, corner_modes),
            "carrier_corner_radial": projector_overlap(present, radial_modes),
            "carrier_corner_phase": projector_overlap(present, phase_modes),
            "plus_copy_envelope": projector_overlap(present, copy_bases["plus"]),
            "minus_copy_envelope": projector_overlap(present, copy_bases["minus"]),
        },
        "top_spatial_fourier_modes": fourier_rows(shaped),
        "unit_connection_quadratics_on_residual": response,
        "verdict": "RESIDUAL_MODE_CLASSIFIED_FOR_ACTION_INVERSE",
        "claim_boundary": (
            "This classifies the fastest finite-hull linear mode after the "
            "natural common-pair connection. It does not prove nonlinear "
            "instability outside the declared orbit, admit an action, or "
            "close V4-006."),
    }
    (HERE/f"density_connection_residual_mode{suffix}.json").write_text(
        json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
