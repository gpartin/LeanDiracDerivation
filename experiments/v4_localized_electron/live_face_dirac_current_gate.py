"""Pull the C19 Dirac density/current through the live face DEL map."""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
from scipy.linalg import lu_factor, lu_solve

import canonical_noncoplanar_bifurcation_audit as native_time
import canonical_transfer_oscillator_spectrum as native_spectrum
import cube_incidence_extended_packet_dirac_current as current
import cube_incidence_extended_packet_gate as gate
import cube_incidence_nonlinear_action as prior_action
import current_action_localized_envelope_closure as envelope
import live_clifford_tangent_gate as tangent


HERE = Path(__file__).resolve().parent


def main() -> None:
    bare, _ = prior_action.expected_blocks()
    rest = bare["rest"]
    theta = float(np.arccos(1-native_time.DT**2*native_time.CHI0**2/2))
    rotation = native_spectrum.rotation_matrix(rest.size, theta)
    previous_block, current_block, following_block = tangent.tangent_blocks(
        (rotation.T@rest, rest, rotation@rest))
    factor = lu_factor(following_block)
    psi_frame, chi_frame, sites = envelope.carrier_frames(rest)
    prior = json.loads(
        (HERE/"cube_incidence_dirac_envelope_gate.json").read_text(
            encoding="utf-8"))
    scales = {}
    for row in prior["rows"]:
        case = next(item for item in row["cases"]
                    if item["envelope"] == "all_125_centers")
        scales[row["flavor"]] = complex(
            case["chi_scale_real"], case["chi_scale_imag"])

    rows = []
    for flavor, basis in gate.flavor_bases().items():
        alpha, beta = gate.compressed_clifford(basis)
        raw = envelope.raw_columns(
            basis, psi_frame, chi_frame, sites, gate.OFFSETS, scales[flavor])
        for axis, sign in itertools.product(range(4), (-1, 1)):
            previous, field, following, momentum = gate.packet(
                axis, sign, alpha, beta, native_time.DT,
                centered_time=True)
            q_previous = raw@gate.real_coordinates(previous)
            q_current = raw@gate.real_coordinates(field)
            q_native = lu_solve(
                factor,
                -(previous_block@q_previous+current_block@q_current))
            recovered, *_ = np.linalg.lstsq(raw, q_native, rcond=1e-10)
            native = current.complex_field(recovered)
            density_native, current_native = current.observables(native, alpha)
            density_target, current_target = current.observables(following, alpha)
            rows.append({
                "flavor": flavor,
                "direction": "diagonal" if axis == 3 else "xyz"[axis],
                "sign": sign,
                "momentum": momentum.tolist(),
                "density_relative_error": float(
                    abs(density_native-density_target)
                    / max(density_target, 1e-30)),
                "current_relative_error": float(
                    np.linalg.norm(current_native-current_target)
                    / max(np.linalg.norm(current_target), 1e-30)),
                "native_current": current_native.tolist(),
                "dirac_current": current_target.tolist(),
            })
    density_error = max(row["density_relative_error"] for row in rows)
    current_error = max(row["current_relative_error"] for row in rows)
    result = {
        "status": "V4_LIVE_FACE_DIRAC_CURRENT_GATE",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "rows": rows,
        "maximum_density_relative_error": density_error,
        "maximum_current_relative_error": current_error,
        "verdict": (
            "LIVE_FACE_READOUT_CURRENT_PASS"
            if density_error < 1e-3 and current_error < 1e-2
            else "LIVE_FACE_READOUT_CURRENT_OPEN"),
        "claim_boundary": (
            "Positive-frequency C19 current through the frozen readout. "
            "The local Noether bond-current identity of the full nonlinear "
            "action remains a separate requirement."),
    }
    path = HERE/"live_face_dirac_current_gate.json"
    path.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n",
                    encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items()
                      if key != "rows"}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
