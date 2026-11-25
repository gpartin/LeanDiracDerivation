"""Timestep refinement of the corrected extended-packet motion gate."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.linalg import lu_factor, lu_solve

import canonical_noncoplanar_bifurcation_audit as native_time
import cube_incidence_extended_packet_gate as packet_gate
import cube_incidence_nonlinear_action as action
import current_action_localized_envelope_closure as envelope


HERE = Path(__file__).resolve().parent


def update(mass, gyro, stiffness, dt):
    factor = lu_factor(mass/dt+gyro/2)
    return (
        -lu_solve(factor, dt*stiffness-2*mass/dt),
        -lu_solve(factor, mass/dt-gyro/2),
    )


def main():
    bare, matrices = action.expected_blocks()
    mass, gyro, stiffness = (matrices[name]
                             for name in ("mass", "gyro", "stiffness"))
    psi_frame, chi_frame, sites = envelope.carrier_frames(bare["rest"])
    prior = json.loads((HERE/"cube_incidence_dirac_envelope_gate.json").read_text())
    scales = {}
    for row in prior["rows"]:
        case = next(item for item in row["cases"]
                    if item["envelope"] == "all_125_centers")
        scales[row["flavor"]] = complex(
            case["chi_scale_real"], case["chi_scale_imag"])

    rows = []
    for divisor in (1, 2, 4):
        dt = native_time.DT/divisor
        advance, memory = update(mass, gyro, stiffness, dt)
        errors = []
        leakages = []
        flavor_rows = []
        for flavor, basis in packet_gate.flavor_bases().items():
            alpha, beta = packet_gate.compressed_clifford(basis)
            raw = envelope.raw_columns(
                basis, psi_frame, chi_frame, sites, packet_gate.OFFSETS,
                scales[flavor])
            q = envelope.orth(raw)
            projector = q@q.T
            flavor_errors = []
            for axis in range(4):
                for sign in (-1, 1):
                    previous, current, following, _momentum = packet_gate.packet(
                        axis, sign, alpha, beta, dt)
                    native = (advance@(
                        raw@packet_gate.real_coordinates(current))
                        + memory@(
                            raw@packet_gate.real_coordinates(previous)))
                    target = raw@packet_gate.real_coordinates(following)
                    error = float(np.linalg.norm(native-target)
                                  / np.linalg.norm(target))
                    leakage = float(np.linalg.norm(native-projector@native)
                                    / np.linalg.norm(native))
                    errors.append(error)
                    leakages.append(leakage)
                    flavor_errors.append(error)
            flavor_rows.append({
                "flavor": flavor,
                "maximum_relative_error": max(flavor_errors),
            })
        rows.append({
            "dt": dt,
            "divisor": divisor,
            "maximum_relative_error": max(errors),
            "maximum_outside_spatial_lift_fraction": max(leakages),
            "flavors": flavor_rows,
        })
    orders = [float(np.log(rows[index-1]["maximum_relative_error"]
                           / rows[index]["maximum_relative_error"])/np.log(2))
              for index in (1, 2)]
    result = {
        "status": "V4_CUBE_INCIDENCE_EXTENDED_PACKET_REFINEMENT",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "rows": rows,
        "observed_error_orders": orders,
        "verdict": (
            "EXTENDED_PACKET_CONTINUUM_TANGENT_SURVIVES"
            if min(orders) > 0.9 and rows[-1]["maximum_relative_error"] < 1e-3
            else "EXTENDED_PACKET_CONTINUUM_TANGENT_OPEN"),
        "claim_boundary": (
            "Fixed side-five tangent refinement with the carrier co-rotating "
            "Dirac comparator. It does not prove nonlinear transport, box "
            "convergence, native current equality, or V4-006."),
    }
    (HERE/"cube_incidence_extended_packet_refinement.json").write_text(
        json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
