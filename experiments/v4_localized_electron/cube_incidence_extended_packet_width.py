"""Width and momentum robustness of the extended native packet bridge."""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
from scipy.linalg import lu_factor, lu_solve

import canonical_noncoplanar_bifurcation_audit as native_time
import cube_incidence_extended_packet_gate as gate
import cube_incidence_nonlinear_action as action
import current_action_localized_envelope_closure as envelope


HERE = Path(__file__).resolve().parent


def main():
    bare, matrices = action.expected_blocks()
    mass, gyro, stiffness = (matrices[name]
                             for name in ("mass", "gyro", "stiffness"))
    dt = native_time.DT/4
    factor = lu_factor(mass/dt+gyro/2)
    advance = -lu_solve(factor, dt*stiffness-2*mass/dt)
    memory = -lu_solve(factor, mass/dt-gyro/2)
    psi_frame, chi_frame, sites = envelope.carrier_frames(bare["rest"])
    prior = json.loads((HERE/"cube_incidence_dirac_envelope_gate.json").read_text())
    scales = {
        row["flavor"]: complex(
            next(case for case in row["cases"]
                 if case["envelope"] == "all_125_centers")["chi_scale_real"],
            next(case for case in row["cases"]
                 if case["envelope"] == "all_125_centers")["chi_scale_imag"])
        for row in prior["rows"]
    }
    rows = []
    for width, wave_number in itertools.product(
            (0.95, 1.15, 1.35, 1.6), (0.2, 0.35)):
        errors = []
        leakages = []
        for flavor, basis in gate.flavor_bases().items():
            alpha, beta = gate.compressed_clifford(basis)
            raw = envelope.raw_columns(
                basis, psi_frame, chi_frame, sites, gate.OFFSETS,
                scales[flavor])
            q = envelope.orth(raw)
            projector = q@q.T
            for axis, sign in itertools.product(range(4), (-1, 1)):
                previous, current, following, _ = gate.packet(
                    axis, sign, alpha, beta, dt, width, wave_number)
                native = advance@(raw@gate.real_coordinates(current))
                native += memory@(raw@gate.real_coordinates(previous))
                target = raw@gate.real_coordinates(following)
                errors.append(float(np.linalg.norm(native-target)
                                    / np.linalg.norm(target)))
                leakages.append(float(np.linalg.norm(native-projector@native)
                                      / np.linalg.norm(native)))
        rows.append({
            "width": width,
            "carrier_wave_number": wave_number,
            "maximum_relative_error": max(errors),
            "maximum_outside_spatial_lift_fraction": max(leakages),
        })
    worst = max(row["maximum_relative_error"] for row in rows)
    result = {
        "status": "V4_CUBE_INCIDENCE_EXTENDED_PACKET_WIDTH",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "dt": dt,
        "rows": rows,
        "maximum_relative_error": worst,
        "verdict": ("EXTENDED_PACKET_WIDTH_ROBUST"
                    if worst < 1e-3 else "EXTENDED_PACKET_WIDTH_OPEN"),
        "claim_boundary": (
            "Side-five tangent packet-width and momentum sensitivity. This "
            "is not box refinement or nonlinear translation."),
    }
    (HERE/"cube_incidence_extended_packet_width.json").write_text(
        json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
