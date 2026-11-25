"""Diagnostic coefficient screen for one-cycle extended-packet transport.

The screen asks whether the existing two reciprocal connection terms span a
long-lived transport repair.  Any optimum is an inverse diagnostic only; a
coefficient fitted to Dirac is not admissible.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import splu

import canonical_noncoplanar_bifurcation_audit as native_time
import cube_incidence_extended_packet_gate as gate
import cube_incidence_memory_channel_ablation as ablation
import cube_incidence_nonlinear_action as action
import current_action_localized_envelope_closure as envelope


HERE = Path(__file__).resolve().parent


def sparse(value):
    result = value.copy()
    result[abs(result) < 1e-13] = 0
    return csc_matrix(result)


def phase_aligned_error(actual, target, target_i):
    basis = np.column_stack((target, target_i))
    coefficients, *_ = np.linalg.lstsq(basis, actual, rcond=1e-12)
    coefficients /= max(np.linalg.norm(coefficients), 1e-30)
    aligned = basis@coefficients
    return float(np.linalg.norm(actual-aligned)/np.linalg.norm(target))


def main():
    bare, _ = action.expected_blocks()
    blocks = ablation.blocks()
    base = blocks["base_without_density_or_cube_connection"]
    with_density = blocks["base_plus_density_connection"]
    with_cube = blocks["base_plus_cube_incidence_connection"]
    density = tuple(value-base[index]
                    for index, value in enumerate(with_density))
    cube = tuple(value-base[index] for index, value in enumerate(with_cube))
    dt = native_time.DT/16
    steps = int(np.ceil(2*np.pi/native_time.CHI0/dt))
    psi_frame, chi_frame, sites = envelope.carrier_frames(bare["rest"])
    prior = json.loads((HERE/"cube_incidence_dirac_envelope_gate.json").read_text())
    row0 = next(row for row in prior["rows"] if row["flavor"] == "plus")
    case = next(item for item in row0["cases"]
                if item["envelope"] == "all_125_centers")
    scale = complex(case["chi_scale_real"], case["chi_scale_imag"])
    basis = gate.flavor_bases()["plus"]
    alpha, beta = gate.compressed_clifford(basis)
    raw = envelope.raw_columns(
        basis, psi_frame, chi_frame, sites, gate.OFFSETS, scale)
    previous, initial, _following, _ = gate.packet(0, 1, alpha, beta, dt)
    q_previous0 = raw@gate.real_coordinates(previous)
    q_current0 = raw@gate.real_coordinates(initial)
    target = gate.dirac_step(initial, alpha, beta, 1, steps*dt)
    target_raw = raw@gate.real_coordinates(target)
    target_i = raw@gate.real_coordinates(1j*target)
    rows = []
    for density_factor, cube_factor in itertools.product(
            (0.0, 1.0, 5.0, 10.0, 20.0, 50.0, 100.0),
            (-0.5, 0.0, 0.1, 0.25, 0.5, 1.0)):
        values = tuple(base[index]
                       + density_factor*density[index]
                       + cube_factor*cube[index] for index in range(3))
        mass, gyro, stiffness = (sparse(value) for value in values)
        lhs = (mass/dt+gyro/2).tocsc()
        cb = (dt*stiffness-2*mass/dt).tocsc()
        pb = (mass/dt-gyro/2).tocsc()
        try:
            factor = splu(lhs)
        except RuntimeError:
            rows.append({"density_factor": density_factor,
                         "cube_factor": cube_factor,
                         "factorization_failed": True})
            continue
        q_previous, q_current = q_previous0.copy(), q_current0.copy()
        maximum_norm = np.linalg.norm(q_current)
        unstable = False
        for _ in range(steps):
            following = factor.solve(-(cb@q_current+pb@q_previous))
            if not np.all(np.isfinite(following)) or np.linalg.norm(following) > 1e100:
                unstable = True
                break
            q_previous, q_current = q_current, following
            maximum_norm = max(maximum_norm, np.linalg.norm(q_current))
        if unstable:
            rows.append({
                "density_factor": density_factor,
                "cube_factor": cube_factor,
                "unstable_or_nonfinite": True,
            })
            continue
        rows.append({
            "density_factor": density_factor,
            "cube_factor": cube_factor,
            "phase_aligned_shape_error": phase_aligned_error(
                q_current, target_raw, target_i),
            "final_norm_ratio": float(
                np.linalg.norm(q_current)/np.linalg.norm(target_raw)),
            "maximum_norm_ratio": float(
                maximum_norm/np.linalg.norm(target_raw)),
        })
    valid = [row for row in rows if "phase_aligned_shape_error" in row]
    best = min(valid, key=lambda row: row["phase_aligned_shape_error"])
    result = {
        "status": "V4_CUBE_INCIDENCE_LONG_TRANSPORT_COEFFICIENT_SCREEN",
        "canonical_action_changed": False,
        "research_action_changed": False,
        "dt": dt,
        "one_cycle_steps": steps,
        "rows": rows,
        "best_diagnostic_row": best,
        "verdict": ("EXISTING_CONNECTION_SPAN_HAS_LOW_RESIDUAL_DIAGNOSTIC"
                    if best["phase_aligned_shape_error"] < 0.05
                    else "EXISTING_CONNECTION_SPAN_LONG_TRANSPORT_DEFICIENT"),
        "claim_boundary": (
            "Coefficients are scanned against Dirac and therefore cannot be "
            "admitted. The result only tests response sufficiency of this "
            "two-term action span."),
    }
    (HERE/"cube_incidence_long_transport_coefficient_screen.json").write_text(
        json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items()
                      if key != "rows"}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
