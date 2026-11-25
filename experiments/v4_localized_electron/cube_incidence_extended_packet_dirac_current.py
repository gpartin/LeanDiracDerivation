"""Pull the positive-frequency density/current through the native packet lift.

The V2 theorem identifies this four-component current with the native C19
bond current in the closed free sector.  Here we test whether the localized
joint Psi--chi lift preserves that same readout under its native tangent
update.  No current is fed back into the update.
"""

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
import localized_stability as c19


HERE = Path(__file__).resolve().parent
SIDE = gate.SIDE


def c19_velocity(momentum):
    """Derivative of the three C19 first symbols with respect to momentum."""
    result = np.zeros((3, 3))
    for offset, weight in c19.DIRECTIONS:
        direction = np.asarray(offset, dtype=float)
        result += (weight/12)*np.outer(direction, direction)*np.cos(
            momentum@direction)
    return result


def observables(field, alpha):
    fourier = np.fft.fftn(field, axes=(0, 1, 2))
    frequencies = 2*np.pi*np.fft.fftfreq(SIDE)
    density = float(np.sum(abs(fourier)**2)/SIDE**3)
    current = np.zeros(3)
    for index in itertools.product(range(SIDE), repeat=3):
        momentum = np.asarray([frequencies[value] for value in index])
        velocity = c19_velocity(momentum)
        spinor = fourier[index]
        for direction in range(3):
            operator = sum(velocity[component, direction]*alpha[component]
                           for component in range(3))
            current[direction] += float(
                np.real(np.vdot(spinor, operator@spinor)))/SIDE**3
    return density, current


def complex_field(real_coordinates):
    return (real_coordinates.reshape(-1, 2)
            @np.asarray((1.0, 1.0j))).reshape(SIDE, SIDE, SIDE, 4)


def main():
    bare, matrices = action.expected_blocks()
    mass, gyro, stiffness = (matrices[name]
                             for name in ("mass", "gyro", "stiffness"))
    # The state and current gates are both evaluated in the proved
    # centered-time refinement regime.  The coarser values are recorded by
    # cube_incidence_extended_packet_refinement.py; this final current sample
    # uses one further halving to separate a genuine current mismatch from the
    # already-observed second-order timestep remainder.
    dt = native_time.DT/8
    factor = lu_factor(mass/dt+gyro/2)
    advance = -lu_solve(factor, dt*stiffness-2*mass/dt)
    memory = -lu_solve(factor, mass/dt-gyro/2)
    psi_frame, chi_frame, sites = envelope.carrier_frames(bare["rest"])
    prior = json.loads((HERE/"cube_incidence_dirac_envelope_gate.json").read_text())
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
            previous, current, following, momentum = gate.packet(
                axis, sign, alpha, beta, dt)
            native_raw = advance@(raw@gate.real_coordinates(current))
            native_raw += memory@(raw@gate.real_coordinates(previous))
            recovered, *_ = np.linalg.lstsq(raw, native_raw, rcond=1e-10)
            native = complex_field(recovered)
            density_native, current_native = observables(native, alpha)
            density_target, current_target = observables(following, alpha)
            current_scale = max(np.linalg.norm(current_target), 1e-30)
            rows.append({
                "flavor": flavor,
                "direction": "diagonal" if axis == 3 else "xyz"[axis],
                "sign": sign,
                "momentum": momentum.tolist(),
                "density_relative_error": abs(density_native-density_target)
                    / density_target,
                "current_relative_error": float(
                    np.linalg.norm(current_native-current_target)/current_scale),
                "native_current": current_native.tolist(),
                "dirac_current": current_target.tolist(),
            })
    density_error = max(row["density_relative_error"] for row in rows)
    current_error = max(row["current_relative_error"] for row in rows)
    result = {
        "status": "V4_CUBE_INCIDENCE_EXTENDED_PACKET_DIRAC_CURRENT",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "dt": dt,
        "rows": rows,
        "maximum_density_relative_error": density_error,
        "maximum_current_relative_error": current_error,
        "verdict": ("EXTENDED_PACKET_DIRAC_CURRENT_PASS"
                    if density_error < 1e-3 and current_error < 1e-2
                    else "EXTENDED_PACKET_DIRAC_CURRENT_OPEN"),
        "claim_boundary": (
            "Positive-frequency C19 density/current pulled through the V2 "
            "readout at tangent order. The failed raw nonlinear phase-charge "
            "contraction is retained separately. This is not a nonlinear "
            "Noether-current or long-time translation proof."),
    }
    (HERE/"cube_incidence_extended_packet_dirac_current.json").write_text(
        json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items()
                      if key != "rows"}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
