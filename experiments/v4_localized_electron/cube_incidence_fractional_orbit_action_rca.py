"""Decompose the finite-translation DEL residual by velocity parity and action term.

At layer zero, positive- and negative-velocity fractional translations share
the exact same current carrier.  Their half-difference isolates the response
odd in translation velocity; their half-sum isolates the even/static part.
Each is then decomposed into the declared terms of the unchanged research
action.  This produces an inverse requirement without installing it.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

import area_bilinear_phase_action_gate as area_phase
import canonical_noncoplanar_bifurcation_audit as native_time
import canonical_noncoplanar_box_transfer as native_box
import cube_incidence_complete_cochain_fractional_orbit_screen as orbit
import cube_incidence_complete_cochain_rigid_orbit_screen as rigid
import cube_incidence_nonlinear_action as action
import cube_incidence_symplectic_basis as cube_basis
import density_oriented_pair_connection as density_connection
import personal_touch_activated_bond_basis as activated


HERE = Path(__file__).resolve().parent
SIDE = 5
CASES = ((64, 1/16), (64, 1/32), (64, 1/64))


def del_of_interval(interval, previous, current, following):
    middle = torch.tensor(current, dtype=torch.float64, requires_grad=True)
    value = (interval(torch.tensor(previous, dtype=torch.float64), middle)
             + interval(middle, torch.tensor(following, dtype=torch.float64)))
    return np.asarray(torch.autograd.grad(value, middle)[0].detach())


def base_interval(left, right):
    return native_box.interval(left, right, SIDE)


def activated_interval(left, right):
    return -native_time.DT*(sum(activated.energies(left))
                            + sum(activated.energies(right)))/2


def area_interval(left, right):
    return -native_time.DT*(area_phase.raw_potential(left)
                            + area_phase.raw_potential(right)) / (
                                2*native_time.CHI0**14)


def density_interval(left, right):
    return (native_time.B*native_time.CHI0
            * density_connection.interval(
                left, right, action.DENSITY_COEFFICIENTS))


def cube_interval(left, right):
    return cube_basis.interval(left, right, action.CUBE_COEFFICIENTS)


TERMS = {
    "canonical_V40": base_interval,
    "activated_positive_bonds": activated_interval,
    "area_bilinear_phase": area_interval,
    "density_oriented_connection": density_interval,
    "cube_incidence_symplectic": cube_interval,
}


def correlation(first, second):
    return float(np.dot(first, second)
                 / max(np.linalg.norm(first)*np.linalg.norm(second), 1e-30))


def main() -> None:
    original_dt = native_time.DT
    bare, _ = action.expected_blocks()
    rest = bare["rest"]
    sites, psi_frame, chi_frame = rigid.carrier_data(rest)
    rows = []
    try:
        for updates, dt in CASES:
            native_time.DT = dt
            theta = float(np.arccos(1-dt**2*native_time.CHI0**2/2))
            speed = 1/(updates*dt)
            for axis in range(3):
                signed = {}
                term_signed = {name: {} for name in TERMS}
                for sign in (-1, 1):
                    previous = orbit.state(
                        rest, axis, sign, -1/updates, -theta)
                    following = orbit.state(
                        rest, axis, sign, 1/updates, theta)
                    current = rest
                    signed[sign] = np.asarray(action.del_residual(
                        torch.tensor(previous, dtype=torch.float64),
                        torch.tensor(current, dtype=torch.float64,
                                     requires_grad=True),
                        torch.tensor(following, dtype=torch.float64)
                    ).detach())
                    for name, interval in TERMS.items():
                        term_signed[name][sign] = del_of_interval(
                            interval, previous, current, following)
                odd = (signed[1]-signed[-1])/2
                even = (signed[1]+signed[-1])/2
                term_rows = []
                reconstructed_odd = np.zeros_like(odd)
                reconstructed_even = np.zeros_like(even)
                for name, values in term_signed.items():
                    term_odd = (values[1]-values[-1])/2
                    term_even = (values[1]+values[-1])/2
                    reconstructed_odd += term_odd
                    reconstructed_even += term_even
                    term_rows.append({
                        "term": name,
                        "odd_norm_over_state_norm": float(
                            np.linalg.norm(term_odd)/np.linalg.norm(rest)),
                        "even_norm_over_state_norm": float(
                            np.linalg.norm(term_even)/np.linalg.norm(rest)),
                        "odd_correlation_with_total": correlation(term_odd, odd),
                        "even_correlation_with_total": correlation(term_even, even),
                    })
                rows.append({
                    "updates_per_cell": updates,
                    "dt": dt,
                    "cell_speed": speed,
                    "axis": axis,
                    "odd_norm_over_state_norm": float(
                        np.linalg.norm(odd)/np.linalg.norm(rest)),
                    "even_norm_over_state_norm": float(
                        np.linalg.norm(even)/np.linalg.norm(rest)),
                    "odd_channels": rigid.projected_channels(
                        odd, sites, psi_frame, chi_frame),
                    "even_channels": rigid.projected_channels(
                        even, sites, psi_frame, chi_frame),
                    "term_rows": term_rows,
                    "term_sum_odd_relative_defect": float(
                        np.linalg.norm(reconstructed_odd-odd)
                        / max(np.linalg.norm(odd), 1e-30)),
                    "term_sum_even_relative_defect": float(
                        np.linalg.norm(reconstructed_even-even)
                        / max(np.linalg.norm(even), 1e-30)),
                })
    finally:
        native_time.DT = original_dt

    speed_rows = []
    for speed in sorted(set(row["cell_speed"] for row in rows)):
        selected = [row for row in rows if row["cell_speed"] == speed]
        speed_rows.append({
            "speed": speed,
            "odd_norm_range": [
                min(row["odd_norm_over_state_norm"] for row in selected),
                max(row["odd_norm_over_state_norm"] for row in selected),
            ],
            "even_norm_range": [
                min(row["even_norm_over_state_norm"] for row in selected),
                max(row["even_norm_over_state_norm"] for row in selected),
            ],
        })
    result = {
        "status": "V4_FRACTIONAL_ORBIT_ACTION_RCA",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "rows": rows,
        "speed_summary": speed_rows,
        "verdict": "FINITE_ORBIT_RESIDUAL_DECOMPOSED",
        "claim_boundary": (
            "Exact DEL inverse diagnosis at the shared rest layer. The odd "
            "residual is a requirement for action research, not an admitted "
            "force or fitted coefficient."),
    }
    output = HERE/"cube_incidence_fractional_orbit_action_rca.json"
    output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n",
                      encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items()
                      if key != "rows"}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
