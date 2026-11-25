"""Locate the first unresolved channel in a stored edge-orbit residual.

The historical filename is retained because other research scripts import it,
but the classifier now reads the number of residual layers from the artifact.
This prevents a timestep-refined orbit from being silently forced back into
the original four-layer shape.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

import edge_snap_action_rank as rank


HERE = Path(__file__).resolve().parent
SIDE = 5


def main(input_stem: str = "edge_snap_four_update_orbit",
         output_stem: str = "edge_snap_four_update_rca") -> None:
    data = np.load(HERE / f"{input_stem}.npz")
    source = json.loads((HERE / f"{input_stem}.json").read_text(
        encoding="utf-8"))
    packed = data["residuals"]
    layer_width = SIDE * SIDE * SIDE * 4 * 3
    if packed.size % layer_width:
        raise ValueError(
            f"residual array of size {packed.size} is not a whole number "
            f"of {layer_width}-entry lattice layers")
    layer_count = packed.size // layer_width
    residuals = packed.reshape(
        layer_count, SIDE, SIDE, SIDE, 4, 3)
    complex_psi = residuals[..., 0, :] + 1j * residuals[..., 1, :]
    complex_chi = residuals[..., 2, :] + 1j * residuals[..., 3, :]
    power_psi = abs(complex_psi)**2
    power_chi = abs(complex_chi)**2
    total = float(np.sum(power_psi) + np.sum(power_chi))

    amplitude_sq = rank.R**2 / (2 + 54 / rank.R**2)
    psi, _ = rank.native.carrier_fields(SIDE, amplitude_sq)
    core = rank.native.norm2(psi) > 0
    shell = np.zeros_like(core)
    for site in zip(*np.nonzero(core)):
        for offset, _ in rank.native.DIRECTIONS:
            neighbor = tuple((site[d] + offset[d]) % SIDE for d in range(3))
            shell[neighbor] = True
    shell &= ~core
    distant = ~(core | shell)

    rows = []
    for layer in range(layer_count):
        layer_psi = power_psi[layer]
        layer_chi = power_chi[layer]
        layer_total = float(np.sum(layer_psi) + np.sum(layer_chi))
        rows.append({
            "layer": layer,
            "fraction_of_total_residual": layer_total / total,
            "psi_fraction": float(np.sum(layer_psi) / layer_total),
            "chi_fraction": float(np.sum(layer_chi) / layer_total),
            "chi_component_fractions": (
                np.sum(layer_chi, axis=(0, 1, 2))
                / max(float(np.sum(layer_chi)), 1e-300)).tolist(),
            "support_fractions": {
                "core": float((np.sum(layer_psi[core])
                               + np.sum(layer_chi[core])) / layer_total),
                "first_shell": float((np.sum(layer_psi[shell])
                                      + np.sum(layer_chi[shell])) / layer_total),
                "distant": float((np.sum(layer_psi[distant])
                                  + np.sum(layer_chi[distant])) / layer_total),
            },
        })

    component_power = np.stack((power_psi, power_chi), axis=-2)
    maximum = np.unravel_index(
        int(np.argmax(component_power)), component_power.shape)
    # layer,x,y,z,field,component
    first_nonzero_layer = next(
        index for index, row in enumerate(rows)
        if row["fraction_of_total_residual"] > 1e-14)
    field_names = ("psi", "chi")
    result = {
        "status": "V4_EDGE_ORBIT_RCA",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "action_variant": source.get("action_variant", "unspecified"),
        "updates": source.get("updates", layer_count),
        "residual_layer_count": int(layer_count),
        "rows": rows,
        "first_departure_layer": first_nonzero_layer,
        "largest_residual": {
            "layer": int(maximum[0]),
            "site": [int(value) for value in maximum[1:4]],
            "field": field_names[int(maximum[4])],
            "component": int(maximum[5]),
            "fraction_of_total_residual": float(
                component_power[maximum] / total),
        },
        "aggregate_field_fractions": {
            "psi": float(np.sum(power_psi) / total),
            "chi": float(np.sum(power_chi) / total),
        },
        "issue": (
            "The charge-guided four-update path advances edge-to-edge but "
            "does not satisfy the native discrete Euler-Lagrange equations."),
        "why_it_matters": (
            "Without closure, the path is a prepared interpolation rather "
            "than a freely moving electron."),
        "symptoms": (
            "The matched search retains a finite action residual even after "
            "enforcing the two native phase-charge continuities."),
        "expected_symptoms": (
            "A moving branch would drive every layer residual to numerical "
            "zero while preserving carrier identity and charge."),
        "root_cause": (
            "The unresolved response is localized below by field, component, "
            "site and layer.  It is evidence for the next action channel, not "
            "for increasing the launch amplitude."),
        "corrective_action": (
            "Project the residual against the remaining cubic C19 three-site "
            "and plaquette action bases and admit only a positive reciprocal "
            "term with nonzero response in this exact channel."),
        "claim_boundary": (
            "This classifies one optimized path; it is not an action-class "
            "no-go or a Dirac movement pass."),
    }
    (HERE / f"{output_stem}.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-stem", default="edge_snap_four_update_orbit")
    parser.add_argument("--output-stem", default="edge_snap_four_update_rca")
    args = parser.parse_args()
    main(args.input_stem, args.output_stem)
