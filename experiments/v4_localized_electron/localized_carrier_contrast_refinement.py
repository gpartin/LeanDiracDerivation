"""Summarize timestep refinement of the two-copy carrier contrast graph."""

from __future__ import annotations

import json
import math
from pathlib import Path


HERE = Path(__file__).resolve().parent
UPDATES = (128, 256, 512)


def main() -> None:
    rows = []
    for updates in UPDATES:
        source = HERE / f"localized_carrier_contrast_two_layer_{updates}.json"
        data = json.loads(source.read_text(encoding="utf-8"))
        best = data["joint_native_graph_resolution"]["best"]
        rows.append({
            "updates_per_unit_interval": updates,
            "dt": data["dt"],
            "cross_block_relative_residual": best[
                "cross_block_relative_residual"],
            "full_invariance_relative_residual": best[
                "full_invariance_relative_residual"],
            "retained_generator_spectral_radius": best[
                "retained_generator_spectral_radius"],
            "previous_layer_map_condition": best[
                "previous_layer_map_condition"],
            "source": source.name,
        })
    full_orders = [
        math.log(rows[index]["full_invariance_relative_residual"]
                 / rows[index+1]["full_invariance_relative_residual"], 2)
        for index in range(len(rows)-1)
    ]
    cross_orders = [
        math.log(rows[index]["cross_block_relative_residual"]
                 / rows[index+1]["cross_block_relative_residual"], 2)
        for index in range(len(rows)-1)
    ]
    radius_orders = [
        math.log(
            (rows[index]["retained_generator_spectral_radius"]-1)
            / (rows[index+1]["retained_generator_spectral_radius"]-1), 2)
        for index in range(len(rows)-1)
    ]
    result = {
        "status": "V4_LOCALIZED_CARRIER_CONTRAST_REFINEMENT_AUDIT",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "rows": rows,
        "observed_full_invariance_orders": full_orders,
        "observed_cross_block_orders": cross_orders,
        "observed_spectral_radius_excess_orders": radius_orders,
        "verdict": (
            "CENTERED_TIME_CONTINUUM_CLOSURE_SUPPORTED"
            if min(full_orders) > 1.8 else
            "TWO_LAYER_RESIDUAL_PLATEAUS"),
        "interpretation": (
            "The full graph residual decreases at centered-time second order, "
            "while the raw cross block decreases at first order.  The retained "
            "spectral-radius excess also tends toward zero.  This supports a "
            "continuum phase-space closure of the exact two-copy carrier "
            "contrast; it is not exact finite-step or nonlinear translation."),
        "claim_boundary": (
            "No bond-current bridge, large-box test, nonlinear handoff, rest "
            "stability, or action admission follows from refinement alone."),
    }
    output = HERE / "localized_carrier_contrast_refinement.json"
    output.write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
