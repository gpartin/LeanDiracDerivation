"""Classify timestep convergence of the triangle carrier-sector diagnostic."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
RATES = (128, 256, 512, 1024)


def source(rate: int) -> Path:
    suffix = "" if rate == 256 else f"_{rate}"
    return HERE / f"c19_triangle_same_embedding_closure{suffix}.json"


def orders(values: list[float]) -> list[float]:
    return [float(np.log(values[index] / values[index + 1]) / np.log(2.0))
            for index in range(len(values) - 1)]


def main() -> None:
    artifacts = [json.loads(source(rate).read_text(encoding="utf-8"))
                 for rate in RATES]
    own = [row["worst_own_copy_previous_layer_leakage"]
           for row in artifacts]
    combined = [row["worst_both_copy_previous_layer_leakage"]
                for row in artifacts]
    target = [row["worst_following_target_embedding_leakage"]
              for row in artifacts]
    third = [[axis["fixed_charge_center_singular_values"][2]
              for axis in row["corrected_physical_coordinate_axis_rows"]]
             for row in artifacts]
    result = {
        "status": "V4_C19_TRIANGLE_SAME_EMBEDDING_REFINEMENT",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "rates": list(RATES),
        "own_copy_previous_layer_leakage": own,
        "own_copy_leakage_orders": orders(own),
        "combined_copy_previous_layer_leakage": combined,
        "combined_copy_leakage_orders": orders(combined),
        "following_target_embedding_leakage": target,
        "smallest_fixed_charge_singular_values_by_rate": third,
        "smallest_singular_value_orders_by_axis": [
            orders([row[axis] for row in third]) for axis in range(3)],
        "all_finite_steps_rank_three": all(
            row["all_axes_rank_three"] for row in artifacts),
        "continuum_sector_closure_supported": bool(
            min(orders(own)[-2:]) > 0.8
            and min(orders(combined)[-2:]) > 0.8
            and max(target) < 2e-12),
        "verdict": (
            "FROZEN_CUBICAL_SECTOR_LEAKAGE_VANISHES_IN_CONTINUUM"
            if (min(orders(own)[-2:]) > 0.8
                and min(orders(combined)[-2:]) > 0.8
                and max(target) < 2e-12) else
            "FROZEN_CUBICAL_SECTOR_REFINEMENT_FAIL"),
        "interpretation": (
            "Each copy and the combined doublet approach invariant continuum "
            "phase-space sectors with approximately first-order layer-graph "
            "error. The smallest fixed-charge singular response in axes zero "
            "and one also scales toward zero, so finite-step rank three is "
            "not by itself a continuum alpha-generator proof."),
        "claim_boundary": (
            "Linear timestep-refinement result. It does not establish the "
            "full compressed alpha matrices, nonlinear endurance, or V4-006."),
    }
    output = HERE / "c19_triangle_same_embedding_refinement.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
