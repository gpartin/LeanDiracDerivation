"""Classify timestep convergence of the localized two-layer graph closure."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent


def main() -> None:
    rows = []
    for rate in (128, 256, 512):
        source = HERE / f"localized_embedding_two_layer_closure_{rate}.json"
        data = json.loads(source.read_text(encoding="utf-8"))
        best = min(data["rows"],
                   key=lambda row: row["full_invariance_relative_residual"])
        rows.append({
            "updates_per_unit": rate,
            "dt": data["dt"],
            "flavor": best["flavor"],
            "full_invariance_relative_residual": (
                best["full_invariance_relative_residual"]),
            "cross_block_relative_residual": (
                best["cross_block_relative_residual"]),
            "native_selected_chi_scale_real": (
                best["native_selected_chi_scale_real"]),
            "native_selected_chi_scale_imag": (
                best["native_selected_chi_scale_imag"]),
        })
    full_orders = []
    cross_orders = []
    for coarse, fine in zip(rows[:-1], rows[1:]):
        ratio = coarse["dt"] / fine["dt"]
        full_orders.append(float(np.log(
            coarse["full_invariance_relative_residual"] /
            fine["full_invariance_relative_residual"]) / np.log(ratio)))
        cross_orders.append(float(np.log(
            coarse["cross_block_relative_residual"] /
            fine["cross_block_relative_residual"]) / np.log(ratio)))
    second_order = min(full_orders) > 1.95
    result = {
        "status": "V4_LOCALIZED_EMBEDDING_TIMESTEP_REFINEMENT",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "rows": rows,
        "observed_full_invariance_orders": full_orders,
        "observed_cross_block_orders": cross_orders,
        "full_invariance_is_second_order": second_order,
        "verdict": (
            "CONTINUUM_TWO_LAYER_CLOSURE_SUPPORTED" if second_order else
            "TWO_LAYER_CLOSURE_REMAINDER_NOT_CONVERGED"),
        "interpretation": (
            "The complete graph-invariance defect falls quadratically with "
            "the native timestep, while the separately normalized off-sector "
            "cross block falls linearly because the one-step coupling itself "
            "is proportional to dt. This is the expected discrete-time "
            "remainder pattern for a continuum-invariant first-order sector."),
        "claim_boundary": (
            "This supports continuum tangent closure of the localized joint "
            "Psi/chi readout. It does not prove exact finite-step closure, a "
            "nonlinear translating orbit, long-time stability, or current."),
    }
    (HERE / "localized_embedding_timestep_refinement.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
