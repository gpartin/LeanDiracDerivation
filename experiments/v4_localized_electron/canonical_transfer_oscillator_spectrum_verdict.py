"""Collect the transfer-oscillator stability scans into one scoped verdict."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent


def read(name: str) -> dict:
    return json.loads((HERE / name).read_text(encoding="utf-8"))


def growth(row: dict) -> float:
    return float(np.log(row["leading_spectral_radius"])/row.get(
        "dt", 0.00390625))


def main() -> None:
    baseline = read("canonical_transfer_oscillator_spectrum_0.json")
    odd = read("canonical_transfer_oscillator_spectrum_0p5.json")
    paired = read(
        "canonical_transfer_oscillator_spectrum_0p5_even_0p15.json")
    half = read(
        "canonical_transfer_oscillator_spectrum_0p5_even_0p15_dt_div_2.json")
    quarter = read(
        "canonical_transfer_oscillator_spectrum_0p5_even_0p15_dt_div_4.json")
    parallel = read(
        "canonical_transfer_oscillator_spectrum_0p5_even_0p15_parallel_0p1.json")
    rows = []
    for label, row in (("canonical", baseline), ("odd", odd),
                       ("odd_even", paired), ("dt_half", half),
                       ("dt_quarter", quarter),
                       ("parallel_stiffness", parallel)):
        rows.append({
            "label": label,
            "odd_strength": row.get("strength", 0.0),
            "even_strength": row.get("even_strength", 0.0),
            "parallel_strength": row.get("parallel_strength", 0.0),
            "dt": row.get("dt", 0.00390625),
            "spectral_radius": row["leading_spectral_radius"],
            "growth_rate": growth(row),
            "ritz_count": row["computed_eigenvalue_count"],
        })
    refinement_rates = [growth(paired), growth(half), growth(quarter)]
    result = {
        "status": "V4_TRANSFER_OSCILLATOR_SPECTRUM_VERDICT",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "rows": rows,
        "canonical_to_best_radius_reduction_fraction": float(
            (baseline["leading_spectral_radius"]
             - paired["leading_spectral_radius"])
            / (baseline["leading_spectral_radius"]-1)),
        "refined_growth_rates": refinement_rates,
        "growth_rate_relative_spread": float(
            (max(refinement_rates)-min(refinement_rates))
            / np.mean(refinement_rates)),
        "leading_mode_classification": paired[
            "leading_mode_classification"],
        "parallel_stiffness_radius_change": float(
            parallel["leading_spectral_radius"]
            - paired["leading_spectral_radius"]),
        "verdict": "TRANSFER_OSCILLATOR_SPECTRUM_GATE_FAIL",
        "promotion_consequence": (
            "Do not admit the action or claim finite localized Dirac motion; "
            "retain it as a rest-preserving rank-three response basis."),
        "claim_boundary": (
            "Leading Ritz spectrum of the side-five exact rest orbit. This "
            "is a scoped failure of the tested odd/even transfer-oscillator "
            "basis, not a universal two-field no-go."),
    }
    output = HERE / "canonical_transfer_oscillator_spectrum_verdict.json"
    output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n",
                      encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
