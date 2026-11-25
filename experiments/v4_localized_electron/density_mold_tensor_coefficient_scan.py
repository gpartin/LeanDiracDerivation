"""Diagnostic strength scan for the traceless C19 mold tensor."""

import json
from pathlib import Path

import density_mold_tensor_probe as probe


HERE = Path(__file__).resolve().parent
STRENGTHS = (-2.0, -1.0, -0.5, 0.0, 0.25, 0.5, 2/3, 1.0,
             1.5, 2.0, 3.0)


def main() -> None:
    probe.rank.lock_inertia()
    rows = [probe.classify("local_plus_traceless", value)
            for value in STRENGTHS]
    best = min(rows, key=lambda row: row["combined_relative_residual"])
    result = {
        "status": "V4_DENSITY_MOLD_TENSOR_STRENGTH_DIAGNOSTIC",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "rows": rows,
        "best_tested_strength": best["tensor_strength"],
        "best_tested_relative_residual": best["combined_relative_residual"],
        "coefficient_rule": (
            "The scan is diagnostic. A movement-minimizing coefficient is "
            "not admissible without independent cubic/normalization provenance."),
    }
    (HERE / "density_mold_tensor_coefficient_scan.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
