"""Full rest-spectrum gate for the transfer-driven chi-frame reflection."""

from __future__ import annotations

import json
from pathlib import Path

import chi_transfer_softening_rest_spectrum as softening
import density_mold_rest_spectrum as rest
import edge_snap_action_rank as rank
import transfer_activated_chi_bond_rest_spectrum as activated
import transfer_neighbor_connection_probe as connection


HERE = Path(__file__).resolve().parent
COEFFICIENT = 1.0 / (2.0 * rank.R)


def interval_action(left, right, coefficient: float):
    return (
        softening.interval_action(left, right, 19.0)
        + 0.5 * activated.extra_action(left, right)
        + connection.interval_action_delta(
            left, right, rest.DT, coefficient, "frame_reflection")
    )


def classify(coefficient: float):
    original = rest.interval_action
    rest.interval_action = interval_action
    try:
        return rest.classify(coefficient)
    finally:
        rest.interval_action = original


def main() -> None:
    rank.lock_inertia()
    rows = []
    for name, coefficient in (
            ("incumbent", 0.0),
            ("native_frame_reflection", COEFFICIENT)):
        row = classify(coefficient)
        row["name"] = name
        row["coefficient"] = coefficient
        rows.append(row)
    incumbent, candidate = rows
    growth_delta = (candidate["growth_rate_per_unit_time"]
                    - incumbent["growth_rate_per_unit_time"])
    passed = growth_delta <= 1e-9
    result = {
        "status": (
            "V4_TRANSFER_NEIGHBOR_REFLECTION_REST_GATE_PASS" if passed else
            "V4_TRANSFER_NEIGHBOR_REFLECTION_REST_GATE_FAIL"),
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "side": rest.SIDE,
        "dt": rest.DT,
        "rows": rows,
        "growth_rate_delta": float(growth_delta),
        "coefficient_provenance": "1/(2 chi0), midpoint and chi0 normalization",
        "verdict": (
            "RETAIN_FOR_MOVEMENT_RCA" if passed else
            "RETIRE_FRAME_REFLECTION"),
        "claim_boundary": (
            "Complete side-three two-layer rest-spectrum gate only; the "
            "carrier remains unstable and no movement or Dirac gate passes."),
    }
    (HERE / "transfer_neighbor_reflection_rest_spectrum.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
