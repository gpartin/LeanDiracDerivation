"""Report the current DIRAC-001--009 derivation contract.

This is a read-only status audit, not a commit hook.  The stronger clean
rebuild, hash, axiom and numerical checks live in ``production_guard.py``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "project"


def load(name: str) -> dict:
    return json.loads((PROJECT / name).read_text(encoding="utf-8"))


def contains(path: str, text: str) -> bool:
    return text in (ROOT / path).read_text(encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    status = load("status.json")
    contract = load("derivation_contract.json")
    review = load("no_bolton_review.json")
    stability = json.loads((
        ROOT / "experiments" / "dirac009_native_coupling" /
        "candidate_007_l4_nondegenerate_stability.json"
    ).read_text(encoding="utf-8"))

    checks = {
        "highest_gate_is_009": status.get("highest_passed_gate") == "DIRAC-009",
        "next_gate_is_010": status.get("next_gate") == "DIRAC-010",
        "action_bridge_present": contains(
            "LeanDirac/Dirac009ActionBridge.lean",
            "theorem prepared_action_bloch_bridge"),
        "completed_transport_present": contains(
            "LeanDirac/Dirac006ActionTransport.lean",
            "theorem dirac006_complete_action_transport"),
        "closed_certificate_present": contains(
            "LeanDirac/Dirac009Closed.lean",
            "theorem dirac009_closed_native_clifford"),
        "publication_stops_at_009": contains(
            "LeanDirac/Publication.lean",
            "theorem current_highest_gate_is_dirac009"),
        "period_two_not_used_as_transport": (
            contract["dirac009_mandatory_gates"]["numerical"]
            ["incidence_must_be_active"] is True),
        "nondegenerate_numerics_pass": stability.get("verdict") == "PASS",
        "zero_critical_high": (
            review.get("status") == "PASS_ZERO_CRITICAL_HIGH"
            and not review.get("critical_findings")
            and not review.get("high_findings")),
    }
    passed = all(checks.values())
    report = {
        "action_id": contract["action_id"],
        "checks": checks,
        "verdict": "DIRAC009_READY" if passed else "DIRAC009_OPEN",
        "claim_boundary": status["claim_boundary"],
    }
    print(json.dumps(report, indent=2))
    if args.strict and not passed:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
