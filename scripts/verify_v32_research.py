"""Verify the V3.2 research bundle without claiming release closure."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LEAN_DIR = ROOT / "LeanDirac" / "V2"
FORBIDDEN = re.compile(r"\b(?:sorry|admit|axiom)\b|sorryAx")
BASE_COMMIT = "61a4798b8b3ebfcffd24065079dc4db9d7b43f75"


def run(*command: str) -> str:
    print("+", " ".join(command), flush=True)
    completed = subprocess.run(
        command,
        cwd=ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    print(completed.stdout, end="")
    return completed.stdout


def source_audit() -> None:
    failures: list[str] = []
    for path in sorted(LEAN_DIR.glob("V32*.lean")):
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if FORBIDDEN.search(line):
                failures.append(f"{path.relative_to(ROOT)}:{number}: {line.strip()}")
    if failures:
        raise RuntimeError("forbidden proof token(s):\n" + "\n".join(failures))


def framework_lock_audit() -> None:
    run("git", "merge-base", "--is-ancestor", BASE_COMMIT, "HEAD")
    changed = run(
        "git", "diff", "--name-only", BASE_COMMIT, "--", "LeanDirac/V2"
    ).splitlines()
    forbidden = [
        path for path in changed
        if not Path(path).name.startswith("V32")
    ]
    if forbidden:
        raise RuntimeError(
            "V3.1/core Lean source changed outside V32 modules:\n"
            + "\n".join(forbidden)
        )


def claim_audit() -> None:
    status = (ROOT / "docs" / "V32_STAGE_STATUS.md").read_text(encoding="utf-8")
    blocked = ("FAILED_RCA_ACTIVE", "BLOCKED_UNKNOWN", "NOT_STARTED")
    present = [word for word in blocked if f"| {word} |" in status]
    if present:
        raise RuntimeError("open stage status(es): " + ", ".join(present))


def main() -> int:
    source_audit()
    framework_lock_audit()
    claim_audit()
    output = run(
        "lake",
        "build",
        "LeanDirac/V2/V32CompleteFreeFermion.lean:olean",
    )
    if "sorryAx" in output:
        raise RuntimeError("semantic axiom audit found sorryAx")
    all_v32 = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted(LEAN_DIR.glob("V32*.lean"))
    )
    required = (
        "v32_no_global_exact_local_oriented_root",
        "v32_no_odd_periodic_sum_of_squares_factorization",
        "v32_exact_c19_symbol_is_local_bond_sum",
        "v32_rest_spin_commutant_is_scalar",
        "v32_positive_lower_component_elimination",
        "v32_c19_fourth_phase_moment_isotropic",
        "v32_c19_sixth_order_anisotropy_witness",
        "v32_rest_spinors_are_positive_energy",
        "v32_group_velocity_is_current_over_density",
        "v32_group_speed_bounded_by_lattice_speed_sq",
    )
    missing = [name for name in required if name not in all_v32]
    if missing:
        raise RuntimeError("Stage-2 audit theorem(s) missing: " + ", ".join(missing))
    print("V32_RESEARCH_BUNDLE_PASS")
    print("V32_ROUTE1_CLAIM_AUDIT_PASS")
    print("V32_RELEASE_CANDIDATE_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
