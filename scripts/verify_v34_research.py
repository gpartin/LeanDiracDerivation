"""Strict local verifier for the V3.4 research branch."""

from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
STAGES = [
    "V34CovariantSpinors",
    "V34SpinorSectorBasis",
    "V34CovariantSpinSums",
    "V34MomentumHelicity",
    "V34NonforwardGordon",
    "V34PropagatorResidues",
    "V34CompleteCovariantFreeSpinor",
]


def fail(message: str) -> None:
    print(f"[FAIL] {message}")
    raise SystemExit(1)


for name in STAGES:
    path = ROOT / "LeanDirac" / f"{name}.lean"
    if not path.is_file():
        fail(f"missing stage: {path}")
    source = path.read_text(encoding="utf-8")
    if "sorry" in source or "admit" in source:
        fail(f"unfinished proof marker in {path.name}")
    if "\naxiom " in source or "\nconstant " in source:
        fail(f"new axiom or opaque constant declaration in {path.name}")

plan = (ROOT / "docs" / "V34_COVARIANT_FREE_SPINOR_CLOSURE_PLAN.md").read_text(
    encoding="utf-8"
)
for required in (
    "Required pass gates",
    "Disallowed live mechanisms",
    "Planned live mechanisms",
    "Claim boundary",
    "Seven meaningful handoffs",
):
    if required not in plan:
        fail(f"plan omits required section: {required}")

commands = [
    ["lake", "build", "LeanDirac.V34CompleteCovariantFreeSpinor"],
    [sys.executable, "scripts/verify_v32_research.py"],
    [sys.executable, "scripts/verify_v33_research.py"],
]
for command in commands:
    print("[RUN] " + " ".join(command))
    result = subprocess.run(command, cwd=ROOT)
    if result.returncode != 0:
        fail("verification command failed: " + " ".join(command))

axiom_command = [
    "lake",
    "env",
    "lean",
    "LeanDirac/V34CompleteCovariantFreeSpinor.lean",
]
print("[RUN] " + " ".join(axiom_command))
axiom_result = subprocess.run(
    axiom_command, cwd=ROOT, text=True, capture_output=True
)
print(axiom_result.stdout, end="")
print(axiom_result.stderr, end="", file=sys.stderr)
if axiom_result.returncode != 0:
    fail("cumulative axiom audit failed to compile")
combined = axiom_result.stdout + axiom_result.stderr
if "sorryAx" in combined:
    fail("cumulative theorem depends on sorryAx")
for expected in ("propext", "Classical.choice", "Quot.sound"):
    if expected not in combined:
        fail(f"axiom audit omitted expected standard axiom: {expected}")

print("[PASS] V3.4 research verification")
sys.exit(0)
