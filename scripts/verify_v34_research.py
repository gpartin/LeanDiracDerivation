"""Strict local verifier for the V3.4 research branch."""

from pathlib import Path
import re
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

REQUIRED_DECLARATIONS = {
    "V34CovariantSpinors": (
        "v34_positive_spinor_covariant_equation",
        "v34_antiparticle_spinor_covariant_equation",
    ),
    "V34SpinorSectorBasis": (
        "v34_positive_spinors_energy_orthogonal",
        "v34_negative_spinors_energy_orthogonal",
        "v34_every_spinor_has_unique_two_plus_two_coordinates",
    ),
    "V34CovariantSpinSums": (
        "v34_positive_covariant_spin_sum",
        "v34_antiparticle_covariant_spin_sum",
    ),
    "V34MomentumHelicity": (
        "v34_helicity_projector_is_eigenbranch",
    ),
    "V34NonforwardGordon": (
        "v34_nonforward_gordon_for_native_spinors",
        "nativeGordon",
    ),
    "V34PropagatorResidues": (
        "v34_resolvent_spectral_decomposition",
        "v34_positive_spin_sum_is_projector_beta",
    ),
    "V34CompleteCovariantFreeSpinor": (
        "priorFreeSpinor",
        "sectorBasis",
        "nonforwardGordon",
    ),
}

REQUIRED_IMPORTS = {
    "V34MomentumHelicity": {"LeanDirac.V34CovariantSpinors"},
    "V34NonforwardGordon": {"LeanDirac.V34CovariantSpinors"},
    "V34PropagatorResidues": {
        "LeanDirac.V34CovariantSpinSums",
        "LeanDirac.V33FreePropagator",
    },
    "V34CompleteCovariantFreeSpinor": {
        "LeanDirac.V34SpinorSectorBasis",
        "LeanDirac.V34CovariantSpinSums",
        "LeanDirac.V34MomentumHelicity",
        "LeanDirac.V34NonforwardGordon",
        "LeanDirac.V34PropagatorResidues",
    },
}

FORBIDDEN_SERIAL_IMPORTS = {
    "V34MomentumHelicity": {"LeanDirac.V34CovariantSpinSums"},
    "V34NonforwardGordon": {"LeanDirac.V34MomentumHelicity"},
    "V34PropagatorResidues": {"LeanDirac.V34NonforwardGordon"},
}


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
    if re.search(r"(?m)^\s*(axiom|constant|opaque)\s+", source):
        fail(f"new axiom or opaque declaration in {path.name}")
    for declaration in REQUIRED_DECLARATIONS.get(name, ()):
        if declaration not in source:
            fail(f"{path.name} omits required declaration: {declaration}")
    imports = {
        line.removeprefix("import ").strip()
        for line in source.splitlines()
        if line.startswith("import ")
    }
    missing_imports = REQUIRED_IMPORTS.get(name, set()) - imports
    if missing_imports:
        fail(f"{path.name} omits direct imports: {sorted(missing_imports)}")
    serial_imports = FORBIDDEN_SERIAL_IMPORTS.get(name, set()) & imports
    if serial_imports:
        fail(f"{path.name} restores artificial serial imports: {sorted(serial_imports)}")

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
