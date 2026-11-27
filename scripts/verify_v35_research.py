"""Strict local verifier for the V3.5 research branch."""

from pathlib import Path
import re
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
STAGES = [
    "V35FrameworkAndGapLock",
    "V35AntiparticleResidueClosure",
    "V35GeneratedLorentzTransformations",
    "V35CovariantSolutionTransport",
    "V35CompleteBilinears",
    "V35BilinearCliffordClosure",
    "V35MovingSpinInvariant",
    "V35CompleteCovariantObservables",
]

REQUIRED_DECLARATIONS = {
    "V35FrameworkAndGapLock": ("v35_framework_and_gap_lock",),
    "V35AntiparticleResidueClosure": (
        "v35_antiparticle_spin_sum_is_negative_projector_beta",
        "v35_antiparticle_spin_sum_is_negative_pole",
    ),
    "V35GeneratedLorentzTransformations": (
        "v35_apply_lorentz_word_append",
        "v35_apply_inverse_lorentz_word",
        "v35_apply_lorentz_word_inverse",
    ),
    "V35CovariantSolutionTransport": (
        "v35_finite_boost_mass_shell",
        "v35_finite_boost_covariant",
        "v35_finite_rotation_covariant",
        "v35_finite_boost_preserves_solutions",
        "v35_lorentz_word_covariant",
        "v35_lorentz_word_preserves_solutions",
    ),
    "V35CompleteBilinears": (
        "v35_gamma_five_square",
        "v35_gamma_five_anticommutes",
        "v35_tensor_bilinear_antisymmetric",
    ),
    "V35BilinearCliffordClosure": (
        "v35_bilinear_channel_count",
        "v35_spinor_channel_normalization_eq_quarter",
        "v35_tensor_pair_normalization_eq_half",
        "v35_sigma_antisymmetric",
        "v35_fierz_completeness",
    ),
    "V35MovingSpinInvariant": (
        "v35_pauli_lubanski_square_at_momentum",
        "v35_on_shell_pauli_lubanski_spin_half",
        "v35_helicity_is_normalized_pauli_lubanski_time",
    ),
    "V35CompleteCovariantObservables": (
        "v35_complete_covariant_observables",
        "antiparticleResidue",
        "movingSpin",
    ),
}


def fail(message: str) -> None:
    print(f"[FAIL] {message}")
    raise SystemExit(1)


for name in STAGES:
    path = ROOT / "LeanDirac" / f"{name}.lean"
    if not path.is_file():
        fail(f"missing stage: {path}")
    source = path.read_text(encoding="utf-8")
    if re.search(r"\b(sorry|admit)\b", source):
        fail(f"unfinished proof marker in {path.name}")
    if re.search(r"(?m)^\s*(axiom|constant|opaque)\s+", source):
        fail(f"new axiom or opaque declaration in {path.name}")
    for declaration in REQUIRED_DECLARATIONS[name]:
        if declaration not in source:
            fail(f"{path.name} omits required declaration: {declaration}")

plan = (ROOT / "docs" / "V35_COVARIANT_OBSERVABLE_CLOSURE_PLAN.md").read_text(
    encoding="utf-8"
)
for required in (
    "Required pass gates",
    "Disallowed live mechanisms",
    "Planned live mechanisms",
    "Claim boundary",
    "Eight meaningful handoffs",
):
    if required not in plan:
        fail(f"plan omits required section: {required}")

commands = [
    ["lake", "build", "LeanDirac.V35CompleteCovariantObservables"],
    [sys.executable, "scripts/verify_v32_research.py"],
    [sys.executable, "scripts/verify_v33_research.py"],
    [sys.executable, "scripts/verify_v34_research.py"],
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
    "LeanDirac/V35CompleteCovariantObservables.lean",
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

print("[PASS] V3.5 research verification")
sys.exit(0)
