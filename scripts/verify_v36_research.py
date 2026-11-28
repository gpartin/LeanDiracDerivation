"""Strict local verifier for the V3.6 geometrically normalized action."""

from pathlib import Path
import re
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
STAGES = [
    "V36CubicalVectorBivectorIsometry",
    "V36FrameworkAndNormalizationLock",
    "V36FreeDiracNormalizationIndependence",
    "V36KineticMetricClassification",
    "V36RadialResponse",
    "V36NormalizationDecisionSurface",
    "V36CompleteGeometricallyNormalizedAction",
]

REQUIRED_DECLARATIONS = {
    "V36CubicalVectorBivectorIsometry": (
        "v36_cubical_vector_bivector_isometry",
        "cube_vector_bivector_density_isometry",
    ),
    "V36FrameworkAndNormalizationLock": (
        "v36_framework_and_normalization_lock",
        "v36_live_foundation_is_equal_metric_pair",
        "v36_live_action_is_B_one",
    ),
    "V36FreeDiracNormalizationIndependence": (
        "v36_free_dirac_normalization_independence",
        "v36_live_research_action_mode_to_free_dirac",
        "periodicV36ResearchLagrangianTangentQuotient",
    ),
    "V36KineticMetricClassification": (
        "v36_kinetic_metric_classification",
        "v36_equal_inherited_metric_forces_B_one",
        "v36_c19_weighted_metric_forces_B_nineteen",
        "v36_common_cubic_turn_preserves_every_relative_weight",
    ),
    "V36RadialResponse": (
        "v36_radial_response",
        "v36_radial_scalar_potential_is_native_site_slice",
        "v36_radial_force_expansion",
        "v36_candidate_radial_frequency_squares_strictly_ordered",
    ),
    "V36NormalizationDecisionSurface": (
        "v36_normalization_decision_surface",
        "v36_candidate_B_values_pairwise_distinct",
    ),
    "V36CompleteGeometricallyNormalizedAction": (
        "v36_complete_geometrically_normalized_action",
        "liveActionToFreeDirac",
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

foundation = (ROOT / "LeanDirac" / "Foundation.lean").read_text(encoding="utf-8")
native_action = (ROOT / "LeanDirac" / "NativeAction.lean").read_text(
    encoding="utf-8"
)
for path_name, source in (
    ("Foundation.lean", foundation),
    ("NativeAction.lean", native_action),
):
    if re.search(r"\b(sorry|admit)\b", source):
        fail(f"unfinished proof marker in {path_name}")
    if re.search(r"(?m)^\s*(axiom|constant|opaque)\s+", source):
        fail(f"new axiom or opaque declaration in {path_name}")

for required in (
    "def chiKineticNormalization",
    "density (cubicalVectorToBivector cubicalReferenceUnitVector)",
    "theorem cube_vector_bivector_density_isometry",
    "theorem kappa_eq_nineteen",
    "def v36ResearchKinetic",
):
    if required not in foundation:
        fail(f"Foundation.lean omits revised-action declaration: {required}")

for required in (
    "def v36ResearchC19Lagrangian",
    "def v36ResearchC19Hamiltonian",
    "v36_research_c19_lagrangian_uses_equal_grade_kinetic_weight",
):
    if required not in native_action:
        fail(f"NativeAction.lean omits revised-action declaration: {required}")

plan = (ROOT / "docs" / "V36_GEOMETRICALLY_NORMALIZED_ACTION_PLAN.md").read_text(
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

release_files = {
    "README.md": (
        "V3.6",
        "v36_live_research_action_mode_to_free_dirac",
        "python scripts/verify_release.py",
    ),
    "docs/REPRODUCIBILITY.md": (
        "V36_RELEASE_VERIFICATION_PASS",
        "python scripts/verify_release.py",
    ),
    "docs/PROOF_CHAIN.md": (
        "V3.2",
        "V3.3",
        "V3.4",
        "V3.5",
        "V3.6",
    ),
    "CITATION.cff": ("3.6.0-research",),
    ".zenodo.json": ("V3.6",),
}
for relative_path, required_fragments in release_files.items():
    source = (ROOT / relative_path).read_text(encoding="utf-8")
    for required in required_fragments:
        if required not in source:
            fail(f"{relative_path} omits release marker: {required}")

# The V3.2--V3.5 release verifiers intentionally reject any change to the
# older framework lock. V3.6 deliberately changes Foundation and NativeAction,
# so those historical seals must remain untouched and are not invoked here.
# Building the cumulative facade and environment audit recompiles and imports
# the complete inherited chain under the revised action.
commands = [
    ["lake", "-Kjobs=2", "build", "LeanDirac"],
    ["lake", "-Kjobs=2", "build", "LeanDiracAudit"],
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
    "LeanDirac/V36CompleteGeometricallyNormalizedAction.lean",
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

if "v36_complete_geometrically_normalized_action" not in combined:
    fail("axiom audit did not print the revised-action cumulative theorem")
if "v36_live_research_action_mode_to_free_dirac" not in combined:
    fail("axiom audit did not print the direct live-action theorem")

print("[PASS] V3.6 geometrically normalized action verification")
sys.exit(0)

