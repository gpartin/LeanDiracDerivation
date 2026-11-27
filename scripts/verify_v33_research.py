"""Strict local verifier for the V3.3 research branch."""

from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
STAGES = [
    "V33FrameworkLock",
    "V33SpinorBasis",
    "V33ProjectedSpinors",
    "V33SpinSums",
    "V33BilinearObservables",
    "V33Helicity",
    "V33GordonIdentity",
    "V33FreePropagator",
    "V33ScopeAudit",
    "V33CompleteFreeSpinor",
]


def fail(message: str) -> None:
    print(f"[FAIL] {message}")
    raise SystemExit(1)


for name in STAGES:
    path = ROOT / "LeanDirac" / f"{name}.lean"
    if not path.is_file():
        fail(f"missing stage: {path}")
    text = path.read_text(encoding="utf-8")
    if "sorry" in text or "admit" in text:
        fail(f"unfinished proof marker in {path.name}")
    if "\naxiom " in text or "\nconstant " in text:
        fail(f"new axiom or opaque constant declaration in {path.name}")

plan = (ROOT / "docs" / "V33_FREE_SPINOR_OBSERVABLE_CLOSURE_PLAN.md").read_text(
    encoding="utf-8"
)
for required in ("Disallowed live mechanisms", "Claim boundary", "Ten meaningful handoffs"):
    if required not in plan:
        fail(f"plan omits required section: {required}")

command = ["lake", "build", "LeanDirac.V33CompleteFreeSpinor"]
print("[RUN] " + " ".join(command))
result = subprocess.run(command, cwd=ROOT)
if result.returncode != 0:
    fail("Lean cumulative build failed")

print("[PASS] V3.3 research verification")
sys.exit(0)
