"""Regenerate the four role-specific release manifests.

Run this only after an intentional reviewed change.  The production guard is
read-only and never rewrites these trust roots.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "project"
ACTION_ID = "DIRAC002_COMPLEX_CHI_COVARIANCE_PLUS_INCIDENCE_V2"

FRAMEWORK_FILES = [
    "LeanDirac/Foundation.lean",
    "LeanDirac/C19Geometry.lean",
    "LeanDirac/NativeAction.lean",
    "LeanDirac/ExteriorArea.lean",
    "LeanDirac/ShapeStability.lean",
    "LeanDirac/ShapeStabilityVariation.lean",
    "LeanDirac/ChiIncidence.lean",
    "LeanDirac/ChiIncidenceVariation.lean",
]

EVIDENCE_FILES = [
    "experiments/dirac009_native_coupling/dirac009_nondegenerate_stability.py",
    "experiments/dirac009_native_coupling/dirac009_nondegenerate_stability.json",
]

RELEASE_FILES = [
    ".gitattributes",
    ".github/workflows/lean.yml",
    ".github/workflows/numerical.yml",
    ".zenodo.json",
    "CHANGELOG.md",
    "CITATION.cff",
    "LICENSE",
    "README.md",
    "docs/CLAIMS.md",
    "docs/FRAMEWORK.md",
    "docs/PROOF_CHAIN.md",
    "docs/REPRODUCIBILITY.md",
    "lake-manifest.json",
    "lakefile.lean",
    "lean-toolchain",
    "project/derivation_contract.json",
    "project/no_bolton_review.json",
    "project/status.json",
    "requirements.txt",
    "scripts/adversarial_guard_test.py",
    "scripts/derivation_contract_audit.py",
    "scripts/production_guard.py",
    "scripts/update_manifests.py",
    "scripts/verify_release.py",
]


def sha256(relative: str) -> str:
    data = (ROOT / relative).read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def hashes(files: list[str]) -> dict[str, str]:
    missing = [relative for relative in files if not (ROOT / relative).is_file()]
    if missing:
        raise FileNotFoundError(f"manifest inputs are missing: {missing}")
    return {relative: sha256(relative) for relative in sorted(files)}


def write(name: str, payload: dict) -> None:
    (PROJECT / name).write_text(
        json.dumps(payload, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def command_output(command: list[str]) -> str:
    result = subprocess.run(
        command, cwd=ROOT, text=True, capture_output=True, check=True
    )
    return result.stdout.strip()


def main() -> int:
    proof_files = sorted(
        path.relative_to(ROOT).as_posix()
        for path in (ROOT / "LeanDirac").rglob("*.lean")
    ) + ["LeanDirac.lean", "LeanDiracAudit.lean"]

    write("framework_lock.json", {
        "schema_version": "2.0.0",
        "role": "native-framework",
        "action_id": ACTION_ID,
        "canonical": False,
        "native_fields": ["Psi:C3", "chi:C3"],
        "field_families": 2,
        "files": hashes(FRAMEWORK_FILES),
    })
    write("proof_manifest.json", {
        "schema_version": "2.0.0",
        "role": "formal-proof",
        "action_id": ACTION_ID,
        "publication_module": "LeanDirac/Publication.lean",
        "publication_theorem": "LeanDirac.current_highest_gate_is_dirac009",
        "highest_gate": "DIRAC-009",
        "files": hashes(proof_files),
    })

    import numpy
    import torch

    write("evidence_manifest.json", {
        "schema_version": "2.0.0",
        "role": "numerical-evidence",
        "action_id": ACTION_ID,
        "numerical_evidence": EVIDENCE_FILES[1],
        "result_origin_commit": "3284aec7db85fd0f5125d0443798373ef318339b",
        "script_origin_commit": "597d4211f2d3822b18dc7a1eead483382c4ab68b",
        "seed": 120009,
        "environment": {
            "python": platform.python_version(),
            "numpy": numpy.__version__,
            "torch": torch.__version__,
            "platform": platform.platform(),
            "device": "cuda" if torch.cuda.is_available() else "cpu",
        },
        "files": hashes(EVIDENCE_FILES),
    })
    write("release_manifest.json", {
        "schema_version": "2.0.0",
        "role": "publication-release",
        "action_id": ACTION_ID,
        "lean": command_output(["lean", "--version"]),
        "files": hashes(RELEASE_FILES),
    })
    print("UPDATED_ROLE_MANIFESTS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
