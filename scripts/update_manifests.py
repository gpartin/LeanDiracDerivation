"""Regenerate the role-specific release manifests after reviewed changes."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "project"
ACTION_ID = "DIRAC002_COMPLEX_CHI_BASE_C19_V3"

FRAMEWORK_FILES = [
    "LeanDirac/Foundation.lean",
    "LeanDirac/C19Geometry.lean",
    "LeanDirac/CubeGeometry.lean",
    "LeanDirac/ExteriorArea.lean",
    "LeanDirac/NativeAction.lean",
    "LeanDirac/PreparedBackground.lean",
]

EVIDENCE_FILES = [
    "experiments/dirac010_standard_map/validate_dirac010.py",
    "experiments/dirac010_standard_map/result.json",
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
    "project/dirac010_standard_map_status.json",
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
    missing = [name for name in files if not (ROOT / name).is_file()]
    if missing:
        raise FileNotFoundError(f"manifest inputs are missing: {missing}")
    return {name: sha256(name) for name in sorted(files)}


def write(name: str, payload: dict) -> None:
    (PROJECT / name).write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n"
    )


def output(command: list[str]) -> str:
    return subprocess.run(
        command, cwd=ROOT, text=True, capture_output=True, check=True
    ).stdout.strip()


def main() -> int:
    proof_files = sorted(
        path.relative_to(ROOT).as_posix()
        for path in (ROOT / "LeanDirac").rglob("*.lean")
    ) + ["LeanDirac.lean", "LeanDiracAudit.lean"]
    write("framework_lock.json", {
        "schema_version": "2.1.0",
        "role": "native-framework",
        "action_id": ACTION_ID,
        "canonical": False,
        "native_fields": ["Psi:C3", "chi:C3"],
        "field_families": 2,
        "files": hashes(FRAMEWORK_FILES),
    })
    write("proof_manifest.json", {
        "schema_version": "2.1.0",
        "role": "formal-proof",
        "action_id": ACTION_ID,
        "publication_module": "LeanDirac/Publication.lean",
        "publication_theorem": "LeanDirac.current_highest_gate_is_dirac010",
        "highest_gate": "DIRAC-010",
        "files": hashes(proof_files),
    })
    import numpy
    write("evidence_manifest.json", {
        "schema_version": "2.1.0",
        "role": "independent-evidence",
        "action_id": ACTION_ID,
        "numerical_evidence": EVIDENCE_FILES[1],
        "environment": {
            "python": platform.python_version(),
            "numpy": numpy.__version__,
            "platform": platform.platform(),
        },
        "files": hashes(EVIDENCE_FILES),
    })
    write("release_manifest.json", {
        "schema_version": "2.1.0",
        "role": "publication-release",
        "action_id": ACTION_ID,
        "lean": output(["lean", "--version"]),
        "files": hashes(RELEASE_FILES),
    })
    print("UPDATED_ROLE_MANIFESTS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
