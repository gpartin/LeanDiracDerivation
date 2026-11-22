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
    "experiments/dirac011_frequency_mass/validate_dirac011.py",
    "experiments/dirac011_frequency_mass/result.json",
    "experiments/dirac012_continuum_evolution/validate_dirac012.py",
    "experiments/dirac012_continuum_evolution/result.json",
    "experiments/dirac013_density_current/validate_dirac013.py",
    "experiments/dirac013_density_current/result.json",
    "experiments/dirac016_copy_translation/validate_dirac016.py",
    "experiments/dirac016_copy_translation/result.json",
    "experiments/dirac017_spin_half/validate_dirac017.py",
    "experiments/dirac017_spin_half/result.json",
    "experiments/dirac018_factorization_classification/validate_dirac018.py",
    "experiments/dirac018_factorization_classification/result.json",
    "experiments/dirac019_wavepacket_transport/validate_dirac019.py",
    "experiments/dirac019_wavepacket_transport/result.json",
    "experiments/dirac020_coherent_spatial_transport/validate_dirac020.py",
    "experiments/dirac020_coherent_spatial_transport/result.json",
    "experiments/dirac021_rest_clock/validate_dirac021.py",
    "experiments/dirac021_rest_clock/result.json",
    "experiments/dirac022_zitterbewegung/validate_dirac022.py",
    "experiments/dirac022_zitterbewegung/result.json",
    "experiments/dirac023_phase_harmony/validate_dirac023.py",
    "experiments/dirac023_phase_harmony/result.json",
    "experiments/dirac024_projective_zitter_spin/validate_dirac024.py",
    "experiments/dirac024_projective_zitter_spin/result.json",
    "experiments/dirac025_velocity_clock/validate_dirac025.py",
    "experiments/dirac025_velocity_clock/result.json",
    "experiments/dirac026_030_free_electron/validate_dirac026_030.py",
    "experiments/dirac026_030_free_electron/result.json",
]

RELEASE_FILES = [
    ".gitattributes",
    ".github/workflows/lean.yml",
    ".github/workflows/numerical.yml",
    ".zenodo.json",
    "CHANGELOG.md",
    "CITATION.cff",
    "LICENSE",
    "LICENSE-CC0",
    "LICENSING.md",
    "README.md",
    "docs/CLAIMS.md",
    "docs/DIRAC018_RED_TEAM.md",
    "docs/FORUM_POST_DIRAC018.md",
    "docs/DIRAC019_RED_TEAM.md",
    "docs/FORUM_POST_DIRAC019.md",
    "docs/DIRAC020_RED_TEAM.md",
    "docs/FORUM_POST_DIRAC020.md",
    "docs/DIRAC021_RED_TEAM.md",
    "docs/FORUM_POST_DIRAC021.md",
    "docs/DIRAC022_PLAN.md",
    "docs/DIRAC022_RED_TEAM.md",
    "docs/FORUM_POST_DIRAC022.md",
    "docs/DIRAC023_RED_TEAM.md",
    "docs/FORUM_POST_DIRAC023.md",
    "docs/DIRAC024_RED_TEAM.md",
    "docs/FORUM_POST_DIRAC024.md",
    "docs/DIRAC025_RESEARCH_TARGET.md",
    "docs/DIRAC025_RED_TEAM.md",
    "docs/FORUM_POST_DIRAC025.md",
    "docs/DIRAC026_RESEARCH_TARGET.md",
    "docs/DIRAC026_RED_TEAM.md",
    "docs/DIRAC027_RED_TEAM.md",
    "docs/DIRAC028_RED_TEAM.md",
    "docs/DIRAC029_RED_TEAM.md",
    "docs/DIRAC030_RED_TEAM.md",
    "docs/DIRAC001_030_RED_TEAM.md",
    "docs/DIRAC_V2_COMPLETION_AUDIT.md",
    "docs/DIRAC_V2_PRODUCTION_PORT.md",
    "docs/DIRAC_V2_STAGE_MAP.md",
    "docs/FORUM_POST_DIRAC026.md",
    "docs/FORUM_POST_DIRAC027.md",
    "docs/FORUM_POST_DIRAC028.md",
    "docs/FORUM_POST_DIRAC029.md",
    "docs/FORUM_POST_DIRAC030.md",
    "docs/FRAMEWORK.md",
    "docs/PROOF_CHAIN.md",
    "docs/RELATED_WORK.md",
    "docs/REPRODUCIBILITY.md",
    "docs/SUBMISSION_TRACKER.md",
    "docs/WORLD_CLASS_AUDIT.md",
    "lake-manifest.json",
    "lakefile.lean",
    "lean-toolchain",
    "project/derivation_contract.json",
    "project/dirac010_standard_map_status.json",
    "project/dirac011_frequency_mass_status.json",
    "project/dirac012_continuum_evolution_status.json",
    "project/dirac013_density_current_status.json",
    "project/dirac014_final_audit_status.json",
    "project/dirac015_clean_reproduction_status.json",
    "project/dirac016_copy_translation_status.json",
    "project/dirac017_spin_half_representation_status.json",
    "project/dirac017_release_readiness_status.json",
    "project/dirac018_factorization_classification_status.json",
    "project/dirac018_release_readiness_status.json",
    "project/dirac019_wavepacket_transport_status.json",
    "project/dirac019_release_readiness_status.json",
    "project/dirac020_coherent_spatial_transport_status.json",
    "project/dirac020_release_readiness_status.json",
    "project/dirac021_rest_clock_status.json",
    "project/dirac021_release_readiness_status.json",
    "project/dirac022_zitterbewegung_status.json",
    "project/dirac022_release_readiness_status.json",
    "project/dirac023_phase_harmony_status.json",
    "project/dirac023_release_readiness_status.json",
    "project/dirac024_projective_zitter_spin_status.json",
    "project/dirac024_release_readiness_status.json",
    "project/dirac025_velocity_clock_status.json",
    "project/dirac025_release_readiness_status.json",
    "project/dirac026_lorentz_boost_status.json",
    "project/dirac027_pauli_lubanski_status.json",
    "project/dirac028_wigner_thomas_status.json",
    "project/dirac029_charge_conjugation_status.json",
    "project/dirac030_gordon_status.json",
    "project/dirac030_release_readiness_status.json",
    "project/dirac_v2_stage_status.json",
    "project/no_bolton_review.json",
    "project/status.json",
    "requirements.txt",
    "scripts/adversarial_guard_test.py",
    "scripts/derivation_contract_audit.py",
    "scripts/final_chain_audit.py",
    "scripts/build_publication_package.py",
    "scripts/production_guard.py",
    "scripts/publication_quality_audit.py",
    "scripts/update_manifests.py",
    "scripts/verify_release.py",
    "experiments/v2_first_principles_free_dirac/AxiomAudit.lean",
    "experiments/v2_first_principles_free_dirac/RCA.md",
    "experiments/v2_first_principles_free_dirac/REPORT.md",
    "experiments/v2_first_principles_free_dirac/artifact_audit.json",
    "experiments/v2_first_principles_free_dirac/summary.json",
    "experiments/v2_first_principles_free_dirac/verify.py",
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
        "publication_theorem": "LeanDirac.current_highest_gate_is_dirac030",
        "highest_gate": "DIRAC-030",
        "files": hashes(proof_files),
    })
    import numpy
    write("evidence_manifest.json", {
        "schema_version": "2.1.0",
        "role": "independent-evidence",
        "action_id": ACTION_ID,
        "numerical_evidence": "experiments/dirac026_030_free_electron/result.json",
        "cumulative_numerical_evidence": [
            "experiments/dirac010_standard_map/result.json",
            "experiments/dirac011_frequency_mass/result.json",
            "experiments/dirac012_continuum_evolution/result.json",
            "experiments/dirac013_density_current/result.json",
            "experiments/dirac016_copy_translation/result.json",
            "experiments/dirac017_spin_half/result.json",
            "experiments/dirac018_factorization_classification/result.json",
            "experiments/dirac019_wavepacket_transport/result.json",
            "experiments/dirac020_coherent_spatial_transport/result.json",
            "experiments/dirac021_rest_clock/result.json",
            "experiments/dirac022_zitterbewegung/result.json",
            "experiments/dirac023_phase_harmony/result.json",
            "experiments/dirac024_projective_zitter_spin/result.json",
            "experiments/dirac025_velocity_clock/result.json",
            "experiments/dirac026_030_free_electron/result.json",
        ],
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
